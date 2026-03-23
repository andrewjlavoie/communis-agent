from __future__ import annotations

import json
import os
import re

import litellm
from temporalio import activity

from shared.constants import DEFAULT_MODEL_STRING

from prompts.communis_prompts import (
    EXTRACT_INSIGHTS_PROMPT,
    PLANNER_PROMPT,
    SUMMARIZE_ARTIFACTS_PROMPT,
    SUMMARIZE_SUBCOMMUNIS_RESULTS_PROMPT,
    VALIDATE_FEEDBACK_PROMPT,
)

# --- Provider defaults (from env, overridable per-call) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", DEFAULT_MODEL_STRING)
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-4-5-20251001")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16384"))

# Override for fast/utility LLM calls (insights, summaries, validation, planning).
# Thinking models (e.g. Qwen3) need higher limits because reasoning tokens consume
# the budget before output is produced. Set to 4096+ for thinking models.
FAST_MAX_TOKENS = int(os.getenv("FAST_MAX_TOKENS", "0"))

# Regex to strip <think>...</think> blocks from thinking-model output
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


# --- Message / tool format conversion ---


def _convert_messages_to_openai(messages: list[dict]) -> list[dict]:
    """Convert Anthropic-format messages to OpenAI chat format.

    Handles:
    - Assistant messages with tool_use content blocks → assistant with tool_calls
    - User messages with tool_result content blocks → role=tool messages
    - Plain text messages → passed through as-is
    """
    oai_messages: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
        elif isinstance(content, list) and role == "assistant":
            # Anthropic-style assistant message with content blocks
            text_parts = []
            tool_calls = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            oai_msg: dict = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            oai_messages.append(oai_msg)
        elif isinstance(content, list) and role == "user":
            # Anthropic-style tool_result blocks → OpenAI tool messages
            for block in content:
                if block.get("type") == "tool_result":
                    oai_messages.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block.get("content", ""),
                    })
                else:
                    # Fallback: stringify non-tool content blocks
                    text = block.get("text", str(block))
                    oai_messages.append({"role": "user", "content": text})
        else:
            oai_messages.append(msg)
    return oai_messages


# --- LiteLLM helpers ---


def _build_litellm_model(
    provider: str, model: str, base_url: str
) -> tuple[str, dict]:
    """Return (litellm_model_string, extra_kwargs) for the given provider.

    LiteLLM uses a "provider/model" prefix convention to route to the right backend.
    """
    extra: dict = {}

    if provider == "openai":
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
        # For custom OpenAI-compatible endpoints, use the openai/ prefix
        model_str = f"openai/{model}" if not model.startswith("openai/") else model
        extra["api_base"] = base_url
        extra["api_key"] = api_key
    elif provider == "gemini":
        model_str = f"gemini/{model}" if not model.startswith("gemini/") else model
    else:
        # Default: Anthropic
        model_str = f"anthropic/{model}" if not model.startswith("anthropic/") else model

    return model_str, extra


def _normalize_litellm_response(response) -> dict:
    """Convert a litellm response (OpenAI format) to our internal dict format.

    Our internal format uses Anthropic-style keys:
    - text: concatenated text content
    - reasoning: reasoning/thinking content (if any)
    - stop_reason: "end_turn", "tool_use", or "max_tokens"
    - content_blocks: list of {type: "text"} and {type: "tool_use"} blocks
    - usage: {input_tokens, output_tokens}
    """
    choice = response.choices[0]
    text = choice.message.content or ""

    # Capture reasoning_content from thinking models (Qwen3, etc.)
    reasoning = ""
    if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
        reasoning = choice.message.reasoning_content.strip()

    # Strip <think>...</think> blocks from thinking models
    text = _THINK_RE.sub("", text).strip()

    # If content is empty but reasoning exists, salvage it so downstream
    # activities don't receive an empty string.
    if not text and reasoning:
        text = reasoning

    # Map finish_reason to Anthropic-style stop_reason
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
        "tool_calls": "tool_use",
    }
    stop_reason = stop_reason_map.get(choice.finish_reason, "end_turn")

    # Build content_blocks in Anthropic format for consistency
    content_blocks: list[dict] = []
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Convert OpenAI tool_calls to Anthropic-style tool_use blocks
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments) if tc.function.arguments else {},
            })

    usage = response.usage
    return {
        "text": text,
        "reasoning": reasoning,
        "stop_reason": stop_reason,
        "content_blocks": content_blocks,
        "usage": {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        },
    }


# --- Unified LLM call ---


async def _call_llm(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
    provider: str = "",
    base_url: str = "",
    tools: list[dict] | None = None,
) -> dict:
    """Call the configured LLM provider via LiteLLM. Returns normalized response dict.

    When tools are provided, response includes 'content_blocks' with full structured
    data (text blocks + tool_use blocks). The 'text' field always contains concatenated
    text content for backward compatibility.

    provider/base_url override module defaults when passed (e.g. from CLI flags).
    """
    provider = (provider or LLM_PROVIDER).strip().lower()
    model_str, extra = _build_litellm_model(provider, model, base_url)

    # Convert Anthropic-format messages to OpenAI format, then prepend system message
    oai_messages = [{"role": "system", "content": system_prompt}] + _convert_messages_to_openai(messages)

    kwargs: dict = {
        "model": model_str,
        "max_tokens": max_tokens,
        "messages": oai_messages,
        **extra,
    }
    if tools:
        # Convert Anthropic-style tool defs to OpenAI format
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    response = await litellm.acompletion(**kwargs)
    return _normalize_litellm_response(response)


def _parse_llm_json(text: str, default: dict | list) -> dict | list:
    """Parse JSON from LLM output, with fallback extraction and default."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        bracket = "{" if isinstance(default, dict) else "["
        close = "}" if isinstance(default, dict) else "]"
        start = text.find(bracket)
        end = text.rfind(close) + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return default
        return default


# --- Activities ---


@activity.defn
async def call_llm(
    messages: list[dict],
    system_prompt: str,
    model: str = "",
    max_tokens: int = 0,
    provider: str = "",
    base_url: str = "",
    tools: list[dict] | None = None,
) -> dict:
    """Call LLM with a system prompt and message history.

    Returns content text, usage, and truncation info. When tools are provided,
    also returns content_blocks with full structured data (text + tool_use blocks).
    """
    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or MAX_OUTPUT_TOKENS
    return await _call_llm(messages, system_prompt, model, max_tokens, provider, base_url, tools)


@activity.defn
async def plan_next_turn(context: str, provider: str = "", base_url: str = "", model: str = "") -> dict:
    """Use LLM to decide the role and instructions for the next turn agent.

    Returns dict with keys: role, instructions, reasoning, goal_complete, action, subcommunis, plan_summary.
    """
    response = await _call_llm(
        messages=[{"role": "user", "content": context}],
        system_prompt=PLANNER_PROMPT,
        model=model or DEFAULT_MODEL,
        max_tokens=FAST_MAX_TOKENS or 2048,
        provider=provider,
        base_url=base_url,
    )

    parsed = _parse_llm_json(response["text"], {
        "role": "Explorer",
        "instructions": "Explore the idea broadly. Identify what it is, what makes it interesting, and what directions it could go.",
        "reasoning": "Defaulting to open exploration.",
    })

    return {
        "role": parsed.get("role", "Explorer"),
        "instructions": parsed.get("instructions", "Explore and develop the idea."),
        "reasoning": parsed.get("reasoning", ""),
        "goal_complete": parsed.get("goal_complete", False),
        "action": parsed.get("action", "step"),
        "subcommunis": parsed.get("subcommunis", []),
        "plan_summary": parsed.get("plan_summary", ""),
    }


@activity.defn
async def extract_key_insights(content: str, provider: str = "", base_url: str = "", model: str = "") -> list[str]:
    """Use a fast model to extract key insights from turn content."""
    if not content or not content.strip():
        return ["(no content produced — model may have exhausted token budget on reasoning)"]

    response = await _call_llm(
        messages=[{"role": "user", "content": content}],
        system_prompt=EXTRACT_INSIGHTS_PROMPT,
        model=model or FAST_MODEL,
        max_tokens=FAST_MAX_TOKENS or 512,
        provider=provider,
        base_url=base_url,
    )

    text = response["text"].strip()
    parsed = _parse_llm_json(text, [])
    if isinstance(parsed, list) and parsed:
        return [str(i) for i in parsed]
    return [text[:200]] if text else []


@activity.defn
async def summarize_artifacts(artifacts_text: str, provider: str = "", base_url: str = "", model: str = "") -> str:
    """Use a fast model to summarize older turn artifacts."""
    if not artifacts_text or not artifacts_text.strip():
        return "(no content to summarize)"

    response = await _call_llm(
        messages=[{"role": "user", "content": artifacts_text}],
        system_prompt=SUMMARIZE_ARTIFACTS_PROMPT,
        model=model or FAST_MODEL,
        max_tokens=FAST_MAX_TOKENS or 1024,
        provider=provider,
        base_url=base_url,
    )
    return response["text"]


@activity.defn
async def validate_user_feedback(feedback: str, idea: str, provider: str = "", base_url: str = "", model: str = "") -> dict:
    """Use a fast model to check if user feedback is relevant."""
    response = await _call_llm(
        messages=[
            {
                "role": "user",
                "content": f"Idea being developed: {idea}\nUser feedback: {feedback}\n\nIs this feedback relevant?",
            }
        ],
        system_prompt=VALIDATE_FEEDBACK_PROMPT,
        model=model or FAST_MODEL,
        max_tokens=FAST_MAX_TOKENS or 256,
        provider=provider,
        base_url=base_url,
    )

    parsed = _parse_llm_json(response["text"], {"relevant": True, "reason": "Could not validate, accepting feedback."})

    return {"relevant": parsed.get("relevant", True), "reason": parsed.get("reason", "")}


@activity.defn
async def summarize_subcommunis_results(
    results_text: str, goal_context: str, provider: str = "", base_url: str = "", model: str = ""
) -> str:
    """Summarize subcommunis results for injection into parent context."""
    response = await _call_llm(
        messages=[
            {
                "role": "user",
                "content": f"Parent goal: {goal_context}\n\nSubcommunis results:\n{results_text}",
            }
        ],
        system_prompt=SUMMARIZE_SUBCOMMUNIS_RESULTS_PROMPT,
        model=model or FAST_MODEL,
        max_tokens=FAST_MAX_TOKENS or 1024,
        provider=provider,
        base_url=base_url,
    )
    return response["text"]
