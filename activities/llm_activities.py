from __future__ import annotations

import json
import os
import re
import uuid

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

# --- Anthropic client ---
_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic

        _anthropic_client = AsyncAnthropic()
    return _anthropic_client


# --- OpenAI-compatible clients (keyed by base_url) ---
_openai_clients: dict[str, object] = {}


def _get_openai_client(base_url: str = ""):
    global _openai_clients
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    if base_url not in _openai_clients:
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
        _openai_clients[base_url] = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return _openai_clients[base_url]


# --- Gemini client ---
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))
    return _gemini_client


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
    """Call the configured LLM provider. Returns normalized {text, stop_reason, usage}.

    When tools are provided, response includes 'content_blocks' with full structured
    data (text blocks + tool_use blocks). The 'text' field always contains concatenated
    text content for backward compatibility.

    provider/base_url override module defaults when passed (e.g. from CLI flags).
    """
    provider = (provider or LLM_PROVIDER).strip().lower()
    if provider == "openai":
        return await _call_openai(messages, system_prompt, model, max_tokens, base_url, tools)
    if provider == "gemini":
        return await _call_gemini(messages, system_prompt, model, max_tokens, tools)
    return await _call_anthropic(messages, system_prompt, model, max_tokens, tools)


async def _call_anthropic(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
    tools: list[dict] | None = None,
) -> dict:
    client = _get_anthropic_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    response = await client.messages.create(**kwargs)

    text = ""
    reasoning = ""
    content_blocks = []
    for block in response.content:
        block_dict = block.model_dump()
        content_blocks.append(block_dict)
        if block.type == "text":
            text += block.text
        elif block.type == "thinking":
            thinking_text = getattr(block, "thinking", "")
            if thinking_text:
                reasoning += thinking_text

    return {
        "text": text,
        "stop_reason": response.stop_reason,
        "content_blocks": content_blocks,
        "reasoning": reasoning,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


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


async def _call_openai(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
    base_url: str = "",
    tools: list[dict] | None = None,
) -> dict:
    client = _get_openai_client(base_url)

    # Convert Anthropic-format messages to OpenAI format, then prepend system message
    oai_messages = [{"role": "system", "content": system_prompt}] + _convert_messages_to_openai(messages)

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": oai_messages,
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

    response = await client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    text = choice.message.content or ""

    # Always capture reasoning_content before stripping think tags
    reasoning = ""
    if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
        reasoning = choice.message.reasoning_content.strip()

    # Strip <think>...</think> blocks from thinking models (Qwen3, etc.)
    text = _THINK_RE.sub("", text).strip()

    # Thinking models (Qwen3, etc.) may exhaust the token budget on reasoning,
    # leaving content empty. If reasoning_content exists, salvage it so
    # downstream activities don't receive an empty string.
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
        "stop_reason": stop_reason,
        "content_blocks": content_blocks,
        "reasoning": reasoning,
        "usage": {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        },
    }


def _find_tool_name_for_id(messages: list[dict], tool_use_id: str) -> str:
    """Scan messages backwards to find the tool name for a given tool_use_id.

    Needed because Gemini's Part.from_function_response() requires the function
    name, but our internal tool_result format only carries tool_use_id.
    """
    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return block.get("name", "unknown")
    return "unknown"


def _convert_messages_to_gemini(messages: list[dict]) -> list:
    """Convert internal Anthropic-format messages to Gemini Content objects.

    Mapping:
    - role="assistant" → role="model"
    - tool_use blocks → Part.from_function_call()
    - tool_result blocks → Part.from_function_response()
    - Plain text → Part.from_text()
    """
    from google.genai import types

    contents: list = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        gemini_role = "model" if role == "assistant" else "user"

        if isinstance(content, str):
            contents.append(
                types.Content(role=gemini_role, parts=[types.Part.from_text(text=content)])
            )
        elif isinstance(content, list):
            parts = []
            for block in content:
                block_type = block.get("type", "")
                if block_type == "text":
                    parts.append(types.Part.from_text(text=block.get("text", "")))
                elif block_type == "tool_use":
                    parts.append(
                        types.Part.from_function_call(
                            name=block["name"],
                            args=block.get("input", {}),
                        )
                    )
                elif block_type == "tool_result":
                    tool_name = _find_tool_name_for_id(messages, block["tool_use_id"])
                    result_content = block.get("content", "")
                    parts.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result_content},
                        )
                    )
            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))
    return contents


def _convert_tools_to_gemini(tools: list[dict]):
    """Convert internal tool definitions to Gemini Tool format.

    Maps input_schema → parameters (same JSON Schema format).
    """
    from google.genai import types

    declarations = []
    for t in tools:
        declarations.append(
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("input_schema", {}),
            )
        )
    return types.Tool(function_declarations=declarations)


async def _call_gemini(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
    tools: list[dict] | None = None,
) -> dict:
    """Call Google Gemini API and return normalized response."""
    from google.genai import types

    client = _get_gemini_client()

    contents = _convert_messages_to_gemini(messages)

    config_kwargs: dict = {
        "system_instruction": system_prompt,
        "max_output_tokens": max_tokens,
    }
    if tools:
        config_kwargs["tools"] = [_convert_tools_to_gemini(tools)]
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True,
        )

    config = types.GenerateContentConfig(**config_kwargs)

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    # Normalize response
    text = ""
    reasoning = ""
    content_blocks: list[dict] = []

    if response.candidates and response.candidates[0].content:
        reasoning_parts: list[str] = []
        for part in response.candidates[0].content.parts:
            if part.function_call:
                call_id = f"gemini_{uuid.uuid4().hex[:12]}"
                # Convert args from proto MapComposite to a plain dict
                args = dict(part.function_call.args) if part.function_call.args else {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": call_id,
                    "name": part.function_call.name,
                    "input": args,
                })
            elif part.text:
                # Capture <think>...</think> inner content before stripping
                think_matches = re.findall(r"<think>(.*?)</think>", part.text, re.DOTALL)
                reasoning_parts.extend(m.strip() for m in think_matches)
                cleaned = _THINK_RE.sub("", part.text).strip()
                if cleaned:
                    text += cleaned
                    content_blocks.append({"type": "text", "text": cleaned})
        reasoning = "\n".join(reasoning_parts).strip()

    # Determine stop reason
    has_tool_calls = any(b["type"] == "tool_use" for b in content_blocks)
    if has_tool_calls:
        stop_reason = "tool_use"
    elif response.candidates:
        finish = response.candidates[0].finish_reason
        # finish_reason is an enum; compare by name
        finish_name = finish.name if hasattr(finish, "name") else str(finish)
        stop_reason_map = {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
        }
        stop_reason = stop_reason_map.get(finish_name, "end_turn")
    else:
        stop_reason = "end_turn"

    # Usage
    usage_meta = response.usage_metadata
    usage = {
        "input_tokens": usage_meta.prompt_token_count if usage_meta else 0,
        "output_tokens": usage_meta.candidates_token_count if usage_meta else 0,
    }

    return {
        "text": text,
        "stop_reason": stop_reason,
        "content_blocks": content_blocks,
        "reasoning": reasoning,
        "usage": usage,
    }


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
