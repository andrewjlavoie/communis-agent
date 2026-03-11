from __future__ import annotations

import json
import os
import re

from temporalio import activity

from prompts.riff_prompts import (
    EXTRACT_INSIGHTS_PROMPT,
    PLANNER_PROMPT,
    SUMMARIZE_ARTIFACTS_PROMPT,
    VALIDATE_FEEDBACK_PROMPT,
)

# --- Provider defaults (from env, overridable per-call) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")
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


# --- Unified LLM call ---


async def _call_llm(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
    provider: str = "",
    base_url: str = "",
) -> dict:
    """Call the configured LLM provider. Returns normalized {text, stop_reason, usage}.

    provider/base_url override module defaults when passed (e.g. from CLI flags).
    """
    provider = provider or LLM_PROVIDER
    if provider == "openai":
        return await _call_openai(messages, system_prompt, model, max_tokens, base_url)
    return await _call_anthropic(messages, system_prompt, model, max_tokens)


async def _call_anthropic(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
) -> dict:
    client = _get_anthropic_client()
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    return {
        "text": text,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


async def _call_openai(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
    base_url: str = "",
) -> dict:
    client = _get_openai_client(base_url)

    # OpenAI format: system message goes in the messages array
    oai_messages = [{"role": "system", "content": system_prompt}, *messages]

    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=oai_messages,
    )

    choice = response.choices[0]
    text = choice.message.content or ""

    # Strip <think>...</think> blocks from thinking models (Qwen3, etc.)
    text = _THINK_RE.sub("", text).strip()

    # Map finish_reason to Anthropic-style stop_reason
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    stop_reason = stop_reason_map.get(choice.finish_reason, "end_turn")

    usage = response.usage
    return {
        "text": text,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        },
    }


# --- Activities ---


@activity.defn
async def call_claude(
    messages: list[dict],
    system_prompt: str,
    model: str = "",
    max_tokens: int = 0,
    provider: str = "",
    base_url: str = "",
) -> dict:
    """Call LLM with a system prompt and message history. Returns content text, usage, and truncation info."""
    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or MAX_OUTPUT_TOKENS
    return await _call_llm(messages, system_prompt, model, max_tokens, provider, base_url)


@activity.defn
async def plan_next_turn(context: str, provider: str = "", base_url: str = "") -> dict:
    """Use LLM to decide the role and instructions for the next turn agent."""
    response = await _call_llm(
        messages=[{"role": "user", "content": context}],
        system_prompt=PLANNER_PROMPT,
        model=DEFAULT_MODEL,
        max_tokens=FAST_MAX_TOKENS or 1024,
        provider=provider,
        base_url=base_url,
    )

    text = response["text"].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
        else:
            parsed = {
                "role": "Explorer",
                "instructions": "Explore the idea broadly. Identify what it is, what makes it interesting, and what directions it could go.",
                "reasoning": "Defaulting to open exploration.",
            }

    return {
        "role": parsed.get("role", "Explorer"),
        "instructions": parsed.get("instructions", "Explore and develop the idea."),
        "reasoning": parsed.get("reasoning", ""),
    }


@activity.defn
async def extract_key_insights(content: str, provider: str = "", base_url: str = "") -> list[str]:
    """Use a fast model to extract key insights from turn content."""
    response = await _call_llm(
        messages=[{"role": "user", "content": content}],
        system_prompt=EXTRACT_INSIGHTS_PROMPT,
        model=FAST_MODEL,
        max_tokens=FAST_MAX_TOKENS or 512,
        provider=provider,
        base_url=base_url,
    )

    text = response["text"].strip()

    try:
        insights = json.loads(text)
        if isinstance(insights, list):
            return [str(i) for i in insights]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                insights = json.loads(text[start:end])
                return [str(i) for i in insights]
            except json.JSONDecodeError:
                pass

    return [text[:200]] if text else []


@activity.defn
async def summarize_artifacts(artifacts_text: str, provider: str = "", base_url: str = "") -> str:
    """Use a fast model to summarize older turn artifacts."""
    response = await _call_llm(
        messages=[{"role": "user", "content": artifacts_text}],
        system_prompt=SUMMARIZE_ARTIFACTS_PROMPT,
        model=FAST_MODEL,
        max_tokens=FAST_MAX_TOKENS or 1024,
        provider=provider,
        base_url=base_url,
    )
    return response["text"]


@activity.defn
async def validate_user_feedback(feedback: str, idea: str, provider: str = "", base_url: str = "") -> dict:
    """Use a fast model to check if user feedback is relevant."""
    response = await _call_llm(
        messages=[
            {
                "role": "user",
                "content": f"Idea being developed: {idea}\nUser feedback: {feedback}\n\nIs this feedback relevant?",
            }
        ],
        system_prompt=VALIDATE_FEEDBACK_PROMPT,
        model=FAST_MODEL,
        max_tokens=FAST_MAX_TOKENS or 256,
        provider=provider,
        base_url=base_url,
    )

    text = response["text"].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                parsed = {"relevant": True, "reason": "Could not validate, accepting feedback."}
        else:
            parsed = {"relevant": True, "reason": "Could not validate, accepting feedback."}

    return {"relevant": parsed.get("relevant", True), "reason": parsed.get("reason", "")}
