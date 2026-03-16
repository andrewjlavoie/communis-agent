"""Temporal activities for the session-based agent REPL."""

from __future__ import annotations

from temporalio import activity

from activities.llm_activities import _call_llm, _parse_llm_json, DEFAULT_MODEL, MAX_OUTPUT_TOKENS
from prompts.session_prompts import FRONT_AGENT_SYSTEM_PROMPT


@activity.defn
async def front_agent_respond(
    conversation: list[dict],
    active_tasks: dict[str, dict],
    model: str = "",
    provider: str = "",
    base_url: str = "",
) -> dict:
    """Front agent LLM: understand intent, respond conversationally, optionally delegate.

    Returns: {text: str, delegate_tasks: [{description: str, context: str}]}
    """
    # Format active tasks for the system prompt
    if active_tasks:
        task_lines = []
        for task_id, task_info in active_tasks.items():
            status = task_info.get("status", "unknown")
            desc = task_info.get("description", "")
            progress = task_info.get("progress", "")
            line = f"- [{task_id}] {desc} (status: {status})"
            if progress:
                line += f" — {progress}"
            task_lines.append(line)
        tasks_text = "\n".join(task_lines)
    else:
        tasks_text = "No active tasks."

    system_prompt = FRONT_AGENT_SYSTEM_PROMPT.format(active_tasks=tasks_text)

    # Build messages for LLM — use conversation history directly
    messages = []
    for msg in conversation:
        messages.append({"role": msg["role"], "content": msg["content"]})

    response = await _call_llm(
        messages=messages,
        system_prompt=system_prompt,
        model=model or DEFAULT_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        provider=provider,
        base_url=base_url,
    )

    default: dict = {"text": response["text"], "delegate_tasks": []}
    parsed = _parse_llm_json(response["text"], default)
    assert isinstance(parsed, dict)

    return {
        "text": parsed.get("text", response["text"]),
        "delegate_tasks": parsed.get("delegate_tasks", []),
    }
