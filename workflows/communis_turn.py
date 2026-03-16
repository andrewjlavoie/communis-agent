from __future__ import annotations

import asyncio

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.llm_activities import call_llm, extract_key_insights
    from activities.tool_activities import execute_run_command
    from activities.workspace_activities import read_turn_context, write_turn_artifact
    from models.data_types import TurnConfig, TurnResult
    from prompts.communis_prompts import TURN_AGENT_PROMPT_WITH_TOOLS
    from tools.run_tool import RUN_TOOL_DEFINITION
    from workflows.constants import (
        FAST_LLM_TIMEOUT,
        IO_TIMEOUT,
        LLM_RETRY_POLICY,
        LLM_TIMEOUT,
        TOOL_TIMEOUT,
    )

# Safety limit on tool call iterations per turn
MAX_TOOL_ITERATIONS = 20


def _build_user_message(config: TurnConfig, ws_context: dict) -> str:
    """Build the user message from turn config and workspace context."""
    parts = [f"## Goal\n{config.idea}"]

    if config.max_turns > 0:
        parts.append(
            f"## Position\n"
            f"You are on step **{config.turn_number}** (max {config.max_turns})."
        )
    else:
        parts.append(
            f"## Position\n"
            f"You are on step **{config.turn_number}**."
        )

    if ws_context["summary"]:
        parts.append(f"## Summary of Earlier Work\n{ws_context['summary']}")

    if ws_context["recent_turns"]:
        recent_text = []
        for turn_data in ws_context["recent_turns"]:
            insights_str = ", ".join(turn_data.get("key_insights", []))
            recent_text.append(
                f"### Turn {turn_data['turn']} — {turn_data['role']}\n"
                f"{turn_data['content']}\n"
                f"Key insights: {insights_str}"
            )
        parts.append("## Recent Work\n\n" + "\n\n".join(recent_text))

    if config.user_feedback:
        parts.append(f"## User Feedback\n{config.user_feedback}")

    parts.append("## Your Task\nProduce your work now.")

    return "\n\n".join(parts)


def _extract_text_from_blocks(content_blocks: list[dict]) -> str:
    """Extract concatenated text from content blocks."""
    return "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )


def _extract_tool_uses(content_blocks: list[dict]) -> list[dict]:
    """Extract tool_use blocks from content blocks."""
    return [block for block in content_blocks if block.get("type") == "tool_use"]


@workflow.defn
class CommunisTurnWorkflow:
    """Child workflow: executes a single riff turn with a dynamically assigned role.

    When tools are available, runs an agent loop: LLM call -> tool execution -> repeat.
    Supports human-in-the-loop approval for tool calls (unless dangerous mode is on).
    """

    def __init__(self):
        self.pending_tool: dict | None = None  # {"command": str, "tool_use_id": str}
        self.tool_approved: bool | None = None
        self.tool_approval_event = asyncio.Event()

    @workflow.signal
    async def approve_tool(self, approved: bool):
        """Signal from CLI to approve or deny a pending tool call."""
        self.tool_approved = approved
        self.tool_approval_event.set()

    @workflow.query
    def get_pending_tool(self) -> dict | None:
        """Query handler: returns pending tool call awaiting approval, or None."""
        return self.pending_tool

    @workflow.run
    async def run(self, config: TurnConfig) -> TurnResult:
        # Read context from workspace files
        ws_context = await workflow.execute_activity(
            read_turn_context,
            args=[config.workspace_dir, config.turn_number],
            start_to_close_timeout=IO_TIMEOUT,
        )

        user_message = _build_user_message(config, ws_context)

        # Use tool-aware prompt and provide tool definitions
        system_prompt = TURN_AGENT_PROMPT_WITH_TOOLS.format(
            role=config.role,
            instructions=config.instructions,
        )
        tools = [RUN_TOOL_DEFINITION]
        messages: list[dict] = [{"role": "user", "content": user_message}]

        # Agent loop: call LLM, handle tool use, repeat
        tool_calls_made = 0
        total_usage = {"input_tokens": 0, "output_tokens": 0}
        final_text_parts: list[str] = []
        truncated = False
        iteration = 0

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            llm_response = await workflow.execute_activity(
                call_llm,
                args=[
                    messages,
                    system_prompt,
                    config.model,
                    config.max_tokens,
                    config.provider,
                    config.base_url,
                    tools,
                ],
                start_to_close_timeout=LLM_TIMEOUT,
                retry_policy=LLM_RETRY_POLICY,
            )

            # Accumulate token usage
            usage = llm_response.get("usage", {})
            total_usage["input_tokens"] += usage.get("input_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)

            content_blocks = llm_response.get("content_blocks", [])
            stop_reason = llm_response.get("stop_reason", "end_turn")

            if stop_reason == "max_tokens":
                truncated = True

            # Collect text content from this response
            text = _extract_text_from_blocks(content_blocks)
            if text:
                final_text_parts.append(text)

            tool_uses = _extract_tool_uses(content_blocks)

            if not tool_uses:
                # No tool calls — LLM is done. Add assistant message and break.
                messages.append({"role": "assistant", "content": content_blocks})
                break

            # LLM wants to use tools — process each tool_use block
            messages.append({"role": "assistant", "content": content_blocks})

            tool_results: list[dict] = []
            for tool_use in tool_uses:
                tool_name = tool_use.get("name", "")
                tool_input = tool_use.get("input", {})
                tool_use_id = tool_use.get("id", "")

                if tool_name != "run":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"[error] unknown tool: {tool_name}. Available: run",
                    })
                    continue

                command = tool_input.get("command", "")
                if not command:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "[error] run: missing required parameter 'command'",
                    })
                    continue

                # Human-in-the-loop approval (unless dangerous mode)
                if not config.dangerous:
                    self.pending_tool = {
                        "command": command,
                        "tool_use_id": tool_use_id,
                    }
                    self.tool_approved = None
                    self.tool_approval_event = asyncio.Event()

                    # Wait for approval signal from CLI
                    await workflow.wait_condition(
                        lambda: self.tool_approved is not None
                    )

                    approved = self.tool_approved
                    self.pending_tool = None
                    self.tool_approved = None

                    if not approved:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "[denied] Command execution denied by user.",
                        })
                        continue

                # Execute the command
                exec_result = await workflow.execute_activity(
                    execute_run_command,
                    args=[command, config.workspace_dir],
                    start_to_close_timeout=TOOL_TIMEOUT,
                    retry_policy=LLM_RETRY_POLICY,
                )

                tool_calls_made += 1
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": exec_result["output"],
                })

            # Add all tool results as a single user message
            messages.append({"role": "user", "content": tool_results})

        # Combine all text output from the agent loop
        content = "\n\n".join(final_text_parts) if final_text_parts else ""

        insights = await workflow.execute_activity(
            extract_key_insights,
            args=[content, config.provider, config.base_url, config.model],
            start_to_close_timeout=FAST_LLM_TIMEOUT,
            retry_policy=LLM_RETRY_POLICY,
        )

        # Write output to workspace
        artifact_path = await workflow.execute_activity(
            write_turn_artifact,
            args=[config.workspace_dir, config.turn_number, config.role, content, insights, total_usage, truncated],
            start_to_close_timeout=IO_TIMEOUT,
        )

        return TurnResult(
            turn_number=config.turn_number,
            role=config.role,
            key_insights=insights,
            token_usage=total_usage,
            truncated=truncated,
            artifact_path=artifact_path,
            tool_calls_made=tool_calls_made,
        )
