from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.llm_activities import call_claude, extract_key_insights
    from activities.workspace_activities import read_turn_context, write_turn_artifact
    from models.data_types import TurnConfig, TurnResult
    from prompts.riff_prompts import TURN_AGENT_PROMPT

LLM_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)

# Timeouts — LLM calls can take minutes for long context; file I/O is fast
LLM_TIMEOUT = timedelta(minutes=30)
FAST_LLM_TIMEOUT = timedelta(minutes=10)
IO_TIMEOUT = timedelta(seconds=30)


def _build_user_message(config: TurnConfig, ws_context: dict) -> str:
    """Build the user message from turn config and workspace context."""
    parts = [f"## Prompt\n{config.idea}"]

    parts.append(
        f"## Position in Chain\n"
        f"You are turn **{config.turn_number} of {config.total_turns}**. "
        + (
            "You are the FINAL agent — synthesize everything into a polished deliverable."
            if config.turn_number == config.total_turns
            else f"There {'is 1 more agent' if config.total_turns - config.turn_number == 1 else f'are {config.total_turns - config.turn_number} more agents'} after you."
        )
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


@workflow.defn
class RiffTurnWorkflow:
    """Child workflow: executes a single riff turn with a dynamically assigned role."""

    @workflow.run
    async def run(self, config: TurnConfig) -> TurnResult:
        # Read context from workspace files
        ws_context = await workflow.execute_activity(
            read_turn_context,
            args=[config.workspace_dir, config.turn_number],
            start_to_close_timeout=IO_TIMEOUT,
        )

        user_message = _build_user_message(config, ws_context)

        system_prompt = TURN_AGENT_PROMPT.format(
            role=config.role,
            instructions=config.instructions,
        )

        llm_response = await workflow.execute_activity(
            call_claude,
            args=[
                [{"role": "user", "content": user_message}],
                system_prompt,
                config.model,
                config.max_tokens,
                config.provider,
                config.base_url,
            ],
            start_to_close_timeout=LLM_TIMEOUT,
            retry_policy=LLM_RETRY_POLICY,
        )

        content = llm_response["text"]
        usage = llm_response["usage"]
        truncated = llm_response.get("stop_reason") == "max_tokens"

        insights = await workflow.execute_activity(
            extract_key_insights,
            args=[content, config.provider, config.base_url],
            start_to_close_timeout=FAST_LLM_TIMEOUT,
            retry_policy=LLM_RETRY_POLICY,
        )

        # Write output to workspace
        artifact_path = await workflow.execute_activity(
            write_turn_artifact,
            args=[config.workspace_dir, config.turn_number, config.role, content, insights, usage, truncated],
            start_to_close_timeout=IO_TIMEOUT,
        )

        return TurnResult(
            turn_number=config.turn_number,
            role=config.role,
            key_insights=insights,
            token_usage=usage,
            truncated=truncated,
            artifact_path=artifact_path,
        )
