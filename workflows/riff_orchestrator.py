from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import is_cancelled_exception

with workflow.unsafe.imports_passed_through():
    from activities.llm_activities import (
        plan_next_turn,
        summarize_artifacts,
        validate_user_feedback,
    )
    from activities.workspace_activities import (
        collect_older_turns_text,
        init_workspace,
        read_turn_context,
        write_workspace_summary,
    )
    from models.data_types import RiffConfig, RiffState, TurnConfig, TurnResult
    from prompts.riff_prompts import FINAL_TURN_PLANNER_ADDENDUM
    from workflows.riff_turn import RiffTurnWorkflow

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

# Summarize older turns once we have more than this many
MAX_RECENT_ARTIFACTS = 3


@workflow.defn
class RiffOrchestratorWorkflow:
    """Parent workflow: orchestrates iterative riff turns toward idea development."""

    def __init__(self):
        self.state = RiffState()
        self.user_feedback: str | None = None
        self.feedback_skipped: bool = False
        self.feedback_event = asyncio.Event()

    # --- Signals ---
    @workflow.signal
    async def receive_user_feedback(self, feedback: str):
        self.user_feedback = feedback
        self.feedback_event.set()

    @workflow.signal
    async def skip_feedback(self):
        self.feedback_skipped = True
        self.feedback_event.set()

    # --- Queries ---
    @workflow.query
    def get_state(self) -> dict:
        return self.state.to_dict()

    @workflow.query
    def get_turn_result(self, turn_number: int) -> dict | None:
        for r in self.state.turn_results:
            if r.turn_number == turn_number:
                return {
                    "turn_number": r.turn_number,
                    "role": r.role,
                    "key_insights": r.key_insights,
                    "token_usage": r.token_usage,
                    "artifact_path": r.artifact_path,
                }
        return None

    @workflow.query
    def get_all_results(self) -> list[dict]:
        return [
            {
                "turn_number": r.turn_number,
                "role": r.role,
                "key_insights": r.key_insights,
                "token_usage": r.token_usage,
                "artifact_path": r.artifact_path,
            }
            for r in self.state.turn_results
        ]

    # --- Main workflow ---
    @workflow.run
    async def run(self, config: RiffConfig) -> dict:
        self.state.idea = config.idea
        self.state.num_turns = config.num_turns
        self.state.status = "running"

        try:
            await self._run_turns(config)
        except (asyncio.CancelledError, Exception) as err:
            if isinstance(err, asyncio.CancelledError) or is_cancelled_exception(err):
                turns_done = len(self.state.turn_results)
                self.state.status = "cancelled"
                self.state.latest_message = (
                    f"Cancelled after {turns_done}/{config.num_turns} turns."
                )
                return self.state.to_dict()
            raise

        self.state.status = "complete"
        self.state.latest_message = "All turns complete!"

        return self.state.to_dict()

    async def _run_turns(self, config: RiffConfig) -> None:
        """Execute the turn loop. Raises CancelledError if workflow is cancelled."""
        # Initialize workspace
        workspace_dir = await workflow.execute_activity(
            init_workspace,
            args=[workflow.info().workflow_id, config.idea, config.num_turns, config.model],
            start_to_close_timeout=IO_TIMEOUT,
        )
        self.state.workspace_dir = workspace_dir

        for turn_number in range(1, config.num_turns + 1):
            self.state.current_turn = turn_number
            is_final = turn_number == config.num_turns

            # Read workspace context for planner
            ws_context = await workflow.execute_activity(
                read_turn_context,
                args=[workspace_dir, turn_number],
                start_to_close_timeout=IO_TIMEOUT,
            )

            # Ask the planner what the next agent should do
            plan_context = self._build_planner_context(config, ws_context, is_final)
            plan = await workflow.execute_activity(
                plan_next_turn,
                args=[plan_context, config.provider, config.base_url],
                start_to_close_timeout=FAST_LLM_TIMEOUT,
                retry_policy=LLM_RETRY_POLICY,
            )

            role = plan["role"]
            instructions = plan["instructions"]
            self.state.current_role = role
            self.state.latest_message = (
                f"Turn {turn_number}/{config.num_turns}: {role} "
                f"\u2014 {plan['reasoning']}"
            )

            # Build turn config — lightweight, just workspace path + metadata
            turn_config = TurnConfig(
                workspace_dir=workspace_dir,
                idea=config.idea,
                role=role,
                instructions=instructions,
                turn_number=turn_number,
                total_turns=config.num_turns,
                user_feedback=self.user_feedback or "",
                model=config.model,
                provider=config.provider,
                base_url=config.base_url,
            )

            # Execute child workflow
            result: TurnResult = await workflow.execute_child_workflow(
                RiffTurnWorkflow.run,
                turn_config,
                id=f"{workflow.info().workflow_id}-turn-{turn_number}",
            )

            self.state.turn_results.append(result)
            self.state.latest_message = (
                f"Turn {turn_number}/{config.num_turns} ({role}) complete."
            )

            # Update rolling summary if enough turns have accumulated
            if len(self.state.turn_results) > MAX_RECENT_ARTIFACTS + 1:
                # Read older turns from files and summarize
                threshold_turn = turn_number - MAX_RECENT_ARTIFACTS + 1
                older_text = await workflow.execute_activity(
                    collect_older_turns_text,
                    args=[workspace_dir, threshold_turn],
                    start_to_close_timeout=IO_TIMEOUT,
                )
                summary = await workflow.execute_activity(
                    summarize_artifacts,
                    args=[older_text, config.provider, config.base_url],
                    start_to_close_timeout=FAST_LLM_TIMEOUT,
                    retry_policy=LLM_RETRY_POLICY,
                )
                await workflow.execute_activity(
                    write_workspace_summary,
                    args=[workspace_dir, summary],
                    start_to_close_timeout=IO_TIMEOUT,
                )

            # Wait for user feedback between turns (unless final or auto mode)
            if not is_final and not config.auto:
                self.state.status = "waiting_for_feedback"
                self.user_feedback = None
                self.feedback_skipped = False
                self.feedback_event = asyncio.Event()

                timed_out = False
                try:
                    await workflow.wait_condition(
                        lambda: self.feedback_event.is_set(),
                        timeout=timedelta(seconds=120),
                    )
                except asyncio.TimeoutError:
                    timed_out = True

                if timed_out or self.feedback_skipped:
                    self.user_feedback = ""
                    self.state.latest_message = "No feedback received, continuing..."
                elif self.user_feedback:
                    validation = await workflow.execute_activity(
                        validate_user_feedback,
                        args=[self.user_feedback, config.idea, config.provider, config.base_url],
                        start_to_close_timeout=FAST_LLM_TIMEOUT,
                        retry_policy=LLM_RETRY_POLICY,
                    )
                    if not validation["relevant"]:
                        self.state.latest_message = (
                            f"Feedback noted but may not be relevant: {validation['reason']}. "
                            "Continuing with it anyway."
                        )

                self.state.status = "running"

    def _build_planner_context(
        self, config: RiffConfig, ws_context: dict, is_final: bool
    ) -> str:
        """Build context string for the planner from workspace data."""
        parts = [f"Prompt: {config.idea}"]

        if ws_context["summary"]:
            parts.append(f"Summary of earlier work:\n{ws_context['summary']}")

        for turn_data in ws_context["recent_turns"]:
            insights = turn_data.get("key_insights", [])
            parts.append(
                f"Turn {turn_data['turn']} (role: {turn_data['role']}) insights: "
                + "; ".join(insights)
            )

        if self.user_feedback:
            parts.append(f"Latest user feedback: {self.user_feedback}")

        parts.append(
            f"This will be turn {self.state.current_turn} of {config.num_turns}. "
            f"Turns completed so far: {len(self.state.turn_results)}. "
            f"Remaining after this one: {config.num_turns - self.state.current_turn}."
        )

        if is_final:
            parts.append(
                FINAL_TURN_PLANNER_ADDENDUM.format(
                    turn=self.state.current_turn,
                    total=config.num_turns,
                )
            )

        return "\n\n".join(parts)
