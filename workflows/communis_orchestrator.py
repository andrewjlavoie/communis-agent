from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.exceptions import is_cancelled_exception

with workflow.unsafe.imports_passed_through():
    from activities.llm_activities import (
        plan_next_turn,
        summarize_artifacts,
        summarize_subcommunis_results,
        validate_user_feedback,
    )
    from activities.workspace_activities import (
        collect_older_turns_text,
        init_workspace,
        read_turn_context,
        write_plan_file,
        write_subcommunis_summary,
        write_workspace_summary,
    )
    from models.data_types import (
        DEFAULT_MAX_TURNS,
        CommunisConfig,
        CommunisState,
        SubCommunisResult,
        TurnConfig,
        TurnResult,
    )
    from prompts.communis_prompts import (
        APPROACHING_LIMIT_ADDENDUM,
        FINAL_TURN_PLANNER_ADDENDUM,
    )
    from workflows.communis_turn import CommunisTurnWorkflow
    from workflows.constants import (
        FAST_LLM_TIMEOUT,
        IO_TIMEOUT,
        LLM_RETRY_POLICY,
    )

# Summarize older turns once we have more than this many
MAX_RECENT_ARTIFACTS = 3


@workflow.defn
class CommunisOrchestratorWorkflow:
    """Parent workflow: orchestrates iterative turns toward goal completion."""

    def __init__(self):
        self.state = CommunisState()
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
                return r.to_dict()
        return None

    @workflow.query
    def get_all_results(self) -> list[dict]:
        return [r.to_dict() for r in self.state.turn_results]

    # --- Main workflow ---
    @workflow.run
    async def run(self, config: CommunisConfig) -> dict:
        self.state.idea = config.idea
        effective_max = config.max_turns if config.max_turns > 0 else DEFAULT_MAX_TURNS
        self.state.max_turns = effective_max
        self.state.dangerous = config.dangerous
        self.state.status = "running"

        try:
            await self._run_turns(config, effective_max)
        except (asyncio.CancelledError, Exception) as err:
            if isinstance(err, asyncio.CancelledError) or is_cancelled_exception(err):
                turns_done = len(self.state.turn_results)
                self.state.status = "cancelled"
                self.state.latest_message = (
                    f"Cancelled after {turns_done} steps."
                )
                return self.state.to_dict()
            raise

        if self.state.goal_complete:
            self.state.status = "complete"
            self.state.latest_message = "Goal complete!"
        else:
            self.state.status = "complete"
            self.state.latest_message = "All steps complete!"

        return self.state.to_dict()

    async def _run_turns(self, config: CommunisConfig, effective_max: int) -> None:
        """Execute the turn loop. Raises CancelledError if workflow is cancelled."""
        # Initialize workspace
        workspace_dir = await workflow.execute_activity(
            init_workspace,
            args=[workflow.info().workflow_id, config.idea, effective_max, config.model],
            start_to_close_timeout=IO_TIMEOUT,
        )
        self.state.workspace_dir = workspace_dir

        turn_number = 0

        while turn_number < effective_max:
            turn_number += 1
            self.state.current_turn = turn_number

            # Read workspace context for planner
            ws_context = await workflow.execute_activity(
                read_turn_context,
                args=[workspace_dir, turn_number],
                start_to_close_timeout=IO_TIMEOUT,
            )

            # Ask the planner what to do next
            plan_context = self._build_planner_context(
                config, ws_context, turn_number, effective_max
            )
            plan = await workflow.execute_activity(
                plan_next_turn,
                args=[plan_context, config.provider, config.base_url, config.model],
                start_to_close_timeout=FAST_LLM_TIMEOUT,
                retry_policy=LLM_RETRY_POLICY,
            )

            # Write plan summary to workspace
            plan_summary = plan.get("plan_summary", "")
            if plan_summary:
                await workflow.execute_activity(
                    write_plan_file,
                    args=[workspace_dir, plan_summary],
                    start_to_close_timeout=IO_TIMEOUT,
                )

            # Check for goal completion
            if config.goal_complete_detection and plan.get("goal_complete", False):
                self.state.goal_complete = True
                self.state.latest_message = (
                    f"Goal complete after {len(self.state.turn_results)} steps "
                    f"— {plan.get('reasoning', '')}"
                )
                break

            action = plan.get("action", "step")

            # --- Handle spawn action ---
            if action == "spawn" and config.max_subcommunis > 0:
                subcommunis_tasks = plan.get("subcommunis", [])
                if subcommunis_tasks:
                    self.state.current_role = "Spawning subcommuniss"
                    self.state.latest_message = (
                        f"Step {turn_number}: Spawning {len(subcommunis_tasks)} subcommuniss "
                        f"— {plan.get('reasoning', '')}"
                    )

                    subcommunis_results = await self._spawn_subcommunis(
                        subcommunis_tasks, config, workspace_dir, turn_number
                    )

                    # Summarize subcommunis results
                    results_text = "\n\n".join(
                        f"Task: {r.task}\nStatus: {r.status}\nSummary: {r.summary}"
                        for r in subcommunis_results
                    )
                    summary = await workflow.execute_activity(
                        summarize_subcommunis_results,
                        args=[results_text, config.idea, config.provider, config.base_url, config.model],
                        start_to_close_timeout=FAST_LLM_TIMEOUT,
                        retry_policy=LLM_RETRY_POLICY,
                    )

                    # Write subcommunis summary to workspace
                    await workflow.execute_activity(
                        write_subcommunis_summary,
                        args=[workspace_dir, turn_number, summary],
                        start_to_close_timeout=IO_TIMEOUT,
                    )

                    self.state.latest_message = (
                        f"Step {turn_number}: Subcommuniss complete."
                    )
                    continue

            # --- Handle step action (default) ---
            role = plan["role"]
            instructions = plan["instructions"]
            self.state.current_role = role
            self.state.latest_message = (
                f"Step {turn_number}: {role} "
                f"\u2014 {plan['reasoning']}"
            )

            # Build turn config
            turn_config = TurnConfig(
                workspace_dir=workspace_dir,
                idea=config.idea,
                role=role,
                instructions=instructions,
                turn_number=turn_number,
                max_turns=effective_max,
                user_feedback=self.user_feedback or "",
                model=config.model,
                provider=config.provider,
                base_url=config.base_url,
                dangerous=config.dangerous,
            )

            # Execute child workflow
            result: TurnResult = await workflow.execute_child_workflow(
                CommunisTurnWorkflow.run,
                turn_config,
                id=f"{workflow.info().workflow_id}-turn-{turn_number}",
            )

            self.state.turn_results.append(result)
            self.state.latest_message = (
                f"Step {turn_number} ({role}) complete."
            )

            # Update rolling summary if enough turns have accumulated
            if len(self.state.turn_results) > MAX_RECENT_ARTIFACTS + 1:
                threshold_turn = turn_number - MAX_RECENT_ARTIFACTS + 1
                older_text = await workflow.execute_activity(
                    collect_older_turns_text,
                    args=[workspace_dir, threshold_turn],
                    start_to_close_timeout=IO_TIMEOUT,
                )
                summary = await workflow.execute_activity(
                    summarize_artifacts,
                    args=[older_text, config.provider, config.base_url, config.model],
                    start_to_close_timeout=FAST_LLM_TIMEOUT,
                    retry_policy=LLM_RETRY_POLICY,
                )
                await workflow.execute_activity(
                    write_workspace_summary,
                    args=[workspace_dir, summary],
                    start_to_close_timeout=IO_TIMEOUT,
                )

            # Wait for user feedback between turns (unless auto mode)
            if not config.auto:
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
                        args=[self.user_feedback, config.idea, config.provider, config.base_url, config.model],
                        start_to_close_timeout=FAST_LLM_TIMEOUT,
                        retry_policy=LLM_RETRY_POLICY,
                    )
                    if not validation["relevant"]:
                        self.state.latest_message = (
                            f"Feedback noted but may not be relevant: {validation['reason']}. "
                            "Continuing with it anyway."
                        )

                self.state.status = "running"

    async def _spawn_subcommunis(
        self,
        subcommunis_tasks: list[dict],
        config: CommunisConfig,
        workspace_dir: str,
        turn_number: int,
    ) -> list[SubCommunisResult]:
        """Spawn subcommunis orchestrator workflows in parallel and collect results."""
        # Cap at max_subcommunis
        tasks = subcommunis_tasks[: config.max_subcommunis]

        parent_id = workflow.info().workflow_id

        handles = []
        for i, task_def in enumerate(tasks):
            sub_id = f"{parent_id}-subcommunis-{turn_number}-{i}"
            sub_config = CommunisConfig(
                idea=task_def.get("task", ""),
                max_turns=task_def.get("max_turns", 5),
                model=config.model,
                auto=True,  # Subcommuniss don't prompt for feedback
                provider=config.provider,
                base_url=config.base_url,
                dangerous=config.dangerous,
                goal_complete_detection=True,
                max_subcommunis=0,  # Prevent recursion bomb
            )
            handle = await workflow.start_child_workflow(
                CommunisOrchestratorWorkflow.run,
                sub_config,
                id=sub_id,
            )
            handles.append((task_def, handle))

        # Wait for all subcommuniss to complete
        results: list[SubCommunisResult] = []
        for task_def, handle in handles:
            try:
                sub_result = await handle
                status = "goal_complete" if sub_result.get("goal_complete", False) else sub_result.get("status", "complete")
                # Build summary from turn results
                turn_summaries = []
                for tr in sub_result.get("turn_results", []):
                    insights = tr.get("key_insights", [])
                    if insights:
                        turn_summaries.append(
                            f"Step {tr['turn_number']} ({tr['role']}): {'; '.join(insights)}"
                        )
                summary = "\n".join(turn_summaries) if turn_summaries else sub_result.get("latest_message", "")

                results.append(SubCommunisResult(
                    task=task_def.get("task", ""),
                    status=status,
                    summary=summary,
                    turn_results=sub_result.get("turn_results", []),
                    workspace_dir=sub_result.get("workspace_dir", ""),
                ))
            except Exception as e:
                results.append(SubCommunisResult(
                    task=task_def.get("task", ""),
                    status="error",
                    summary=f"Error: {e}",
                ))

        return results

    def _build_planner_context(
        self, config: CommunisConfig, ws_context: dict, turn_number: int, effective_max: int
    ) -> str:
        """Build context string for the planner from workspace data."""
        parts = [f"Goal: {config.idea}"]

        if ws_context.get("plan"):
            parts.append(f"Current plan summary:\n{ws_context['plan']}")

        if ws_context["summary"]:
            parts.append(f"Summary of earlier work:\n{ws_context['summary']}")

        for turn_data in ws_context["recent_turns"]:
            insights = turn_data.get("key_insights", [])
            parts.append(
                f"Step {turn_data['turn']} (role: {turn_data['role']}) insights: "
                + "; ".join(insights)
            )

        if self.user_feedback:
            parts.append(f"Latest user feedback: {self.user_feedback}")

        # Step position info
        if config.max_turns > 0:
            remaining = effective_max - turn_number
            parts.append(
                f"This will be step {turn_number} (max {effective_max}). "
                f"Steps completed so far: {len(self.state.turn_results)}. "
                f"Remaining after this one: {remaining}."
            )
        else:
            parts.append(
                f"This will be step {turn_number}. "
                f"Steps completed so far: {len(self.state.turn_results)}."
            )

        # Subcommunis capability
        if config.max_subcommunis > 0:
            parts.append(
                f"You can spawn up to {config.max_subcommunis} subcommuniss in parallel "
                f"for independent tasks using the 'spawn' action."
            )

        # Approaching limit warning (only in fixed-turn mode)
        if config.max_turns > 0:
            remaining = effective_max - turn_number
            if remaining <= 2 and remaining > 0:
                parts.append(
                    APPROACHING_LIMIT_ADDENDUM.format(
                        turn=turn_number,
                        total=effective_max,
                        remaining=remaining,
                    )
                )

        # Final turn
        if turn_number == effective_max:
            parts.append(
                FINAL_TURN_PLANNER_ADDENDUM.format(
                    turn=turn_number,
                    total=effective_max,
                )
            )

        return "\n\n".join(parts)
