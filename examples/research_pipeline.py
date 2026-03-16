"""Example: Multi-phase research pipeline using communis as a child workflow.

Runs three sequential riff sessions — explore, deep dive, synthesize —
where each phase feeds its insights into the next.

Usage:
    # Make sure the worker is running first:
    #   uv run python scripts/run_worker.py

    # Then run this from the project root:
    #   uv run python examples/research_pipeline.py "your topic here"

    # Or register the workflow with the existing worker and start via Temporal CLI:
    #   temporal workflow start --type ResearchPipelineWorkflow \
    #     --task-queue communis-task-queue --workflow-id "research-test" \
    #     --input '"your topic here"'
"""

from __future__ import annotations

import asyncio
import sys

from temporalio import workflow
from temporalio.client import Client

with workflow.unsafe.imports_passed_through():
    from models.data_types import CommunisConfig
    from workflows.communis_orchestrator import CommunisOrchestratorWorkflow


@workflow.defn
class ResearchPipelineWorkflow:
    """Run three communis phases: explore → deep dive → synthesize."""

    @workflow.run
    async def run(self, topic: str) -> dict:
        wf_id = workflow.info().workflow_id

        # Phase 1: Broad exploration
        explore = await workflow.execute_child_workflow(
            CommunisOrchestratorWorkflow.run,
            CommunisConfig(
                idea=f"Research and explore: {topic}",
                max_turns=3,
                auto=True,
            ),
            id=f"{wf_id}-explore",
        )

        explore_insights = [
            i
            for turn in explore["turn_results"]
            for i in turn.get("key_insights", [])
        ]

        # Phase 2: Deep dive on findings
        deep_dive = await workflow.execute_child_workflow(
            CommunisOrchestratorWorkflow.run,
            CommunisConfig(
                idea=(
                    f"Deep dive on: {topic}\n\n"
                    f"Prior research found:\n"
                    + "\n".join(f"- {i}" for i in explore_insights)
                    + "\n\nGo deeper. Challenge assumptions. Find what was missed."
                ),
                max_turns=4,
                auto=True,
            ),
            id=f"{wf_id}-deep-dive",
        )

        deep_insights = [
            i
            for turn in deep_dive["turn_results"]
            for i in turn.get("key_insights", [])
        ]

        # Phase 3: Final synthesis
        report = await workflow.execute_child_workflow(
            CommunisOrchestratorWorkflow.run,
            CommunisConfig(
                idea=(
                    f"Create a comprehensive report on: {topic}\n\n"
                    f"Exploration insights:\n"
                    + "\n".join(f"- {i}" for i in explore_insights)
                    + f"\n\nDeep dive insights:\n"
                    + "\n".join(f"- {i}" for i in deep_insights)
                    + "\n\nSynthesize into a polished, actionable report."
                ),
                max_turns=3,
                auto=True,
            ),
            id=f"{wf_id}-report",
        )

        return {
            "topic": topic,
            "phases": {
                "exploration": explore,
                "deep_dive": deep_dive,
                "report": report,
            },
            "total_turns": sum(
                len(r["turn_results"])
                for r in [explore, deep_dive, report]
            ),
        }


# --- Standalone runner (not needed if registered with a worker) ---


async def main():
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "the future of local-first software"

    client = await Client.connect("localhost:7233")

    # Import worker deps to register everything in one process
    from temporalio.worker import Worker

    from activities.llm_activities import (
        call_llm,
        extract_key_insights,
        plan_next_turn,
        summarize_artifacts,
        validate_user_feedback,
    )
    from activities.workspace_activities import (
        collect_older_turns_text,
        init_workspace,
        read_turn_context,
        read_turn_file,
        write_turn_artifact,
        write_workspace_summary,
    )
    from shared.constants import TASK_QUEUE
    from workflows.communis_turn import CommunisTurnWorkflow

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ResearchPipelineWorkflow, CommunisOrchestratorWorkflow, CommunisTurnWorkflow],
        activities=[
            call_llm, plan_next_turn, extract_key_insights,
            summarize_artifacts, validate_user_feedback,
            init_workspace, read_turn_context, write_turn_artifact,
            write_workspace_summary, read_turn_file, collect_older_turns_text,
        ],
    ):
        result = await client.execute_workflow(
            ResearchPipelineWorkflow.run,
            topic,
            id=f"research-pipeline-example",
            task_queue=TASK_QUEUE,
        )

    print(f"\nTopic: {result['topic']}")
    print(f"Total turns across all phases: {result['total_turns']}")
    for phase_name, phase_data in result["phases"].items():
        print(f"\n  {phase_name}: {phase_data['status']} ({len(phase_data['turn_results'])} turns)")
        for turn in phase_data["turn_results"]:
            print(f"    Turn {turn['turn_number']}: {turn['role']}")
            for insight in turn.get("key_insights", []):
                print(f"      - {insight}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
