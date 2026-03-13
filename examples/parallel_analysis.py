"""Example: Parallel multi-perspective analysis using communis.

Launches three riff sessions concurrently (technical, business, UX),
then synthesizes all findings in a fourth session.

Usage:
    uv run python examples/parallel_analysis.py "your problem statement"
"""

from __future__ import annotations

import asyncio
import sys

from temporalio import workflow
from temporalio.client import Client

with workflow.unsafe.imports_passed_through():
    from models.data_types import CommunisConfig
    from workflows.communis_orchestrator import CommunisOrchestratorWorkflow


PERSPECTIVES = [
    ("technical", "Analyze from a TECHNICAL perspective — architecture, scalability, implementation risk"),
    ("business", "Analyze from a BUSINESS perspective — market fit, revenue model, competitive landscape"),
    ("ux", "Analyze from a USER EXPERIENCE perspective — usability, onboarding, retention"),
]


@workflow.defn
class ParallelAnalysisWorkflow:
    """Analyze a problem from multiple angles simultaneously, then synthesize."""

    @workflow.run
    async def run(self, problem: str) -> dict:
        wf_id = workflow.info().workflow_id

        # Fan out: run all perspectives in parallel
        tasks = [
            workflow.execute_child_workflow(
                CommunisOrchestratorWorkflow.run,
                CommunisConfig(
                    idea=f"{angle}: {problem}",
                    num_turns=3,
                    auto=True,
                ),
                id=f"{wf_id}-{name}",
            )
            for name, angle in PERSPECTIVES
        ]

        results = await asyncio.gather(*tasks)
        perspective_results = {name: result for (name, _), result in zip(PERSPECTIVES, results)}

        # Collect all insights with labels
        all_insights = []
        for (name, _), result in zip(PERSPECTIVES, results):
            for turn in result["turn_results"]:
                for insight in turn.get("key_insights", []):
                    all_insights.append(f"[{name.upper()}] {insight}")

        # Synthesize
        synthesis = await workflow.execute_child_workflow(
            CommunisOrchestratorWorkflow.run,
            CommunisConfig(
                idea=(
                    f"Synthesize multi-perspective analysis of: {problem}\n\n"
                    f"Findings from three parallel analyses:\n"
                    + "\n".join(f"- {i}" for i in all_insights)
                    + "\n\nCreate a unified recommendation that balances all perspectives. "
                    "Identify where they agree, where they conflict, and what to prioritize."
                ),
                num_turns=2,
                auto=True,
            ),
            id=f"{wf_id}-synthesis",
        )

        return {
            "problem": problem,
            "perspectives": perspective_results,
            "synthesis": synthesis,
            "total_turns": sum(len(r["turn_results"]) for r in results) + len(synthesis["turn_results"]),
        }


async def main():
    problem = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "should we build or buy an internal analytics platform"

    client = await Client.connect("localhost:7233")

    from temporalio.worker import Worker

    from activities.llm_activities import (
        call_claude, extract_key_insights, plan_next_turn,
        summarize_artifacts, validate_user_feedback,
    )
    from activities.workspace_activities import (
        collect_older_turns_text, init_workspace, read_turn_context,
        read_turn_file, write_turn_artifact, write_workspace_summary,
    )
    from workflows.communis_turn import CommunisTurnWorkflow

    task_queue = "communis-task-queue"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ParallelAnalysisWorkflow, CommunisOrchestratorWorkflow, CommunisTurnWorkflow],
        activities=[
            call_claude, plan_next_turn, extract_key_insights,
            summarize_artifacts, validate_user_feedback,
            init_workspace, read_turn_context, write_turn_artifact,
            write_workspace_summary, read_turn_file, collect_older_turns_text,
        ],
    ):
        result = await client.execute_workflow(
            ParallelAnalysisWorkflow.run,
            problem,
            id="parallel-analysis-example",
            task_queue=task_queue,
        )

    print(f"\nProblem: {result['problem']}")
    print(f"Total turns: {result['total_turns']}")
    for name, data in result["perspectives"].items():
        print(f"\n  {name}: {len(data['turn_results'])} turns")
        for turn in data["turn_results"]:
            print(f"    {turn['turn_number']}. {turn['role']}")
    print(f"\n  synthesis: {len(result['synthesis']['turn_results'])} turns")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
