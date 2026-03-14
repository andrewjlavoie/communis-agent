"""Example: Start and monitor a communis session from outside Temporal.

This is a plain Python script — not a workflow. It uses the Temporal client
SDK to start a workflow, poll its state, and read the result. Use this pattern
from web apps, cron jobs, Slack bots, or any service.

Usage:
    # Make sure the worker is running:
    #   uv run python scripts/run_worker.py

    uv run python examples/external_client.py "your prompt here"
"""

from __future__ import annotations

import asyncio
import sys

from temporalio.client import Client

from models.data_types import CommunisConfig
from shared.constants import TASK_QUEUE
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow


async def main():
    idea = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "design a notification system for a SaaS platform"

    client = await Client.connect("localhost:7233")

    # Start the workflow
    handle = await client.start_workflow(
        CommunisOrchestratorWorkflow.run,
        CommunisConfig(idea=idea, max_turns=3, auto=True),
        id="external-client-example",
        task_queue=TASK_QUEUE,
    )
    print(f"Started workflow: {handle.id}")

    # Poll for progress
    while True:
        try:
            state = await handle.query(CommunisOrchestratorWorkflow.get_state)
        except Exception:
            break

        turn = state["current_turn"]
        total = state["max_turns"]
        role = state["current_role"] or "starting"
        status = state["status"]
        print(f"  [{status}] Turn {turn}/{total} — {role}")

        if status in ("complete", "cancelled", "error"):
            break
        await asyncio.sleep(3)

    # Get final result
    result = await handle.result()
    print(f"\nDone! Status: {result['status']}")
    print(f"Workspace: {result['workspace_dir']}")

    for turn in result["turn_results"]:
        print(f"\n  Turn {turn['turn_number']}: {turn['role']}")
        for insight in turn.get("key_insights", []):
            print(f"    - {insight}")

    # Read the final turn's artifact
    final_turn = result["turn_results"][-1] if result["turn_results"] else None
    if final_turn and final_turn.get("artifact_path"):
        from pathlib import Path

        path = Path(final_turn["artifact_path"])
        if path.exists():
            from shared.frontmatter import parse_frontmatter

            _, text = parse_frontmatter(path.read_text())
            print(f"\n{'='*60}")
            print(f"Final output ({final_turn['role']}):")
            print(f"{'='*60}")
            print(text[:2000])


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
