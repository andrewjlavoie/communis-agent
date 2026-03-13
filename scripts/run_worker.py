import asyncio
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

from activities.llm_activities import (
    call_claude,
    extract_key_insights,
    plan_next_turn,
    summarize_artifacts,
    summarize_subagent_results,
    validate_user_feedback,
)
from activities.tool_activities import execute_run_command
from activities.workspace_activities import (
    collect_older_turns_text,
    init_workspace,
    read_turn_context,
    read_turn_file,
    write_plan_file,
    write_subagent_summary,
    write_turn_artifact,
    write_workspace_summary,
)
from workflows.riff_orchestrator import RiffOrchestratorWorkflow
from workflows.riff_turn import RiffTurnWorkflow

TASK_QUEUE = "autoriff-task-queue"


async def main():
    load_dotenv()

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")

    client = await Client.connect(address, namespace=namespace)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RiffOrchestratorWorkflow, RiffTurnWorkflow],
        activities=[
            # LLM activities
            call_claude,
            extract_key_insights,
            plan_next_turn,
            summarize_artifacts,
            summarize_subagent_results,
            validate_user_feedback,
            # Tool activities
            execute_run_command,
            # Workspace activities
            init_workspace,
            write_turn_artifact,
            read_turn_context,
            write_workspace_summary,
            read_turn_file,
            collect_older_turns_text,
            write_plan_file,
            write_subagent_summary,
        ],
    )
    print(f"autoRiff worker started on task queue: {TASK_QUEUE}")
    print("Waiting for workflows...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
