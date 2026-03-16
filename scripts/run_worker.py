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
    summarize_subcommunis_results,
    validate_user_feedback,
)
from activities.session_activities import front_agent_respond
from activities.tool_activities import execute_run_command
from activities.workspace_activities import (
    collect_older_turns_text,
    init_workspace,
    read_turn_context,
    read_turn_file,
    write_plan_file,
    write_subcommunis_summary,
    write_turn_artifact,
    write_workspace_summary,
)
from shared.constants import TASK_QUEUE
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow
from workflows.communis_turn import CommunisTurnWorkflow
from workflows.session_workflow import SessionWorkflow
from workflows.task_workflow import TaskWorkflow


async def main():
    load_dotenv()

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")

    client = await Client.connect(address, namespace=namespace)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            CommunisOrchestratorWorkflow,
            CommunisTurnWorkflow,
            SessionWorkflow,
            TaskWorkflow,
        ],
        activities=[
            # LLM activities
            call_claude,
            extract_key_insights,
            plan_next_turn,
            summarize_artifacts,
            summarize_subcommunis_results,
            validate_user_feedback,
            # Session activities
            front_agent_respond,
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
            write_subcommunis_summary,
        ],
    )
    print(f"communis worker started on task queue: {TASK_QUEUE}")
    print("Waiting for workflows...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
