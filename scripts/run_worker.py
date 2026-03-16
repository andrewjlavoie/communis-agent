import asyncio
import os

from dotenv import load_dotenv

# Load .env BEFORE any imports that read os.getenv() at module level
# (llm_activities reads LLM_PROVIDER, DEFAULT_MODEL, etc. on import)
load_dotenv()

from temporalio.client import Client  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from activities.llm_activities import (  # noqa: E402
    call_claude,
    extract_key_insights,
    plan_next_turn,
    summarize_artifacts,
    summarize_subcommunis_results,
    validate_user_feedback,
)
from activities.tool_activities import execute_run_command  # noqa: E402
from activities.workspace_activities import (  # noqa: E402
    collect_older_turns_text,
    init_workspace,
    read_turn_context,
    read_turn_file,
    write_plan_file,
    write_subcommunis_summary,
    write_turn_artifact,
    write_workspace_summary,
)
from shared.constants import TASK_QUEUE  # noqa: E402
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow  # noqa: E402
from workflows.communis_turn import CommunisTurnWorkflow  # noqa: E402
from workflows.session_workflow import SessionWorkflow  # noqa: E402
from workflows.task_workflow import TaskWorkflow  # noqa: E402


async def main():

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
