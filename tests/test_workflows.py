"""Tests for workflows using Temporal test framework with mock activities."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models.data_types import RiffConfig, TurnConfig
from workflows.riff_orchestrator import RiffOrchestratorWorkflow
from workflows.riff_turn import RiffTurnWorkflow

TASK_QUEUE = "test-queue"

# Shared temp dir for workspace files during tests
_test_workspace_dir: str = ""


def _get_test_workspace() -> str:
    global _test_workspace_dir
    if not _test_workspace_dir:
        _test_workspace_dir = tempfile.mkdtemp(prefix="autoriff-test-")
    return _test_workspace_dir


# --- Mock LLM activities ---


@activity.defn(name="call_claude")
async def mock_call_claude(
    messages: list[dict],
    system_prompt: str,
    model: str = "",
    max_tokens: int = 0,
    provider: str = "",
    base_url: str = "",
) -> dict:
    return {
        "text": f"Mock output for role. System prompt length: {len(system_prompt)}",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


@activity.defn(name="extract_key_insights")
async def mock_extract_key_insights(content: str, provider: str = "", base_url: str = "") -> list[str]:
    return ["mock insight 1", "mock insight 2"]


@activity.defn(name="plan_next_turn")
async def mock_plan_next_turn(context: str, provider: str = "", base_url: str = "") -> dict:
    return {
        "role": "Explorer",
        "instructions": "Explore the idea broadly.",
        "reasoning": "Starting with open exploration.",
    }


@activity.defn(name="summarize_artifacts")
async def mock_summarize_artifacts(artifacts_text: str, provider: str = "", base_url: str = "") -> str:
    return f"Summary of {len(artifacts_text)} chars of artifacts."


@activity.defn(name="validate_user_feedback")
async def mock_validate_user_feedback(feedback: str, idea: str, provider: str = "", base_url: str = "") -> dict:
    return {"relevant": True, "reason": "Feedback is relevant."}


# --- Mock workspace activities ---


@activity.defn(name="init_workspace")
async def mock_init_workspace(workflow_id: str, idea: str, num_turns: int, model: str) -> str:
    ws = Path(_get_test_workspace()) / workflow_id
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "riff.md").write_text(f"# {idea}\n")
    return str(ws)


@activity.defn(name="write_turn_artifact")
async def mock_write_turn_artifact(
    workspace_dir: str,
    turn_number: int,
    role: str,
    content: str,
    key_insights: list[str],
    token_usage: dict[str, int],
    truncated: bool,
) -> str:
    ws = Path(workspace_dir)
    safe_role = role.lower().replace(" ", "-").replace("'", "")[:30]
    filename = f"turn-{turn_number:02d}-{safe_role}.md"
    path = ws / filename
    path.write_text(f"---\nturn: {turn_number}\nrole: {role}\n---\n{content}")
    return str(path)


@activity.defn(name="read_turn_context")
async def mock_read_turn_context(workspace_dir: str, current_turn: int) -> dict:
    # Return empty context for simplicity — planner and turn agent still work fine
    return {"summary": "", "recent_turns": []}


@activity.defn(name="write_workspace_summary")
async def mock_write_workspace_summary(workspace_dir: str, summary: str) -> None:
    Path(workspace_dir, "summary.md").write_text(summary)


@activity.defn(name="collect_older_turns_text")
async def mock_collect_older_turns_text(workspace_dir: str, before_turn: int) -> str:
    return "Older turns text placeholder."


ALL_MOCK_ACTIVITIES = [
    # LLM
    mock_call_claude,
    mock_extract_key_insights,
    mock_plan_next_turn,
    mock_summarize_artifacts,
    mock_validate_user_feedback,
    # Workspace
    mock_init_workspace,
    mock_write_turn_artifact,
    mock_read_turn_context,
    mock_write_workspace_summary,
    mock_collect_older_turns_text,
]

ALL_WORKFLOWS = [RiffOrchestratorWorkflow, RiffTurnWorkflow]


@pytest.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


@pytest.fixture(autouse=True)
def reset_workspace():
    """Reset the test workspace dir for each test."""
    global _test_workspace_dir
    _test_workspace_dir = ""
    yield


@pytest.mark.asyncio
async def test_riff_turn_workflow(env):
    """Test that a single turn child workflow executes and returns a TurnResult."""
    # Create a temp workspace for this test
    ws = Path(tempfile.mkdtemp(prefix="autoriff-test-turn-"))

    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[RiffTurnWorkflow],
        activities=[mock_call_claude, mock_extract_key_insights, mock_read_turn_context, mock_write_turn_artifact],
    ):
        config = TurnConfig(
            workspace_dir=str(ws),
            idea="A mobile app for finding quiet workspaces",
            role="Explorer",
            instructions="Explore the idea broadly and identify key opportunities.",
            turn_number=1,
            total_turns=3,
        )
        result = await env.client.execute_workflow(
            RiffTurnWorkflow.run,
            config,
            id="test-turn-1",
            task_queue=TASK_QUEUE,
        )

    assert result.turn_number == 1
    assert result.role == "Explorer"
    assert len(result.key_insights) == 2
    assert result.artifact_path != ""


@pytest.mark.asyncio
async def test_orchestrator_single_turn(env):
    """Test orchestrator with a single turn."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_MOCK_ACTIVITIES,
    ):
        config = RiffConfig(idea="Test idea", num_turns=1)
        result = await env.client.execute_workflow(
            RiffOrchestratorWorkflow.run,
            config,
            id="test-orch-1",
            task_queue=TASK_QUEUE,
        )

    assert result["status"] == "complete"
    assert len(result["turn_results"]) == 1
    assert result["turn_results"][0]["role"] == "Explorer"
    assert result["workspace_dir"] != ""


@pytest.mark.asyncio
async def test_orchestrator_multi_turn_auto(env):
    """Test orchestrator with multiple turns in auto mode (no feedback waits)."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_MOCK_ACTIVITIES,
    ):
        config = RiffConfig(idea="Test idea", num_turns=3, auto=True)
        result = await env.client.execute_workflow(
            RiffOrchestratorWorkflow.run,
            config,
            id="test-orch-auto",
            task_queue=TASK_QUEUE,
        )

    assert result["status"] == "complete"
    assert len(result["turn_results"]) == 3


@pytest.mark.asyncio
async def test_orchestrator_multi_turn_with_skip(env):
    """Test orchestrator with multiple turns, skipping feedback."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_MOCK_ACTIVITIES,
    ):
        handle = await env.client.start_workflow(
            RiffOrchestratorWorkflow.run,
            RiffConfig(idea="Test idea", num_turns=3),
            id="test-orch-multi",
            task_queue=TASK_QUEUE,
        )

        for _ in range(2):  # 2 feedback waits for 3 turns
            for _attempt in range(30):
                state = await handle.query(RiffOrchestratorWorkflow.get_state)
                if state["status"] == "waiting_for_feedback":
                    break
                await asyncio.sleep(0.3)
            await handle.signal(RiffOrchestratorWorkflow.skip_feedback)

        result = await handle.result()

    assert result["status"] == "complete"
    assert len(result["turn_results"]) == 3


@pytest.mark.asyncio
async def test_orchestrator_with_feedback(env):
    """Test orchestrator processes user feedback."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_MOCK_ACTIVITIES,
    ):
        handle = await env.client.start_workflow(
            RiffOrchestratorWorkflow.run,
            RiffConfig(idea="Test idea", num_turns=2),
            id="test-orch-feedback",
            task_queue=TASK_QUEUE,
        )

        # Wait for feedback prompt after turn 1
        for _attempt in range(30):
            state = await handle.query(RiffOrchestratorWorkflow.get_state)
            if state["status"] == "waiting_for_feedback":
                break
            await asyncio.sleep(0.3)

        await handle.signal(
            RiffOrchestratorWorkflow.receive_user_feedback,
            "Focus on enterprise market",
        )

        result = await handle.result()

    assert result["status"] == "complete"
    assert len(result["turn_results"]) == 2


@pytest.mark.asyncio
async def test_query_state(env):
    """Test that query handlers work correctly."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_MOCK_ACTIVITIES,
    ):
        handle = await env.client.start_workflow(
            RiffOrchestratorWorkflow.run,
            RiffConfig(idea="Query test idea", num_turns=2),
            id="test-orch-query",
            task_queue=TASK_QUEUE,
        )

        # Wait for first turn to complete
        for _attempt in range(30):
            state = await handle.query(RiffOrchestratorWorkflow.get_state)
            if state["status"] == "waiting_for_feedback":
                break
            await asyncio.sleep(0.3)

        state = await handle.query(RiffOrchestratorWorkflow.get_state)
        assert state["idea"] == "Query test idea"
        assert state["num_turns"] == 2
        assert state["status"] == "waiting_for_feedback"
        assert len(state["turn_results"]) == 1
        assert state["workspace_dir"] != ""

        turn1 = await handle.query(RiffOrchestratorWorkflow.get_turn_result, 1)
        assert turn1 is not None
        assert turn1["turn_number"] == 1

        all_results = await handle.query(RiffOrchestratorWorkflow.get_all_results)
        assert len(all_results) == 1

        await handle.signal(RiffOrchestratorWorkflow.skip_feedback)
        await handle.result()


@pytest.mark.asyncio
async def test_orchestrator_summarization(env):
    """Test that summarization triggers when turn count exceeds threshold."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_MOCK_ACTIVITIES,
    ):
        config = RiffConfig(idea="Summarization test", num_turns=6, auto=True)
        result = await env.client.execute_workflow(
            RiffOrchestratorWorkflow.run,
            config,
            id="test-orch-summarize",
            task_queue=TASK_QUEUE,
        )

    assert result["status"] == "complete"
    assert len(result["turn_results"]) == 6


@pytest.mark.asyncio
async def test_orchestrator_cancel_during_feedback(env):
    """Test that cancelling during feedback wait returns partial results with cancelled status."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_MOCK_ACTIVITIES,
    ):
        handle = await env.client.start_workflow(
            RiffOrchestratorWorkflow.run,
            RiffConfig(idea="Cancel test", num_turns=3),
            id="test-orch-cancel",
            task_queue=TASK_QUEUE,
        )

        # Wait for feedback prompt after turn 1
        for _attempt in range(30):
            state = await handle.query(RiffOrchestratorWorkflow.get_state)
            if state["status"] == "waiting_for_feedback":
                break
            await asyncio.sleep(0.3)

        assert state["status"] == "waiting_for_feedback"
        assert len(state["turn_results"]) == 1

        # Cancel the workflow
        await handle.cancel()

        result = await handle.result()

    assert result["status"] == "cancelled"
    assert len(result["turn_results"]) == 1
    assert "Cancelled" in result["latest_message"]


@pytest.mark.asyncio
async def test_orchestrator_cancel_during_turn(env):
    """Test that cancelling during a running turn returns partial results."""
    # Use a slow mock to simulate a long-running turn
    @activity.defn(name="call_claude")
    async def slow_call_claude(
        messages: list[dict],
        system_prompt: str,
        model: str = "",
        max_tokens: int = 0,
        provider: str = "",
        base_url: str = "",
    ) -> dict:
        await asyncio.sleep(30)  # Simulate slow LLM call
        return {
            "text": "Should not reach here",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

    slow_activities = [
        slow_call_claude,
        mock_extract_key_insights,
        mock_plan_next_turn,
        mock_summarize_artifacts,
        mock_validate_user_feedback,
        mock_init_workspace,
        mock_write_turn_artifact,
        mock_read_turn_context,
        mock_write_workspace_summary,
        mock_collect_older_turns_text,
    ]

    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=slow_activities,
    ):
        handle = await env.client.start_workflow(
            RiffOrchestratorWorkflow.run,
            RiffConfig(idea="Cancel during turn test", num_turns=3, auto=True),
            id="test-orch-cancel-turn",
            task_queue=TASK_QUEUE,
        )

        # Wait for it to start running
        for _attempt in range(30):
            state = await handle.query(RiffOrchestratorWorkflow.get_state)
            if state["status"] == "running" and state["current_turn"] == 1:
                break
            await asyncio.sleep(0.3)

        # Cancel while turn 1 is in progress
        await handle.cancel()

        result = await handle.result()

    assert result["status"] == "cancelled"
    assert len(result["turn_results"]) == 0  # Turn 1 never completed
