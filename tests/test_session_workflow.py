"""Tests for SessionWorkflow — front agent with task management."""

from __future__ import annotations

import asyncio

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models.session_types import SessionConfig
from workflows.session_workflow import SessionWorkflow
from workflows.task_workflow import TaskWorkflow

TASK_QUEUE = "test-queue"


# --- Mock activities ---


@activity.defn(name="front_agent_respond")
async def mock_front_agent_respond(
    conversation: list[dict],
    active_tasks: dict[str, dict],
    model: str = "",
    provider: str = "",
    base_url: str = "",
) -> dict:
    """Mock front agent that responds directly."""
    last_msg = conversation[-1]["content"] if conversation else ""
    return {"text": f"Response to: {last_msg}", "delegate_tasks": []}


@activity.defn(name="front_agent_respond")
async def mock_delegating_agent(
    conversation: list[dict],
    active_tasks: dict[str, dict],
    model: str = "",
    provider: str = "",
    base_url: str = "",
) -> dict:
    """Mock front agent that delegates tasks."""
    last_msg = conversation[-1]["content"] if conversation else ""
    if "delegate" in last_msg.lower():
        return {
            "text": "I'll delegate that work.",
            "delegate_tasks": [
                {"description": "Do the work", "context": last_msg},
            ],
        }
    return {"text": f"Response to: {last_msg}", "delegate_tasks": []}


@activity.defn(name="call_claude")
async def mock_call_claude(
    messages: list[dict],
    system_prompt: str,
    model: str = "",
    max_tokens: int = 0,
    provider: str = "",
    base_url: str = "",
    tools: list[dict] | None = None,
) -> dict:
    return {
        "text": "Sub-agent done.",
        "stop_reason": "end_turn",
        "content_blocks": [{"type": "text", "text": "Sub-agent done."}],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


@activity.defn(name="execute_run_command")
async def mock_execute_run_command(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
) -> dict:
    return {"output": f"mock: {command}\n[exit:0]", "exit_code": 0, "duration_ms": 5}


@pytest.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


@pytest.mark.asyncio
async def test_session_user_message_response(env):
    """User sends a message and gets a response event."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[SessionWorkflow, TaskWorkflow],
        activities=[mock_front_agent_respond, mock_call_claude, mock_execute_run_command],
    ):
        handle = await env.client.start_workflow(
            SessionWorkflow.run,
            SessionConfig(),
            id="session-msg-1",
            task_queue=TASK_QUEUE,
        )

        await handle.signal(SessionWorkflow.user_message, "Hello")

        # Wait for response event
        for _attempt in range(50):
            events = await handle.query(SessionWorkflow.get_events_since, 0)
            if any(e["event_type"] == "assistant_message" for e in events):
                break
            await asyncio.sleep(0.2)

        events = await handle.query(SessionWorkflow.get_events_since, 0)
        msg_events = [e for e in events if e["event_type"] == "assistant_message"]
        assert len(msg_events) >= 1
        assert "Hello" in msg_events[0]["data"]["text"]

        await handle.signal(SessionWorkflow.end_session)
        result = await handle.result()
        assert result["status"] == "ended"


@pytest.mark.asyncio
async def test_session_delegation_spawns_task(env):
    """Front agent delegates work, spawning a TaskWorkflow."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[SessionWorkflow, TaskWorkflow],
        activities=[mock_delegating_agent, mock_call_claude, mock_execute_run_command],
    ):
        handle = await env.client.start_workflow(
            SessionWorkflow.run,
            SessionConfig(),
            id="session-delegate-1",
            task_queue=TASK_QUEUE,
        )

        await handle.signal(SessionWorkflow.user_message, "Please delegate this work")

        # Wait for task_started event
        for _attempt in range(50):
            events = await handle.query(SessionWorkflow.get_events_since, 0)
            if any(e["event_type"] == "task_started" for e in events):
                break
            await asyncio.sleep(0.2)

        events = await handle.query(SessionWorkflow.get_events_since, 0)
        started = [e for e in events if e["event_type"] == "task_started"]
        assert len(started) == 1
        task_id = started[0]["data"]["task_id"]

        # Wait for task_completed
        for _attempt in range(50):
            events = await handle.query(SessionWorkflow.get_events_since, 0)
            if any(e["event_type"] == "task_completed" for e in events):
                break
            await asyncio.sleep(0.2)

        events = await handle.query(SessionWorkflow.get_events_since, 0)
        completed = [e for e in events if e["event_type"] == "task_completed"]
        assert len(completed) == 1
        assert completed[0]["data"]["task_id"] == task_id

        # Check state
        state = await handle.query(SessionWorkflow.get_state)
        assert task_id in state["tasks"]
        assert state["tasks"][task_id]["status"] == "completed"

        await handle.signal(SessionWorkflow.end_session)
        await handle.result()


@pytest.mark.asyncio
async def test_session_end_signal(env):
    """End session signal causes workflow to return."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[SessionWorkflow, TaskWorkflow],
        activities=[mock_front_agent_respond, mock_call_claude, mock_execute_run_command],
    ):
        handle = await env.client.start_workflow(
            SessionWorkflow.run,
            SessionConfig(),
            id="session-end-1",
            task_queue=TASK_QUEUE,
        )

        await handle.signal(SessionWorkflow.end_session)
        result = await handle.result()

        assert result["status"] == "ended"


@pytest.mark.asyncio
async def test_session_multiple_messages(env):
    """Multiple messages build up conversation history."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[SessionWorkflow, TaskWorkflow],
        activities=[mock_front_agent_respond, mock_call_claude, mock_execute_run_command],
    ):
        handle = await env.client.start_workflow(
            SessionWorkflow.run,
            SessionConfig(),
            id="session-multi-1",
            task_queue=TASK_QUEUE,
        )

        await handle.signal(SessionWorkflow.user_message, "First message")

        # Wait for first response
        for _attempt in range(50):
            events = await handle.query(SessionWorkflow.get_events_since, 0)
            if len([e for e in events if e["event_type"] == "assistant_message"]) >= 1:
                break
            await asyncio.sleep(0.2)

        await handle.signal(SessionWorkflow.user_message, "Second message")

        # Wait for second response
        for _attempt in range(50):
            events = await handle.query(SessionWorkflow.get_events_since, 0)
            if len([e for e in events if e["event_type"] == "assistant_message"]) >= 2:
                break
            await asyncio.sleep(0.2)

        state = await handle.query(SessionWorkflow.get_state)
        # Should have at least 4 conversation entries (2 user + 2 assistant)
        assert len(state["conversation"]) >= 4

        await handle.signal(SessionWorkflow.end_session)
        await handle.result()


@pytest.mark.asyncio
async def test_session_get_events_since(env):
    """get_events_since query returns only events after given ID."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[SessionWorkflow, TaskWorkflow],
        activities=[mock_front_agent_respond, mock_call_claude, mock_execute_run_command],
    ):
        handle = await env.client.start_workflow(
            SessionWorkflow.run,
            SessionConfig(),
            id="session-events-1",
            task_queue=TASK_QUEUE,
        )

        await handle.signal(SessionWorkflow.user_message, "Hello")

        # Wait for response
        events: list[dict] = []
        for _attempt in range(50):
            events = await handle.query(SessionWorkflow.get_events_since, 0)
            if events:
                break
            await asyncio.sleep(0.2)

        first_event_id = events[0]["event_id"]

        await handle.signal(SessionWorkflow.user_message, "World")

        # Wait for second response
        new_events: list[dict] = []
        for _attempt in range(50):
            new_events = await handle.query(SessionWorkflow.get_events_since, first_event_id)
            if new_events:
                break
            await asyncio.sleep(0.2)

        # New events should only be after first_event_id
        assert all(e["event_id"] > first_event_id for e in new_events)

        await handle.signal(SessionWorkflow.end_session)
        await handle.result()
