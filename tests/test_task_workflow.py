"""Tests for TaskWorkflow — sub-agent with LLM + tool loop."""

from __future__ import annotations

import asyncio

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models.session_types import SessionConfig, TaskSpec
from workflows.session_workflow import SessionWorkflow
from workflows.task_workflow import TaskWorkflow

TASK_QUEUE = "test-queue"


# --- Mock activities ---


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
        "text": "Task complete. Here's what I did.",
        "stop_reason": "end_turn",
        "content_blocks": [
            {"type": "text", "text": "Task complete. Here's what I did."},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


@activity.defn(name="execute_run_command")
async def mock_execute_run_command(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
) -> dict:
    return {
        "output": f"mock output for: {command}\n[exit:0 | 5ms]",
        "exit_code": 0,
        "duration_ms": 5,
    }


MOCK_ACTIVITIES = [mock_call_claude, mock_execute_run_command]


@pytest.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


@pytest.mark.asyncio
async def test_task_workflow_simple_completion(env):
    """Task completes without tool use and signals parent."""
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[TaskWorkflow, SessionWorkflow],
        activities=MOCK_ACTIVITIES,
    ):
        # Start session (task needs a parent to signal)
        session_handle = await env.client.start_workflow(
            SessionWorkflow.run,
            SessionConfig(),
            id="session-test-1",
            task_queue=TASK_QUEUE,
        )

        spec = TaskSpec(
            task_id="task-simple-1",
            description="Write a hello world script",
            context="User wants a simple Python script",
            parent_session_id="session-test-1",
        )
        result = await env.client.execute_workflow(
            TaskWorkflow.run,
            spec,
            id="task-simple-1",
            task_queue=TASK_QUEUE,
        )

        assert result["status"] == "completed"
        assert "Task complete" in result["summary"]

        try:
            await session_handle.signal(SessionWorkflow.end_session)
        except Exception:
            pass
        try:
            await session_handle.result()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_task_workflow_with_tool_use_dangerous(env):
    """Task uses tools in dangerous mode (auto-approve)."""
    call_count = 0

    @activity.defn(name="call_claude")
    async def tool_call_claude(
        messages: list[dict],
        system_prompt: str,
        model: str = "",
        max_tokens: int = 0,
        provider: str = "",
        base_url: str = "",
        tools: list[dict] | None = None,
    ) -> dict:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return {
                "text": "",
                "stop_reason": "tool_use",
                "content_blocks": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "tu_001",
                        "name": "run",
                        "input": {"command": "ls -la"},
                    },
                ],
                "usage": {"input_tokens": 200, "output_tokens": 30},
            }
        else:
            return {
                "text": "Based on the listing, done.",
                "stop_reason": "end_turn",
                "content_blocks": [
                    {"type": "text", "text": "Based on the listing, done."},
                ],
                "usage": {"input_tokens": 300, "output_tokens": 100},
            }

    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[TaskWorkflow, SessionWorkflow],
        activities=[tool_call_claude, mock_execute_run_command],
    ):
        session_handle = await env.client.start_workflow(
            SessionWorkflow.run,
            SessionConfig(),
            id="session-test-tool",
            task_queue=TASK_QUEUE,
        )

        spec = TaskSpec(
            task_id="task-tool-1",
            description="List files",
            context="List the directory",
            parent_session_id="session-test-tool",
            dangerous=True,
        )

        result = await env.client.execute_workflow(
            TaskWorkflow.run,
            spec,
            id="task-tool-1",
            task_queue=TASK_QUEUE,
        )

        assert result["status"] == "completed"
        assert result["tool_calls_made"] == 1
        assert call_count == 2

        try:
            await session_handle.signal(SessionWorkflow.end_session)
        except Exception:
            pass
        try:
            await session_handle.result()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_task_workflow_delegation_spawns_child(env):
    """Front agent delegates via delegate_task tool, spawning a TaskWorkflow child."""
    @activity.defn(name="call_claude")
    async def delegate_claude(
        messages: list[dict],
        system_prompt: str,
        model: str = "",
        max_tokens: int = 0,
        provider: str = "",
        base_url: str = "",
        tools: list[dict] | None = None,
    ) -> dict:
        # Sub-agent calls (system prompt has "Task:")
        if "Task:" in system_prompt:
            return {
                "text": "Sub-agent done.",
                "stop_reason": "end_turn",
                "content_blocks": [{"type": "text", "text": "Sub-agent done."}],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        # Front agent: delegate, then finish
        has_tool_result = any(
            isinstance(m.get("content"), list)
            and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in m["content"])
            for m in messages
        )
        if not has_tool_result:
            return {
                "text": "",
                "stop_reason": "tool_use",
                "content_blocks": [{
                    "type": "tool_use",
                    "id": "tu_del",
                    "name": "delegate_task",
                    "input": {"description": "Do some work"},
                }],
                "usage": {"input_tokens": 100, "output_tokens": 30},
            }
        return {
            "text": "Task delegated.",
            "stop_reason": "end_turn",
            "content_blocks": [{"type": "text", "text": "Task delegated."}],
            "usage": {"input_tokens": 200, "output_tokens": 30},
        }

    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[TaskWorkflow, SessionWorkflow],
        activities=[delegate_claude, mock_execute_run_command],
    ):
        handle = await env.client.start_workflow(
            SessionWorkflow.run,
            SessionConfig(dangerous=True),
            id="session-deleg-test",
            task_queue=TASK_QUEUE,
        )

        await handle.signal(SessionWorkflow.user_message, "Do some work")

        # Wait for task_started and task_completed
        for _attempt in range(60):
            events = await handle.query(SessionWorkflow.get_events_since, 0)
            if any(e["event_type"] == "task_completed" for e in events):
                break
            await asyncio.sleep(0.3)

        events = await handle.query(SessionWorkflow.get_events_since, 0)
        assert any(e["event_type"] == "task_started" for e in events)
        assert any(e["event_type"] == "task_completed" for e in events)

        try:
            await handle.signal(SessionWorkflow.end_session)
        except Exception:
            pass
        try:
            await handle.result()
        except Exception:
            pass
