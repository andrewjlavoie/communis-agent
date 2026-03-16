"""Tests for session activities with mocked LLM backend."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from activities.session_activities import front_agent_respond


def _make_llm_response(text: str):
    return {
        "text": text,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


@pytest.mark.asyncio
async def test_front_agent_direct_response():
    """Front agent answers directly without delegation."""
    response_json = json.dumps({
        "text": "Hello! How can I help you today?",
        "delegate_tasks": [],
    })
    mock_response = _make_llm_response(response_json)

    with patch("activities.session_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await front_agent_respond(
            conversation=[{"role": "user", "content": "Hi"}],
            active_tasks={},
        )

    assert result["text"] == "Hello! How can I help you today?"
    assert result["delegate_tasks"] == []


@pytest.mark.asyncio
async def test_front_agent_delegates_task():
    """Front agent delegates work to a sub-agent."""
    response_json = json.dumps({
        "text": "I'll set up a Python project for you.",
        "delegate_tasks": [
            {"description": "Create Python project structure", "context": "User wants a new project"},
        ],
    })
    mock_response = _make_llm_response(response_json)

    with patch("activities.session_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await front_agent_respond(
            conversation=[
                {"role": "user", "content": "Create a Python project for me"},
            ],
            active_tasks={},
        )

    assert result["text"] == "I'll set up a Python project for you."
    assert len(result["delegate_tasks"]) == 1
    assert result["delegate_tasks"][0]["description"] == "Create Python project structure"


@pytest.mark.asyncio
async def test_front_agent_with_active_tasks():
    """Front agent receives info about active tasks."""
    response_json = json.dumps({
        "text": "Task abc123 is still running — it's setting up the database.",
        "delegate_tasks": [],
    })
    mock_response = _make_llm_response(response_json)

    with patch("activities.session_activities._call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
        result = await front_agent_respond(
            conversation=[{"role": "user", "content": "How's the task going?"}],
            active_tasks={
                "abc123": {
                    "description": "Set up database",
                    "status": "running",
                    "progress": "Creating tables...",
                },
            },
        )

    # Verify active tasks were included in the system prompt
    call_args = mock_llm.call_args
    system_prompt = call_args.kwargs.get("system_prompt", call_args[1].get("system_prompt", ""))
    assert "abc123" in system_prompt
    assert "Set up database" in system_prompt
    assert result["text"] == "Task abc123 is still running — it's setting up the database."


@pytest.mark.asyncio
async def test_front_agent_fallback_on_bad_json():
    """Front agent falls back gracefully if LLM returns malformed JSON."""
    mock_response = _make_llm_response("I'll help you with that right away!")

    with patch("activities.session_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await front_agent_respond(
            conversation=[{"role": "user", "content": "Help me"}],
            active_tasks={},
        )

    # Should fall back to raw text
    assert "help" in result["text"].lower()
    assert result["delegate_tasks"] == []


@pytest.mark.asyncio
async def test_front_agent_multiple_delegations():
    """Front agent can delegate multiple tasks at once."""
    response_json = json.dumps({
        "text": "I'll handle both tasks in parallel.",
        "delegate_tasks": [
            {"description": "Write unit tests", "context": "Testing module X"},
            {"description": "Update docs", "context": "Module X documentation"},
        ],
    })
    mock_response = _make_llm_response(response_json)

    with patch("activities.session_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await front_agent_respond(
            conversation=[{"role": "user", "content": "Write tests and update docs"}],
            active_tasks={},
        )

    assert len(result["delegate_tasks"]) == 2
