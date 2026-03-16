"""Tests for LLM activities with mocked LLM backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from activities.llm_activities import (
    call_llm,
    extract_key_insights,
    plan_next_turn,
    summarize_artifacts,
    summarize_subcommunis_results,
    validate_user_feedback,
)


def _make_llm_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a normalized LLM response dict (matches _call_llm return format)."""
    return {
        "text": text,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


@pytest.mark.asyncio
async def test_call_llm():
    mock_response = _make_llm_response("Hello world", 150, 75)

    with patch("activities.llm_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await call_llm(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
            model="claude-sonnet-4-5-20250929",
        )

    assert result["text"] == "Hello world"
    assert result["usage"]["input_tokens"] == 150
    assert result["usage"]["output_tokens"] == 75


@pytest.mark.asyncio
async def test_plan_next_turn():
    plan_json = '{"role": "Devil\'s Advocate", "instructions": "Challenge assumptions.", "reasoning": "Need critical analysis."}'
    mock_response = _make_llm_response(plan_json)

    with patch("activities.llm_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await plan_next_turn("Some context about the idea")

    assert result["role"] == "Devil's Advocate"
    assert result["instructions"] == "Challenge assumptions."
    assert result["reasoning"] == "Need critical analysis."
    assert result["goal_complete"] is False
    assert result["action"] == "step"
    assert result["subcommunis"] == []


@pytest.mark.asyncio
async def test_plan_next_turn_fallback():
    """Test that malformed JSON falls back to Explorer."""
    mock_response = _make_llm_response("I think we should explore more")

    with patch("activities.llm_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await plan_next_turn("Some context")

    assert result["role"] == "Explorer"


@pytest.mark.asyncio
async def test_extract_key_insights():
    insights_json = '["Market is growing", "Users want simplicity", "Mobile-first approach"]'
    mock_response = _make_llm_response(insights_json)

    with patch("activities.llm_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await extract_key_insights("Some content about the idea")

    assert len(result) == 3
    assert "Market is growing" in result


@pytest.mark.asyncio
async def test_summarize_artifacts():
    mock_response = _make_llm_response("Summarized content here.")

    with patch("activities.llm_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await summarize_artifacts("Turn 1: research\nTurn 2: prototype")

    assert result == "Summarized content here."


@pytest.mark.asyncio
async def test_validate_user_feedback_relevant():
    mock_response = _make_llm_response('{"relevant": true, "reason": "Directly related to idea."}')

    with patch("activities.llm_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await validate_user_feedback("Focus on enterprise users", "SaaS product")

    assert result["relevant"] is True


@pytest.mark.asyncio
async def test_validate_user_feedback_irrelevant():
    mock_response = _make_llm_response('{"relevant": false, "reason": "Not related to the idea."}')

    with patch("activities.llm_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await validate_user_feedback("What's the weather?", "SaaS product")

    assert result["relevant"] is False


@pytest.mark.asyncio
async def test_summarize_subcommunis_results():
    mock_response = _make_llm_response("Subcommunis A found X. Subcommunis B found Y.")

    with patch("activities.llm_activities._call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await summarize_subcommunis_results(
            "Task: Research A\nSummary: Found X", "Build a product"
        )

    assert "Subcommunis" in result
