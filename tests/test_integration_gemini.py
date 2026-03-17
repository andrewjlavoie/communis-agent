"""Integration tests for Gemini provider (requires GOOGLE_API_KEY).

Tests actual Gemini API calls through our activity layer to verify:
- Basic connectivity and response format
- Planner returns valid JSON
- Insight extraction returns list of strings
- Tool use works end-to-end (call + result round-trip)

Run:
    uv sync --extra gemini --extra dev
    GOOGLE_API_KEY=... uv run pytest tests/test_integration_gemini.py -v -s
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("google.genai", reason="google-genai not installed (uv sync --extra gemini)")

GEMINI_MODEL = "gemini-2.5-flash-lite-preview"

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)


@pytest.fixture(autouse=True)
def configure_gemini():
    """Patch llm_activities to use Gemini provider."""
    import activities.llm_activities as mod

    saved = {
        "provider": mod.LLM_PROVIDER,
        "default_model": mod.DEFAULT_MODEL,
        "fast_model": mod.FAST_MODEL,
        "fast_max_tokens": mod.FAST_MAX_TOKENS,
        "gemini_client": mod._gemini_client,
    }

    mod.LLM_PROVIDER = "gemini"
    mod.DEFAULT_MODEL = GEMINI_MODEL
    mod.FAST_MODEL = GEMINI_MODEL
    mod.FAST_MAX_TOKENS = 0
    mod._gemini_client = None

    yield

    mod.LLM_PROVIDER = saved["provider"]
    mod.DEFAULT_MODEL = saved["default_model"]
    mod.FAST_MODEL = saved["fast_model"]
    mod.FAST_MAX_TOKENS = saved["fast_max_tokens"]
    mod._gemini_client = saved["gemini_client"]


@pytest.mark.asyncio
async def test_basic_llm_call():
    """Verify basic connectivity and response format."""
    from activities.llm_activities import _call_llm

    result = await _call_llm(
        messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
        system_prompt="You are a helpful assistant.",
        model=GEMINI_MODEL,
        max_tokens=256,
        provider="gemini",
    )

    assert "text" in result
    assert "stop_reason" in result
    assert "usage" in result
    assert "content_blocks" in result
    assert len(result["text"]) > 0
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] > 0
    print(f"\n  Response: {result['text']}")
    print(f"  Usage: {result['usage']}")


@pytest.mark.asyncio
async def test_plan_next_turn():
    """Planner activity returns valid JSON with expected keys."""
    from activities.llm_activities import _call_llm
    from prompts.communis_prompts import PLANNER_PROMPT

    context = (
        "Idea: Build a CLI tool for managing bookmarks.\n"
        "Turn 1 (Explorer): Explored the concept. Key features: tagging, search, sync.\n"
        "Insights so far: Users want fast access, cross-device sync is important."
    )

    result = await _call_llm(
        messages=[{"role": "user", "content": context}],
        system_prompt=PLANNER_PROMPT,
        model=GEMINI_MODEL,
        max_tokens=2048,
        provider="gemini",
    )

    text = result["text"]
    print(f"\n  Planner response: {text[:500]}")
    assert len(text) > 0, "Planner returned empty response"

    import json
    from activities.llm_activities import _parse_llm_json

    parsed = _parse_llm_json(text, {})
    assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"
    assert "role" in parsed or "instructions" in parsed, (
        f"Planner JSON missing expected keys, got: {list(parsed.keys())}"
    )


@pytest.mark.asyncio
async def test_extract_key_insights():
    """Insight extraction returns a list of strings."""
    from activities.llm_activities import _call_llm
    from prompts.communis_prompts import EXTRACT_INSIGHTS_PROMPT
    from activities.llm_activities import _parse_llm_json

    content = (
        "We explored building a bookmark manager CLI. The key features are: "
        "1) Fast fuzzy search using sqlite FTS5. "
        "2) Tag-based organization with hierarchical tags. "
        "3) Import/export in Netscape format for browser compatibility. "
        "4) Optional sync via git repository."
    )

    result = await _call_llm(
        messages=[{"role": "user", "content": content}],
        system_prompt=EXTRACT_INSIGHTS_PROMPT,
        model=GEMINI_MODEL,
        max_tokens=512,
        provider="gemini",
    )

    text = result["text"].strip()
    print(f"\n  Insights response: {text[:500]}")

    parsed = _parse_llm_json(text, [])
    assert isinstance(parsed, list), f"Expected list, got {type(parsed)}: {text[:200]}"
    assert len(parsed) > 0, "No insights extracted"
    for item in parsed:
        assert isinstance(item, str), f"Insight should be string, got {type(item)}"
    print(f"  Extracted {len(parsed)} insights")


@pytest.mark.asyncio
async def test_tool_use_basic():
    """Model should return tool_use blocks when given tool definitions."""
    from activities.llm_activities import _call_llm

    tools = [
        {
            "name": "search_files",
            "description": "Search for files matching a pattern in a directory. Use this tool when the user asks about files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match"},
                    "directory": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
            },
        }
    ]

    result = await _call_llm(
        messages=[{"role": "user", "content": "Find all Python files in /tmp"}],
        system_prompt="You are a file management assistant. Use tools to answer questions about files.",
        model=GEMINI_MODEL,
        max_tokens=1024,
        provider="gemini",
        tools=tools,
    )

    blocks = result.get("content_blocks", [])
    print(f"\n  Stop reason: {result['stop_reason']}")
    print(f"  Blocks: {blocks}")

    tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
    assert len(tool_blocks) > 0, (
        f"Expected tool_use block, got: {blocks}"
    )
    assert tool_blocks[0]["name"] == "search_files"
    assert "id" in tool_blocks[0]
    assert tool_blocks[0]["id"].startswith("gemini_")
    assert result["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_tool_result_round_trip():
    """Send a tool result back and verify the model incorporates it."""
    from activities.llm_activities import _call_llm

    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature and conditions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }
    ]

    # Step 1: Get tool call
    result1 = await _call_llm(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        system_prompt="You are a weather assistant. Always use the get_weather tool to answer weather questions.",
        model=GEMINI_MODEL,
        max_tokens=1024,
        provider="gemini",
        tools=tools,
    )

    blocks1 = result1.get("content_blocks", [])
    tool_blocks = [b for b in blocks1 if b.get("type") == "tool_use"]
    if not tool_blocks:
        pytest.skip("Model did not use tool on first call")

    tool_call = tool_blocks[0]
    print(f"\n  Tool call: {tool_call}")

    # Step 2: Send tool result back
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": blocks1},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": '{"temperature": 18, "conditions": "partly cloudy", "humidity": 65}',
                },
            ],
        },
    ]

    result2 = await _call_llm(
        messages=messages,
        system_prompt="You are a weather assistant. Always use the get_weather tool to answer weather questions.",
        model=GEMINI_MODEL,
        max_tokens=1024,
        provider="gemini",
        tools=tools,
    )

    text = result2["text"]
    print(f"  Final response: {text[:300]}")

    # Model should mention the temperature or conditions from the tool result
    assert "18" in text or "partly cloudy" in text or "cloudy" in text.lower(), (
        f"Model should incorporate tool result data, got: {text[:200]}"
    )
