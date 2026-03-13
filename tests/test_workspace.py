"""Tests for workspace file I/O activities."""

from __future__ import annotations

from pathlib import Path

import pytest

from activities.workspace_activities import (
    _build_turn_file,
    _parse_turn_file,
    _turn_filename,
    collect_older_turns_text,
    init_workspace,
    read_turn_context,
    read_turn_file,
    write_plan_file,
    write_subcommunis_summary,
    write_turn_artifact,
    write_workspace_summary,
)


@pytest.fixture
def workspace(tmp_path):
    """Provide a temporary workspace directory."""
    return tmp_path / "test-workspace"


def test_turn_filename():
    assert _turn_filename(1, "Explorer") == "turn-01-explorer.md"
    assert _turn_filename(2, "Devil's Advocate") == "turn-02-devils-advocate.md"
    assert _turn_filename(10, "Systems Thinker") == "turn-10-systems-thinker.md"


def test_build_and_parse_turn_file():
    content = "# Research Output\n\nSome findings here."
    file_text = _build_turn_file(
        turn_number=1,
        role="Explorer",
        content=content,
        key_insights=["insight 1", "insight 2"],
        token_usage={"input_tokens": 100, "output_tokens": 50},
        truncated=False,
    )

    parsed = _parse_turn_file(file_text)
    assert parsed["meta"]["turn"] == 1
    assert parsed["meta"]["role"] == "Explorer"
    assert parsed["meta"]["key_insights"] == ["insight 1", "insight 2"]
    assert parsed["meta"]["truncated"] is False
    assert parsed["content"] == content


def test_parse_turn_file_no_frontmatter():
    parsed = _parse_turn_file("Just plain content.")
    assert parsed["meta"] == {}
    assert parsed["content"] == "Just plain content."


@pytest.mark.asyncio
async def test_init_workspace(workspace):
    import activities.workspace_activities as mod
    orig = mod.WORKSPACE_BASE
    mod.WORKSPACE_BASE = str(workspace.parent)

    try:
        ws = await init_workspace("test-workspace", "A cool idea", 3, "claude-sonnet-4-5-20250929")
        assert Path(ws).exists()
        riff_md = (Path(ws) / "communis.md").read_text()
        assert "A cool idea" in riff_md
    finally:
        mod.WORKSPACE_BASE = orig


@pytest.mark.asyncio
async def test_write_and_read_turn_artifact(workspace):
    workspace.mkdir(parents=True)

    path = await write_turn_artifact(
        str(workspace), 1, "Explorer",
        "# My Output\nHere it is.",
        ["insight a", "insight b"],
        {"input_tokens": 200, "output_tokens": 100},
        False,
    )

    assert Path(path).exists()
    assert "turn-01-explorer.md" in path

    # Read it back
    result = await read_turn_file(str(workspace), 1)
    assert result is not None
    assert result["turn"] == 1
    assert result["role"] == "Explorer"
    assert "My Output" in result["content"]
    assert result["key_insights"] == ["insight a", "insight b"]


@pytest.mark.asyncio
async def test_read_turn_context_empty(workspace):
    workspace.mkdir(parents=True)
    ctx = await read_turn_context(str(workspace), 1)
    assert ctx["summary"] == ""
    assert ctx["recent_turns"] == []
    assert ctx["plan"] == ""


@pytest.mark.asyncio
async def test_read_turn_context_with_turns(workspace):
    workspace.mkdir(parents=True)

    # Write 3 turn files
    for i in range(1, 4):
        await write_turn_artifact(
            str(workspace), i, f"Role{i}",
            f"Content for turn {i}.",
            [f"insight {i}"],
            {"input_tokens": 100, "output_tokens": 50},
            False,
        )

    # Reading context for turn 4 should see all 3
    ctx = await read_turn_context(str(workspace), 4)
    assert len(ctx["recent_turns"]) == 3
    assert ctx["recent_turns"][0]["turn"] == 1
    assert ctx["recent_turns"][2]["turn"] == 3

    # Reading context for turn 2 should only see turn 1
    ctx = await read_turn_context(str(workspace), 2)
    assert len(ctx["recent_turns"]) == 1
    assert ctx["recent_turns"][0]["turn"] == 1


@pytest.mark.asyncio
async def test_read_turn_context_with_summary(workspace):
    workspace.mkdir(parents=True)
    await write_workspace_summary(str(workspace), "Summary of earlier work.")
    await write_turn_artifact(
        str(workspace), 5, "Builder",
        "Building stuff.",
        ["built it"],
        {"input_tokens": 100, "output_tokens": 50},
        False,
    )

    ctx = await read_turn_context(str(workspace), 6)
    assert ctx["summary"] == "Summary of earlier work."
    assert len(ctx["recent_turns"]) == 1


@pytest.mark.asyncio
async def test_collect_older_turns_text(workspace):
    workspace.mkdir(parents=True)

    for i in range(1, 5):
        await write_turn_artifact(
            str(workspace), i, f"Role{i}",
            f"Content {i}.",
            [f"insight {i}"],
            {"input_tokens": 100, "output_tokens": 50},
            False,
        )

    # Collect turns before turn 3
    text = await collect_older_turns_text(str(workspace), 3)
    assert "Turn 1" in text
    assert "Turn 2" in text
    assert "Turn 3" not in text
    assert "Turn 4" not in text


@pytest.mark.asyncio
async def test_read_turn_file_missing(workspace):
    workspace.mkdir(parents=True)
    result = await read_turn_file(str(workspace), 99)
    assert result is None


@pytest.mark.asyncio
async def test_write_plan_file(workspace):
    workspace.mkdir(parents=True)
    await write_plan_file(str(workspace), "Step 1 done. Step 2 next.")
    plan_path = workspace / "plan.md"
    assert plan_path.exists()
    assert plan_path.read_text() == "Step 1 done. Step 2 next."


@pytest.mark.asyncio
async def test_write_subcommunis_summary(workspace):
    workspace.mkdir(parents=True)
    await write_subcommunis_summary(str(workspace), 3, "Subcommunis results here.")
    summary_path = workspace / "subcommunis-step-03.md"
    assert summary_path.exists()
    assert summary_path.read_text() == "Subcommunis results here."


@pytest.mark.asyncio
async def test_read_turn_context_with_plan(workspace):
    workspace.mkdir(parents=True)
    await write_plan_file(str(workspace), "Current plan: do X then Y.")
    ctx = await read_turn_context(str(workspace), 1)
    assert ctx["plan"] == "Current plan: do X then Y."
