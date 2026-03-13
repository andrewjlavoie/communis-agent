"""Tests for data models — serialization round-trips and state management."""

from models.data_types import RiffConfig, RiffState, SubAgentResult, SubAgentTask, TurnConfig, TurnResult


def test_riff_config_defaults():
    config = RiffConfig(idea="test idea")
    assert config.idea == "test idea"
    assert config.max_turns == 0
    assert config.model == "claude-sonnet-4-5-20250929"
    assert config.auto is False
    assert config.goal_complete_detection is True
    assert config.max_subagents == 3


def test_turn_config_defaults():
    config = TurnConfig(
        workspace_dir="/tmp/test-ws",
        idea="test",
        role="Explorer",
        instructions="Explore the idea broadly.",
        turn_number=1,
        max_turns=3,
    )
    assert config.user_feedback == ""
    assert config.max_tokens == 0
    assert config.workspace_dir == "/tmp/test-ws"


def test_turn_result_defaults():
    result = TurnResult(turn_number=1, role="Explorer")
    assert result.key_insights == []
    assert result.token_usage == {}
    assert result.truncated is False
    assert result.artifact_path == ""


def test_riff_state_to_dict():
    state = RiffState(
        idea="test idea",
        max_turns=3,
        current_turn=1,
        current_role="Explorer",
        status="running",
        workspace_dir="/tmp/test-ws",
    )
    d = state.to_dict()
    assert d["idea"] == "test idea"
    assert d["max_turns"] == 3
    assert d["status"] == "running"
    assert d["turn_results"] == []
    assert d["workspace_dir"] == "/tmp/test-ws"
    assert d["goal_complete"] is False


def test_riff_state_with_turn_results():
    state = RiffState(idea="test")
    result = TurnResult(
        turn_number=1,
        role="Explorer",
        key_insights=["insight1", "insight2"],
        token_usage={"input_tokens": 100, "output_tokens": 200},
        artifact_path="/tmp/test-ws/turn-01-explorer.md",
    )
    state.turn_results.append(result)
    d = state.to_dict()
    assert len(d["turn_results"]) == 1
    assert d["turn_results"][0]["role"] == "Explorer"
    assert d["turn_results"][0]["key_insights"] == ["insight1", "insight2"]
    assert d["turn_results"][0]["artifact_path"] == "/tmp/test-ws/turn-01-explorer.md"


def test_turn_config_with_feedback():
    config = TurnConfig(
        workspace_dir="/tmp/test-ws",
        idea="test",
        role="Devil's Advocate",
        instructions="Challenge the assumptions from the exploration.",
        turn_number=2,
        max_turns=3,
        user_feedback="Focus on mobile",
    )
    assert config.user_feedback == "Focus on mobile"
    assert config.role == "Devil's Advocate"


def test_subagent_task_defaults():
    task = SubAgentTask(task="Research API docs")
    assert task.task == "Research API docs"
    assert task.max_turns == 5


def test_subagent_result_defaults():
    result = SubAgentResult(task="Do something", status="complete", summary="Done")
    assert result.task == "Do something"
    assert result.status == "complete"
    assert result.turn_results == []
    assert result.workspace_dir == ""
