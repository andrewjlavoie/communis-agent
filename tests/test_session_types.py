"""Tests for session data types — defaults, serialization, event management."""

from models.session_types import (
    ApprovalRequest,
    MAX_EVENTS,
    SessionConfig,
    SessionEvent,
    SessionState,
    TaskSpec,
    TaskStatus,
    TaskUpdate,
)


def test_session_config_defaults():
    config = SessionConfig()
    assert config.model == ""  # empty = env DEFAULT_MODEL resolved at activity layer
    assert config.provider == ""
    assert config.base_url == ""
    assert config.dangerous is False
    assert config.max_concurrent_tasks == 5
    assert config.max_task_depth == 2


def test_session_event_to_dict():
    event = SessionEvent(
        event_id=1,
        timestamp="2026-01-01T00:00:00+00:00",
        event_type="assistant_message",
        data={"text": "Hello"},
    )
    d = event.to_dict()
    assert d["event_id"] == 1
    assert d["event_type"] == "assistant_message"
    assert d["data"]["text"] == "Hello"


def test_task_spec_defaults():
    spec = TaskSpec(
        task_id="task-1",
        description="Do something",
        context="some context",
        parent_session_id="session-1",
    )
    assert spec.model == ""  # empty = inherited from session, resolved at activity layer
    assert spec.dangerous is False
    assert spec.max_tool_iterations == 20
    assert spec.depth == 0
    assert spec.max_depth == 2


def test_task_status_to_dict():
    status = TaskStatus(task_id="task-1", description="Test task", status="running", progress="50%")
    d = status.to_dict()
    assert d["task_id"] == "task-1"
    assert d["status"] == "running"
    assert d["progress"] == "50%"


def test_task_update_defaults():
    update = TaskUpdate(task_id="task-1", update_type="progress", message="Working...")
    assert update.result_summary == ""
    assert update.approval_request == {}


def test_approval_request_to_dict():
    req = ApprovalRequest(
        approval_id="apr-1",
        task_id="task-1",
        task_description="Test task",
        tool_name="run",
        tool_input={"command": "ls"},
        timestamp="2026-01-01T00:00:00+00:00",
    )
    d = req.to_dict()
    assert d["approval_id"] == "apr-1"
    assert d["tool_name"] == "run"
    assert d["resolved"] is False
    assert d["approved"] is None


def test_session_state_defaults():
    state = SessionState()
    assert state.session_id == ""
    assert state.conversation == []
    assert state.tasks == {}
    assert state.pending_approvals == []
    assert state.event_counter == 0
    assert state.events == []
    assert state.status == "active"


def test_session_state_add_event():
    state = SessionState()
    event = state.add_event("assistant_message", {"text": "Hi"})
    assert event.event_id == 1
    assert event.event_type == "assistant_message"
    assert event.data["text"] == "Hi"
    assert len(state.events) == 1
    assert state.event_counter == 1

    event2 = state.add_event("task_started", {"task_id": "task-1"})
    assert event2.event_id == 2
    assert state.event_counter == 2
    assert len(state.events) == 2


def test_session_state_add_event_trims_old():
    state = SessionState()
    for i in range(MAX_EVENTS + 50):
        state.add_event("test", {"i": i})

    assert len(state.events) == MAX_EVENTS
    assert state.event_counter == MAX_EVENTS + 50
    # Oldest event should be trimmed, newest kept
    assert state.events[0].event_id == 51
    assert state.events[-1].event_id == MAX_EVENTS + 50


def test_session_state_to_dict():
    state = SessionState(session_id="s-1", status="active")
    task = TaskStatus(task_id="t-1", description="A task", status="running")
    state.tasks["t-1"] = task
    approval = ApprovalRequest(
        approval_id="a-1", task_id="t-1",
        task_description="A task", tool_name="run",
    )
    state.pending_approvals.append(approval)

    d = state.to_dict()
    assert d["session_id"] == "s-1"
    assert d["status"] == "active"
    assert "t-1" in d["tasks"]
    assert d["tasks"]["t-1"]["status"] == "running"
    assert len(d["pending_approvals"]) == 1
    assert d["pending_approvals"][0]["approval_id"] == "a-1"
