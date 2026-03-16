"""Data types for the session-based agent REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_EVENTS = 500


@dataclass
class SessionConfig:
    """Input to the SessionWorkflow. Acts as the singleton config holder for the
    entire session — all values are passed through to every front agent activity
    call and inherited by every child TaskSpec.

    Model resolution flow:
        CLI (--model flag or "")
          → SessionConfig(model="")
            → SessionWorkflow passes config.model to:
                1. front_agent_respond(model="")   — front agent LLM calls
                2. TaskSpec(model="")              — inherited by sub-agents
              → TaskWorkflow passes spec.model to:
                  call_llm(model="")
                → Activity layer resolves:
                    model = model or DEFAULT_MODEL
                    # DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")

    Empty string means "use env default". The workflow layer never reads env vars
    directly (Temporal sandbox forbids it) — resolution happens in activities.
    """

    user: str = ""  # username for workflow IDs (e.g. "andrew")
    model: str = ""  # "" = env DEFAULT_MODEL → DEFAULT_MODEL_STRING fallback
    provider: str = ""  # "" = env LLM_PROVIDER
    base_url: str = ""  # "" = env OPENAI_BASE_URL
    dangerous: bool = False
    max_concurrent_tasks: int = 5
    max_task_depth: int = 2


@dataclass
class SessionEvent:
    """A single event in the session event stream."""

    event_id: int
    timestamp: str  # ISO8601
    event_type: str  # assistant_message | task_started | task_progress |
    # task_completed | task_failed | approval_requested |
    # approval_resolved | session_ended
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "data": self.data,
        }


@dataclass
class TaskSpec:
    """Input to TaskWorkflow — describes what a sub-agent should do.

    model/provider/base_url are inherited from the parent SessionConfig.
    Empty string means "use env default" — resolved at the activity layer.
    """

    task_id: str
    description: str
    context: str  # relevant conversation context
    parent_session_id: str  # session workflow ID (for signaling back)
    model: str = ""  # inherited from session config; "" = env default
    provider: str = ""
    base_url: str = ""
    dangerous: bool = False
    max_tool_iterations: int = 20
    depth: int = 0
    max_depth: int = 2


@dataclass
class TaskStatus:
    """Tracks the current status of a sub-agent task."""

    task_id: str
    description: str
    status: str = "pending"  # pending | running | waiting_approval | completed | failed | cancelled
    progress: str = ""
    result_summary: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "progress": self.progress,
            "result_summary": self.result_summary,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class TaskUpdate:
    """Signal payload from TaskWorkflow to SessionWorkflow."""

    task_id: str
    update_type: str  # progress | completed | failed | approval_request
    message: str = ""
    result_summary: str = ""
    approval_request: dict = field(default_factory=dict)  # {approval_id, tool_name, tool_input}


@dataclass
class ApprovalRequest:
    """A pending tool approval request from a sub-agent."""

    approval_id: str
    task_id: str
    task_description: str
    tool_name: str
    tool_input: dict = field(default_factory=dict)
    timestamp: str = ""
    resolved: bool = False
    approved: bool | None = None

    def to_dict(self) -> dict:
        return {
            "approval_id": self.approval_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "approved": self.approved,
        }


@dataclass
class SessionState:
    """Full queryable state for a SessionWorkflow."""

    session_id: str = ""
    conversation: list[dict] = field(default_factory=list)  # [{role, content, timestamp}]
    tasks: dict[str, TaskStatus] = field(default_factory=dict)
    pending_approvals: list[ApprovalRequest] = field(default_factory=list)
    event_counter: int = 0
    events: list[SessionEvent] = field(default_factory=list)  # bounded ring buffer
    status: str = "active"

    def add_event(self, event_type: str, data: dict, timestamp: str = "") -> SessionEvent:
        """Create and append a new event, trimming old events if needed.

        timestamp should be passed explicitly from workflow code (via workflow.now()).
        """
        self.event_counter += 1
        event = SessionEvent(
            event_id=self.event_counter,
            timestamp=timestamp,
            event_type=event_type,
            data=data,
        )
        self.events.append(event)
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]
        return event

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "conversation": self.conversation,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "pending_approvals": [a.to_dict() for a in self.pending_approvals],
            "event_counter": self.event_counter,
            "status": self.status,
        }
