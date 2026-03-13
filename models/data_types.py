from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_TURNS = 50


@dataclass
class CommunisConfig:
    """Input to the orchestrator workflow."""

    idea: str
    max_turns: int = 0  # 0 = indefinite (capped by DEFAULT_MAX_TURNS)
    model: str = "claude-sonnet-4-5-20250929"
    auto: bool = False  # Skip all feedback prompts
    provider: str = ""  # "anthropic" or "openai" — empty = use env default
    base_url: str = ""  # OpenAI-compatible base URL — empty = use env default
    dangerous: bool = False  # Auto-approve all tool calls (no human-in-the-loop)
    goal_complete_detection: bool = True  # Detect goal completion and stop early
    max_subcommunis: int = 3  # Max parallel subcommuniss the planner can spawn (0 = disabled)


@dataclass
class TurnConfig:
    """Input to a single communis turn child workflow."""

    workspace_dir: str  # Path to workspace directory with turn files
    idea: str
    role: str  # Dynamic role assigned by the planner (e.g., "Devil's Advocate")
    instructions: str  # What this agent should do this turn
    turn_number: int
    max_turns: int
    user_feedback: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 0  # 0 = use model max (configured via MAX_OUTPUT_TOKENS env var)
    provider: str = ""
    base_url: str = ""
    dangerous: bool = False  # Auto-approve tool calls without human confirmation


@dataclass
class TurnResult:
    """Output from a single communis turn child workflow. Metadata only — full content lives in workspace files."""

    turn_number: int
    role: str
    key_insights: list[str] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    artifact_path: str = ""  # Path to the turn file in workspace
    tool_calls_made: int = 0  # Number of tool calls executed this turn


@dataclass
class NextTurnPlan:
    """Output from the plan_next_turn activity."""

    role: str
    instructions: str
    reasoning: str


@dataclass
class SubCommunisTask:
    """A task to be executed by a subcommunis."""

    task: str  # What the subcommunis should accomplish
    max_turns: int = 5  # Turn budget for this subcommunis


@dataclass
class SubCommunisResult:
    """Result from a completed subcommunis."""

    task: str
    status: str  # "complete" | "goal_complete" | "cancelled" | "error"
    summary: str  # Key findings/outputs summarized
    turn_results: list[dict] = field(default_factory=list)
    workspace_dir: str = ""


@dataclass
class CommunisState:
    """Queryable state for the orchestrator workflow."""

    idea: str = ""
    max_turns: int = 0
    current_turn: int = 0
    current_role: str = ""
    status: str = "initializing"  # initializing | running | waiting_for_feedback | complete | cancelled | error
    turn_results: list[TurnResult] = field(default_factory=list)
    latest_message: str = ""
    workspace_dir: str = ""
    dangerous: bool = False  # Whether tool calls auto-approve
    goal_complete: bool = False

    def to_dict(self) -> dict:
        return {
            "idea": self.idea,
            "max_turns": self.max_turns,
            "current_turn": self.current_turn,
            "current_role": self.current_role,
            "status": self.status,
            "turn_results": [
                {
                    "turn_number": r.turn_number,
                    "role": r.role,
                    "key_insights": r.key_insights,
                    "token_usage": r.token_usage,
                    "truncated": r.truncated,
                    "artifact_path": r.artifact_path,
                    "tool_calls_made": r.tool_calls_made,
                }
                for r in self.turn_results
            ],
            "latest_message": self.latest_message,
            "workspace_dir": self.workspace_dir,
            "dangerous": self.dangerous,
            "goal_complete": self.goal_complete,
        }
