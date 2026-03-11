from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiffConfig:
    """Input to the orchestrator workflow."""

    idea: str
    num_turns: int = 3
    model: str = "claude-sonnet-4-5-20250929"
    auto: bool = False  # Skip all feedback prompts
    provider: str = ""  # "anthropic" or "openai" — empty = use env default
    base_url: str = ""  # OpenAI-compatible base URL — empty = use env default


@dataclass
class TurnConfig:
    """Input to a single riff turn child workflow."""

    workspace_dir: str  # Path to workspace directory with turn files
    idea: str
    role: str  # Dynamic role assigned by the planner (e.g., "Devil's Advocate")
    instructions: str  # What this agent should do this turn
    turn_number: int
    total_turns: int
    user_feedback: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 0  # 0 = use model max (configured via MAX_OUTPUT_TOKENS env var)
    provider: str = ""
    base_url: str = ""


@dataclass
class TurnResult:
    """Output from a single riff turn child workflow. Metadata only — full content lives in workspace files."""

    turn_number: int
    role: str
    key_insights: list[str] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    artifact_path: str = ""  # Path to the turn file in workspace


@dataclass
class NextTurnPlan:
    """Output from the plan_next_turn activity."""

    role: str
    instructions: str
    reasoning: str


@dataclass
class RiffState:
    """Queryable state for the orchestrator workflow."""

    idea: str = ""
    num_turns: int = 0
    current_turn: int = 0
    current_role: str = ""
    status: str = "initializing"  # initializing | running | waiting_for_feedback | complete | cancelled | error
    turn_results: list[TurnResult] = field(default_factory=list)
    latest_message: str = ""
    workspace_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "idea": self.idea,
            "num_turns": self.num_turns,
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
                }
                for r in self.turn_results
            ],
            "latest_message": self.latest_message,
            "workspace_dir": self.workspace_dir,
        }
