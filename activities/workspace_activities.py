from __future__ import annotations

import os
from pathlib import Path

import yaml
from temporalio import activity

WORKSPACE_BASE = os.getenv("AUTORIFF_WORKSPACE", ".autoriff")

# How many recent turn files to include as full content in context
MAX_RECENT_FILES = 3


def _workspace_dir(workflow_id: str) -> Path:
    return Path(WORKSPACE_BASE) / workflow_id


def _turn_filename(turn_number: int, role: str) -> str:
    safe_role = role.lower().replace(" ", "-").replace("'", "")[:30]
    return f"turn-{turn_number:02d}-{safe_role}.md"


def _parse_turn_file(text: str) -> dict:
    """Parse a turn markdown file with YAML frontmatter."""
    if not text.startswith("---"):
        return {"meta": {}, "content": text}

    end = text.find("\n---\n", 3)
    if end == -1:
        return {"meta": {}, "content": text}

    frontmatter = text[4:end]
    content = text[end + 5:]

    try:
        meta = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        meta = {}

    return {"meta": meta, "content": content}


def _build_turn_file(
    turn_number: int,
    role: str,
    content: str,
    key_insights: list[str],
    token_usage: dict[str, int],
    truncated: bool,
) -> str:
    """Build a turn markdown file with YAML frontmatter."""
    meta = {
        "turn": turn_number,
        "role": role,
        "key_insights": key_insights,
        "token_usage": token_usage,
        "truncated": truncated,
    }
    frontmatter = yaml.dump(meta, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{frontmatter}\n---\n{content}"


@activity.defn
async def init_workspace(workflow_id: str, idea: str, num_turns: int, model: str) -> str:
    """Create workspace directory and write riff.md manifest. Returns workspace_dir path."""
    ws = _workspace_dir(workflow_id)
    ws.mkdir(parents=True, exist_ok=True)

    manifest = {
        "idea": idea,
        "num_turns": num_turns,
        "model": model,
        "workflow_id": workflow_id,
    }
    manifest_yaml = yaml.dump(manifest, default_flow_style=False, sort_keys=False).strip()
    riff_md = f"---\n{manifest_yaml}\n---\n# {idea}\n"
    (ws / "riff.md").write_text(riff_md)

    return str(ws)


@activity.defn
async def write_turn_artifact(
    workspace_dir: str,
    turn_number: int,
    role: str,
    content: str,
    key_insights: list[str],
    token_usage: dict[str, int],
    truncated: bool,
) -> str:
    """Write turn output to markdown file with YAML frontmatter. Returns file path."""
    ws = Path(workspace_dir)
    filename = _turn_filename(turn_number, role)
    file_text = _build_turn_file(turn_number, role, content, key_insights, token_usage, truncated)
    path = ws / filename
    path.write_text(file_text)
    return str(path)


@activity.defn
async def read_turn_context(workspace_dir: str, current_turn: int) -> dict:
    """Read workspace to build context for the current turn.

    Returns:
        {
            "summary": str,            # contents of summary.md (empty if none)
            "recent_turns": [           # last N turn files, parsed
                {"turn": int, "role": str, "content": str, "key_insights": [str]}
            ]
        }
    """
    ws = Path(workspace_dir)

    # Read rolling summary
    summary_path = ws / "summary.md"
    summary = summary_path.read_text() if summary_path.exists() else ""

    # Find and read recent turn files (the ones before current_turn)
    turn_files: list[tuple[int, Path]] = []
    for path in sorted(ws.glob("turn-*.md")):
        parsed = _parse_turn_file(path.read_text())
        meta = parsed["meta"]
        tn = meta.get("turn", 0)
        if isinstance(tn, int) and tn < current_turn:
            turn_files.append((tn, path))

    # Take last N
    recent_paths = turn_files[-MAX_RECENT_FILES:]

    recent_turns = []
    for _, path in recent_paths:
        parsed = _parse_turn_file(path.read_text())
        meta = parsed["meta"]
        recent_turns.append({
            "turn": meta.get("turn", 0),
            "role": meta.get("role", "Unknown"),
            "content": parsed["content"],
            "key_insights": meta.get("key_insights", []),
        })

    return {"summary": summary, "recent_turns": recent_turns}


@activity.defn
async def write_workspace_summary(workspace_dir: str, summary: str) -> None:
    """Write or update summary.md in the workspace."""
    ws = Path(workspace_dir)
    (ws / "summary.md").write_text(summary)


@activity.defn
async def read_turn_file(workspace_dir: str, turn_number: int) -> dict | None:
    """Read a specific turn file by number. Returns parsed metadata + content, or None."""
    ws = Path(workspace_dir)
    for path in ws.glob(f"turn-{turn_number:02d}-*.md"):
        parsed = _parse_turn_file(path.read_text())
        return {**parsed["meta"], "content": parsed["content"]}
    return None


@activity.defn
async def collect_older_turns_text(workspace_dir: str, before_turn: int) -> str:
    """Read older turn files (before a threshold) and return them as concatenated text for summarization."""
    ws = Path(workspace_dir)
    parts = []
    for path in sorted(ws.glob("turn-*.md")):
        parsed = _parse_turn_file(path.read_text())
        meta = parsed["meta"]
        tn = meta.get("turn", 0)
        if isinstance(tn, int) and tn < before_turn:
            parts.append(f"Turn {tn} ({meta.get('role', 'Unknown')}):\n{parsed['content']}")
    return "\n\n---\n\n".join(parts)
