"""Unix-style run(command="...") tool for the communis agent.

Implements the *nix agent philosophy: a single run tool that executes shell
commands, with a two-layer architecture:
  Layer 1 (Execution): Pure Unix semantics — subprocess with shell=True,
    supporting pipes, &&, ||, ; natively via bash.
  Layer 2 (Presentation): LLM-friendly output — binary guard, overflow
    truncation, metadata footer, stderr attachment.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass

# --- Tool definition (Claude tool use JSON schema) ---

RUN_TOOL_DEFINITION = {
    "name": "run",
    "description": (
        "Execute a shell command and return its output. "
        "Supports full Unix semantics: pipes (|), chaining (&&, ||, ;), "
        "redirection, and all standard CLI tools. "
        "Use this tool whenever you need to interact with the filesystem, "
        "run code, search files, or perform any system operation.\n\n"
        "Examples:\n"
        "  run(command=\"ls -la\")  — list files\n"
        "  run(command=\"cat file.txt | grep ERROR | wc -l\")  — count error lines\n"
        "  run(command=\"python3 script.py\")  — run a script\n"
        "  run(command=\"find . -name '*.py' | head 20\")  — find Python files\n"
        "  run(command=\"cat log.txt | sort | uniq -c | sort -rn | head 10\")  — top patterns\n"
        "  run(command=\"curl -sL $URL -o data.csv && head -5 data.csv\")  — download and inspect"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute. Supports pipes (|), chaining (&&, ||, ;), and all standard Unix tools.",
            },
        },
        "required": ["command"],
    },
}

# --- Presentation layer constants ---

MAX_OUTPUT_LINES = 200
MAX_OUTPUT_BYTES = 50 * 1024  # 50KB
BINARY_CONTROL_RATIO = 0.10  # >10% control chars = binary
OVERFLOW_DIR = os.path.join(tempfile.gettempdir(), "communis-cmd-output")

# Track overflow file counter per-process
_overflow_counter = 0


@dataclass
class CommandResult:
    """Raw result from Layer 1 (execution layer)."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    timed_out: bool = False


# --- Layer 1: Execution ---


async def execute_command(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
    env_vars: dict[str, str] | None = None,
) -> CommandResult:
    """Execute a shell command via bash subprocess.

    Layer 1: Pure Unix semantics. Pipes, chains, and redirection are handled
    natively by bash. Output is raw — no truncation or metadata.
    """
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    proc = None
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        return CommandResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
            duration_ms=duration_ms,
        )

    except asyncio.TimeoutError:
        duration_ms = int((time.monotonic() - start) * 1000)
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        return CommandResult(
            stderr=f"Command timed out after {timeout}s",
            exit_code=124,  # Standard timeout exit code
            duration_ms=duration_ms,
            timed_out=True,
        )

    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        return CommandResult(
            stderr=str(e),
            exit_code=1,
            duration_ms=duration_ms,
        )


# --- Layer 2: Presentation ---


def is_binary(data: str) -> bool:
    """Check if data appears to be binary content.

    Detects null bytes and high ratio of control characters.
    """
    if not data:
        return False

    if "\x00" in data:
        return True

    # Check control character ratio (exclude common whitespace)
    sample = data[:4096]  # Only check first 4KB
    control_count = sum(
        1 for c in sample
        if ord(c) < 32 and c not in ("\n", "\r", "\t")
    )
    if len(sample) > 0 and control_count / len(sample) > BINARY_CONTROL_RATIO:
        return True

    return False


def _format_duration(ms: int) -> str:
    """Format milliseconds into human-readable duration."""
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60_000:
        return f"{ms / 1000:.1f}s"
    else:
        mins = ms // 60_000
        secs = (ms % 60_000) / 1000
        return f"{mins}m{secs:.0f}s"


def _save_overflow(full_output: str) -> str:
    """Save full output to temp file and return the path."""
    global _overflow_counter
    os.makedirs(OVERFLOW_DIR, exist_ok=True)
    _overflow_counter += 1
    path = os.path.join(OVERFLOW_DIR, f"cmd-{_overflow_counter}.txt")
    with open(path, "w") as f:
        f.write(full_output)
    return path


def present_output(result: CommandResult) -> str:
    """Process raw command output for LLM consumption.

    Layer 2: Applies binary guard, overflow truncation, metadata footer,
    and stderr attachment. Returns a string safe for LLM context.
    """
    parts = []

    # Binary guard
    if is_binary(result.stdout):
        size_kb = len(result.stdout.encode("utf-8", errors="replace")) / 1024
        parts.append(f"[error] binary output ({size_kb:.1f}KB). The command produced binary data that cannot be displayed as text.")
        parts.append(f"[exit:{result.exit_code} | {_format_duration(result.duration_ms)}]")
        return "\n".join(parts)

    output = result.stdout

    # Overflow handling
    lines = output.split("\n")
    total_lines = len(lines)
    total_bytes = len(output.encode("utf-8", errors="replace"))
    truncated = False

    if total_lines > MAX_OUTPUT_LINES or total_bytes > MAX_OUTPUT_BYTES:
        truncated = True
        # Truncate to MAX_OUTPUT_LINES, rune-safe (split on newlines, not bytes)
        truncated_lines = lines[:MAX_OUTPUT_LINES]
        output = "\n".join(truncated_lines)

        # Also enforce byte limit on the truncated output
        output_bytes = output.encode("utf-8", errors="replace")
        if len(output_bytes) > MAX_OUTPUT_BYTES:
            output = output_bytes[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")

    if output:
        parts.append(output.rstrip("\n"))

    if truncated:
        overflow_path = _save_overflow(result.stdout)
        size_str = f"{total_bytes / 1024:.1f}KB" if total_bytes > 1024 else f"{total_bytes}B"
        parts.append("")
        parts.append(f"--- output truncated ({total_lines} lines, {size_str}) ---")
        parts.append(f"Full output: {overflow_path}")
        parts.append(f"Explore: cat {overflow_path} | grep <pattern>")
        parts.append(f"         cat {overflow_path} | tail 100")

    # Stderr attachment (always include on failure, include on success if non-empty)
    stderr = result.stderr.strip()
    if stderr:
        if result.exit_code != 0:
            parts.append(f"[stderr] {stderr}")
        else:
            # On success, only attach stderr if it has content (warnings, etc.)
            parts.append(f"[stderr] {stderr}")

    # Timeout notice
    if result.timed_out:
        parts.append(f"[timeout] Command timed out and was killed.")

    # Metadata footer
    parts.append(f"[exit:{result.exit_code} | {_format_duration(result.duration_ms)}]")

    return "\n".join(parts)
