"""Temporal activities for tool execution."""

from __future__ import annotations

from temporalio import activity

from tools.run_tool import execute_command, present_output


@activity.defn
async def execute_run_command(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
) -> dict:
    """Execute a shell command and return LLM-friendly output.

    This activity wraps the two-layer execution engine:
      Layer 1: Raw Unix execution via subprocess
      Layer 2: Presentation (binary guard, overflow, metadata footer, stderr)

    Returns dict with:
      - output: str — presented output for LLM consumption
      - exit_code: int — process exit code
      - duration_ms: int — execution time in milliseconds
    """
    result = await execute_command(command, cwd=cwd, timeout=timeout)
    presented = present_output(result)

    return {
        "output": presented,
        "exit_code": result.exit_code,
        "duration_ms": result.duration_ms,
    }
