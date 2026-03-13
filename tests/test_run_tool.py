"""Tests for the run tool execution engine and presentation layer."""

from __future__ import annotations

import pytest

from tools.run_tool import (
    CommandResult,
    execute_command,
    is_binary,
    present_output,
)


# --- is_binary tests ---


def test_is_binary_empty():
    assert not is_binary("")


def test_is_binary_text():
    assert not is_binary("hello world\nthis is normal text\n")


def test_is_binary_null_bytes():
    assert is_binary("hello\x00world")


def test_is_binary_control_chars():
    # >10% control characters
    data = "\x01\x02\x03\x04\x05\x06\x07\x08" + "ab"  # 80% control
    assert is_binary(data)


def test_is_binary_whitespace_ok():
    # Tabs and newlines should not trigger binary detection
    assert not is_binary("line1\tcolumn2\nline2\tcolumn2\n")


# --- present_output tests ---


def test_present_success():
    result = CommandResult(stdout="hello\n", exit_code=0, duration_ms=12)
    output = present_output(result)
    assert "hello" in output
    assert "[exit:0 | 12ms]" in output


def test_present_failure_with_stderr():
    result = CommandResult(
        stdout="",
        stderr="command not found: foo",
        exit_code=127,
        duration_ms=5,
    )
    output = present_output(result)
    assert "[stderr] command not found: foo" in output
    assert "[exit:127 | 5ms]" in output


def test_present_binary_guard():
    result = CommandResult(
        stdout="PNG\x89\x00\x01binary data here",
        exit_code=0,
        duration_ms=10,
    )
    output = present_output(result)
    assert "[error] binary output" in output
    assert "[exit:0 | 10ms]" in output


def test_present_overflow_truncation():
    # Create output with >200 lines
    lines = [f"line {i}" for i in range(500)]
    big_output = "\n".join(lines)

    result = CommandResult(stdout=big_output, exit_code=0, duration_ms=45)
    output = present_output(result)

    assert "--- output truncated" in output
    assert "500 lines" in output
    assert "Full output:" in output
    assert "Explore:" in output
    assert "[exit:0 | 45ms]" in output
    # Should contain first lines but not all
    assert "line 0" in output
    assert "line 199" in output


def test_present_timeout():
    result = CommandResult(
        stderr="Command timed out after 30s",
        exit_code=124,
        duration_ms=30000,
        timed_out=True,
    )
    output = present_output(result)
    assert "[timeout]" in output
    assert "[exit:124 |" in output


def test_present_duration_formatting():
    # Milliseconds
    result = CommandResult(stdout="ok\n", exit_code=0, duration_ms=45)
    assert "45ms" in present_output(result)

    # Seconds
    result = CommandResult(stdout="ok\n", exit_code=0, duration_ms=3200)
    assert "3.2s" in present_output(result)

    # Minutes
    result = CommandResult(stdout="ok\n", exit_code=0, duration_ms=125000)
    assert "2m" in present_output(result)


def test_present_stderr_on_success():
    result = CommandResult(
        stdout="output\n",
        stderr="warning: something\n",
        exit_code=0,
        duration_ms=10,
    )
    output = present_output(result)
    assert "[stderr] warning: something" in output
    assert "[exit:0 |" in output


# --- execute_command tests ---


@pytest.mark.asyncio
async def test_execute_echo():
    result = await execute_command("echo hello")
    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_execute_pipe():
    result = await execute_command("echo 'line1\nline2\nline3' | wc -l")
    assert result.exit_code == 0
    assert result.stdout.strip() in ("3", "4")  # Depends on echo behavior


@pytest.mark.asyncio
async def test_execute_chain_and():
    result = await execute_command("echo first && echo second")
    assert result.exit_code == 0
    assert "first" in result.stdout
    assert "second" in result.stdout


@pytest.mark.asyncio
async def test_execute_chain_or():
    result = await execute_command("false || echo fallback")
    assert result.exit_code == 0
    assert "fallback" in result.stdout


@pytest.mark.asyncio
async def test_execute_nonexistent_command():
    result = await execute_command("nonexistent_command_xyz_12345")
    assert result.exit_code != 0
    assert result.stderr != ""


@pytest.mark.asyncio
async def test_execute_timeout():
    result = await execute_command("sleep 10", timeout=1)
    assert result.timed_out
    assert result.exit_code == 124


@pytest.mark.asyncio
async def test_execute_with_cwd(tmp_path):
    # Create a file in tmp_path
    (tmp_path / "test.txt").write_text("hello from tmp")
    result = await execute_command("cat test.txt", cwd=str(tmp_path))
    assert result.exit_code == 0
    assert "hello from tmp" in result.stdout
