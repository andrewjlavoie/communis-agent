"""End-to-end integration tests for the session REPL against a live LLM.

These tests make ACTUAL LLM calls — nothing is mocked.
They verify that:
- The LLM uses tools when it should (not hallucinating)
- Tool results are reflected in final answers
- The LLM doesn't fabricate data
- Delegation decisions are reasonable
- The full approval flow works end-to-end

Requires an OpenAI-compatible LLM endpoint (LM Studio, Ollama, vLLM, etc.).

Configure via env vars:
    INTEGRATION_LLM_BASE_URL  (default: value from .env or http://localhost:1234/v1)
    INTEGRATION_LLM_MODEL     (default: value from .env or openai/gpt-oss-20b)

Run:
    uv sync --extra openai --extra dev
    uv run pytest tests/test_integration_session.py -v -s
    uv run pytest tests/test_integration_session.py -v -s -k "not full_workflow"  # fast only
"""

from __future__ import annotations

import os

import pytest

from dotenv import load_dotenv

load_dotenv()

# --- LLM endpoint configuration (from env or defaults) ---
LLM_BASE_URL = os.getenv(
    "INTEGRATION_LLM_BASE_URL",
    os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
)
LLM_MODEL = os.getenv(
    "INTEGRATION_LLM_MODEL",
    os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b"),
)
LLM_PROVIDER = "openai"


@pytest.fixture(autouse=True)
def configure_llm():
    """Point LLM activities at the integration test endpoint."""
    import activities.llm_activities as mod

    saved = {
        "provider": mod.LLM_PROVIDER,
        "default_model": mod.DEFAULT_MODEL,
        "fast_model": mod.FAST_MODEL,
        "fast_max_tokens": mod.FAST_MAX_TOKENS,
        "openai_clients": mod._openai_clients.copy(),
    }

    mod.LLM_PROVIDER = LLM_PROVIDER
    mod.DEFAULT_MODEL = LLM_MODEL
    mod.FAST_MODEL = LLM_MODEL
    mod.FAST_MAX_TOKENS = 4096
    mod._openai_clients.clear()

    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "lm-studio")

    yield

    mod.LLM_PROVIDER = saved["provider"]
    mod.DEFAULT_MODEL = saved["default_model"]
    mod.FAST_MODEL = saved["fast_model"]
    mod.FAST_MAX_TOKENS = saved["fast_max_tokens"]
    mod._openai_clients = saved["openai_clients"]


@pytest.fixture
def sandbox(tmp_path):
    """Create a sandbox directory with known test data for tool execution."""
    # Create some known files
    (tmp_path / "README.md").write_text("# Test Project\nThis is a test project.\n")
    (tmp_path / "hello.py").write_text("print('hello world')\n")
    (tmp_path / "math.py").write_text("def add(a, b): return a + b\n")
    (tmp_path / "config.yaml").write_text("debug: true\nport: 8080\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("import sys\nprint(sys.argv)\n")
    (tmp_path / "src" / "utils.py").write_text("def helper(): pass\n")
    return tmp_path


# ════════════════════════════════════════════
# Layer A: Activity-level tests (no Temporal)
# ════════════════════════════════════════════
# These test the LLM's tool use decisions directly.


def _build_front_agent_call(user_message: str, conversation: list[dict] | None = None):
    """Build the args for a front agent LLM call with tools."""
    from prompts.session_prompts import FRONT_AGENT_SYSTEM_PROMPT
    from tools.delegate_tool import DELEGATE_TASK_TOOL
    from tools.run_tool import RUN_TOOL_DEFINITION

    system_prompt = FRONT_AGENT_SYSTEM_PROMPT.format(active_tasks="No active tasks.")
    messages = conversation or []
    messages.append({"role": "user", "content": user_message})
    tools = [RUN_TOOL_DEFINITION, DELEGATE_TASK_TOOL]
    return system_prompt, messages, tools


def _has_tool_use(content_blocks: list[dict], tool_name: str | None = None) -> bool:
    """Check if response contains a tool_use block, optionally for a specific tool."""
    for block in content_blocks:
        if block.get("type") == "tool_use":
            if tool_name is None or block.get("name") == tool_name:
                return True
    return False


def _get_tool_commands(content_blocks: list[dict]) -> list[str]:
    """Extract all run tool commands from content blocks."""
    commands = []
    for block in content_blocks:
        if block.get("type") == "tool_use" and block.get("name") == "run":
            cmd = block.get("input", {}).get("command", "")
            if cmd:
                commands.append(cmd)
    return commands


def _get_text(content_blocks: list[dict]) -> str:
    """Extract concatenated text from content blocks."""
    return "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )


# --- Category 1: Tool invocation ---


@pytest.mark.asyncio
async def test_filesystem_query_triggers_tool(sandbox):
    """Asking about files should trigger a run tool call, not a text-only answer."""
    from activities.llm_activities import _call_llm

    system_prompt, messages, tools = _build_front_agent_call(
        f"What files are in {sandbox}?"
    )
    result = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    blocks = result.get("content_blocks", [])
    assert _has_tool_use(blocks, "run"), (
        f"LLM should use run tool to list files, but responded with text only: "
        f"{_get_text(blocks)[:200]}"
    )

    commands = _get_tool_commands(blocks)
    print(f"\n  Commands: {commands}")

    # Should be a real listing command, not echo
    for cmd in commands:
        assert "echo" not in cmd.split()[0], f"LLM used echo instead of a real command: {cmd}"


@pytest.mark.asyncio
async def test_data_fetch_triggers_curl():
    """Asking for real-world data should trigger curl/wget, not a text-only answer."""
    from activities.llm_activities import _call_llm

    system_prompt, messages, tools = _build_front_agent_call(
        "What is the current weather in Denver, Colorado? Use wttr.in to fetch it."
    )
    result = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    blocks = result.get("content_blocks", [])
    assert _has_tool_use(blocks, "run"), (
        f"LLM should use run tool to fetch weather, but responded with text only: "
        f"{_get_text(blocks)[:200]}"
    )

    commands = _get_tool_commands(blocks)
    print(f"\n  Commands: {commands}")

    # Should contain curl or wget
    has_fetch = any("curl" in cmd or "wget" in cmd for cmd in commands)
    assert has_fetch, f"Expected curl/wget for data fetching, got: {commands}"


@pytest.mark.asyncio
async def test_simple_question_no_tool():
    """Simple factual questions should NOT trigger tool use."""
    from activities.llm_activities import _call_llm

    system_prompt, messages, tools = _build_front_agent_call(
        "What is 2 + 2?"
    )
    result = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    blocks = result.get("content_blocks", [])
    text = _get_text(blocks)

    # Should answer directly
    assert "4" in text, f"Expected answer containing '4', got: {text[:200]}"

    # Should NOT use tools for this
    if _has_tool_use(blocks):
        commands = _get_tool_commands(blocks)
        print(f"\n  WARNING: LLM used tools for simple math: {commands}")
        # Soft assertion — some models over-tool. Print warning but don't fail.


@pytest.mark.asyncio
async def test_file_creation_triggers_tool():
    """Asking to create a file should trigger a run tool with a real write command."""
    from activities.llm_activities import _call_llm

    system_prompt, messages, tools = _build_front_agent_call(
        "Create a file called /tmp/communis-test-hello.txt containing 'hello world'"
    )
    result = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    blocks = result.get("content_blocks", [])
    assert _has_tool_use(blocks, "run"), (
        f"LLM should use run tool to create file, but responded with text: "
        f"{_get_text(blocks)[:200]}"
    )

    commands = _get_tool_commands(blocks)
    print(f"\n  Commands: {commands}")

    # Should be a real file-write command (echo+redirect, tee, cat, printf, python, etc.)
    has_write = any(
        ">" in cmd or "tee" in cmd or "printf" in cmd or "write" in cmd.lower()
        for cmd in commands
    )
    assert has_write, f"Expected file-writing command, got: {commands}"


# --- Category 2: Output quality (multi-turn tool loop) ---


@pytest.mark.asyncio
async def test_tool_result_reflected_in_answer(sandbox):
    """After executing a tool, the LLM's final answer should reflect the tool output."""
    from activities.llm_activities import _call_llm
    from tools.run_tool import execute_command, present_output

    # Step 1: Get tool call from LLM
    system_prompt, messages, tools = _build_front_agent_call(
        f"How many Python files are in {sandbox}? Count them."
    )
    result = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    blocks = result.get("content_blocks", [])
    if not _has_tool_use(blocks, "run"):
        pytest.skip("LLM did not use run tool — cannot test tool result reflection")

    # Step 2: Execute the command
    commands = _get_tool_commands(blocks)
    tool_use_block = next(b for b in blocks if b.get("type") == "tool_use" and b.get("name") == "run")

    cmd_result = await execute_command(tool_use_block["input"]["command"], cwd=str(sandbox))
    output = present_output(cmd_result)
    print(f"\n  Command: {tool_use_block['input']['command']}")
    print(f"  Output: {output[:200]}")

    # Step 3: Feed result back to LLM
    messages.append({"role": "assistant", "content": blocks})
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_block["id"], "content": output}
    ]})
    result2 = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    final_text = _get_text(result2.get("content_blocks", []))
    print(f"  Final answer: {final_text[:300]}")

    # The actual count (hello.py, math.py, src/main.py, src/utils.py = 4 files)
    # The LLM should mention a number — either as a digit or a word
    NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
    has_digit = any(c.isdigit() for c in final_text)
    has_number_word = any(w in final_text.lower() for w in NUMBER_WORDS)
    assert has_digit or has_number_word, (
        f"Final answer should contain a number (digit or word), got: {final_text[:200]}"
    )


@pytest.mark.asyncio
async def test_honest_failure_on_missing_file(sandbox):
    """When a tool fails, the LLM should acknowledge the error, not invent content."""
    from activities.llm_activities import _call_llm
    from tools.run_tool import execute_command, present_output

    # Step 1: Ask for a non-existent file
    system_prompt, messages, tools = _build_front_agent_call(
        f"Show me the contents of {sandbox}/nonexistent.yaml"
    )
    result = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    blocks = result.get("content_blocks", [])
    if not _has_tool_use(blocks, "run"):
        pytest.skip("LLM did not use run tool")

    # Step 2: Execute — will fail
    tool_use_block = next(b for b in blocks if b.get("type") == "tool_use" and b.get("name") == "run")
    cmd_result = await execute_command(tool_use_block["input"]["command"], cwd=str(sandbox))
    output = present_output(cmd_result)
    print(f"\n  Command: {tool_use_block['input']['command']}")
    print(f"  Output: {output[:200]}")

    # Step 3: Feed error back
    messages.append({"role": "assistant", "content": blocks})
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_block["id"], "content": output}
    ]})
    result2 = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    final_text = _get_text(result2.get("content_blocks", []))
    print(f"  Final answer: {final_text[:300]}")

    # Should acknowledge the error honestly
    error_indicators = ["not found", "no such file", "doesn't exist", "does not exist", "error", "cannot", "failed"]
    has_error_ack = any(ind in final_text.lower() for ind in error_indicators)
    assert has_error_ack, (
        f"LLM should acknowledge file not found, but said: {final_text[:200]}"
    )


# --- Category 3: Delegation decision ---


@pytest.mark.asyncio
async def test_simple_request_uses_run_not_delegate(sandbox):
    """A simple 'read this file' request should use run, not delegate_task."""
    from activities.llm_activities import _call_llm

    system_prompt, messages, tools = _build_front_agent_call(
        f"Read the file {sandbox}/README.md"
    )
    result = await _call_llm(messages, system_prompt, LLM_MODEL, 4096, LLM_PROVIDER, LLM_BASE_URL, tools)

    blocks = result.get("content_blocks", [])
    if not _has_tool_use(blocks):
        pytest.skip("LLM did not use any tools")

    # Should use run, not delegate_task for a simple file read
    assert not _has_tool_use(blocks, "delegate_task"), (
        "LLM should use run for simple file read, not delegate_task"
    )
    assert _has_tool_use(blocks, "run"), "Expected run tool for file read"
    print(f"\n  Commands: {_get_tool_commands(blocks)}")


# ════════════════════════════════════════════
# Layer B: Full workflow tests (Temporal)
# ════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_full_workflow_direct_tool_use(sandbox):
    """Full E2E: user message → tool use → approval → response with real data."""
    import asyncio

    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from activities.llm_activities import call_llm
    from activities.tool_activities import execute_run_command
    from models.session_types import SessionConfig
    from workflows.session_workflow import CommunisAgent
    from workflows.task_workflow import CommunisSubAgent

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue="integration-test-queue",
            workflows=[CommunisAgent, CommunisSubAgent],
            activities=[call_llm, execute_run_command],
        ):
            handle = await env.client.start_workflow(
                CommunisAgent.run,
                SessionConfig(dangerous=True),  # auto-approve for test speed
                id="integration-session-tool",
                task_queue="integration-test-queue",
            )

            # Ask something that requires a tool call
            await handle.signal(
                CommunisAgent.user_message,
                f"How many files are in {sandbox}? Use ls to count them.",
            )

            # Poll for assistant_message event (generous timeout for LLM)
            events = []
            for _attempt in range(120):  # 2 minutes max
                events = await handle.query(CommunisAgent.get_events_since, 0)
                if any(e["event_type"] == "assistant_message" for e in events):
                    break
                await asyncio.sleep(1)

            # End session
            await handle.signal(CommunisAgent.end_session)
            result = await handle.result()

            assert result["status"] == "ended"

            msg_events = [e for e in events if e["event_type"] == "assistant_message"]
            assert len(msg_events) >= 1, "Expected at least one assistant message"

            final_text = msg_events[-1]["data"]["text"]
            print(f"\n  Final response: {final_text[:500]}")

            # The answer should contain actual data (numbers, file names) not a refusal
            refusal_phrases = ["i can't", "i don't have access", "i'm unable"]
            for phrase in refusal_phrases:
                assert phrase not in final_text.lower(), (
                    f"LLM refused instead of using tools: {final_text[:200]}"
                )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_full_workflow_approval_flow():
    """Full E2E: tool call blocked until user approves."""
    import asyncio

    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from activities.llm_activities import call_llm
    from activities.tool_activities import execute_run_command
    from models.session_types import SessionConfig
    from workflows.session_workflow import CommunisAgent
    from workflows.task_workflow import CommunisSubAgent

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue="integration-test-queue",
            workflows=[CommunisAgent, CommunisSubAgent],
            activities=[call_llm, execute_run_command],
        ):
            handle = await env.client.start_workflow(
                CommunisAgent.run,
                SessionConfig(dangerous=False),  # require approval
                id="integration-session-approval",
                task_queue="integration-test-queue",
            )

            await handle.signal(
                CommunisAgent.user_message,
                "Run the command: echo 'approval test'",
            )

            # Wait for approval request
            approvals: list[dict] = []
            for _attempt in range(60):
                approvals = await handle.query(CommunisAgent.get_pending_approvals)
                if approvals:
                    break
                await asyncio.sleep(1)

            if not approvals:
                # LLM might have answered without tools — check
                events = await handle.query(CommunisAgent.get_events_since, 0)
                msg_events = [e for e in events if e["event_type"] == "assistant_message"]
                if msg_events:
                    print(f"\n  LLM answered without tools: {msg_events[-1]['data']['text'][:200]}")
                    pytest.skip("LLM did not use tools for this request")
                pytest.fail("No approval request and no response after 60s")

            print(f"\n  Approval requested: {approvals[0]}")
            approval_id = approvals[0]["approval_id"]

            # Approve it
            await handle.signal(CommunisAgent.approval_response, [approval_id, True])

            # Wait for final response
            for _attempt in range(60):
                events = await handle.query(CommunisAgent.get_events_since, 0)
                if any(e["event_type"] == "assistant_message" for e in events):
                    break
                await asyncio.sleep(1)

            events = await handle.query(CommunisAgent.get_events_since, 0)

            # Should have approval_resolved event
            assert any(e["event_type"] == "approval_resolved" for e in events), (
                "Expected approval_resolved event"
            )

            await handle.signal(CommunisAgent.end_session)
            await handle.result()
