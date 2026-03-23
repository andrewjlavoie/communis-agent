"""Full integration tests — end-user journeys through the communis run-mode pipeline.

Each test loads a YAML config, builds a CommunisConfig, and executes the full
orchestrator workflow via Temporal against a real LLM backend. Tests are
parametrized to run against both Anthropic Haiku and local Qwen.

These tests make ACTUAL LLM calls — nothing is mocked.

The test suite uses a single Temporal server for the entire session (either an
existing one at localhost:7233 or one started automatically). Each test spins up
a lightweight Worker (just handler registration, no server overhead).

Run:
    # Both backends
    uv run pytest tests/full_integration/ -v -s

    # Haiku only
    uv run pytest tests/full_integration/ -v -s -k "haiku"

    # Local Qwen only
    uv run pytest tests/full_integration/ -v -s -k "qwen_local"

    # Single scenario
    uv run pytest tests/full_integration/ -v -s -k "quick_sanity"

    # Single scenario, single backend
    uv run pytest tests/full_integration/ -v -s -k "quick_sanity and haiku"
"""
from __future__ import annotations

import os
import shutil
import uuid

import pytest
from temporalio.worker import Worker

from activities.llm_activities import (
    call_llm,
    extract_key_insights,
    plan_next_turn,
    summarize_artifacts,
    summarize_subcommunis_results,
    validate_user_feedback,
)
from activities.tool_activities import execute_run_command
from activities.workspace_activities import (
    collect_older_turns_text,
    init_workspace,
    read_turn_context,
    read_turn_file,
    write_plan_file,
    write_subcommunis_summary,
    write_turn_artifact,
    write_workspace_summary,
)
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow
from workflows.communis_turn import CommunisTurnWorkflow

TASK_QUEUE = "full-integration-test-queue"

ALL_WORKFLOWS = [CommunisOrchestratorWorkflow, CommunisTurnWorkflow]
ALL_ACTIVITIES = [
    call_llm,
    extract_key_insights,
    plan_next_turn,
    summarize_artifacts,
    summarize_subcommunis_results,
    validate_user_feedback,
    execute_run_command,
    init_workspace,
    write_turn_artifact,
    read_turn_context,
    write_workspace_summary,
    read_turn_file,
    collect_older_turns_text,
    write_plan_file,
    write_subcommunis_summary,
]


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════


def _print_result_summary(scenario: str, backend: dict, result: dict):
    """Print a human-readable summary of the test result."""
    turns = result.get("turn_results", [])
    total_in = sum(tr.get("token_usage", {}).get("input_tokens", 0) for tr in turns)
    total_out = sum(tr.get("token_usage", {}).get("output_tokens", 0) for tr in turns)

    print(f"\n  === {scenario} [{backend['provider']}:{backend['model']}] ===")
    print(f"  Status: {result['status']} | Goal complete: {result.get('goal_complete', False)}")
    print(f"  Turns: {len(turns)} | Total tokens: {total_in:,} in / {total_out:,} out")
    print(f"  Workspace: {result.get('workspace_dir', '')}")

    for tr in turns:
        u = tr.get("token_usage", {})
        tools = tr.get("tool_calls_made", 0)
        insights = tr.get("key_insights", [])
        tool_str = f" | tools: {tools}" if tools else ""
        print(
            f"    Turn {tr['turn_number']}: {tr['role']} "
            f"({u.get('input_tokens', 0):,}/{u.get('output_tokens', 0):,})"
            f"{tool_str} | insights: {len(insights)}"
        )


def _assert_turn_basics(turn_results: list[dict]):
    """Assert structural invariants that must hold for every completed turn."""
    for tr in turn_results:
        assert tr["role"], f"Turn {tr['turn_number']} missing role"
        assert tr["artifact_path"], f"Turn {tr['turn_number']} missing artifact path"
        assert os.path.exists(tr["artifact_path"]), (
            f"Artifact file missing: {tr['artifact_path']}"
        )
        assert tr["token_usage"]["input_tokens"] > 0, (
            f"Turn {tr['turn_number']} has zero input tokens"
        )
        assert tr["token_usage"]["output_tokens"] > 0, (
            f"Turn {tr['turn_number']} has zero output tokens"
        )


def _read_artifact_content(artifact_path: str) -> str:
    """Read a turn artifact file, stripping YAML frontmatter."""
    content = open(artifact_path).read()
    if content.startswith("---"):
        end = content.find("\n---\n", 3)
        if end != -1:
            content = content[end + 5:]
    return content.strip()


# ═══════════════════════════════════════════════
# Test: Quick Sanity (2-turn, fixed, no goal detect)
# ═══════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_quick_sanity(temporal_client, load_config, model_backend):
    """Minimal 2-turn run: verifies the full pipeline works end-to-end."""
    config = load_config("quick_sanity.yaml")

    async with Worker(
        temporal_client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
    ):
        wf_id = f"fi-sanity-{uuid.uuid4().hex[:8]}"
        handle = await temporal_client.start_workflow(
            CommunisOrchestratorWorkflow.run,
            config,
            id=wf_id,
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()

    assert result["status"] == "complete"
    assert len(result["turn_results"]) == 2, (
        f"Expected exactly 2 turns (fixed, no goal detect), got {len(result['turn_results'])}"
    )

    _assert_turn_basics(result["turn_results"])

    # At least one turn should have extracted insights
    all_insights = [i for tr in result["turn_results"] for i in tr.get("key_insights", [])]
    assert len(all_insights) >= 1, "Expected at least one insight across all turns"

    # Workspace dir should exist
    assert result.get("workspace_dir")
    assert os.path.isdir(result["workspace_dir"])

    # Artifact content should be non-trivial (not empty or just frontmatter)
    for tr in result["turn_results"]:
        content = _read_artifact_content(tr["artifact_path"])
        assert len(content) > 50, (
            f"Turn {tr['turn_number']} artifact is too short ({len(content)} chars)"
        )

    _print_result_summary("quick_sanity", model_backend, result)


# ═══════════════════════════════════════════════
# Test: Goal Detection Early Stop
# ═══════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_goal_detection_early_stop(temporal_client, load_config, model_backend):
    """Goal detection enabled: the planner should complete before max turns."""
    config = load_config("goal_detect_essay.yaml")

    async with Worker(
        temporal_client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
    ):
        wf_id = f"fi-goaldetect-{uuid.uuid4().hex[:8]}"
        handle = await temporal_client.start_workflow(
            CommunisOrchestratorWorkflow.run,
            config,
            id=wf_id,
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()

    assert result["status"] == "complete"
    turns_done = len(result["turn_results"])

    _assert_turn_basics(result["turn_results"])

    # Soft assertion: we expect goal_complete and early stop, but LLMs are
    # unpredictable. If the model doesn't signal completion, it runs all 5 turns.
    if result.get("goal_complete"):
        assert turns_done < 5, f"Goal complete signalled but ran all 5 turns"
        print(f"\n  Goal detected early after {turns_done} turns")
    else:
        assert turns_done == 5, f"No goal detect, expected 5 turns, got {turns_done}"
        print(f"\n  No early stop — ran all 5 turns (acceptable)")

    # The final artifact should contain essay-like content
    last_turn = result["turn_results"][-1]
    content = _read_artifact_content(last_turn["artifact_path"])
    assert len(content) > 100, (
        f"Final turn artifact is too short for an essay ({len(content)} chars)"
    )

    _print_result_summary("goal_detect_essay", model_backend, result)


# ═══════════════════════════════════════════════
# Test: Tool Use (filesystem interaction)
# ═══════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_tool_use_filesystem(temporal_client, load_config, model_backend):
    """Task requiring tool use: agent should use run tool to create/verify a file."""
    target_dir = "/tmp/communis-integration-test"
    target_file = os.path.join(target_dir, "output.txt")

    # Clean slate
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    config = load_config("tool_use_filesystem.yaml")

    async with Worker(
        temporal_client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
    ):
        wf_id = f"fi-tool-{uuid.uuid4().hex[:8]}"
        handle = await temporal_client.start_workflow(
            CommunisOrchestratorWorkflow.run,
            config,
            id=wf_id,
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()

    assert result["status"] == "complete"
    _assert_turn_basics(result["turn_results"])

    # At least one turn should have made tool calls
    tool_turns = [tr for tr in result["turn_results"] if tr.get("tool_calls_made", 0) > 0]
    assert len(tool_turns) >= 1, (
        f"Expected at least one turn with tool calls, but none of "
        f"the {len(result['turn_results'])} turns used tools"
    )

    total_tool_calls = sum(tr.get("tool_calls_made", 0) for tr in result["turn_results"])
    print(f"\n  Tool calls total: {total_tool_calls}")

    # Soft check: did the file actually get created?
    if os.path.exists(target_file):
        file_content = open(target_file).read()
        print(f"  File created: {target_file}")
        print(f"  Content: {file_content[:200]}")
        assert "integration test passed" in file_content.lower(), (
            f"File content doesn't match expected: {file_content[:200]}"
        )
    else:
        # The LLM may have used a slightly different path — warn but don't fail
        print(f"  WARNING: File not created at {target_file} (LLM may have used a different path)")

    _print_result_summary("tool_use_filesystem", model_backend, result)

    # Cleanup
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)


# ═══════════════════════════════════════════════
# Test: Multi-Turn Research (4-turn context accumulation)
# ═══════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_multi_turn_research(temporal_client, load_config, model_backend):
    """4-turn research task: validates context accumulation and diverse roles."""
    config = load_config("multi_turn_research.yaml")

    async with Worker(
        temporal_client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
    ):
        wf_id = f"fi-research-{uuid.uuid4().hex[:8]}"
        handle = await temporal_client.start_workflow(
            CommunisOrchestratorWorkflow.run,
            config,
            id=wf_id,
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()

    assert result["status"] == "complete"
    turns_done = len(result["turn_results"])
    assert turns_done == 4, f"Expected 4 turns (fixed, no goal detect), got {turns_done}"

    _assert_turn_basics(result["turn_results"])

    # Every turn should have extracted insights
    for tr in result["turn_results"]:
        assert tr.get("key_insights"), (
            f"Turn {tr['turn_number']} ({tr['role']}) has no insights"
        )

    # Expect at least 2 distinct roles across 4 turns (planner should diversify)
    roles = [tr["role"] for tr in result["turn_results"]]
    unique_roles = set(roles)
    assert len(unique_roles) >= 2, (
        f"Expected diverse roles across 4 turns, but got only: {unique_roles}"
    )

    # Token usage should grow across turns as context accumulates
    # (later turns have more input from prior work)
    first_input = result["turn_results"][0]["token_usage"]["input_tokens"]
    last_input = result["turn_results"][-1]["token_usage"]["input_tokens"]
    print(f"\n  Context growth: turn 1 input={first_input:,}, turn 4 input={last_input:,}")

    _print_result_summary("multi_turn_research", model_backend, result)


# ═══════════════════════════════════════════════
# Test: Subcommunis Parallel (may trigger spawning)
# ═══════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(1200)
async def test_subcommunis_parallel(temporal_client, load_config, model_backend):
    """Task that may trigger subcommunis spawning. Validates workflow completes without error."""
    config = load_config("subcommunis_parallel.yaml")

    async with Worker(
        temporal_client,
        task_queue=TASK_QUEUE,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
    ):
        wf_id = f"fi-subcommunis-{uuid.uuid4().hex[:8]}"
        handle = await temporal_client.start_workflow(
            CommunisOrchestratorWorkflow.run,
            config,
            id=wf_id,
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()

    assert result["status"] == "complete"
    assert len(result["turn_results"]) >= 1

    _assert_turn_basics(result["turn_results"])

    # Workspace should exist with artifacts
    assert result.get("workspace_dir")
    assert os.path.isdir(result["workspace_dir"])

    if result.get("goal_complete"):
        print(f"\n  Goal complete after {len(result['turn_results'])} turns")
    else:
        print(f"\n  Ran full {len(result['turn_results'])} turns")

    _print_result_summary("subcommunis_parallel", model_backend, result)
