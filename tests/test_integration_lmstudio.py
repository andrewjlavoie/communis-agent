"""Integration tests against a real LM Studio instance.

These tests make ACTUAL LLM calls — nothing is mocked.
Requires LM Studio running at http://192.168.5.71:1234 with qwen/qwen3.5-9b loaded.

Run:
    uv sync --extra openai
    uv run pytest tests/test_integration_lmstudio.py -v -s
"""

from __future__ import annotations

import os
import shutil

import pytest

# --- LM Studio configuration ---
LM_STUDIO_BASE_URL = "http://192.168.5.71:1234/v1"
LM_STUDIO_MODEL = "qwen/qwen3.5-9b"


@pytest.fixture(autouse=True)
def configure_lmstudio():
    """Reconfigure LLM activities to use LM Studio for each test."""
    import activities.llm_activities as mod

    saved = {
        "provider": mod.LLM_PROVIDER,
        "default_model": mod.DEFAULT_MODEL,
        "fast_model": mod.FAST_MODEL,
        "fast_max_tokens": mod.FAST_MAX_TOKENS,
        "openai_client": mod._openai_client,
    }

    mod.LLM_PROVIDER = "openai"
    mod.DEFAULT_MODEL = LM_STUDIO_MODEL
    mod.FAST_MODEL = LM_STUDIO_MODEL
    # Thinking models (Qwen3, etc.) need extra headroom for reasoning tokens
    mod.FAST_MAX_TOKENS = 4096
    mod._openai_client = None

    os.environ["OPENAI_BASE_URL"] = LM_STUDIO_BASE_URL
    os.environ["OPENAI_API_KEY"] = "lm-studio"

    yield

    mod.LLM_PROVIDER = saved["provider"]
    mod.DEFAULT_MODEL = saved["default_model"]
    mod.FAST_MODEL = saved["fast_model"]
    mod.FAST_MAX_TOKENS = saved["fast_max_tokens"]
    mod._openai_client = saved["openai_client"]


# ────────────────────────────────────────────
# Activity-level tests (no Temporal needed)
# ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_llm_call():
    """Verify basic connectivity and response format from LM Studio."""
    from activities.llm_activities import _call_llm

    result = await _call_llm(
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        system_prompt="You are a helpful assistant. Be concise.",
        model=LM_STUDIO_MODEL,
        max_tokens=4096,  # Thinking models need room for reasoning tokens
    )

    assert isinstance(result, dict)
    assert "text" in result and len(result["text"]) > 0
    assert "stop_reason" in result
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] > 0
    print(f"\n  Response: {result['text'][:200]}")
    print(f"  Stop reason: {result['stop_reason']}")
    print(f"  Tokens: {result['usage']}")


@pytest.mark.asyncio
async def test_plan_next_turn_real():
    """Planner returns a valid role/instructions/reasoning dict."""
    from activities.llm_activities import plan_next_turn

    result = await plan_next_turn(
        "Prompt: Design a simple REST API for a todo app.\n\n"
        "This will be turn 1 of 3. Turns completed so far: 0. "
        "Remaining after this one: 2."
    )

    assert "role" in result and len(result["role"]) > 0
    assert "instructions" in result and len(result["instructions"]) > 0
    assert "reasoning" in result
    print(f"\n  Role: {result['role']}")
    print(f"  Instructions: {result['instructions'][:200]}")
    print(f"  Reasoning: {result['reasoning'][:200]}")


@pytest.mark.asyncio
async def test_extract_key_insights_real():
    """Insight extractor returns a list of strings."""
    from activities.llm_activities import extract_key_insights

    content = (
        "REST APIs should follow resource-oriented design. Key decisions:\n"
        "1. Use nouns for endpoints (/todos, /users), not verbs\n"
        "2. HTTP methods convey the action (GET, POST, PUT, DELETE)\n"
        "3. Return proper status codes (201 for creation, 404 for not found)\n"
        "4. Version the API from day one (/v1/todos)\n"
        "5. Use JSON for request/response bodies\n"
    )

    result = await extract_key_insights(content)

    assert isinstance(result, list)
    assert len(result) >= 1
    for insight in result:
        assert isinstance(insight, str) and len(insight) > 0
    print(f"\n  Insights ({len(result)}):")
    for i, insight in enumerate(result, 1):
        print(f"    {i}. {insight}")


@pytest.mark.asyncio
async def test_summarize_artifacts_real():
    """Summarizer returns non-empty text."""
    from activities.llm_activities import summarize_artifacts

    artifacts = (
        "Turn 1 (Explorer): Researched REST API design patterns. Key findings: "
        "resource-oriented URLs, proper HTTP method usage, status code conventions, "
        "pagination strategies.\n\n"
        "Turn 2 (Architect): Designed the data model with User and Todo entities. "
        "Defined endpoints: GET/POST /todos, GET/PUT/DELETE /todos/:id, POST /users, "
        "POST /auth/login. Chose JWT for authentication."
    )

    result = await summarize_artifacts(artifacts)

    assert isinstance(result, str)
    assert len(result) > 20
    print(f"\n  Summary ({len(result)} chars):\n  {result[:400]}")


@pytest.mark.asyncio
async def test_validate_feedback_relevant():
    """Feedback validator identifies relevant feedback."""
    from activities.llm_activities import validate_user_feedback

    result = await validate_user_feedback(
        "Add pagination to the list endpoint",
        "Design a REST API for a todo app",
    )

    assert "relevant" in result
    assert "reason" in result
    assert isinstance(result["relevant"], bool)
    print(f"\n  Relevant: {result['relevant']}")
    print(f"  Reason: {result['reason']}")


@pytest.mark.asyncio
async def test_validate_feedback_irrelevant():
    """Feedback validator identifies irrelevant feedback."""
    from activities.llm_activities import validate_user_feedback

    result = await validate_user_feedback(
        "What's the weather like in Tokyo?",
        "Design a REST API for a todo app",
    )

    assert "relevant" in result
    assert isinstance(result["relevant"], bool)
    print(f"\n  Relevant: {result['relevant']}")
    print(f"  Reason: {result['reason']}")


# ────────────────────────────────────────────
# Full 3-turn workflow (needs Temporal test env)
# ────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(300)  # 5 minute ceiling for 3 LLM-powered turns
async def test_full_3_turn_workflow():
    """Run a complete 3-turn communis session against LM Studio via Temporal.

    This exercises the entire pipeline end-to-end:
      planner → turn agent → insight extractor → (repeat x3) → final result
    """
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from activities.llm_activities import (
        call_claude,
        extract_key_insights,
        plan_next_turn,
        summarize_artifacts,
        validate_user_feedback,
    )
    from activities.workspace_activities import (
        collect_older_turns_text,
        init_workspace,
        read_turn_context,
        read_turn_file,
        write_turn_artifact,
        write_workspace_summary,
    )
    from models.data_types import CommunisConfig
    from workflows.communis_orchestrator import CommunisOrchestratorWorkflow
    from workflows.communis_turn import CommunisTurnWorkflow

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue="integration-test-queue",
            workflows=[CommunisOrchestratorWorkflow, CommunisTurnWorkflow],
            activities=[
                call_claude,
                plan_next_turn,
                extract_key_insights,
                summarize_artifacts,
                validate_user_feedback,
                init_workspace,
                read_turn_context,
                write_turn_artifact,
                write_workspace_summary,
                read_turn_file,
                collect_older_turns_text,
            ],
        ):
            config = CommunisConfig(
                idea="Design a simple CLI calculator in Python that supports add, subtract, multiply, divide",
                num_turns=3,
                model=LM_STUDIO_MODEL,
                auto=True,
            )

            handle = await env.client.start_workflow(
                CommunisOrchestratorWorkflow.run,
                config,
                id="integration-test-3turn",
                task_queue="integration-test-queue",
            )

            result = await handle.result()

    # --- Assertions ---
    assert result["status"] == "complete", f"Workflow ended with status: {result['status']}"
    assert len(result["turn_results"]) == 3, f"Expected 3 turns, got {len(result['turn_results'])}"

    # Every turn should have a role, insights, and an artifact file
    for tr in result["turn_results"]:
        assert tr["role"], f"Turn {tr['turn_number']} has no role"
        assert tr["artifact_path"], f"Turn {tr['turn_number']} has no artifact"
        assert os.path.exists(tr["artifact_path"]), f"Artifact missing: {tr['artifact_path']}"

    # --- Print summary ---
    print(f"\n  Status: {result['status']}")
    print(f"  Workspace: {result.get('workspace_dir', '')}")
    print(f"  Turns completed: {len(result['turn_results'])}")

    total_input = 0
    total_output = 0
    for tr in result["turn_results"]:
        usage = tr.get("token_usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        total_input += in_tok
        total_output += out_tok
        print(f"\n  Turn {tr['turn_number']}: {tr['role']}")
        print(f"    Tokens: {in_tok:,} in / {out_tok:,} out")
        print(f"    Truncated: {tr.get('truncated', False)}")
        print(f"    Insights: {tr.get('key_insights', [])}")
        print(f"    Artifact: {tr['artifact_path']}")

        # Read and show a snippet of the actual output
        content = open(tr["artifact_path"]).read()
        # Skip YAML frontmatter
        if content.startswith("---"):
            end = content.find("\n---\n", 3)
            if end != -1:
                content = content[end + 5:]
        snippet = content.strip()[:300]
        print(f"    Output preview:\n      {snippet}...")

    print(f"\n  Total tokens: {total_input:,} in / {total_output:,} out")

    # Cleanup workspace
    ws_dir = result.get("workspace_dir", "")
    if ws_dir and os.path.isdir(ws_dir):
        shutil.rmtree(ws_dir)
        print(f"  Cleaned up workspace: {ws_dir}")
