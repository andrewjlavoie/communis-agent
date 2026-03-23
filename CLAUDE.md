# CLAUDE.md — communis Project Reference

> Instructions and context for AI assistants (and humans) working in this codebase.

---

## Project Overview

**communis** is a durable AI agent system built on Temporal workflows. It has two modes:

- **`run`** — A self-directing iterative work loop. Give it a prompt and a turn count; an LLM planner decomposes work into steps, assigns dynamic agent roles each turn, and builds on prior output.
- **`chat`** — An interactive agent REPL. A front agent handles conversation and can delegate complex multi-step work to background sub-agents running as child Temporal workflows.

Both modes share the same LLM layer, tool system, and Temporal infrastructure. All LLM calls go through LiteLLM, making the system provider-agnostic — Claude, Gemini, or any OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, etc.).

---

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| **Language** | Python 3.11+ | `uv` for dependency management |
| **LLM** | LiteLLM (`litellm`) | Universal provider — routes to Anthropic, OpenAI-compat, Gemini |
| **Orchestration** | Temporal (`temporalio`) | Durable workflow execution |
| **CLI / TUI** | Rich (`rich`) | Terminal rendering for both modes |
| **Config** | PyYAML (`pyyaml`) | YAML config files for run mode |
| **Testing** | `pytest` + `pytest-asyncio` | With Temporal time-skipping support |

### Model Selection

Two-tier model strategy to balance capability and cost:

| Tier | Used for | Default | Why |
|------|----------|---------|-----|
| **DEFAULT_MODEL** | Planner, turn agent, front agent, sub-agents | `claude-sonnet-4-5-20250929` | Needs reasoning and creativity |
| **FAST_MODEL** | Insight extraction, summarization, feedback validation | `claude-haiku-4-5-20251001` | Mechanical tasks — structured extraction, compression, yes/no checks |

With Claude, Haiku is ~60× cheaper than Sonnet. For local models, set both to the same model.

---

## Architecture

Two parallel systems, both on Temporal:

### Run Mode — Orchestrator + Turn Workflows

```
CommunisOrchestratorWorkflow (parent)
  ├── init_workspace → .communis/<id>/
  ├── while not complete:
  │     ├── read_turn_context → summary.md + plan.md + recent turns
  │     ├── plan_next_turn → LLM picks role + instructions (or goal_complete)
  │     ├── CommunisTurnWorkflow (child):
  │     │     ├── call_llm (with tool use loop)
  │     │     ├── extract_key_insights
  │     │     └── write_turn_artifact → turn-NN-role.md
  │     ├── summarize_artifacts → compress older turns
  │     └── wait for feedback signal (120s timeout)
  └── return results
```

### Chat Mode — Session Agent + Sub-Agents

```
CommunisAgent (front agent — long-running workflow)
  ├── wait for user_message signal
  ├── agent loop: LLM → tool dispatch → repeat
  │     ├── text response → assistant_message event
  │     ├── run tool → execute shell command (with approval gate)
  │     └── delegate_task → spawn CommunisSubAgent as child workflow
  ├── CommunisSubAgent (background):
  │     ├── autonomous LLM + tool loop (up to max_tool_iterations)
  │     ├── approval_requested → signal parent → wait
  │     └── task_update signals → parent emits events
  ├── CLI polls events via get_events_since query
  └── end_session signal → complete
```

### Shared Infrastructure

Both modes share:
- `call_llm` activity — the single LLM entry point (`activities/llm_activities.py`)
- `execute_run_command` activity — shell command execution (`activities/tool_activities.py`)
- `LLM_RETRY_POLICY`, timeout constants (`workflows/constants.py`)
- `TASK_QUEUE` (`shared/constants.py`)

### Key Files

| File | Description |
|------|-------------|
| `models/data_types.py` | CommunisConfig, TurnConfig, TurnResult, CommunisState |
| `models/session_types.py` | SessionConfig, SessionEvent, TaskSpec, TaskStatus, TaskUpdate, ApprovalRequest, SessionState |
| `activities/llm_activities.py` | All LLM calls via LiteLLM — `call_llm`, `plan_next_turn`, `extract_key_insights`, `summarize_artifacts`, `validate_user_feedback`, `summarize_subcommunis_results` |
| `activities/tool_activities.py` | Shell command execution — `execute_run_command` |
| `activities/workspace_activities.py` | Workspace file I/O — 8 activities for reading/writing turn files, summaries, plans |
| `workflows/communis_orchestrator.py` | Run mode — turn loop, feedback, subcommunis, cancellation |
| `workflows/communis_turn.py` | Run mode — single turn execution with tool use loop |
| `workflows/session_workflow.py` | Chat mode — CommunisAgent (front agent) |
| `workflows/task_workflow.py` | Chat mode — CommunisSubAgent (sub-agent) |
| `workflows/constants.py` | LLM_RETRY_POLICY, LLM_TIMEOUT, FAST_LLM_TIMEOUT, IO_TIMEOUT, TOOL_TIMEOUT |
| `tools/run_tool.py` | `run` tool — definition + execution + presentation layers |
| `tools/delegate_tool.py` | `delegate_task` tool definition |
| `prompts/communis_prompts.py` | Run-mode prompts (planner, extraction, summarization, validation) |
| `prompts/session_prompts.py` | Front agent system prompt |
| `cli/main.py` | CLI entry point, run-mode REPL with Rich output |
| `cli/session_cli.py` | Chat-mode REPL with event polling + approval UX |
| `scripts/run_worker.py` | Temporal worker — registers all workflows + activities |
| `shared/constants.py` | TASK_QUEUE, DEFAULT_MODEL_STRING |

---

## LLM Integration — LiteLLM

All LLM calls go through a single function: `_call_llm()` in `activities/llm_activities.py`.

### Message Flow

```
Workflow code (Anthropic-format messages)
  → call_llm activity
    → _convert_messages_to_openai()   # Anthropic → OpenAI format
    → _build_litellm_model()          # provider routing (anthropic/, openai/, gemini/)
    → litellm.acompletion()           # universal LLM call
    → _normalize_litellm_response()   # OpenAI → internal format (Anthropic-style keys)
  ← dict with: text, reasoning, stop_reason, content_blocks, usage
```

### Internal Response Format

The normalized response uses Anthropic-style keys for consistency:

```python
{
    "text": "concatenated text content",
    "reasoning": "thinking/reasoning tokens (if any)",
    "stop_reason": "end_turn" | "tool_use" | "max_tokens",
    "content_blocks": [
        {"type": "text", "text": "..."},
        {"type": "tool_use", "id": "...", "name": "run", "input": {...}},
    ],
    "usage": {"input_tokens": N, "output_tokens": N},
}
```

### Provider Routing

`_build_litellm_model()` maps `(provider, model)` to LiteLLM's `provider/model` prefix convention:
- `anthropic` → `anthropic/claude-sonnet-4-5-20250929`
- `openai` → `openai/your-model` + `api_base` + `api_key`
- `gemini` → `gemini/gemini-2.5-flash-lite-preview`

Provider/model can be set via env vars (`LLM_PROVIDER`, `DEFAULT_MODEL`) or CLI flags (`--provider`, `--model`). CLI flags flow through workflow configs down to activities, where they override env defaults.

### Thinking Model Support

Thinking models (Qwen3, DeepSeek-R1, etc.) generate `<think>...</think>` reasoning tokens that consume the max_tokens budget. The LLM layer handles this:
- `reasoning_content` field is captured if present
- `<think>...</think>` blocks are stripped from text output via regex
- If text is empty but reasoning exists, reasoning is promoted to text
- Set `FAST_MAX_TOKENS=4096+` to give utility calls headroom

---

## Tool System

Two tools, shared across both modes:

### `run` — Shell Command Execution

Defined in `tools/run_tool.py`. A single shell tool instead of many specialized tools — Unix already has composable tools for file I/O, search, processing. Two-layer architecture:

- **Layer 1 (Execution)**: `asyncio.create_subprocess_shell` with configurable timeout (120s default). Pipes, chaining, redirection all handled natively by bash.
- **Layer 2 (Presentation)**: Binary detection, overflow truncation (200 lines / 50KB), stderr attachment, metadata footer (`[exit:N | Ns]`). Overflow saves full output to temp file with exploration hints.

Every call goes through an approval gate (user confirms y/n) unless `--dangerous` is set.

### `delegate_task` — Background Sub-Agents (Chat Mode)

Defined in `tools/delegate_tool.py`. Spawns a `CommunisSubAgent` child workflow for complex multi-step work. The sub-agent gets its own LLM + tool loop, approval propagation, and progress reporting.

### Adding a New Tool

Three steps:

1. **Define the schema** in `tools/my_tool.py` — Anthropic-format tool definition with `name`, `description`, and `input_schema`
2. **Add dispatch** in the relevant workflow (`session_workflow.py` for chat, `communis_turn.py` for run) — add to the `tools` list and add an `elif tool_name == "my_tool"` branch
3. **Implement execution** — as a Temporal activity if it does I/O (register in `scripts/run_worker.py`), or inline if pure computation

Tool results must be dicts with `type: "tool_result"`, `tool_use_id`, and `content`.

---

## Temporal Patterns — Critical Rules

### Deterministic Sandbox

Workflow code runs in Temporal's deterministic sandbox. These cause `RestrictedWorkflowAccessError`:

| Forbidden | Use instead |
|-----------|-------------|
| `datetime.now()` | `workflow.now()` |
| `uuid.uuid4()` | `workflow.uuid4()` |
| `os.getenv()` | Only in activities, never in workflow code |

### Architecture Rules

- **ALL LLM calls MUST be Temporal activities** — workflows are deterministic; LLM calls are non-deterministic.
- **ALL external I/O MUST be Temporal activities** — API calls, file operations, subprocess execution.
- **Workflow code is the orchestrator** — it contains the agent loop logic (conditionals, loops, wait conditions) but delegates all real work to activities.
- **Signals for human input** — workflows wait for user messages, approval decisions, and feedback via Temporal signals.
- **Queries for state inspection** — the CLI reads agent state via Temporal queries (synchronous, read-only).
- **Child workflows inherit parent's task queue** — don't hardcode `TASK_QUEUE` in `start_child_workflow`.
- **Temporal signals accept a single argument** — wrap multi-arg payloads in a list or dataclass.

### Worker Import Ordering

In `scripts/run_worker.py`, `load_dotenv()` **must** execute before importing activities. Activities read `os.getenv()` at module level (e.g., `LLM_PROVIDER`, `DEFAULT_MODEL`), so the env must be loaded first:

```python
from dotenv import load_dotenv
load_dotenv()

# Now safe to import activities
from activities.llm_activities import call_llm  # noqa: E402
```

### Retry Policy and Timeouts

Defined in `workflows/constants.py`:

```python
LLM_RETRY_POLICY = RetryPolicy(
    initial_interval=1s, backoff_coefficient=2.0,
    maximum_interval=30s, maximum_attempts=5
)
LLM_TIMEOUT = 30 minutes    # main LLM calls (long context)
FAST_LLM_TIMEOUT = 10 minutes  # utility LLM calls
IO_TIMEOUT = 30 seconds     # workspace file operations
TOOL_TIMEOUT = 5 minutes    # shell command execution
```

### Production Considerations

- **Payload size**: Full content lives in workspace files (`.communis/<id>/`), not Temporal payloads. Only metadata and file paths flow through workflows (claim-check pattern).
- **Workflow IDs**: Generated as `communis-<uuid-hex[:8]>` for uniqueness.
- **Context window management**: Run mode compresses older turns into `summary.md` via `summarize_artifacts`. Chat mode doesn't yet implement conversation summarization.
- **Cost control**: FAST_MODEL (Haiku) handles mechanical tasks; DEFAULT_MODEL (Sonnet) handles reasoning. Don't pay Sonnet prices to extract bullet points.

---

## Testing

Three tiers:

### Unit Tests (~90 tests)

Mocked LLM calls, no external dependencies. Fast (~10s).

```bash
uv run pytest tests/ --ignore=tests/test_integration_lmstudio.py \
                     --ignore=tests/test_integration_session.py \
                     --ignore=tests/test_integration_gemini.py \
                     --ignore=tests/full_integration/ -v
```

### Integration Tests (~16 tests)

Activity-level and workflow-level tests against a real LLM endpoint.

```bash
uv run pytest tests/test_integration_lmstudio.py tests/test_integration_session.py -v -s  # OpenAI-compat
uv run pytest tests/test_integration_gemini.py -v -s                                      # Gemini
```

### Full Integration Tests (~10 tests)

End-to-end user journey tests. YAML config scenarios × model backends (Anthropic Haiku + local Qwen).

```bash
uv run pytest tests/full_integration/ -v -s                    # all
uv run pytest tests/full_integration/ -v -s -k "haiku"         # Anthropic only
uv run pytest tests/full_integration/ -v -s -k "qwen_local"    # local only
uv run pytest tests/full_integration/ -v -s -k "quick_sanity"  # single scenario
```

### All Tests

```bash
uv run pytest tests/ -v -s  # requires LLM endpoints
```

---

## Development Commands

```bash
# Setup
uv sync

# Run services
temporal server start-dev             # Terminal 1 — Temporal dev server
uv run python scripts/run_worker.py   # Terminal 2 — Temporal worker

# Use
uv run python cli/main.py run "your prompt" -t 3
uv run python cli/main.py chat
uv run python cli/main.py chat --verbose --dangerous

# Unit tests only
uv run pytest tests/ --ignore=tests/test_integration_lmstudio.py \
                     --ignore=tests/test_integration_session.py \
                     --ignore=tests/test_integration_gemini.py \
                     --ignore=tests/full_integration/ -v
```

---

## Environment Configuration

See `.env.example` for all variables. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM backend: `anthropic`, `openai`, or `gemini` |
| `ANTHROPIC_API_KEY` | *(required for anthropic)* | Anthropic API key |
| `GOOGLE_API_KEY` | *(required for gemini)* | Google AI API key |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1` | Base URL for OpenAI-compatible API |
| `OPENAI_API_KEY` | `lm-studio` | API key for OpenAI-compatible endpoint |
| `DEFAULT_MODEL` | `claude-sonnet-4-5-20250929` | Model for planning, turns, front agent, sub-agents |
| `FAST_MODEL` | `claude-haiku-4-5-20251001` | Model for insights, summaries, validation |
| `MAX_OUTPUT_TOKENS` | `16384` | Max tokens per LLM output |
| `FAST_MAX_TOKENS` | *(unset)* | Override for fast calls (`4096+` for thinking models) |
| `TEMPORAL_ADDRESS` | `localhost:7233` | Temporal server address |
| `TEMPORAL_NAMESPACE` | `default` | Temporal namespace |
| `COMMUNIS_WORKSPACE` | `.communis` | Base directory for workspace files |

### Switching Providers at Runtime

```bash
# Claude (default)
uv run python cli/main.py run "prompt" -t 3

# Gemini
uv run python cli/main.py run "prompt" -t 3 -p gemini -m gemini-2.5-flash-lite-preview

# Local model (LM Studio)
uv run python cli/main.py run "prompt" -t 3 -p openai -m your-model-name
```

---

## Reference Links

- **Temporal Python SDK**: https://docs.temporal.io/develop/python
- **LiteLLM Docs**: https://docs.litellm.ai/
- **Claude Tool Use**: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- **Reference Architecture**: https://github.com/temporal-community/temporal-ai-agent
