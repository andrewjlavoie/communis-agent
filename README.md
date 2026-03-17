# communis

Durable AI agents on [Temporal](https://temporal.io/). Two modes: a self-directing iterative work loop (`run`) and an interactive agent REPL (`chat`) with background task delegation. Works with [Claude](https://docs.anthropic.com/) or any OpenAI-compatible API (LM Studio, Ollama, vLLM, etc.).

## Quick Start

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [Temporal CLI](https://docs.temporal.io/cli#install)

```bash
uv sync
cp .env.example .env   # add your ANTHROPIC_API_KEY (or configure a local model)
```

```bash
# Terminal 1 — Temporal dev server
temporal server start-dev

# Terminal 2 — worker
uv run python scripts/run_worker.py

# Terminal 3 — pick a mode
uv run python cli/main.py run "design a rate limiter for a multi-tenant API" -t 3
uv run python cli/main.py chat
```

## Two Modes

### `run` — Self-Directing Work Loop

Give it a prompt and a turn count. It decomposes the work into steps, assigns a different agent role each turn ("Researcher", "Devil's Advocate", "Architect", "Synthesizer" — whatever the task needs), builds on its own prior output, and optionally takes human feedback between turns.

The core primitive: **an LLM plans what to do, does it, reads what it just did, plans the next step, repeats.** Context compression and workspace files keep it from drowning in its own output.

```bash
uv run python cli/main.py run "why do startups fail in their second year" -t 4
uv run python cli/main.py run "write a short story about a librarian" -t 5 --auto
uv run python cli/main.py "design an API schema" -t 3   # 'run' is the default
```

All outputs live as markdown files in `.communis/<workflow-id>/`.

### `chat` — Interactive Agent REPL

A conversational agent session with real-time event streaming. The front agent (CommunisAgent) handles direct conversation and can delegate complex tasks to background sub-agents (CommunisSubAgent) that run as separate Temporal workflows.

```bash
uv run python cli/main.py chat
uv run python cli/main.py chat --user alice --verbose
```

Chat features:
- **Background tasks** — the agent spawns sub-agents for complex multi-step work via `delegate_task`
- **Approval gates** — tool calls require human approval (y/n/wait); deferred approvals queue up
- **Slash commands** — `/tasks`, `/approvals`, `/status`, `/clear`, and more
- **Event stream** — the CLI polls workflow events for real-time task progress, tool calls, and results

## CLI Reference

### `run` (default)

```
uv run python cli/main.py [run] <prompt> [options]

Options:
  --turns, -t         Max steps (0 = indefinite with goal detection, default: 0)
  --model, -m         Model name (default: claude-sonnet-4-5-20250929)
  --provider, -p      LLM provider: 'anthropic' or 'openai' (overrides env var)
  --base-url          Base URL for OpenAI-compatible API (overrides env var)
  --auto, -a          Skip feedback prompts, run straight through
  --output, -o        Save session output to a markdown file
  --verbose, -v       Timing, file paths, token breakdown table
  --dangerous         Auto-approve all tool calls (no human confirmation)
  --no-goal-detect    Disable goal completion detection (requires --turns > 0)
  --max-subcommunis   Max parallel subcommunis (0-5, default: 3)
```

### `chat`

```
uv run python cli/main.py chat [options]

Options:
  --user, -u          Username for workflow IDs (default: $USER)
  --model, -m         Model name (default: env DEFAULT_MODEL or claude-sonnet-4-5-20250929)
  --provider, -p      LLM provider: 'anthropic' or 'openai' (overrides env var)
  --base-url          Base URL for OpenAI-compatible API (overrides env var)
  --verbose, -v       Show tool calls, results, and agent thinking
  --dangerous         Auto-approve all tool calls (no human confirmation)
```

### Chat Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Session info — messages, tasks, events, approvals |
| `/tasks` | List all tasks with status |
| `/task <id>` | Show task details |
| `/cancel <id>` | Cancel a running task |
| `/approvals` | View and manage deferred approvals |
| `/approve <n>` | Approve a deferred approval by number |
| `/deny <n>` | Deny a deferred approval by number |
| `/approveall` | Approve all deferred approvals |
| `/denyall` | Deny all deferred approvals |
| `/clear` | Clear conversation history (keep tasks) |
| `/quit` | End session |

During active approvals, input is locked to `y`/`n`/`wait`. Use `wait` to defer an approval and continue chatting.

## Architecture

### Run Mode — Orchestrator + Turn Workflows

```
CommunisOrchestratorWorkflow (parent)
  │
  ├── init_workspace ─── creates .communis/<id>/
  │
  ├── while not complete:
  │     ├── read_turn_context ─── reads summary.md + plan.md + recent turn files
  │     ├── plan_next_turn ────── LLM picks role + instructions (or goal_complete/spawn)
  │     ├── write_plan_file ───── persist rolling plan summary
  │     ├── CommunisTurnWorkflow ──── child workflow:
  │     │     ├── read_turn_context ── read prior work from files
  │     │     ├── call_llm ─────────── generate turn output (with tool use loop)
  │     │     │     └── [run tool] ─── execute shell commands (with approval)
  │     │     ├── extract_key_insights ─ compact bullet points
  │     │     └── write_turn_artifact ── save to turn-NN-role.md
  │     ├── summarize_artifacts ── compress older turns → summary.md
  │     └── wait for feedback ──── signal from CLI (120s timeout)
  │
  └── return results
```

### Chat Mode — Session Agent + Sub-Agents

```
CommunisAgent (front agent — long-running session workflow)
  │
  ├── wait for user_message signal
  │
  ├── front_agent_respond ─── LLM decides: reply, use tool, or delegate
  │     ├── text response ──── emit assistant_message event
  │     ├── run tool ──────── execute command (with approval gate)
  │     └── delegate_task ──── spawn CommunisSubAgent as child workflow
  │
  ├── CommunisSubAgent (background task workflow)
  │     ├── autonomous LLM + tool loop (up to max_tool_iterations)
  │     ├── approval_requested → signal parent → wait for decision
  │     └── task_update signal → parent emits progress/completion events
  │
  ├── event stream ──── CLI polls get_events_since query
  │
  └── end_session signal → complete
```

### The Two Model Tiers

Not every LLM call needs the same capability. communis splits calls into two tiers:

| Tier | Used for | Default | Why |
|------|----------|---------|-----|
| **DEFAULT_MODEL** | Planner, turn agent, front agent, sub-agents | Sonnet | Needs reasoning and creativity |
| **FAST_MODEL** | Insight extraction, summarization, feedback validation | Haiku | Mechanical tasks — structured extraction, compression, yes/no checks |

With Claude, Haiku is ~60x cheaper than Sonnet. The utility tasks don't need intelligence — paying Sonnet prices to pull bullet points is wasteful. For local models, set both to the same model.

### Workspace Files (Run Mode)

Each session writes to `.communis/<workflow-id>/`:

```
.communis/communis-a1b2c3d4/
├── communis.md                    # Session manifest
├── plan.md                        # Rolling plan summary
├── turn-01-explorer.md            # YAML frontmatter + full output
├── turn-02-devils-advocate.md
├── turn-03-synthesizer.md
├── subcommunis-step-03.md         # Subcommunis results summary
├── subcommunis/                   # Subcommunis workspaces
│   └── <id>-subcommunis-3-0/
└── summary.md                     # Rolling summary of older turns
```

Turn files have YAML frontmatter (role, key insights, token usage) and the full output below. Only paths and metadata flow through Temporal — not content. This is the claim-check pattern: files are the source of truth, workflows stay lightweight.

## Tool System

Tools are how the LLM takes action beyond generating text. Each tool is a JSON schema definition (so the LLM knows what it can call and how) paired with an execution handler in the workflow. When the LLM returns a `tool_use` block, the workflow dispatches it to the matching handler, runs it as a Temporal activity, and feeds the result back to the LLM for the next iteration.

### Current Tools

#### `run` — Shell Command Execution

Both modes share the `run` tool (`tools/run_tool.py`). It gives the agent full Unix shell access — pipes, chaining, redirection, all standard CLI tools. The implementation has two layers:

- **Execution layer** — `asyncio.create_subprocess_shell` with configurable timeout (default 120s)
- **Presentation layer** — binary detection, overflow truncation (200 lines / 50KB), stderr attachment, and a metadata footer so the LLM knows exit code and duration

Every call goes through an approval gate (human confirms `y`/`n`) unless `--dangerous` is set.

**Why a single `run` tool instead of separate `read_file`, `write_file`, `grep`, etc.?** Unix already has composable tools for all of those. A single shell tool avoids duplicating that surface area in custom tool definitions and lets the LLM compose commands naturally (`cat file.txt | grep ERROR | wc -l`).

#### `delegate_task` — Background Sub-Agents (Chat Mode Only)

Defined in `tools/delegate_tool.py`. The front agent can spawn a background sub-agent (`CommunisSubAgent`) as a child Temporal workflow. Each sub-agent gets its own LLM + tool loop, approval propagation, and progress reporting. Use this for multi-step work that shouldn't block the conversation.

The `run` mode doesn't use `delegate_task` — it has its own multi-turn orchestration via `CommunisOrchestratorWorkflow`.

### Adding a New Tool

Three steps: define the schema, implement the handler, and wire it into the workflow.

**1. Define the tool schema** — Create a file in `tools/` with a Claude-compatible tool definition:

```python
# tools/my_tool.py
MY_TOOL_DEFINITION = {
    "name": "my_tool",
    "description": (
        "What this tool does, when the LLM should use it, "
        "and any constraints. Be detailed — the LLM uses this "
        "description to decide when and how to call the tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "param": {
                "type": "string",
                "description": "What this parameter is for.",
            },
        },
        "required": ["param"],
    },
}
```

**2. Add a handler in the workflow** — In the workflow that should have access to the tool (e.g., `workflows/session_workflow.py` for chat mode, `workflows/communis_turn.py` for run mode), add the tool definition to the `tools` list and add a dispatch branch:

```python
# In the workflow's unsafe imports block:
from tools.my_tool import MY_TOOL_DEFINITION

# Add to the tools list passed to the LLM:
tools = [RUN_TOOL_DEFINITION, DELEGATE_TASK_TOOL, MY_TOOL_DEFINITION]

# Add a dispatch branch in the tool processing loop:
elif tool_name == "my_tool":
    result = await self._handle_my_tool(tool_use_id, tool_input)
    tool_results.append(result)
```

**3. Implement execution** — If your tool needs I/O or external calls, implement it as a Temporal activity (in `activities/`) so it gets retry policies and durable execution. If it's pure computation, you can run it inline in the workflow handler. Return a `tool_result` dict:

```python
{
    "type": "tool_result",
    "tool_use_id": tool_use_id,
    "content": "result string for the LLM",
}
```

Don't forget to register any new activities in `scripts/run_worker.py`.

## Composability

communis workflows are standard Temporal workflows. They can be called as child workflows, started via SDK from any service (Python, Go, TypeScript, Java), triggered by signals, or queried mid-execution.

```python
result = await workflow.execute_child_workflow(
    CommunisOrchestratorWorkflow.run,
    CommunisConfig(idea="design the API schema", max_turns=4, auto=True),
    id="sub-communis-api-design",
)
```

See [COMPOSABILITY.md](COMPOSABILITY.md) for detailed patterns and working code — child workflows, parallel fan-out, external clients, Go/TypeScript integration, REST API wrapper, and Temporal CLI usage. Runnable examples in `examples/`.

## Project Structure

```
communis/
├── models/
│   ├── data_types.py              # CommunisConfig, TurnConfig, TurnResult, CommunisState
│   └── session_types.py           # SessionConfig, SessionEvent, TaskSpec, TaskStatus, etc.
├── prompts/
│   ├── communis_prompts.py        # Run-mode prompts (see PROMPTS.md)
│   └── session_prompts.py         # Front agent system prompt
├── activities/
│   ├── llm_activities.py          # LLM calls — Anthropic + OpenAI backends (6 activities)
│   ├── session_activities.py      # Front agent LLM call (1 activity)
│   ├── tool_activities.py         # Shell command execution (1 activity)
│   └── workspace_activities.py    # Workspace file I/O (8 activities)
├── tools/
│   ├── run_tool.py                # Unix-style run(command="...") tool
│   └── delegate_tool.py           # Spawn background sub-agents
├── workflows/
│   ├── communis_orchestrator.py   # Run mode — turn loop, feedback, cancellation
│   ├── communis_turn.py           # Run mode — single turn execution
│   ├── session_workflow.py        # Chat mode — CommunisAgent (front agent)
│   └── task_workflow.py           # Chat mode — CommunisSubAgent (sub-agent)
├── shared/
│   ├── constants.py               # Task queue, default model string
│   └── frontmatter.py             # YAML frontmatter parser
├── cli/
│   ├── main.py                    # CLI entry point, run-mode REPL with Rich output
│   └── session_cli.py             # Chat-mode REPL with event polling + approval UX
├── scripts/run_worker.py          # Temporal worker registration
├── examples/                      # Runnable composability examples
│   ├── research_pipeline.py       # Sequential multi-phase research
│   ├── parallel_analysis.py       # Concurrent multi-perspective analysis
│   ├── external_client.py         # Start/monitor from any Python service
│   └── api_server.py              # FastAPI REST wrapper
├── tests/                         # 106 tests (90 unit + 16 integration)
├── COMPOSABILITY.md               # Patterns for using communis as a building block
├── PROMPTS.md                     # All prompts extracted for reference
└── CLAUDE.md                      # AI agent development instructions
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM backend: `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | *(required for anthropic)* | Anthropic API key |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1` | Base URL for OpenAI-compatible API |
| `OPENAI_API_KEY` | `lm-studio` | API key for OpenAI-compatible endpoint |
| `DEFAULT_MODEL` | `claude-sonnet-4-5-20250929` | Model for planning, turns, front agent, sub-agents |
| `FAST_MODEL` | `claude-haiku-4-5-20251001` | Model for insights, summaries, validation |
| `FAST_MAX_TOKENS` | *(unset)* | Override token budget for fast calls (set to `4096` for thinking models) |
| `MAX_OUTPUT_TOKENS` | `16384` | Max tokens per turn output |
| `TEMPORAL_ADDRESS` | `localhost:7233` | Temporal server |
| `COMMUNIS_WORKSPACE` | `.communis` | Base directory for workspace files |

### Using Local Models (LM Studio, Ollama, etc.)

Any OpenAI-compatible API works. Install the optional dependency, set a few env vars, and go:

```bash
uv sync --extra openai

# In your .env:
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:1234/v1   # LM Studio default
DEFAULT_MODEL=your-model-name              # Must match what your server exposes
FAST_MODEL=your-model-name                 # Can be the same model
FAST_MAX_TOKENS=4096                       # Needed for thinking models (Qwen3, etc.)
```

Common endpoints:
- **LM Studio**: `http://localhost:1234/v1`
- **Ollama**: `http://localhost:11434/v1`
- **vLLM**: `http://localhost:8000/v1`

**Thinking models** (Qwen3, DeepSeek-R1, etc.) generate `<think>...</think>` reasoning tokens that consume the max_tokens budget before producing output. communis automatically strips these from responses. Set `FAST_MAX_TOKENS=4096` to give the utility calls enough headroom for reasoning + output.

You can keep both providers configured in `.env` and switch at runtime:

```bash
# Use Claude (default)
uv run python cli/main.py run "your prompt" -t 3

# Use local model
uv run python cli/main.py run "your prompt" -t 3 -p openai -m qwen/qwen3.5-9b
```

## Tests

```bash
uv run pytest tests/ -v                                                         # all 106 tests
uv run pytest tests/ --ignore=tests/test_integration_lmstudio.py \
                     --ignore=tests/test_integration_session.py -v               # 90 unit tests only
uv run pytest tests/test_integration_lmstudio.py tests/test_integration_session.py -v -s  # 16 integration tests (real LLM)
```

## Why Temporal

This could be a Python script with a for loop. Temporal gives you:

- **Crash recovery** — worker dies mid-turn, restarts exactly where it left off
- **Human-in-the-loop** — workflows pause for feedback and approvals via signals, no polling or external state
- **Visibility** — every activity, signal, and state change in the Web UI at `localhost:8233`
- **Cancellation** — Ctrl+C sends a proper cancel; workflow catches it and returns partial results
- **Composability** — these workflows are callable building blocks for larger agent systems

## License

MIT
