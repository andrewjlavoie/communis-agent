# communis

A self-directing iterative work loop. Give it a prompt and a turn count — it decomposes the work into steps, assigns a different agent role each turn, builds on its own prior output, and optionally takes human feedback between turns.

The core primitive: **an LLM plans what to do, does it, reads what it just did, plans the next step, repeats.** Context compression and workspace files keep it from drowning in its own output.

Built on [Temporal](https://temporal.io/) for durable execution. Works with [Claude](https://docs.anthropic.com/) or any OpenAI-compatible API (LM Studio, Ollama, vLLM, etc.). Designed as a composable building block — this workflow can be called by other Temporal workflows in a larger system.

## Quick Start

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [Temporal CLI](https://docs.temporal.io/cli#install)

```bash
uv sync
cp .env.example .env   # add your ANTHROPIC_API_KEY (or configure a local model)

# Terminal 1
temporal server start-dev

# Terminal 2
uv run python scripts/run_worker.py

# Terminal 3
uv run python cli/main.py "your prompt here" --turns 3
```

## What It Does

1. You give it any prompt and a number of turns.
2. Each turn, an LLM **planner** picks a role and writes specific instructions — "Researcher", "Devil's Advocate", "Architect", "Editor" — whatever the task needs right now.
3. A **turn agent** with that role produces one round of work, reading prior turns from workspace files.
4. Between turns, you can steer with feedback or skip.
5. The final turn synthesizes everything into a deliverable.
6. All outputs live as markdown files in `.communis/<workflow-id>/`.

It doesn't know what it's going to do before it starts. The planner reads what's been produced so far and decides the next move each turn. You can throw anything at it:

```bash
# Research and analysis
uv run python cli/main.py "why do startups fail in their second year" -t 4

# Creative writing
uv run python cli/main.py "write a short story about a librarian who finds a book that writes itself" -t 5

# Technical planning
uv run python cli/main.py "design a rate limiter for a multi-tenant API" -t 3

# Open-ended exploration
uv run python cli/main.py "what would a city look like if it was designed for 15-minute commutes" -t 6 --auto
```

## CLI

```
uv run python cli/main.py <prompt> [options]

Options:
  --turns, -t       Number of turns (default: 3, max: 10)
  --model, -m       Model name (default: claude-sonnet-4-5-20250929)
  --provider, -p    LLM provider: 'anthropic' or 'openai' (overrides env var)
  --base-url        Base URL for OpenAI-compatible API (overrides env var)
  --auto, -a        Skip feedback prompts, run straight through
  --output, -o      Save session output to a markdown file
  --verbose, -v     Timing, file paths, token breakdown table
```

You can keep both providers configured in `.env` and switch at runtime:

```bash
# Use Claude (default)
uv run python cli/main.py "your prompt" -t 3

# Use local model
uv run python cli/main.py "your prompt" -t 3 -p openai -m qwen/qwen3.5-9b
```

**Ctrl+C** cancels gracefully — the workflow stops, preserves completed turns, and returns partial results.

## Architecture

```
CommunisOrchestratorWorkflow (parent)
  │
  ├── init_workspace ─── creates .communis/<id>/
  │
  ├── for each turn:
  │     ├── read_turn_context ─── reads summary.md + recent turn files
  │     ├── plan_next_turn ────── LLM picks role + instructions
  │     ├── CommunisTurnWorkflow ──── child workflow:
  │     │     ├── read_turn_context ── read prior work from files
  │     │     ├── call_claude ──────── generate turn output
  │     │     ├── extract_key_insights ─ compact bullet points
  │     │     └── write_turn_artifact ── save to turn-NN-role.md
  │     ├── summarize_artifacts ── compress older turns → summary.md
  │     └── wait for feedback ──── signal from CLI (120s timeout)
  │
  └── return results
```

### The Two Model Tiers

Not every LLM call needs the same capability. communis splits calls into two tiers:

| Tier | Used for | Default | Why |
|------|----------|---------|-----|
| **DEFAULT_MODEL** | Planner (picks roles), Turn Agent (produces work) | Sonnet | Needs reasoning and creativity |
| **FAST_MODEL** | Insight extraction, summarization, feedback validation | Haiku | Mechanical tasks — structured extraction, compression, yes/no checks |

With Claude, Haiku is ~60x cheaper than Sonnet. The utility tasks don't need intelligence — paying Sonnet prices to pull bullet points is wasteful. For local models, set both to the same model.

### Workspace Files

Each session writes to `.communis/<workflow-id>/`:

```
.communis/communis-a1b2c3d4/
├── communis.md                    # Session manifest
├── turn-01-explorer.md        # YAML frontmatter + full output
├── turn-02-devils-advocate.md
├── turn-03-synthesizer.md
└── summary.md                 # Rolling summary of older turns
```

Turn files have YAML frontmatter (role, key insights, token usage) and the full output below. Only paths and metadata flow through Temporal — not content. This is the claim-check pattern: files are the source of truth, workflows stay lightweight.

### Composability

This is a standard Temporal workflow. It can be:

- **Called as a child workflow** from a larger orchestrator that chains multiple riff sessions
- **Started via the Temporal SDK** from any service — Python, Go, TypeScript, Java
- **Triggered by signals** from other workflows or external systems
- **Queried mid-execution** for state from any Temporal client

```python
# Example: calling communis from another Temporal workflow
result = await workflow.execute_child_workflow(
    CommunisOrchestratorWorkflow.run,
    CommunisConfig(idea="design the API schema", num_turns=4, auto=True),
    id="sub-communis-api-design",
)
```

See [COMPOSABILITY.md](COMPOSABILITY.md) for detailed patterns and working code — child workflows, parallel fan-out, external clients, Go/TypeScript integration, REST API wrapper, and Temporal CLI usage. Runnable examples in `examples/`.

## Project Structure

```
communis/
├── models/data_types.py          # CommunisConfig, TurnConfig, TurnResult, CommunisState
├── prompts/communis_prompts.py       # All system prompts (see PROMPTS.md)
├── activities/
│   ├── llm_activities.py         # LLM calls — Anthropic + OpenAI backends (5 activities)
│   └── workspace_activities.py   # Workspace file I/O (6 activities)
├── workflows/
│   ├── communis_orchestrator.py      # Parent workflow — turn loop, feedback, cancellation
│   └── communis_turn.py              # Child workflow — single turn execution
├── cli/main.py                   # CLI with signal handling, Rich output
├── scripts/run_worker.py         # Temporal worker registration
├── examples/                     # Runnable composability examples
│   ├── research_pipeline.py      # Sequential multi-phase research
│   ├── parallel_analysis.py      # Concurrent multi-perspective analysis
│   ├── external_client.py        # Start/monitor from any Python service
│   └── api_server.py             # FastAPI REST wrapper
├── tests/                        # 32 unit tests (offline) + 7 integration tests (real LLM)
├── COMPOSABILITY.md              # Patterns for using communis as a building block
├── PROMPTS.md                    # All prompts extracted for reference
└── CLAUDE.md                     # AI agent development instructions
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM backend: `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | *(required for anthropic)* | Anthropic API key |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1` | Base URL for OpenAI-compatible API |
| `OPENAI_API_KEY` | `lm-studio` | API key for OpenAI-compatible endpoint |
| `DEFAULT_MODEL` | `claude-sonnet-4-5-20250929` | Model for planning + turn generation |
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

## Tests

```bash
uv run pytest tests/ -v                                              # 32 unit tests, all offline
uv run pytest tests/test_integration_lmstudio.py -v -s               # 7 integration tests, real LLM
uv run pytest tests/ --ignore=tests/test_integration_lmstudio.py -v  # unit tests only
```

## Why Temporal

This could be a Python script with a for loop. Temporal gives you:

- **Crash recovery** — worker dies mid-turn, restarts exactly where it left off
- **Human-in-the-loop** — workflow pauses for feedback via signals, no polling or external state
- **Visibility** — every activity, signal, and state change in the Web UI at `localhost:8233`
- **Cancellation** — Ctrl+C sends a proper cancel; workflow catches it and returns partial results
- **Composability** — this workflow is a callable building block for larger agent systems

## License

MIT
