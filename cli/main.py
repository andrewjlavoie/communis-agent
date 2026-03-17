from __future__ import annotations

import argparse
import asyncio
import os
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from temporalio.client import Client, WorkflowHandle

from models.data_types import DEFAULT_MAX_TURNS, CommunisConfig
from shared.constants import DEFAULT_MODEL_STRING, TASK_QUEUE
from shared.frontmatter import parse_frontmatter
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow
from workflows.communis_turn import CommunisTurnWorkflow
POLL_INTERVAL = 1.5

console = Console()


def _read_turn_file(artifact_path: str) -> tuple[str | None, dict]:
    """Read turn content and frontmatter from a workspace artifact file.

    Returns (content, metadata) tuple.
    """
    path = Path(artifact_path)
    if not path.exists():
        return None, {}

    meta, content = parse_frontmatter(path.read_text())
    return content, meta


def _elapsed(start: float) -> str:
    secs = time.monotonic() - start
    if secs < 60:
        return f"{secs:.1f}s"
    mins = int(secs // 60)
    remaining = secs % 60
    return f"{mins}m{remaining:.0f}s"


def _install_cancel_handlers(
    loop: asyncio.AbstractEventLoop,
    handle: WorkflowHandle,
    verbose: bool,
) -> None:
    """Register SIGINT/SIGTERM handlers that cancel the workflow gracefully."""
    cancel_requested = False

    def on_signal(sig: int, _frame: object) -> None:
        nonlocal cancel_requested
        sig_name = signal.Signals(sig).name
        if cancel_requested:
            console.print(f"\n[red]Received {sig_name} again — forcing exit.[/red]")
            raise SystemExit(1)

        cancel_requested = True
        console.print(f"\n[yellow]Received {sig_name} — cancelling workflow...[/yellow]")
        if verbose:
            console.print("[dim]  Sending cancel request to Temporal. Press Ctrl+C again to force quit.[/dim]")

        # Schedule the cancel coroutine on the running event loop
        asyncio.run_coroutine_threadsafe(_cancel_workflow(handle, verbose), loop)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)


async def _cancel_workflow(handle: WorkflowHandle, verbose: bool) -> None:
    """Send cancellation request to the workflow."""
    try:
        await handle.cancel()
        if verbose:
            console.print("[dim]  Cancel request sent to workflow.[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to cancel workflow: {e}[/red]")


# --- Markdown output builder ---


class MarkdownLog:
    """Accumulates session output as markdown for --output."""

    def __init__(self, idea: str, max_turns: int, model: str, provider: str, goal_detect: bool):
        self.parts: list[str] = []
        self.parts.append(f"# communis Session\n")
        self.parts.append(f"**Prompt:** {idea}\n")
        mode = "goal-detect" if goal_detect else f"fixed {max_turns} steps"
        if max_turns == 0:
            mode = f"indefinite (max {DEFAULT_MAX_TURNS})"
        self.parts.append(
            f"**Config:** {mode} | model: `{model}` | "
            f"provider: {provider or 'env default'} | "
            f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        )

    def add_planner(self, turn: int, role: str, reasoning: str):
        self.parts.append(f"---\n\n## Step {turn} — {role}\n")
        if reasoning:
            self.parts.append(f"*Planner reasoning: {reasoning}*\n")

    def add_turn_output(self, turn: int, role: str, content: str | None, insights: list[str], usage: dict, truncated: bool):
        if content:
            self.parts.append(f"{content}\n")
        else:
            self.parts.append("*Content not available*\n")

        if truncated:
            self.parts.append("**WARNING: Output was truncated (hit max_tokens)**\n")

        if insights:
            self.parts.append("**Key Insights:**\n")
            for i in insights:
                self.parts.append(f"- {i}")
            self.parts.append("")

        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        self.parts.append(f"*Tokens: {in_tok:,} in / {out_tok:,} out*\n")

    def add_status(self, status: str, message: str = ""):
        if status == "complete":
            self.parts.append("---\n\n**Status:** Complete\n")
        elif status == "cancelled":
            self.parts.append(f"---\n\n**Status:** Cancelled — {message}\n")
        elif status == "error":
            self.parts.append(f"---\n\n**Status:** Error — {message}\n")

    def add_stats(self, turn_results: list[dict], elapsed: str):
        total_in = sum(r.get("token_usage", {}).get("input_tokens", 0) for r in turn_results)
        total_out = sum(r.get("token_usage", {}).get("output_tokens", 0) for r in turn_results)
        self.parts.append("## Summary\n")
        self.parts.append(f"| Step | Role | Input | Output |")
        self.parts.append(f"|------|------|------:|-------:|")
        for r in turn_results:
            u = r.get("token_usage", {})
            trunc = " *" if r.get("truncated") else ""
            self.parts.append(
                f"| {r['turn_number']} | {r['role']}{trunc} | "
                f"{u.get('input_tokens', 0):,} | {u.get('output_tokens', 0):,} |"
            )
        self.parts.append(f"| | **Total** | **{total_in:,}** | **{total_out:,}** |")
        self.parts.append(f"\n*Elapsed: {elapsed}*\n")

    def write(self, path: str):
        Path(path).write_text("\n".join(self.parts))


async def run_cli(
    idea: str, *, max_turns: int, model: str, auto: bool = False, verbose: bool = False,
    provider: str = "", base_url: str = "", output: str = "", dangerous: bool = False,
    goal_complete_detection: bool = True, max_subcommunis: int = 3,
):
    load_dotenv()

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")

    effective_max = max_turns if max_turns > 0 else DEFAULT_MAX_TURNS
    md_log = MarkdownLog(idea, max_turns, model, provider, goal_complete_detection) if output else None

    console.print(Panel("[bold]communis[/bold] — Self-Directing Work Loop", style="blue"))
    console.print(f"Prompt: [bold]{idea}[/bold]")
    mode_parts = []
    if auto:
        mode_parts.append("auto")
    if dangerous:
        mode_parts.append("DANGEROUS (tool auto-approve)")
    if goal_complete_detection:
        mode_parts.append("goal-detect")
    mode = ", ".join(mode_parts) if mode_parts else "interactive"
    provider_label = provider or "env default"

    # Build turns display
    if max_turns == 0:
        turns_display = f"indefinite (max {DEFAULT_MAX_TURNS})"
    elif goal_complete_detection:
        turns_display = f"up to {max_turns}"
    else:
        turns_display = f"fixed {max_turns}"

    console.print(f"Steps: {turns_display} | Model: {model} | Provider: {provider_label} | Mode: {mode}")
    if max_subcommunis > 0:
        console.print(f"[dim]Subcommunis: up to {max_subcommunis} parallel[/dim]")

    if verbose:
        console.print(f"[dim]Temporal: {address} (namespace: {namespace})[/dim]")
        console.print(f"[dim]Task queue: {TASK_QUEUE}[/dim]")
        console.print(f"[dim]Poll interval: {POLL_INTERVAL}s[/dim]")

    if output:
        console.print(f"[dim]Output: {output}[/dim]")

    console.print()

    session_start = time.monotonic()

    # Connect to Temporal
    try:
        client = await Client.connect(address, namespace=namespace)
    except Exception as e:
        console.print(f"[red]Failed to connect to Temporal at {address}: {e}[/red]")
        console.print("Make sure Temporal dev server is running: temporal server start-dev")
        return

    if verbose:
        console.print(f"[dim]Connected to Temporal ({_elapsed(session_start)})[/dim]")

    # Start the orchestrator workflow
    workflow_id = f"communis-{uuid.uuid4().hex[:8]}"
    config = CommunisConfig(
        idea=idea, max_turns=max_turns, model=model, auto=auto,
        provider=provider, base_url=base_url, dangerous=dangerous,
        goal_complete_detection=goal_complete_detection,
        max_subcommunis=max_subcommunis,
    )

    handle = await client.start_workflow(
        CommunisOrchestratorWorkflow.run,
        config,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    console.print(f"[dim]Workflow started: {workflow_id}[/dim]")

    if verbose:
        console.print("[dim]Press Ctrl+C to cancel the workflow gracefully.[/dim]")

    console.print()

    # Install signal handlers for graceful cancellation
    loop = asyncio.get_running_loop()
    _install_cancel_handlers(loop, handle, verbose)

    last_turn_displayed = 0
    last_status = ""
    last_message = ""
    turn_start: float | None = None
    poll_count = 0

    while True:
        await asyncio.sleep(POLL_INTERVAL)
        poll_count += 1

        try:
            state = await handle.query(CommunisOrchestratorWorkflow.get_state)
        except Exception:
            if verbose:
                console.print(f"[dim]Query failed — workflow may have completed ({_elapsed(session_start)})[/dim]")
            break

        status = state["status"]
        current_msg = state.get("latest_message", "")

        # In verbose mode, show every status message change (not just status transitions)
        if verbose and current_msg and current_msg != last_message:
            last_message = current_msg
            # Don't double-print messages that will be shown by the status change handlers below
            if status == last_status:
                console.print(f"[dim]  [{_elapsed(session_start)}] {current_msg}[/dim]")

        # Poll child workflow for pending tool approval (non-dangerous mode only)
        if (
            status == "running"
            and not dangerous
            and state.get("current_turn", 0) > 0
        ):
            child_id = f"{workflow_id}-turn-{state['current_turn']}"
            try:
                child_handle = client.get_workflow_handle(child_id)
                pending = await child_handle.query(CommunisTurnWorkflow.get_pending_tool)
                if pending:
                    approved = _prompt_for_tool_approval(pending["command"])
                    await child_handle.signal(CommunisTurnWorkflow.approve_tool, approved)
            except Exception:
                pass  # Child not started yet or already completed

        # Display new turn results — read content from workspace files
        for result in state["turn_results"]:
            if result["turn_number"] > last_turn_displayed:
                turn_elapsed = ""
                if turn_start is not None:
                    turn_elapsed = _elapsed(turn_start)
                last_turn_displayed = result["turn_number"]
                artifact_path = result.get("artifact_path", "")
                content, _ = _read_turn_file(artifact_path)
                _display_turn_result(result, effective_max, content, verbose, turn_elapsed)

                if md_log:
                    md_log.add_turn_output(
                        result["turn_number"], result["role"], content,
                        result.get("key_insights", []), result.get("token_usage", {}),
                        result.get("truncated", False),
                    )

                if verbose and artifact_path:
                    console.print(f"[dim]  Artifact: {artifact_path}[/dim]")

                turn_start = time.monotonic()

        # Show status changes
        if status != last_status:
            last_status = status

            if status == "waiting_for_feedback":
                if verbose:
                    console.print(f"[dim]  [{_elapsed(session_start)}] Waiting for user feedback (timeout: 120s)[/dim]")
                console.print()
                feedback = _prompt_for_feedback()
                if feedback is None:
                    await handle.signal(CommunisOrchestratorWorkflow.skip_feedback)
                    console.print("[dim]Skipping feedback, continuing...[/dim]\n")
                else:
                    await handle.signal(CommunisOrchestratorWorkflow.receive_user_feedback, feedback)
                    console.print(f"[dim]Feedback sent.[/dim]\n")

            elif status == "running":
                if current_msg:
                    last_message = current_msg
                    # Planner reasoning contains an em dash separator
                    if "\u2014" in current_msg:
                        parts = current_msg.split("\u2014", 1)
                        console.print(
                            f"[bold cyan]{parts[0].strip()}[/bold cyan]"
                            f"[dim] \u2014 {parts[1].strip()}[/dim]"
                        )
                        # Log planner decision to markdown
                        if md_log:
                            turn_num = state.get("current_turn", 0)
                            role = state.get("current_role", "")
                            md_log.add_planner(turn_num, role, parts[1].strip())
                    else:
                        console.print(f"[cyan]{current_msg}[/cyan]")

                if turn_start is None:
                    turn_start = time.monotonic()

            elif status == "complete":
                console.print()
                if state.get("goal_complete"):
                    console.print(Panel("[bold green]Goal complete![/bold green]", style="green"))
                else:
                    console.print(Panel("[bold green]All steps complete![/bold green]", style="green"))
                ws_dir = state.get("workspace_dir", "")
                if ws_dir:
                    console.print(f"[dim]Workspace: {ws_dir}[/dim]")
                if md_log:
                    md_log.add_status("complete")
                break

            elif status == "cancelled":
                console.print()
                turns_done = len(state.get("turn_results", []))
                console.print(
                    Panel(
                        f"[bold yellow]Workflow cancelled after {turns_done} steps.[/bold yellow]",
                        style="yellow",
                    )
                )
                ws_dir = state.get("workspace_dir", "")
                if ws_dir:
                    console.print(f"[dim]Workspace (partial): {ws_dir}[/dim]")
                if md_log:
                    md_log.add_status("cancelled", f"{turns_done} steps")
                break

            elif status == "error":
                console.print(f"[red]Error: {state['latest_message']}[/red]")
                if md_log:
                    md_log.add_status("error", state["latest_message"])
                break

    # Get final result
    try:
        final = await handle.result()
        turn_results = final.get("turn_results", [])
        _print_final_stats(turn_results, verbose, session_start, poll_count)

        if md_log:
            md_log.add_stats(turn_results, _elapsed(session_start))
    except asyncio.CancelledError:
        # Workflow was cancelled — still show partial stats from last known state
        if verbose:
            console.print(f"[dim]Session elapsed: {_elapsed(session_start)} | Polls: {poll_count}[/dim]")
    except Exception as e:
        err_str = str(e)
        if "cancel" in err_str.lower():
            # Workflow cancelled — show partial stats from last polled state
            pass
        else:
            console.print(f"[red]Workflow error: {e}[/red]")

    # Write markdown output
    if md_log and output:
        md_log.write(output)
        console.print(f"[dim]Output saved to: {output}[/dim]")


def _print_final_stats(
    turn_results: list[dict], verbose: bool, session_start: float, poll_count: int
):
    total_input = sum(r.get("token_usage", {}).get("input_tokens", 0) for r in turn_results)
    total_output = sum(r.get("token_usage", {}).get("output_tokens", 0) for r in turn_results)

    if verbose:
        if turn_results:
            console.print()
            _print_summary_table(turn_results, total_input, total_output)
        console.print(f"[dim]Session elapsed: {_elapsed(session_start)} | Polls: {poll_count}[/dim]")
    else:
        if turn_results:
            console.print(
                f"\n[dim]Total tokens — input: {total_input:,} | output: {total_output:,}[/dim]"
            )


ROLE_COLORS = ["blue", "yellow", "magenta", "cyan", "green", "red", "bright_blue", "bright_magenta"]


def _display_turn_result(
    result: dict,
    max_turns: int,
    content: str | None,
    verbose: bool = False,
    turn_elapsed: str = "",
):
    role = result["role"]
    turn = result["turn_number"]
    usage = result.get("token_usage", {})
    truncated = result.get("truncated", False)

    color = ROLE_COLORS[(turn - 1) % len(ROLE_COLORS)]

    # Build subtitle with token usage
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    subtitle_parts = [f"tokens: {in_tok:,} in / {out_tok:,} out"]
    tool_calls = result.get("tool_calls_made", 0)
    if tool_calls > 0:
        subtitle_parts.append(f"tool calls: {tool_calls}")
    if turn_elapsed:
        subtitle_parts.append(turn_elapsed)
    if truncated:
        subtitle_parts.append("[bold red]TRUNCATED[/bold red]")
    subtitle = " | ".join(subtitle_parts)

    display_content = content or "[dim]Content not available[/dim]"

    if max_turns > 0:
        title = f"[bold]Step {turn}/{max_turns} — {role}[/bold]"
    else:
        title = f"[bold]Step {turn} — {role}[/bold]"

    console.print(
        Panel(
            Markdown(display_content),
            title=title,
            subtitle=f"[dim]{subtitle}[/dim]",
            border_style=color,
            padding=(1, 2),
        )
    )

    if truncated:
        console.print(
            f"[bold red]Warning: Step {turn} output was truncated (hit max_tokens). "
            "The content above is incomplete.[/bold red]"
        )

    if result.get("key_insights"):
        insights = "\n".join(f"  - {i}" for i in result["key_insights"])
        console.print(f"[{color}]Key Insights:\n{insights}[/{color}]\n")

    if verbose:
        content_len = len(content) if content else 0
        console.print(
            f"[dim]  Content: {content_len:,} chars | "
            f"Input context: ~{in_tok:,} tokens[/dim]"
        )


def _print_summary_table(
    turn_results: list[dict],
    total_input: int,
    total_output: int,
):
    table = Table(title="Session Summary", show_lines=False, border_style="dim")
    table.add_column("Step", style="bold", justify="right")
    table.add_column("Role")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Insights", justify="right")

    for r in turn_results:
        usage = r.get("token_usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        n_insights = len(r.get("key_insights", []))
        trunc = " [red]*[/red]" if r.get("truncated") else ""
        table.add_row(
            str(r["turn_number"]),
            r["role"] + trunc,
            f"{in_tok:,}",
            f"{out_tok:,}",
            str(n_insights),
        )

    table.add_section()
    table.add_row(
        "",
        "[bold]Total[/bold]",
        f"[bold]{total_input:,}[/bold]",
        f"[bold]{total_output:,}[/bold]",
        "",
    )

    console.print(table)


def _prompt_for_tool_approval(command: str) -> bool:
    """Prompt user to approve or deny a tool call."""
    console.print()
    console.print(
        Panel(
            f"[bold]{command}[/bold]",
            title="[yellow]Tool Call Pending Approval[/yellow]",
            border_style="yellow",
        )
    )
    console.print("[bold]Approve?[/bold] (y/n): ", end="")
    try:
        response = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    approved = response in ("y", "yes", "")
    if approved:
        console.print("[dim]Approved.[/dim]")
    else:
        console.print("[dim]Denied.[/dim]")
    return approved


def _prompt_for_feedback() -> str | None:
    console.print(
        "[bold]Feedback?[/bold] Enter your feedback, or press Enter to skip:"
    )
    try:
        feedback = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return feedback if feedback else None


def _load_config_file(path: str) -> dict:
    """Load a YAML config file and return a dict of run-mode settings.

    Supported keys (all optional):
        prompt, turns, model, provider, base_url, auto, output,
        verbose, dangerous, goal_detect, max_subcommunis
    """
    p = Path(path)
    if p.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Config file must be .yaml or .yml, got: {p.suffix}")
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got: {type(data).__name__}")

    return data


def _add_common_args(parser: argparse.ArgumentParser, model_default: str = ""):
    """Add args shared between 'run' and 'chat' subcommands.

    model_default: for 'run', pass DEFAULT_MODEL_STRING for backward compat.
                   for 'chat', pass "" so env DEFAULT_MODEL is used.
    """
    help_text = "LLM model (default: env DEFAULT_MODEL or claude-sonnet-4-5-20250929)"
    if model_default:
        help_text = f"Claude model to use (default: {model_default})"
    parser.add_argument(
        "--model", "-m",
        default=model_default,
        help=help_text,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed progress",
    )
    parser.add_argument(
        "--provider", "-p",
        default="",
        help="LLM provider: 'anthropic' or 'openai' (overrides LLM_PROVIDER env var)",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Base URL for OpenAI-compatible API (overrides OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--dangerous",
        action="store_true",
        help="Auto-approve all tool calls without human confirmation (use with caution!)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="communis — Self-Directing Iterative Work Loop"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- 'chat' subcommand ---
    chat_parser = subparsers.add_parser("chat", help="Start interactive session REPL")
    chat_parser.add_argument(
        "--user", "-u",
        default="",
        help="Username for workflow IDs (default: $USER env var)",
    )
    _add_common_args(chat_parser)  # model defaults to "" → env DEFAULT_MODEL

    # --- 'run' subcommand (also default when positional arg is given) ---
    # For backward compatibility, we support both `communis run "prompt"` and `communis "prompt"`
    run_parser = subparsers.add_parser("run", help="Run a single-shot task (default)")
    run_parser.add_argument("idea", metavar="prompt", nargs="?", default=None, help="The prompt or task to work on")
    run_parser.add_argument(
        "--config", "-c",
        default="",
        help="Path to a YAML config file (see README for format)",
    )
    _add_common_args(run_parser, model_default=DEFAULT_MODEL_STRING)
    run_parser.add_argument(
        "--turns", "-t", type=int, default=0,
        help="Max steps (0 = indefinite with goal detection, default: 0)",
    )
    run_parser.add_argument(
        "--auto", "-a", action="store_true",
        help="Skip all feedback prompts and run straight through",
    )
    run_parser.add_argument(
        "--output", "-o",
        default="",
        help="Save session output to a markdown file (e.g. --output result.md)",
    )
    run_parser.add_argument(
        "--no-goal-detect",
        action="store_true",
        help="Disable goal completion detection (requires --turns > 0)",
    )
    run_parser.add_argument(
        "--max-subcommunis",
        type=int,
        default=3,
        help="Max parallel subcommunis (0 = disabled, default: 3, max: 5)",
    )

    # Parse args — handle backward compatibility (no subcommand = 'run')
    args = parser.parse_args()

    if args.command == "chat":
        from cli.session_cli import run_session_cli

        asyncio.run(run_session_cli(
            user=args.user,
            model=args.model,
            provider=args.provider,
            base_url=args.base_url,
            dangerous=args.dangerous,
            verbose=args.verbose,
        ))
        return

    if args.command == "run":
        config_path = args.config
    elif args.command is None:
        # Backward compatibility: if no subcommand, treat first positional as idea
        # Re-parse with the old-style parser
        legacy_parser = argparse.ArgumentParser(
            description="communis — Self-Directing Iterative Work Loop"
        )
        legacy_parser.add_argument("idea", metavar="prompt", nargs="?", default=None, help="The prompt or task to work on")
        legacy_parser.add_argument("--config", "-c", default="", help="Path to a YAML config file")
        _add_common_args(legacy_parser, model_default=DEFAULT_MODEL_STRING)
        legacy_parser.add_argument("--turns", "-t", type=int, default=0)
        legacy_parser.add_argument("--auto", "-a", action="store_true")
        legacy_parser.add_argument("--output", "-o", default="")
        legacy_parser.add_argument("--no-goal-detect", action="store_true")
        legacy_parser.add_argument("--max-subcommunis", type=int, default=3)
        args = legacy_parser.parse_args()
        config_path = args.config
    else:
        parser.print_help()
        return

    # Load YAML config file if provided, then let CLI args override
    file_cfg: dict = {}
    if config_path:
        try:
            file_cfg = _load_config_file(config_path)
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    # Resolve each setting: CLI arg (if explicitly set) > config file > default
    idea = args.idea if args.idea is not None else file_cfg.get("prompt", "")
    if not idea:
        console.print("[red]Error: No prompt provided. Pass a prompt argument or set 'prompt' in the config file.[/red]")
        return

    turns = args.turns if args.turns != 0 else file_cfg.get("turns", 0)
    model = args.model if args.model != DEFAULT_MODEL_STRING else file_cfg.get("model", DEFAULT_MODEL_STRING)
    auto = args.auto or file_cfg.get("auto", False)
    verbose = args.verbose or file_cfg.get("verbose", False)
    provider = args.provider if args.provider else file_cfg.get("provider", "")
    base_url = args.base_url if args.base_url else file_cfg.get("base_url", "")
    output = args.output if args.output else file_cfg.get("output", "")
    dangerous = args.dangerous or file_cfg.get("dangerous", False)
    no_goal_detect = args.no_goal_detect if hasattr(args, "no_goal_detect") and args.no_goal_detect else file_cfg.get("no_goal_detect", False)
    max_sub = args.max_subcommunis if args.max_subcommunis != 3 else file_cfg.get("max_subcommunis", 3)

    # Validation
    if no_goal_detect and turns <= 0:
        console.print("[red]Error: no_goal_detect requires turns > 0[/red]")
        return

    max_subcommunis = max(0, min(5, max_sub))
    goal_detect = not no_goal_detect

    asyncio.run(run_cli(
        idea,
        max_turns=turns,
        model=model,
        auto=auto,
        verbose=verbose,
        provider=provider,
        base_url=base_url,
        output=output,
        dangerous=dangerous,
        goal_complete_detection=goal_detect,
        max_subcommunis=max_subcommunis,
    ))


if __name__ == "__main__":
    main()
