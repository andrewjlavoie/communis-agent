from __future__ import annotations

import argparse
import asyncio
import os
import signal
import time
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from temporalio.client import Client, WorkflowHandle

from models.data_types import RiffConfig
from workflows.riff_orchestrator import RiffOrchestratorWorkflow

TASK_QUEUE = "autoriff-task-queue"
POLL_INTERVAL = 1.5

console = Console()


def _read_turn_file(artifact_path: str) -> tuple[str | None, dict]:
    """Read turn content and frontmatter from a workspace artifact file.

    Returns (content, metadata) tuple.
    """
    path = Path(artifact_path)
    if not path.exists():
        return None, {}

    text = path.read_text()
    if not text.startswith("---"):
        return text, {}

    end = text.find("\n---\n", 3)
    if end == -1:
        return text, {}

    frontmatter = text[4:end]
    content = text[end + 5:]

    try:
        meta = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        meta = {}

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


async def run_cli(
    idea: str, num_turns: int, model: str, auto: bool = False, verbose: bool = False,
    provider: str = "", base_url: str = "",
):
    load_dotenv()

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")

    console.print(Panel("[bold]autoRiff[/bold] — Self-Directing Work Loop", style="blue"))
    console.print(f"Prompt: [bold]{idea}[/bold]")
    mode = "auto (no feedback prompts)" if auto else "interactive"
    provider_label = provider or "env default"
    console.print(f"Turns: {num_turns} | Model: {model} | Provider: {provider_label} | Mode: {mode}")

    if verbose:
        console.print(f"[dim]Temporal: {address} (namespace: {namespace})[/dim]")
        console.print(f"[dim]Task queue: {TASK_QUEUE}[/dim]")
        console.print(f"[dim]Poll interval: {POLL_INTERVAL}s[/dim]")

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
    workflow_id = f"autoriff-{uuid.uuid4().hex[:8]}"
    config = RiffConfig(idea=idea, num_turns=num_turns, model=model, auto=auto, provider=provider, base_url=base_url)

    handle = await client.start_workflow(
        RiffOrchestratorWorkflow.run,
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
            state = await handle.query(RiffOrchestratorWorkflow.get_state)
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

        # Display new turn results — read content from workspace files
        for result in state["turn_results"]:
            if result["turn_number"] > last_turn_displayed:
                turn_elapsed = ""
                if turn_start is not None:
                    turn_elapsed = _elapsed(turn_start)
                last_turn_displayed = result["turn_number"]
                artifact_path = result.get("artifact_path", "")
                content, _ = _read_turn_file(artifact_path)
                _display_turn_result(result, state["num_turns"], content, verbose, turn_elapsed)

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
                    await handle.signal(RiffOrchestratorWorkflow.skip_feedback)
                    console.print("[dim]Skipping feedback, continuing...[/dim]\n")
                else:
                    await handle.signal(RiffOrchestratorWorkflow.receive_user_feedback, feedback)
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
                    else:
                        console.print(f"[cyan]{current_msg}[/cyan]")

                if turn_start is None:
                    turn_start = time.monotonic()

            elif status == "complete":
                console.print()
                console.print(Panel("[bold green]All turns complete![/bold green]", style="green"))
                ws_dir = state.get("workspace_dir", "")
                if ws_dir:
                    console.print(f"[dim]Workspace: {ws_dir}[/dim]")
                break

            elif status == "cancelled":
                console.print()
                turns_done = len(state.get("turn_results", []))
                console.print(
                    Panel(
                        f"[bold yellow]Workflow cancelled after {turns_done}/{state['num_turns']} turns.[/bold yellow]",
                        style="yellow",
                    )
                )
                ws_dir = state.get("workspace_dir", "")
                if ws_dir:
                    console.print(f"[dim]Workspace (partial): {ws_dir}[/dim]")
                break

            elif status == "error":
                console.print(f"[red]Error: {state['latest_message']}[/red]")
                break

    # Get final result
    try:
        final = await handle.result()
        turn_results = final.get("turn_results", [])
        _print_final_stats(turn_results, verbose, session_start, poll_count)
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
    total_turns: int,
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
    if turn_elapsed:
        subtitle_parts.append(turn_elapsed)
    if truncated:
        subtitle_parts.append("[bold red]TRUNCATED[/bold red]")
    subtitle = " | ".join(subtitle_parts)

    display_content = content or "[dim]Content not available[/dim]"

    console.print(
        Panel(
            Markdown(display_content),
            title=f"[bold]Turn {turn}/{total_turns} — {role}[/bold]",
            subtitle=f"[dim]{subtitle}[/dim]",
            border_style=color,
            padding=(1, 2),
        )
    )

    if truncated:
        console.print(
            f"[bold red]Warning: Turn {turn} output was truncated (hit max_tokens). "
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
    table.add_column("Turn", style="bold", justify="right")
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


def _prompt_for_feedback() -> str | None:
    console.print(
        "[bold]Feedback?[/bold] Enter your feedback, or press Enter to skip:"
    )
    try:
        feedback = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return feedback if feedback else None


def main():
    parser = argparse.ArgumentParser(
        description="autoRiff — Self-Directing Iterative Work Loop"
    )
    parser.add_argument("idea", metavar="prompt", help="The prompt or task to work on")
    parser.add_argument(
        "--turns", "-t", type=int, default=3, help="Number of riff turns (default: 3)"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="claude-sonnet-4-5-20250929",
        help="Claude model to use (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--auto", "-a", action="store_true",
        help="Skip all feedback prompts and run straight through",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed progress: timing, file paths, token breakdown, and session summary table",
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
    args = parser.parse_args()

    if args.turns < 1:
        console.print("[red]Error: turns must be at least 1[/red]")
        return
    if args.turns > 10:
        console.print("[red]Error: turns must be at most 10[/red]")
        return

    asyncio.run(run_cli(args.idea, args.turns, args.model, args.auto, args.verbose, args.provider, args.base_url))


if __name__ == "__main__":
    main()
