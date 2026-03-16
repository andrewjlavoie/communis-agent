"""Session CLI — interactive REPL for the communis session agent.

Two concurrent asyncio tasks:
1. Input loop: reads user input via run_in_executor
2. Event poll loop: queries get_events_since every 0.5s and renders events
"""

from __future__ import annotations

import asyncio
import os
import signal
import uuid

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from temporalio.client import Client

from models.session_types import SessionConfig
from shared.constants import TASK_QUEUE
from workflows.session_workflow import CommunisAgent

POLL_INTERVAL = 0.5

console = Console()


class SessionCLI:
    """Interactive REPL that communicates with a CommunisAgent via Temporal signals/queries."""

    def __init__(
        self,
        client: Client,
        config: SessionConfig,
        verbose: bool = False,
    ):
        self.client = client
        self.config = config
        self.verbose = verbose
        self.running = True
        self.last_event_id = 0
        self.handle = None
        self.session_id = ""
        self._pending_approval: dict | None = None

    async def run(self):
        """Start the session workflow and run input + poll loops concurrently."""
        user_part = f"-{self.config.user}" if self.config.user else ""
        self.session_id = f"communis-agent{user_part}-{uuid.uuid4().hex[:8]}"

        self.handle = await self.client.start_workflow(
            CommunisAgent.run,
            self.config,
            id=self.session_id,
            task_queue=TASK_QUEUE,
        )

        console.print(Panel("[bold]communis chat[/bold] — Interactive Session", style="blue"))
        console.print(f"[dim]Session: {self.session_id}[/dim]")
        if self.verbose:
            mode_parts = []
            if self.config.dangerous:
                mode_parts.append("DANGEROUS")
            mode = ", ".join(mode_parts) if mode_parts else "interactive"
            model_display = self.config.model or "env default"
            provider_display = self.config.provider or "env default"
            console.print(f"[dim]Model: {model_display} | Provider: {provider_display} | Mode: {mode}[/dim]")
        console.print("[dim]Type /help for commands, /quit to exit.[/dim]")
        console.print()

        # Install signal handler
        loop = asyncio.get_running_loop()
        self._install_cancel_handler(loop)

        input_task = asyncio.create_task(self._input_loop())
        poll_task = asyncio.create_task(self._event_poll_loop())

        try:
            done, pending = await asyncio.wait(
                [input_task, poll_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            pass

        # End session
        try:
            await self.handle.signal(CommunisAgent.end_session)
            await self.handle.result()
        except Exception:
            pass

        console.print("\n[dim]Session ended.[/dim]")

    def _install_cancel_handler(self, loop: asyncio.AbstractEventLoop):
        cancel_requested = False

        def on_signal(sig: int, _frame: object) -> None:
            nonlocal cancel_requested
            if cancel_requested:
                raise SystemExit(1)
            cancel_requested = True
            self.running = False
            console.print("\n[yellow]Ending session...[/yellow]")

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

    async def _input_loop(self):
        loop = asyncio.get_running_loop()
        while self.running:
            try:
                line = await loop.run_in_executor(None, self._get_input)
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

            if line is None:
                continue

            line = line.strip()
            if not line:
                continue

            # Handle commands
            if line.startswith("/"):
                await self._handle_command(line)
                continue

            # Handle approval responses
            if self._pending_approval and line.lower() in ("y", "yes", "n", "no"):
                approved = line.lower() in ("y", "yes")
                approval_id = self._pending_approval["approval_id"]
                await self.handle.signal(
                    CommunisAgent.approval_response,
                    [approval_id, approved],
                )
                status = "[green]Approved[/green]" if approved else "[red]Denied[/red]"
                console.print(f"[dim]{status}[/dim]")
                self._pending_approval = None
                continue

            # Regular message
            await self.handle.signal(CommunisAgent.user_message, line)

    def _get_input(self) -> str | None:
        try:
            if self._pending_approval:
                return input("[y/n] > ")
            return input("> ")
        except (EOFError, KeyboardInterrupt):
            return None

    async def _event_poll_loop(self):
        while self.running:
            try:
                events = await self.handle.query(
                    CommunisAgent.get_events_since,
                    self.last_event_id,
                )
                for event in events:
                    self._render_event(event)
                    self.last_event_id = event["event_id"]
            except Exception:
                if not self.running:
                    break
            await asyncio.sleep(POLL_INTERVAL)

    def _render_event(self, event: dict):
        event_type = event["event_type"]
        data = event.get("data", {})

        if event_type == "assistant_message":
            text = data.get("text", "")
            console.print()
            console.print(Panel(
                Markdown(text),
                border_style="blue",
                padding=(0, 1),
            ))
            console.print()

        elif event_type == "task_started":
            task_id = data.get("task_id", "")[:12]
            desc = data.get("description", "")
            console.print(f"[cyan][task:{task_id}] Started: {desc}[/cyan]")

        elif event_type == "task_progress":
            task_id = data.get("task_id", "")[:12]
            msg = data.get("message", "")
            console.print(f"[dim][task:{task_id}] {msg}[/dim]")

        elif event_type == "task_completed":
            task_id = data.get("task_id", "")[:12]
            desc = data.get("description", "")
            summary = data.get("result_summary", "")
            display = summary[:500] if summary else "Done"
            console.print()
            console.print(Panel(
                Markdown(display),
                title=f"[bold green]Task Complete: {desc}[/bold green]",
                border_style="green",
                padding=(0, 1),
            ))
            console.print()

        elif event_type == "task_failed":
            task_id = data.get("task_id", "")[:12]
            desc = data.get("description", "")
            error = data.get("error", "unknown error")
            console.print()
            console.print(Panel(
                f"[red]{error}[/red]",
                title=f"[bold red]Task Failed: {desc}[/bold red]",
                border_style="red",
            ))
            console.print()

        elif event_type == "approval_requested":
            approval_id = data.get("approval_id", "")
            task_desc = data.get("task_description", "")
            tool_name = data.get("tool_name", "")
            tool_input = data.get("tool_input", {})
            command = tool_input.get("command", str(tool_input))

            self._pending_approval = {"approval_id": approval_id}
            console.print()
            console.print(Panel(
                f"[bold]{command}[/bold]",
                title=f"[yellow]Approval Required[/yellow] — {task_desc}",
                subtitle="[dim]y/yes to approve, n/no to deny[/dim]",
                border_style="yellow",
            ))

        elif event_type == "approval_resolved":
            approved = data.get("approved", False)
            status = "[green]approved[/green]" if approved else "[red]denied[/red]"
            console.print(f"[dim]Approval {status}[/dim]")

        elif event_type == "session_ended":
            pass  # Handled by main loop

    async def _handle_command(self, line: str):
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/quit" or cmd == "/exit":
            self.running = False

        elif cmd == "/help":
            console.print(Panel(
                "[bold]/tasks[/bold] — List all tasks\n"
                "[bold]/task <id>[/bold] — Show task details\n"
                "[bold]/cancel <id>[/bold] — Cancel a running task\n"
                "[bold]/quit[/bold] — End session\n"
                "[bold]/help[/bold] — Show this help",
                title="Commands",
                border_style="dim",
            ))

        elif cmd == "/tasks":
            state = await self.handle.query(CommunisAgent.get_state)
            tasks = state.get("tasks", {})
            if not tasks:
                console.print("[dim]No tasks.[/dim]")
                return

            table = Table(title="Tasks", show_lines=False, border_style="dim")
            table.add_column("ID", style="bold")
            table.add_column("Description")
            table.add_column("Status")
            table.add_column("Progress")

            status_colors = {
                "pending": "dim",
                "running": "cyan",
                "waiting_approval": "yellow",
                "completed": "green",
                "failed": "red",
                "cancelled": "dim",
            }

            for tid, task_info in tasks.items():
                status = task_info.get("status", "")
                color = status_colors.get(status, "white")
                table.add_row(
                    tid[:12],
                    task_info.get("description", "")[:50],
                    f"[{color}]{status}[/{color}]",
                    task_info.get("progress", "")[:40] or task_info.get("result_summary", "")[:40],
                )
            console.print(table)

        elif cmd == "/task":
            if len(parts) < 2:
                console.print("[dim]Usage: /task <task-id>[/dim]")
                return
            task_id_prefix = parts[1].strip()
            state = await self.handle.query(CommunisAgent.get_state)
            tasks = state.get("tasks", {})
            matches = [t for tid, t in tasks.items() if tid.startswith(task_id_prefix)]
            if not matches:
                console.print(f"[dim]No task found matching '{task_id_prefix}'[/dim]")
                return
            task_info = matches[0]
            console.print(Panel(
                f"[bold]ID:[/bold] {task_info['task_id']}\n"
                f"[bold]Description:[/bold] {task_info['description']}\n"
                f"[bold]Status:[/bold] {task_info['status']}\n"
                f"[bold]Progress:[/bold] {task_info.get('progress', '')}\n"
                f"[bold]Result:[/bold] {task_info.get('result_summary', '')}\n"
                f"[bold]Error:[/bold] {task_info.get('error', '')}\n"
                f"[bold]Started:[/bold] {task_info.get('started_at', '')}\n"
                f"[bold]Completed:[/bold] {task_info.get('completed_at', '')}",
                title=f"Task: {task_info['task_id'][:12]}",
                border_style="dim",
            ))

        elif cmd == "/cancel":
            if len(parts) < 2:
                console.print("[dim]Usage: /cancel <task-id>[/dim]")
                return
            task_id_prefix = parts[1].strip()
            state = await self.handle.query(CommunisAgent.get_state)
            tasks = state.get("tasks", {})
            matches = [(tid, t) for tid, t in tasks.items() if tid.startswith(task_id_prefix)]
            if not matches:
                console.print(f"[dim]No task found matching '{task_id_prefix}'[/dim]")
                return
            task_id = matches[0][0]
            try:
                task_handle = self.client.get_workflow_handle(task_id)
                await task_handle.cancel()
                console.print(f"[yellow]Cancel request sent to {task_id[:12]}[/yellow]")
            except Exception as e:
                console.print(f"[red]Failed to cancel: {e}[/red]")

        else:
            console.print(f"[dim]Unknown command: {cmd}. Type /help for available commands.[/dim]")


async def run_session_cli(
    *,
    user: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    dangerous: bool = False,
    verbose: bool = False,
):
    """Entry point for the session CLI.

    Empty model/provider/base_url means "use env default" — resolved at the activity layer.
    """
    load_dotenv()

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")

    try:
        client = await Client.connect(address, namespace=namespace)
    except Exception as e:
        console.print(f"[red]Failed to connect to Temporal at {address}: {e}[/red]")
        console.print("Make sure Temporal dev server is running: temporal server start-dev")
        return

    # Pass through as-is. Empty string = env default, resolved in activities.
    config = SessionConfig(
        user=user or os.getenv("USER", ""),
        model=model,
        provider=provider,
        base_url=base_url,
        dangerous=dangerous,
    )

    cli = SessionCLI(client, config, verbose=verbose)
    await cli.run()
