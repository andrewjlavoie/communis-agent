"""Session CLI — interactive REPL for the communis session agent.

Two concurrent asyncio tasks:
1. Input loop: reads user input via run_in_executor
2. Event poll loop: queries get_events_since every 0.5s and renders events

Approval UX:
- Approvals pop up and lock the chat (y/n/wait)
- "wait" defers the approval and unlocks the chat
- /approvals shows deferred approvals and lets the user act on them
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

        # Approval state
        self._active_approval: dict | None = None  # currently blocking input
        self._deferred_approvals: list[dict] = []  # user said "wait"
        self._approval_queue: list[dict] = []  # arrived while another is active

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

    # --- Approval management ---

    def _push_approval(self, approval: dict):
        """New approval arrived. Either activate it or queue it."""
        if self._active_approval is None:
            self._active_approval = approval
            self._render_approval_prompt(approval)
        else:
            # Already have an active approval — queue this one
            self._approval_queue.append(approval)
            aid_short = approval["approval_id"][:8]
            console.print(
                f"  [yellow]+ Queued approval {aid_short}[/yellow] "
                f"[dim]({len(self._approval_queue) + len(self._deferred_approvals)} waiting — /approvals to view)[/dim]"
            )

    def _activate_next_approval(self):
        """After resolving active approval, activate next from queue."""
        self._active_approval = None
        if self._approval_queue:
            next_approval = self._approval_queue.pop(0)
            self._active_approval = next_approval
            self._render_approval_prompt(next_approval)

    def _render_approval_prompt(self, approval: dict):
        """Render the approval panel that locks input."""
        task_desc = approval.get("task_description", "")
        tool_input = approval.get("tool_input", {})
        command = tool_input.get("command", str(tool_input))

        console.print()
        console.print(Panel(
            f"[bold]{command}[/bold]",
            title=f"[yellow]Approval Required[/yellow] — {task_desc}",
            subtitle="[dim]y/yes to approve, n/no to deny, w/wait to defer[/dim]",
            border_style="yellow",
        ))

    async def _resolve_approval(self, approval_id: str, approved: bool):
        """Send approval decision to the workflow."""
        await self.handle.signal(
            CommunisAgent.approval_response,
            [approval_id, approved],
        )
        status = "[green]Approved[/green]" if approved else "[red]Denied[/red]"
        console.print(f"  {status}")

    async def _defer_approval(self):
        """Move the active approval to deferred list."""
        if self._active_approval:
            self._deferred_approvals.append(self._active_approval)
            aid_short = self._active_approval["approval_id"][:8]
            console.print(
                f"  [yellow]Deferred {aid_short}[/yellow] "
                f"[dim]({len(self._deferred_approvals)} deferred — /approvals to manage)[/dim]"
            )
            self._activate_next_approval()

    # --- Input loop ---

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

            # Commands always work, even during approvals
            if line.startswith("/"):
                await self._handle_command(line)
                continue

            # Active approval locks input to y/n/wait
            if self._active_approval:
                lower = line.lower()
                if lower in ("y", "yes"):
                    await self._resolve_approval(self._active_approval["approval_id"], True)
                    self._activate_next_approval()
                elif lower in ("n", "no"):
                    await self._resolve_approval(self._active_approval["approval_id"], False)
                    self._activate_next_approval()
                elif lower in ("w", "wait"):
                    await self._defer_approval()
                else:
                    console.print("[dim]Approval pending — respond y/n/wait (or /approvals to view all)[/dim]")
                continue

            # Regular message
            await self.handle.signal(CommunisAgent.user_message, line)

    def _get_input(self) -> str | None:
        try:
            if self._active_approval:
                return input("[y/n/wait] > ")
            n_deferred = len(self._deferred_approvals) + len(self._approval_queue)
            if n_deferred:
                return input(f"({n_deferred} waiting) > ")
            return input("> ")
        except (EOFError, KeyboardInterrupt):
            return None

    # --- Event poll loop ---

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

            self._push_approval({
                "approval_id": approval_id,
                "task_description": task_desc,
                "tool_name": tool_name,
                "tool_input": tool_input,
            })

        elif event_type == "approval_resolved":
            approved = data.get("approved", False)
            status = "[green]approved[/green]" if approved else "[red]denied[/red]"
            console.print(f"[dim]Approval {status}[/dim]")

        elif event_type == "tool_call":
            if self.verbose:
                source = data.get("source", "agent")
                tool_name = data.get("tool_name", "run")
                tool_input = data.get("tool_input", {})
                command = data.get("command", "") or tool_input.get("command", "")
                thinking = data.get("thinking", "")

                agent_name = "agent" if source == "front_agent" else f"task:{source[:12]}"

                if thinking:
                    console.print(f"  [magenta][thinking - {agent_name}][/magenta] [dim italic]{thinking}[/dim italic]")

                if tool_name == "delegate_task":
                    desc = tool_input.get("description", "")
                    console.print(f"  [dim][{agent_name}][/dim] [cyan]delegate_task[/cyan] {desc}")
                else:
                    console.print(f"  [dim][{agent_name}][/dim] [yellow]$ {command}[/yellow]")

        elif event_type == "tool_result":
            if self.verbose:
                source = data.get("source", "agent")
                exit_code = data.get("exit_code", 0)
                duration_ms = data.get("duration_ms", data.get("duration", ""))
                preview = data.get("output_preview", "")
                truncated = data.get("truncated", False)

                agent_name = "agent" if source == "front_agent" else f"task:{source[:12]}"

                exit_style = "green" if exit_code == 0 else "red"

                # Show output preview (first few lines, dimmed)
                if preview.strip():
                    lines = preview.strip().split("\n")
                    display = lines[:8]
                    for line in display:
                        console.print(f"  [dim]{line}[/dim]")
                    if len(lines) > 8 or truncated:
                        console.print(f"  [dim]  ... (truncated)[/dim]")

                console.print(
                    f"  [dim][{agent_name}] [{exit_style}]exit:{exit_code}[/{exit_style}]"
                    f" | {duration_ms}[/dim]"
                )

        elif event_type == "conversation_cleared":
            pass  # Already printed by /clear handler

        elif event_type == "session_ended":
            pass  # Handled by main loop

    # --- Commands ---

    async def _handle_command(self, line: str):
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/quit" or cmd == "/exit":
            self.running = False

        elif cmd == "/help":
            console.print(Panel(
                "[bold]/approvals[/bold] — View and manage deferred approvals\n"
                "[bold]/approve <n>[/bold] — Approve a deferred approval by number\n"
                "[bold]/deny <n>[/bold] — Deny a deferred approval by number\n"
                "[bold]/clear[/bold] — Clear conversation history (start fresh, keep tasks)\n"
                "[bold]/tasks[/bold] — List all tasks\n"
                "[bold]/task <id>[/bold] — Show task details\n"
                "[bold]/cancel <id>[/bold] — Cancel a running task\n"
                "[bold]/status[/bold] — Show session info\n"
                "[bold]/quit[/bold] — End session\n"
                "[bold]/help[/bold] — Show this help\n"
                "\n[dim]During approvals: y/yes, n/no, w/wait[/dim]",
                title="Commands",
                border_style="dim",
            ))

        elif cmd == "/approvals":
            await self._show_approvals()

        elif cmd == "/approve":
            if len(parts) < 2:
                console.print("[dim]Usage: /approve <number> (see /approvals for list)[/dim]")
                return
            await self._resolve_deferred(parts[1].strip(), approved=True)

        elif cmd == "/deny":
            if len(parts) < 2:
                console.print("[dim]Usage: /deny <number> (see /approvals for list)[/dim]")
                return
            await self._resolve_deferred(parts[1].strip(), approved=False)

        elif cmd == "/approveall":
            await self._resolve_all_deferred(approved=True)

        elif cmd == "/denyall":
            await self._resolve_all_deferred(approved=False)

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

        elif cmd == "/clear":
            await self.handle.signal(CommunisAgent.clear_conversation)
            console.print("[dim]Conversation cleared.[/dim]\n")

        elif cmd == "/status":
            state = await self.handle.query(CommunisAgent.get_state)
            n_msgs = len(state.get("conversation", []))
            n_tasks = len(state.get("tasks", {}))
            n_active = len([
                t for t in state.get("tasks", {}).values()
                if t.get("status") in ("pending", "running", "waiting_approval")
            ])
            n_events = state.get("event_counter", 0)
            n_deferred = len(self._deferred_approvals)
            n_queued = len(self._approval_queue)
            approval_line = ""
            if n_deferred or n_queued:
                approval_line = f"\n[bold]Approvals:[/bold] {n_deferred} deferred, {n_queued} queued"
            console.print(Panel(
                f"[bold]Session:[/bold] {self.session_id}\n"
                f"[bold]Status:[/bold] {state.get('status', 'unknown')}\n"
                f"[bold]Messages:[/bold] {n_msgs}\n"
                f"[bold]Tasks:[/bold] {n_tasks} total, {n_active} active\n"
                f"[bold]Events:[/bold] {n_events}"
                f"{approval_line}",
                title="Session Info",
                border_style="dim",
            ))

        else:
            console.print(f"[dim]Unknown command: {cmd}. Type /help for available commands.[/dim]")

    # --- Approval management commands ---

    async def _show_approvals(self):
        """Show all deferred and queued approvals."""
        all_approvals = self._deferred_approvals + self._approval_queue
        if not all_approvals and not self._active_approval:
            console.print("[dim]No pending approvals.[/dim]")
            return

        if self._active_approval:
            tool_input = self._active_approval.get("tool_input", {})
            command = tool_input.get("command", str(tool_input))
            task_desc = self._active_approval.get("task_description", "")
            console.print(Panel(
                f"[bold]{command}[/bold]\n"
                f"[dim]Source: {task_desc}[/dim]",
                title="[yellow]Active (blocking input)[/yellow]",
                border_style="yellow",
            ))

        if all_approvals:
            table = Table(title="Deferred Approvals", show_lines=True, border_style="yellow")
            table.add_column("#", style="bold yellow", width=4)
            table.add_column("Source", style="dim")
            table.add_column("Command")
            table.add_column("ID", style="dim")

            for i, approval in enumerate(all_approvals, 1):
                tool_input = approval.get("tool_input", {})
                command = tool_input.get("command", str(tool_input))
                task_desc = approval.get("task_description", "")
                aid = approval["approval_id"][:8]
                table.add_row(str(i), task_desc, command, aid)

            console.print(table)
            console.print("[dim]Use /approve <n> or /deny <n> to resolve, /approveall or /denyall for batch[/dim]")

    async def _resolve_deferred(self, index_str: str, approved: bool):
        """Resolve a deferred approval by its 1-based index."""
        all_approvals = self._deferred_approvals + self._approval_queue

        try:
            idx = int(index_str) - 1
        except ValueError:
            console.print("[dim]Usage: /approve <number> or /deny <number>[/dim]")
            return

        if idx < 0 or idx >= len(all_approvals):
            console.print(f"[dim]Invalid number. {len(all_approvals)} approvals pending.[/dim]")
            return

        approval = all_approvals[idx]
        # Remove from whichever list it's in
        if idx < len(self._deferred_approvals):
            self._deferred_approvals.pop(idx)
        else:
            self._approval_queue.pop(idx - len(self._deferred_approvals))

        await self._resolve_approval(approval["approval_id"], approved)

    async def _resolve_all_deferred(self, approved: bool):
        """Approve or deny all deferred + queued approvals."""
        all_approvals = self._deferred_approvals + self._approval_queue
        if not all_approvals:
            console.print("[dim]No deferred approvals.[/dim]")
            return

        count = len(all_approvals)
        for approval in all_approvals:
            await self._resolve_approval(approval["approval_id"], approved)

        self._deferred_approvals.clear()
        self._approval_queue.clear()

        action = "Approved" if approved else "Denied"
        console.print(f"  [{'green' if approved else 'red'}]{action} {count} approvals[/{'green' if approved else 'red'}]")


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
