"""CommunisAgent — front agent for the interactive session REPL.

Long-lived entity workflow that:
- Receives user messages via signals
- Runs an agent loop: LLM call → direct tool use or delegation → repeat
- Manages CommunisSubAgent child workflows
- Routes approval requests between sub-agents and the CLI
- Exposes state via queries for the CLI event poll loop
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.llm_activities import call_llm
    from activities.tool_activities import execute_run_command
    from models.session_types import (
        ApprovalRequest,
        SessionConfig,
        SessionState,
        TaskSpec,
        TaskStatus,
        TaskUpdate,
    )
    from prompts.session_prompts import FRONT_AGENT_SYSTEM_PROMPT
    from tools.delegate_tool import DELEGATE_TASK_TOOL
    from tools.run_tool import RUN_TOOL_DEFINITION
    from workflows.constants import LLM_RETRY_POLICY, LLM_TIMEOUT, TOOL_TIMEOUT
    from workflows.task_workflow import CommunisSubAgent

# Safety limit on tool iterations per user message
MAX_FRONT_AGENT_ITERATIONS = 10


def _extract_text_from_blocks(content_blocks: list[dict]) -> str:
    return "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )


def _extract_tool_uses(content_blocks: list[dict]) -> list[dict]:
    return [block for block in content_blocks if block.get("type") == "tool_use"]


@workflow.defn
class CommunisAgent:
    """Front agent workflow for the interactive session REPL."""

    def __init__(self):
        self.state = SessionState()
        self.config = SessionConfig()
        self.message_queue: list[str] = []
        self.task_updates: list[TaskUpdate] = []
        self.approval_responses: list[tuple[str, bool]] = []
        self.should_exit = False

    def _now(self) -> str:
        """Get current time as ISO string (deterministic within workflow)."""
        return workflow.now().isoformat()

    def _add_event(self, event_type: str, data: dict) -> None:
        """Add an event with the current workflow time."""
        self.state.add_event(event_type, data, timestamp=self._now())

    # --- Signals ---

    @workflow.signal
    async def user_message(self, message: str):
        """Signal from CLI: user typed a message."""
        self.message_queue.append(message)

    @workflow.signal
    async def task_update(self, update: TaskUpdate):
        """Signal from a sub-agent TaskWorkflow with a status update."""
        self.task_updates.append(update)

    @workflow.signal
    async def approval_response(self, response: list):
        """Signal from CLI: user approved/denied a tool call.

        response is [approval_id: str, approved: bool].
        """
        approval_id, approved = response[0], response[1]
        self.approval_responses.append((approval_id, approved))

    @workflow.signal
    async def clear_conversation(self):
        """Signal from CLI: clear conversation history, start fresh."""
        self.state.conversation.clear()
        self._add_event("conversation_cleared", {})

    @workflow.signal
    async def end_session(self):
        """Signal from CLI: user wants to end the session."""
        self.should_exit = True

    # --- Queries ---

    @workflow.query
    def get_state(self) -> dict:
        return self.state.to_dict()

    @workflow.query
    def get_events_since(self, since_event_id: int) -> list[dict]:
        """Return all events after the given event_id."""
        return [
            e.to_dict() for e in self.state.events
            if e.event_id > since_event_id
        ]

    @workflow.query
    def get_pending_approvals(self) -> list[dict]:
        return [a.to_dict() for a in self.state.pending_approvals if not a.resolved]

    # --- Main loop ---

    @workflow.run
    async def run(self, config: SessionConfig) -> dict:
        self.config = config
        self.state.session_id = workflow.info().workflow_id
        self.state.status = "active"

        while not self.should_exit:
            # Wait until there's something to process
            await workflow.wait_condition(
                lambda: bool(self.message_queue)
                or bool(self.task_updates)
                or bool(self.approval_responses)
                or self.should_exit,
            )

            if self.should_exit:
                break

            # Priority 1: Approval responses — forward to sub-agents (unblock fast)
            # Only forward approvals that belong to sub-agents (not the front agent).
            # Front agent approvals are consumed inline during _handle_user_message.
            while self.approval_responses:
                approval_id, approved = self.approval_responses[0]
                # Check if this belongs to a sub-agent
                is_subagent = any(
                    a.approval_id == approval_id and not a.resolved
                    for a in self.state.pending_approvals
                    if self.state.tasks.get(a.task_id, TaskStatus(task_id="", description="")).status == "waiting_approval"
                )
                if is_subagent:
                    self.approval_responses.pop(0)
                    await self._handle_approval_response(approval_id, approved)
                else:
                    break  # Front agent approval — leave for inline consumption

            # Priority 2: Task updates — update state, emit events
            while self.task_updates:
                update = self.task_updates.pop(0)
                self._handle_task_update(update)

            # Priority 3: User messages — run agent loop
            while self.message_queue:
                message = self.message_queue.pop(0)
                await self._handle_user_message(message)

        # Session ending
        self.state.status = "ended"
        self._add_event("session_ended", {})
        return self.state.to_dict()

    # --- Front agent loop ---

    def _build_system_prompt(self) -> str:
        """Build the front agent system prompt with current task state."""
        active = {
            k: v for k, v in self.state.tasks.items()
            if v.status not in ("completed", "failed", "cancelled")
        }
        if active:
            lines = []
            for task_id, task in active.items():
                line = f"- [{task_id}] {task.description} (status: {task.status})"
                if task.progress:
                    line += f" — {task.progress}"
                lines.append(line)
            tasks_text = "\n".join(lines)
        else:
            tasks_text = "No active tasks."
        return FRONT_AGENT_SYSTEM_PROMPT.format(active_tasks=tasks_text)

    async def _handle_user_message(self, message: str):
        """Process a user message through the front agent loop.

        The front agent can:
        1. Respond with text (direct conversation)
        2. Use the `run` tool directly (with approval unless dangerous)
        3. Use `delegate_task` to spawn a background sub-agent
        """
        # Add user message to conversation
        self.state.conversation.append({
            "role": "user",
            "content": message,
            "timestamp": self._now(),
        })

        system_prompt = self._build_system_prompt()
        tools = [RUN_TOOL_DEFINITION, DELEGATE_TASK_TOOL]

        # Build LLM messages from full conversation history.
        # Conversation entries with list content (tool_use/tool_result blocks)
        # are passed through as-is to preserve tool call context across turns.
        messages: list[dict] = [
            {"role": m["role"], "content": m["content"]}
            for m in self.state.conversation
        ]

        final_text = ""

        for _ in range(MAX_FRONT_AGENT_ITERATIONS):
            self._add_event("agent_status", {"status": "thinking", "detail": "Generating response..."})

            llm_response = await workflow.execute_activity(
                call_llm,
                args=[
                    messages,
                    system_prompt,
                    self.config.model,
                    0,  # max_tokens (0 = default)
                    self.config.provider,
                    self.config.base_url,
                    tools,
                ],
                start_to_close_timeout=LLM_TIMEOUT,
                retry_policy=LLM_RETRY_POLICY,
            )

            # Emit reasoning/thinking tokens if present
            reasoning = llm_response.get("reasoning", "")
            if reasoning:
                self._add_event("thinking", {"text": reasoning, "source": "front_agent"})

            content_blocks = llm_response.get("content_blocks", [])
            text = _extract_text_from_blocks(content_blocks)
            tool_uses = _extract_tool_uses(content_blocks)

            if not tool_uses:
                # No tools — LLM is done responding. Use this final text.
                final_text = text
                break

            # Persist assistant message (with tool_use blocks) to conversation
            # so the model retains tool call context across future turns.
            self.state.conversation.append({
                "role": "assistant",
                "content": content_blocks,
                "timestamp": self._now(),
            })
            messages.append({"role": "assistant", "content": content_blocks})

            # Process each tool call
            tool_results: list[dict] = []
            for tool_use in tool_uses:
                tool_name = tool_use.get("name", "")
                tool_input = tool_use.get("input", {})
                tool_use_id = tool_use.get("id", "")

                # Emit tool_call event for visibility
                self._add_event("tool_call", {
                    "source": "front_agent",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "thinking": text,  # LLM text alongside tool call
                })

                # Emit status so user sees which tool is running
                command = tool_input.get("command", tool_input.get("description", tool_name))
                self._add_event("agent_status", {
                    "status": "running_tool",
                    "detail": f"Running: {command[:60]}",
                })

                if tool_name == "run":
                    result = await self._handle_run_tool(tool_use_id, tool_input)
                    tool_results.append(result)

                elif tool_name == "delegate_task":
                    result = await self._handle_delegate_tool(tool_use_id, tool_input)
                    tool_results.append(result)

                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"[error] unknown tool: {tool_name}. Available: run, delegate_task",
                    })

            # Persist tool results to conversation history
            self.state.conversation.append({
                "role": "user",
                "content": tool_results,
                "timestamp": self._now(),
            })
            messages.append({"role": "user", "content": tool_results})

        # Add final text response to conversation
        if final_text:
            self.state.conversation.append({
                "role": "assistant",
                "content": final_text,
                "timestamp": self._now(),
            })
            self._add_event("assistant_message", {"text": final_text})

    async def _handle_run_tool(self, tool_use_id: str, tool_input: dict) -> dict:
        """Execute a run tool call, with approval unless dangerous mode."""
        command = tool_input.get("command", "")
        if not command:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": "[error] run: missing required parameter 'command'",
            }

        # Approval flow (unless dangerous mode)
        if not self.config.dangerous:
            approval_id = f"apr-{workflow.uuid4().hex[:8]}"

            # Surface approval request to CLI
            req = ApprovalRequest(
                approval_id=approval_id,
                task_id=self.state.session_id,  # front agent's own approval
                task_description="Front agent direct tool use",
                tool_name="run",
                tool_input=tool_input,
                timestamp=self._now(),
            )
            self.state.pending_approvals.append(req)
            self._add_event("approval_requested", {
                "approval_id": approval_id,
                "task_id": self.state.session_id,
                "task_description": "Front agent direct tool use",
                "tool_name": "run",
                "tool_input": tool_input,
            })

            # Wait for approval signal
            await workflow.wait_condition(
                lambda: any(
                    aid == approval_id for aid, _ in self.approval_responses
                ) or self.should_exit,
            )

            if self.should_exit:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": "[denied] Session ending.",
                }

            # Consume the approval from the queue
            approved = False
            for i, (aid, appr) in enumerate(self.approval_responses):
                if aid == approval_id:
                    approved = appr
                    self.approval_responses.pop(i)
                    break

            # Mark resolved
            req.resolved = True
            req.approved = approved
            self._add_event("approval_resolved", {
                "approval_id": approval_id,
                "task_id": self.state.session_id,
                "approved": approved,
            })

            if not approved:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": "[denied] Command execution denied by user.",
                }

        # Execute the command
        exec_result = await workflow.execute_activity(
            execute_run_command,
            args=[command, None],
            start_to_close_timeout=TOOL_TIMEOUT,
            retry_policy=LLM_RETRY_POLICY,
        )

        # Emit tool_result event for visibility
        output = exec_result["output"]
        self._add_event("tool_result", {
            "source": "front_agent",
            "tool_name": "run",
            "command": command,
            "exit_code": exec_result["exit_code"],
            "duration_ms": exec_result["duration_ms"],
            "output_preview": output[:500],
            "truncated": len(output) > 500,
        })

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output,
        }

    async def _handle_delegate_tool(self, tool_use_id: str, tool_input: dict) -> dict:
        """Spawn a sub-agent TaskWorkflow for complex work."""
        description = tool_input.get("description", "")
        if not description:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": "[error] delegate_task: missing required parameter 'description'",
            }

        context = tool_input.get("context", "")
        if not context:
            # Fall back to recent conversation
            recent = self.state.conversation[-6:]
            context = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in recent
            )

        user_part = f"-{self.config.user}" if self.config.user else ""
        task_id = f"communis-subagent{user_part}-{workflow.uuid4().hex[:8]}"

        spec = TaskSpec(
            task_id=task_id,
            description=description,
            context=context,
            parent_session_id=self.state.session_id,
            model=self.config.model,
            provider=self.config.provider,
            base_url=self.config.base_url,
            dangerous=self.config.dangerous,
        )

        self.state.tasks[task_id] = TaskStatus(
            task_id=task_id,
            description=description,
            status="pending",
            started_at=self._now(),
        )

        # Start child workflow (fire-and-forget — updates come via signals)
        await workflow.start_child_workflow(
            CommunisSubAgent.run,
            spec,
            id=task_id,
        )

        self.state.tasks[task_id].status = "running"
        self._add_event("task_started", {
            "task_id": task_id,
            "description": description,
        })

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": f"Task spawned: {task_id} — \"{description}\"\nRunning in background. You'll be notified on completion.",
        }

    # --- Task update handling ---

    def _handle_task_update(self, update: TaskUpdate):
        """Process a status update from a sub-agent task."""
        task = self.state.tasks.get(update.task_id)
        if not task:
            return

        if update.update_type == "progress":
            task.progress = update.message
            task.status = "running"
            self._add_event("task_progress", {
                "task_id": update.task_id,
                "message": update.message,
            })

        elif update.update_type == "completed":
            task.status = "completed"
            task.result_summary = update.result_summary
            task.completed_at = self._now()
            self._add_event("task_completed", {
                "task_id": update.task_id,
                "description": task.description,
                "result_summary": update.result_summary,
            })

        elif update.update_type == "failed":
            task.status = "failed"
            task.error = update.message
            task.completed_at = self._now()
            self._add_event("task_failed", {
                "task_id": update.task_id,
                "description": task.description,
                "error": update.message,
            })

        elif update.update_type == "tool_call":
            self._add_event("tool_call", {
                "source": update.task_id,
                "tool_name": "run",
                "command": update.message,
                "thinking": update.result_summary,  # LLM's reasoning text
            })

        elif update.update_type == "tool_result":
            # Parse structured metadata from result_summary (exit:N|Nms)
            parts = update.result_summary.split("|") if update.result_summary else []
            exit_str = parts[0].replace("exit:", "") if parts else "0"
            duration = parts[1] if len(parts) > 1 else ""
            self._add_event("tool_result", {
                "source": update.task_id,
                "tool_name": "run",
                "exit_code": int(exit_str) if exit_str.lstrip("-").isdigit() else 0,
                "duration": duration,
                "output_preview": update.message,
                "truncated": len(update.message) >= 500,
            })

        elif update.update_type == "approval_request":
            task.status = "waiting_approval"
            approval_info = update.approval_request
            approval_id = approval_info.get("approval_id", f"apr-{workflow.uuid4().hex[:8]}")
            req = ApprovalRequest(
                approval_id=approval_id,
                task_id=update.task_id,
                task_description=task.description,
                tool_name=approval_info.get("tool_name", ""),
                tool_input=approval_info.get("tool_input", {}),
                timestamp=self._now(),
            )
            self.state.pending_approvals.append(req)
            self._add_event("approval_requested", {
                "approval_id": approval_id,
                "task_id": update.task_id,
                "task_description": task.description,
                "tool_name": req.tool_name,
                "tool_input": req.tool_input,
            })

    async def _handle_approval_response(self, approval_id: str, approved: bool):
        """Forward an approval decision to the relevant sub-agent task."""
        req = None
        for a in self.state.pending_approvals:
            if a.approval_id == approval_id and not a.resolved:
                req = a
                break

        if not req:
            return

        req.resolved = True
        req.approved = approved

        # Signal the child task workflow
        task_handle = workflow.get_external_workflow_handle(req.task_id)
        await task_handle.signal("approval_decision", approved)

        # Update task status
        task = self.state.tasks.get(req.task_id)
        if task:
            task.status = "running"

        self._add_event("approval_resolved", {
            "approval_id": approval_id,
            "task_id": req.task_id,
            "approved": approved,
        })
