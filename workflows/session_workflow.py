"""SessionWorkflow — front agent for the interactive session REPL.

Long-lived entity workflow that:
- Receives user messages via signals
- Calls a front agent LLM to decide: answer directly vs delegate to sub-agents
- Manages sub-agent TaskWorkflows as child workflows
- Routes approval requests between sub-agents and the CLI
- Exposes state via queries for the CLI event poll loop
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.session_activities import front_agent_respond
    from models.session_types import (
        ApprovalRequest,
        SessionConfig,
        SessionState,
        TaskSpec,
        TaskStatus,
        TaskUpdate,
    )
    from workflows.constants import LLM_RETRY_POLICY, LLM_TIMEOUT
    from workflows.task_workflow import TaskWorkflow


@workflow.defn
class SessionWorkflow:
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
            while self.approval_responses:
                approval_id, approved = self.approval_responses.pop(0)
                await self._handle_approval_response(approval_id, approved)

            # Priority 2: Task updates — update state, emit events
            while self.task_updates:
                update = self.task_updates.pop(0)
                self._handle_task_update(update)

            # Priority 3: User messages — call front agent, emit events, spawn tasks
            while self.message_queue:
                message = self.message_queue.pop(0)
                await self._handle_user_message(message)

        # Session ending
        self.state.status = "ended"
        self._add_event("session_ended", {})
        return self.state.to_dict()

    async def _handle_user_message(self, message: str):
        """Process a user message: call front agent LLM, handle response."""
        # Add to conversation
        self.state.conversation.append({
            "role": "user",
            "content": message,
            "timestamp": self._now(),
        })

        # Call front agent activity
        active_tasks = {
            k: v.to_dict() for k, v in self.state.tasks.items()
            if v.status not in ("completed", "failed", "cancelled")
        }

        response = await workflow.execute_activity(
            front_agent_respond,
            args=[
                self.state.conversation,
                active_tasks,
                self.config.model,
                self.config.provider,
                self.config.base_url,
            ],
            start_to_close_timeout=LLM_TIMEOUT,
            retry_policy=LLM_RETRY_POLICY,
        )

        # Add assistant response to conversation
        text = response.get("text", "")
        if text:
            self.state.conversation.append({
                "role": "assistant",
                "content": text,
                "timestamp": self._now(),
            })
            self._add_event("assistant_message", {"text": text})

        # Spawn delegated tasks
        delegate_tasks = response.get("delegate_tasks", [])
        for task_def in delegate_tasks:
            if len([t for t in self.state.tasks.values()
                    if t.status in ("pending", "running", "waiting_approval")]) >= self.config.max_concurrent_tasks:
                self._add_event("assistant_message", {
                    "text": f"Cannot start more tasks — limit of {self.config.max_concurrent_tasks} concurrent tasks reached.",
                })
                break
            await self._spawn_task(task_def)

    async def _spawn_task(self, task_def: dict):
        """Start a new TaskWorkflow as a child workflow."""
        task_id = f"task-{workflow.uuid4().hex[:8]}"
        description = task_def.get("description", "")
        context = task_def.get("context", "")

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

        # Track the task
        self.state.tasks[task_id] = TaskStatus(
            task_id=task_id,
            description=description,
            status="pending",
            started_at=self._now(),
        )

        # Start child workflow (fire-and-forget — updates come via signals)
        # Don't specify task_queue — inherit from parent workflow
        await workflow.start_child_workflow(
            TaskWorkflow.run,
            spec,
            id=task_id,
        )

        self.state.tasks[task_id].status = "running"
        self._add_event("task_started", {
            "task_id": task_id,
            "description": description,
        })

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
