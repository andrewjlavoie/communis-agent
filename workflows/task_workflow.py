"""TaskWorkflow — sub-agent that executes a task with an LLM + tool loop.

Adapted from CommunisTurnWorkflow but:
- Multi-turn within a single workflow (no child workflow per iteration)
- Signals parent session with updates via external workflow handle
- Receives approval signals from parent session
"""

from __future__ import annotations

import asyncio

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.llm_activities import call_llm
    from activities.tool_activities import execute_run_command
    from models.session_types import TaskSpec, TaskUpdate
    from tools.run_tool import RUN_TOOL_DEFINITION
    from workflows.constants import (
        LLM_RETRY_POLICY,
        LLM_TIMEOUT,
        TOOL_TIMEOUT,
    )


TASK_AGENT_PROMPT = """\
You are a sub-agent working on a specific task. Complete the task thoroughly and autonomously.

Task: {description}

Context: {context}

You have access to a `run` tool that executes shell commands. Use it to:
- Read, write, and search files
- Run code, scripts, or tests
- Inspect the filesystem or system state
- Download or process data
- Create files and directories

You can compose commands with pipes and chains:
  run(command="cat file.txt | grep pattern | wc -l")
  run(command="ls -la && cat README.md")

Focus on producing concrete results. When done, provide a clear summary of what was accomplished.
"""


def _extract_text_from_blocks(content_blocks: list[dict]) -> str:
    return "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )


def _extract_tool_uses(content_blocks: list[dict]) -> list[dict]:
    return [block for block in content_blocks if block.get("type") == "tool_use"]


@workflow.defn
class CommunisSubAgent:
    """Sub-agent workflow: executes a task using an LLM + tool loop.

    Signals its parent CommunisAgent with progress updates, completion,
    failures, and approval requests.
    """

    def __init__(self):
        self.approval_result: bool | None = None
        self.cancelled = False
        self.status = "pending"
        self.progress = ""

    @workflow.signal
    async def approval_decision(self, approved: bool):
        """Signal from parent session with tool approval decision."""
        self.approval_result = approved

    @workflow.signal
    async def cancel_task(self):
        """Signal from parent session to cancel this task."""
        self.cancelled = True

    @workflow.query
    def get_status(self) -> dict:
        return {"status": self.status, "progress": self.progress}

    @workflow.run
    async def run(self, spec: TaskSpec) -> dict:
        parent = workflow.get_external_workflow_handle(spec.parent_session_id)
        self.status = "running"

        # Signal parent: started
        await parent.signal(
            "task_update",
            TaskUpdate(
                task_id=spec.task_id,
                update_type="progress",
                message="Starting task...",
            ),
        )

        try:
            result = await self._run_agent_loop(spec, parent)
            return result
        except asyncio.CancelledError:
            await parent.signal(
                "task_update",
                TaskUpdate(
                    task_id=spec.task_id,
                    update_type="failed",
                    message="Task was cancelled.",
                ),
            )
            self.status = "cancelled"
            return {"status": "cancelled", "summary": "Task was cancelled."}
        except Exception as e:
            error_msg = str(e)
            await parent.signal(
                "task_update",
                TaskUpdate(
                    task_id=spec.task_id,
                    update_type="failed",
                    message=error_msg,
                ),
            )
            self.status = "failed"
            return {"status": "failed", "summary": error_msg}

    async def _run_agent_loop(self, spec: TaskSpec, parent) -> dict:
        system_prompt = TASK_AGENT_PROMPT.format(
            description=spec.description,
            context=spec.context,
        )
        tools = [RUN_TOOL_DEFINITION]
        messages: list[dict] = [
            {"role": "user", "content": f"Complete this task: {spec.description}"}
        ]

        tool_calls_made = 0
        total_usage = {"input_tokens": 0, "output_tokens": 0}
        final_text_parts: list[str] = []
        iteration = 0

        while iteration < spec.max_tool_iterations:
            if self.cancelled:
                break

            iteration += 1

            llm_response = await workflow.execute_activity(
                call_llm,
                args=[
                    messages,
                    system_prompt,
                    spec.model,
                    0,  # max_tokens (0 = default)
                    spec.provider,
                    spec.base_url,
                    tools,
                ],
                start_to_close_timeout=LLM_TIMEOUT,
                retry_policy=LLM_RETRY_POLICY,
            )

            usage = llm_response.get("usage", {})
            total_usage["input_tokens"] += usage.get("input_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)

            content_blocks = llm_response.get("content_blocks", [])
            text = _extract_text_from_blocks(content_blocks)
            if text:
                final_text_parts.append(text)

            tool_uses = _extract_tool_uses(content_blocks)

            if not tool_uses:
                messages.append({"role": "assistant", "content": content_blocks})
                break

            messages.append({"role": "assistant", "content": content_blocks})

            tool_results: list[dict] = []
            for tool_use in tool_uses:
                if self.cancelled:
                    break

                tool_name = tool_use.get("name", "")
                tool_input = tool_use.get("input", {})
                tool_use_id = tool_use.get("id", "")

                if tool_name != "run":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"[error] unknown tool: {tool_name}. Available: run",
                    })
                    continue

                command = tool_input.get("command", "")
                if not command:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "[error] run: missing required parameter 'command'",
                    })
                    continue

                # Signal parent: tool_call for visibility (thinking = LLM text alongside tool call)
                await parent.signal(
                    "task_update",
                    TaskUpdate(
                        task_id=spec.task_id,
                        update_type="tool_call",
                        message=command,
                        result_summary=text,  # LLM's reasoning text
                    ),
                )

                # Human-in-the-loop approval (unless dangerous mode)
                if not spec.dangerous:
                    approval_id = f"apr-{workflow.uuid4().hex[:8]}"
                    self.approval_result = None
                    self.status = "waiting_approval"
                    self.progress = f"Waiting for approval: {command}"

                    await parent.signal(
                        "task_update",
                        TaskUpdate(
                            task_id=spec.task_id,
                            update_type="approval_request",
                            message=f"Tool call requires approval: {command}",
                            approval_request={
                                "approval_id": approval_id,
                                "tool_name": tool_name,
                                "tool_input": tool_input,
                            },
                        ),
                    )

                    # Wait for approval signal from parent
                    await workflow.wait_condition(
                        lambda: self.approval_result is not None or self.cancelled
                    )

                    if self.cancelled:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "[denied] Task was cancelled.",
                        })
                        continue

                    if not self.approval_result:
                        self.status = "running"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "[denied] Command execution denied by user.",
                        })
                        continue

                    self.status = "running"

                # Execute the command
                exec_result = await workflow.execute_activity(
                    execute_run_command,
                    args=[command, None],
                    start_to_close_timeout=TOOL_TIMEOUT,
                    retry_policy=LLM_RETRY_POLICY,
                )

                tool_calls_made += 1
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": exec_result["output"],
                })

                # Signal parent: tool_result for visibility
                output = exec_result["output"]
                await parent.signal(
                    "task_update",
                    TaskUpdate(
                        task_id=spec.task_id,
                        update_type="tool_result",
                        message=output[:500],
                        result_summary=f"exit:{exec_result['exit_code']}|{exec_result['duration_ms']}ms",
                    ),
                )

                # Signal progress after every few tool calls
                if tool_calls_made % 3 == 0:
                    self.progress = f"Executed {tool_calls_made} commands..."
                    await parent.signal(
                        "task_update",
                        TaskUpdate(
                            task_id=spec.task_id,
                            update_type="progress",
                            message=f"Executed {tool_calls_made} commands so far.",
                        ),
                    )

            messages.append({"role": "user", "content": tool_results})

        # Build result summary — prefer LLM text, fall back to last tool results
        content = "\n\n".join(final_text_parts) if final_text_parts else ""
        if not content:
            # LLM never produced text (just tool calls). Extract the last tool
            # results from the conversation so the user sees what happened.
            for msg in reversed(messages):
                if isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            result_text = block.get("content", "")
                            if result_text and not result_text.startswith("[error]"):
                                content = result_text
                                break
                if content:
                    break
        summary = content[:2000] if content else "Task completed but produced no output."

        self.status = "completed"
        self.progress = "Done"

        # Signal parent: completed
        await parent.signal(
            "task_update",
            TaskUpdate(
                task_id=spec.task_id,
                update_type="completed",
                message="Task completed successfully.",
                result_summary=summary,
            ),
        )

        return {
            "status": "completed",
            "summary": summary,
            "tool_calls_made": tool_calls_made,
            "token_usage": total_usage,
        }
