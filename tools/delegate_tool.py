"""delegate_task tool — spawns a durable sub-agent (TaskWorkflow) for complex work."""

DELEGATE_TASK_TOOL = {
    "name": "delegate_task",
    "description": (
        "Spawn a background sub-agent to handle a complex, multi-step task autonomously. "
        "The sub-agent runs as a durable Temporal workflow with its own tool access. "
        "Use this for work that requires many commands, long-running operations, "
        "or tasks that benefit from autonomous multi-step execution. "
        "For simple one-off commands (reading a file, quick lookup), use the run tool directly instead.\n\n"
        "The task runs in the background — you can continue the conversation while it executes. "
        "You'll receive updates when the task completes or needs approval."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Clear description of what the sub-agent should accomplish.",
            },
            "context": {
                "type": "string",
                "description": "Relevant context from the conversation to help the sub-agent understand the task.",
            },
        },
        "required": ["description"],
    },
}
