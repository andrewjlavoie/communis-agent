# CLAUDE.md — AI Agent Development with Claude SDK + Temporal

> This file instructs Claude Code on how to build AI agents using the Anthropic Claude SDK (and Claude Agent SDK) following the agentic architecture patterns from the [temporal-community/temporal-ai-agent](https://github.com/temporal-community/temporal-ai-agent)reference implementation.

---

## Project Overview

We are building **durable, goal-driven AI agents** that use **Temporal workflows** for orchestration and the **Anthropic Claude SDK** for LLM interactions. The architecture follows the temporal-ai-agent reference design: agents pursue goals through iterative tool use, human feedback, and LLM-powered decision-making, all wrapped in Temporal's durable execution guarantees.

### Core Principles

- **Goals, not prompts**: Agents work toward defined goals composed of tool sequences, not open-ended chat
- **Durable execution**: All agent orchestration runs inside Temporal workflows; LLM calls and tool invocations run as Temporal activities
- **Human-in-the-loop**: Support pausing for human input, approval, and confirmation at any point
- **Claude-native**: Use the Anthropic Python SDK (`anthropic`) for all LLM interactions with tool use, or the Claude Agent SDK (`claude-agent-sdk`) for higher-level agent loops
- **MCP + Native tools**: Support both custom native tools and Model Context Protocol (MCP) integrations

---

## Technology Stack

|Layer|Technology|Notes|
|---|---|---|
|**Language**|Python 3.11+|Use `uv` for dependency management|
|**LLM SDK**|`anthropic` (Python SDK)|For direct Messages API + tool use|
|**Agent SDK**|`claude-agent-sdk`|For higher-level agent loops (optional)|
|**Orchestration**|Temporal (Python SDK `temporalio`)|Durable workflow execution|
|**API Layer**|FastAPI / `uvicorn`|REST API for frontend communication|
|**MCP**|Model Context Protocol|External tool integrations|
|**Testing**|`pytest` with Temporal test framework|Including time-skipping support|

### Model Selection

Use the latest Claude models. Current model strings:

- **Primary**: `claude-sonnet-4-5-20250929` — best balance of speed and capability for agent loops
- **Complex reasoning**: `claude-opus-4-6` — for complex tool selection and ambiguous queries
- **Fast/cheap**: `claude-haiku-4-5-20251001` — for input validation, summarization, and guardrail checks

---

## Architecture — Following temporal-ai-agent Patterns

### Directory Structure

```
project-root/
├── CLAUDE.md              # This file
├── pyproject.toml         # Dependencies (use uv)
├── .env                   # Environment config (ANTHROPIC_API_KEY, TEMPORAL_*, etc.)
├── workflows/             # Temporal workflows (agent orchestration logic)
│   ├── __init__.py
│   └── agent_goal_workflow.py
├── activities/            # Temporal activities (LLM calls, tool execution)
│   ├── __init__.py
│   ├── llm_activities.py  # Claude API interactions
│   └── tool_activities.py # Tool execution (native + MCP)
├── tools/                 # Native tool implementations by category
│   ├── __init__.py
│   ├── finance/
│   ├── hr/
│   ├── travel/
│   └── ecommerce/
├── goals/                 # Agent goal definitions by category
│   ├── __init__.py
│   ├── finance/
│   ├── hr/
│   ├── travel/
│   └── ecommerce/
├── models/                # Data types, tool definitions, shared types
│   ├── __init__.py
│   ├── tool_definitions.py
│   └── agent_state.py
├── prompts/               # System prompts and prompt templates
│   └── system_prompts.py
├── shared/                # Shared config (MCP server definitions, etc.)
│   ├── __init__.py
│   └── mcp_config.py
├── api/                   # FastAPI REST API
│   ├── __init__.py
│   └── main.py
├── scripts/               # Worker runner, utility scripts
│   ├── run_worker.py
│   └── run_api.py
├── tests/                 # Comprehensive test suite
│   ├── test_workflows.py
│   ├── test_activities.py
│   └── test_tools.py
└── frontend/              # Optional web UI
```

### The Seven Elements of Agentic AI

Every agent we build must implement these seven elements from the reference architecture:

1. **Goals** — High-level objectives accomplished through sequences of tool calls. Defined in `/goals/` by category. Each goal specifies required tools, system prompt context, and completion criteria.
    
2. **Agent Loop** — The core cycle: call LLM → execute tools → get human input → repeat until goal is complete. This loop lives in the **Temporal workflow**.
    
3. **Tool Approval** — Sensitive tool calls require explicit human confirmation before execution. Controlled via a `SHOW_CONFIRM` flag or per-tool configuration.
    
4. **Input Validation** — Use a lightweight LLM call (Haiku) to check human input for relevance before passing it to the main agent LLM. This prevents prompt injection and keeps the conversation on track.
    
5. **Conversation Summarization** — Use LLM-powered summarization to compact conversation history when it grows too large, preventing context window overflow.
    
6. **Prompt Construction** — Build prompts from three components: system prompt + conversation history + tool metadata. All sent to Claude to generate questions, confirmations, and tool calls.
    
7. **Durability** — Temporal workflows provide crash-proof execution. If the worker dies mid-conversation, it resumes exactly where it left off.
    

---

## Claude SDK Integration Patterns

### Pattern 1: Direct Messages API with Tool Use (Recommended for Temporal)

Use the `anthropic` Python SDK with the Messages API for maximum control inside Temporal activities.

```python
# activities/llm_activities.py
import json
from temporalio import activity
from anthropic import AsyncAnthropic

client = AsyncAnthropic()  # Uses ANTHROPIC_API_KEY env var

@activity.defn
async def call_claude_with_tools(
    messages: list[dict],
    tools: list[dict],
    system_prompt: str,
    model: str = "claude-sonnet-4-5-20250929",
) -> dict:
    """Temporal activity that calls Claude with tool definitions."""
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        tools=tools,
    )
    
    return {
        "content": [block.model_dump() for block in response.content],
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }

@activity.defn
async def validate_user_input(user_input: str, goal_context: str) -> dict:
    """Use a fast model to validate user input relevance."""
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="You are an input validator. Determine if the user input is relevant to the current goal. Respond with JSON: {\"relevant\": true/false, \"reason\": \"...\"}",
        messages=[{
            "role": "user",
            "content": f"Goal: {goal_context}\nUser input: {user_input}\n\nIs this input relevant?",
        }],
    )
    
    text = response.content[0].text
    return json.loads(text)

@activity.defn
async def summarize_conversation(messages: list[dict]) -> str:
    """Summarize conversation history to compact context."""
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="Summarize this conversation history concisely, preserving all key information, decisions made, and data collected. Output only the summary.",
        messages=[{
            "role": "user",
            "content": f"Conversation to summarize:\n{json.dumps(messages, indent=2)}",
        }],
    )
    return response.content[0].text
```

### Pattern 2: Tool Use with Tool Runner (SDK Beta)

The tool runner automates the tool call loop. Useful for activities that need multi-step tool execution:

```python
from anthropic import AsyncAnthropic
from anthropic.types.tool import Tool

client = AsyncAnthropic()

# Define tools using SDK helpers
tools = [
    {
        "name": "search_flights",
        "description": "Search for available flights between two airports on a given date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Origin airport IATA code"},
                "destination": {"type": "string", "description": "Destination airport IATA code"},
                "date": {"type": "string", "description": "Travel date in YYYY-MM-DD format"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
]
```

### Pattern 3: Claude Agent SDK (Higher-Level)

For agents that need filesystem access, code execution, or MCP integration out of the box:

```python
from claude_agent_sdk import query, ClaudeAgentOptions, ClaudeSDKClient

# Simple one-shot query
async for message in query(
    prompt="Analyze the CSV file and generate a report",
    options=ClaudeAgentOptions(
        model="claude-sonnet-4-5-20250929",
        allowed_tools=["Read", "Write", "Bash"],
        system_prompt="You are a data analysis agent.",
        permission_mode="acceptEdits",
        max_turns=10,
    ),
):
    if hasattr(message, "result"):
        print(message.result)

# Interactive client with custom MCP tools
from claude_agent_sdk import create_sdk_mcp_server

async def lookup_employee(employee_id: str) -> str:
    """Look up employee information by ID."""
    # Your implementation here
    return json.dumps({"name": "Jane Doe", "department": "Engineering"})

hr_server = create_sdk_mcp_server(
    name="hr_tools",
    tools=[lookup_employee],
)

client = ClaudeSDKClient(
    options=ClaudeAgentOptions(
        system_prompt="You are an HR assistant agent.",
        mcp_servers={"hr": hr_server},
    )
)
```

---

## Temporal Workflow Pattern — The Agent Loop

The workflow is the heart of the agent. It orchestrates the loop: LLM call → tool execution → human input → repeat.

```python
# workflows/agent_goal_workflow.py
import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.llm_activities import (
        call_claude_with_tools,
        validate_user_input,
        summarize_conversation,
    )
    from activities.tool_activities import execute_tool
    from models.agent_state import AgentState

LLM_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)

@workflow.defn
class AgentGoalWorkflow:
    """Main agent workflow. Runs an agentic loop toward a defined goal."""

    def __init__(self):
        self.state = AgentState()
        self.user_input: str | None = None
        self.user_input_event = asyncio.Event()
        self.is_complete = False

    # --- Signals: receive input from the outside world ---
    @workflow.signal
    async def receive_user_input(self, input_text: str):
        """Signal handler for human input."""
        self.user_input = input_text
        self.user_input_event.set()

    # --- Queries: expose state for the API/UI ---
    @workflow.query
    def get_state(self) -> dict:
        """Return current agent state for the frontend."""
        return self.state.to_dict()

    @workflow.query
    def get_conversation_history(self) -> list[dict]:
        return self.state.conversation_history

    # --- Main workflow logic ---
    @workflow.run
    async def run(self, goal_config: dict) -> dict:
        """Execute the agent loop toward the defined goal."""
        self.state.initialize(goal_config)

        while not self.is_complete:
            # Step 1: Build prompt and call Claude with tools
            llm_response = await workflow.execute_activity(
                call_claude_with_tools,
                args=[
                    self.state.conversation_history,
                    self.state.get_tool_definitions(),
                    self.state.system_prompt,
                ],
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=LLM_RETRY_POLICY,
            )

            # Step 2: Process Claude's response
            for block in llm_response["content"]:
                if block["type"] == "text":
                    # Claude wants to communicate with the user
                    self.state.add_assistant_message(block["text"])

                elif block["type"] == "tool_use":
                    # Claude wants to execute a tool
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_use_id = block["id"]

                    # Optional: require human confirmation for sensitive tools
                    if self.state.requires_confirmation(tool_name):
                        self.state.set_pending_confirmation(tool_name, tool_input)
                        # Wait for human approval signal
                        await workflow.wait_condition(
                            lambda: self.user_input is not None
                        )
                        if self.user_input.lower() != "approve":
                            self.state.add_tool_result(
                                tool_use_id, "Tool execution denied by user."
                            )
                            self.user_input = None
                            continue
                        self.user_input = None

                    # Execute the tool as a Temporal activity
                    tool_result = await workflow.execute_activity(
                        execute_tool,
                        args=[tool_name, tool_input],
                        start_to_close_timeout=timedelta(seconds=60),
                        retry_policy=LLM_RETRY_POLICY,
                    )
                    self.state.add_tool_result(tool_use_id, tool_result)

            # Step 3: Check if the goal is complete
            if llm_response["stop_reason"] == "end_turn":
                if self.state.is_goal_complete():
                    self.is_complete = True
                    continue

                # Need more user input — wait for signal
                self.user_input_event.clear()
                await workflow.wait_condition(
                    lambda: self.user_input is not None
                )

                # Validate user input before adding to conversation
                validation = await workflow.execute_activity(
                    validate_user_input,
                    args=[self.user_input, self.state.goal_description],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=LLM_RETRY_POLICY,
                )
                if validation["relevant"]:
                    self.state.add_user_message(self.user_input)
                else:
                    self.state.add_assistant_message(
                        f"That doesn't seem relevant. {validation['reason']} "
                        "Could you provide information related to the current goal?"
                    )
                self.user_input = None

            # Step 4: Compact conversation if it's getting too long
            if self.state.should_summarize():
                summary = await workflow.execute_activity(
                    summarize_conversation,
                    args=[self.state.conversation_history],
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=LLM_RETRY_POLICY,
                )
                self.state.compact_history(summary)

        return self.state.get_final_result()
```

---

## Defining Goals

Goals define what the agent is trying to accomplish. Each goal specifies its system prompt, available tools, required information, and completion criteria.

```python
# goals/travel/goal_flight_booking.py
GOAL_FLIGHT_BOOKING = {
    "name": "flight_booking",
    "category": "travel",
    "description": "Book a flight for the user",
    "system_prompt": (
        "You are a flight booking agent. Your goal is to help the user book a flight. "
        "You need to collect: origin city, destination city, travel date, "
        "number of passengers, and seating preference. "
        "Use the search_flights tool to find options, then present them to the user. "
        "Once the user selects a flight, use book_flight to complete the reservation. "
        "Always confirm the booking details before finalizing."
    ),
    "tools": ["search_flights", "book_flight", "get_airport_code"],
    "mcp_tools": [],  # Or reference MCP server tools
    "required_info": [
        "origin", "destination", "date", "passengers", "seat_preference"
    ],
    "show_confirm": True,  # Require confirmation before booking
    "completion_check": "booking_confirmation_received",
}
```

---

## Defining Tools

### Native Tools

Native tools are implemented directly in the codebase. Each tool has a definition (JSON schema for Claude) and an implementation function.

```python
# tools/travel/flight_tools.py
SEARCH_FLIGHTS_TOOL = {
    "name": "search_flights",
    "description": (
        "Search for available flights between two cities on a given date. "
        "Returns a list of flight options with prices, times, and airlines. "
        "Use this when the user wants to find flight options. "
        "Requires origin and destination airport codes (IATA format) and a date."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "Origin airport IATA code (e.g., 'SFO', 'LAX')",
            },
            "destination": {
                "type": "string",
                "description": "Destination airport IATA code (e.g., 'JFK', 'ORD')",
            },
            "date": {
                "type": "string",
                "description": "Travel date in YYYY-MM-DD format",
            },
            "passengers": {
                "type": "integer",
                "description": "Number of passengers (default: 1)",
                "default": 1,
            },
        },
        "required": ["origin", "destination", "date"],
    },
}


async def search_flights(origin: str, destination: str, date: str, passengers: int = 1) -> str:
    """Implementation of the search_flights tool."""
    # Your actual flight search API integration here
    import json
    results = [
        {
            "flight": "UA-1234",
            "airline": "United",
            "departure": f"{date}T08:00:00",
            "arrival": f"{date}T11:30:00",
            "price_per_person": 299.99,
            "total": 299.99 * passengers,
        },
    ]
    return json.dumps(results)
```

### Tool Execution Activity

```python
# activities/tool_activities.py
from temporalio import activity
from tools.travel.flight_tools import search_flights
# Import other tools...

TOOL_REGISTRY = {
    "search_flights": search_flights,
    # Register all native tools here
}

@activity.defn
async def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a native tool by name with the given input."""
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{tool_name}'"
    
    try:
        result = await TOOL_REGISTRY[tool_name](**tool_input)
        return result
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
```

### MCP Tool Integration

```python
# shared/mcp_config.py
MCP_SERVERS = {
    "stripe": {
        "command": "npx",
        "args": ["-y", "@stripe/mcp", "--tools=all"],
        "env": {"STRIPE_SECRET_KEY": "sk-..."},
    },
    "postgres": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"],
        "env": {"DATABASE_URL": "postgresql://..."},
    },
}
```

---

## Agent State Management

```python
# models/agent_state.py
from dataclasses import dataclass, field
import json

MAX_HISTORY_TOKENS_ESTIMATE = 50_000  # Trigger summarization threshold

@dataclass
class AgentState:
    """Tracks the full state of an agent conversation."""
    
    goal_config: dict = field(default_factory=dict)
    goal_description: str = ""
    system_prompt: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    collected_info: dict = field(default_factory=dict)
    tools_executed: list[dict] = field(default_factory=list)
    pending_confirmation: dict | None = None
    is_done: bool = False
    
    def initialize(self, goal_config: dict):
        self.goal_config = goal_config
        self.goal_description = goal_config["description"]
        self.system_prompt = goal_config["system_prompt"]

    def add_user_message(self, text: str):
        self.conversation_history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str):
        self.conversation_history.append({"role": "assistant", "content": text})

    def add_tool_result(self, tool_use_id: str, result: str):
        self.conversation_history.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
            ],
        })

    def get_tool_definitions(self) -> list[dict]:
        """Return Claude-formatted tool definitions for this goal's tools."""
        from tools import get_tools_for_names
        return get_tools_for_names(self.goal_config.get("tools", []))

    def requires_confirmation(self, tool_name: str) -> bool:
        return self.goal_config.get("show_confirm", False)

    def set_pending_confirmation(self, tool_name: str, tool_input: dict):
        self.pending_confirmation = {"tool": tool_name, "input": tool_input}

    def is_goal_complete(self) -> bool:
        """Check if all required information has been collected."""
        required = self.goal_config.get("required_info", [])
        return all(key in self.collected_info for key in required)

    def should_summarize(self) -> bool:
        """Estimate if conversation is getting too long."""
        text_length = sum(
            len(json.dumps(msg)) for msg in self.conversation_history
        )
        return text_length > MAX_HISTORY_TOKENS_ESTIMATE * 4  # Rough char estimate

    def compact_history(self, summary: str):
        """Replace conversation history with a summary + recent messages."""
        recent = self.conversation_history[-4:]  # Keep last 2 exchanges
        self.conversation_history = [
            {"role": "user", "content": f"[Previous conversation summary]: {summary}"},
            {"role": "assistant", "content": "I understand. Let me continue helping you with your goal."},
            *recent,
        ]

    def get_final_result(self) -> dict:
        return {
            "goal": self.goal_config["name"],
            "collected_info": self.collected_info,
            "tools_executed": self.tools_executed,
            "conversation_length": len(self.conversation_history),
        }

    def to_dict(self) -> dict:
        return {
            "goal": self.goal_description,
            "is_done": self.is_done,
            "collected_info": self.collected_info,
            "pending_confirmation": self.pending_confirmation,
            "message_count": len(self.conversation_history),
        }
```

---

## Worker and API Setup

### Temporal Worker

```python
# scripts/run_worker.py
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflows.agent_goal_workflow import AgentGoalWorkflow
from activities.llm_activities import (
    call_claude_with_tools,
    validate_user_input,
    summarize_conversation,
)
from activities.tool_activities import execute_tool

TASK_QUEUE = "ai-agent-task-queue"

async def main():
    client = await Client.connect("localhost:7233")  # Or Temporal Cloud address

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentGoalWorkflow],
        activities=[
            call_claude_with_tools,
            validate_user_input,
            summarize_conversation,
            execute_tool,
        ],
    )
    print(f"Worker started on task queue: {TASK_QUEUE}")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### FastAPI Server

```python
# api/main.py
from fastapi import FastAPI
from temporalio.client import Client

app = FastAPI()
temporal_client: Client = None

@app.on_event("startup")
async def startup():
    global temporal_client
    temporal_client = await Client.connect("localhost:7233")

@app.post("/agent/start")
async def start_agent(goal_name: str, workflow_id: str):
    """Start a new agent workflow for a goal."""
    from goals import get_goal_config
    goal_config = get_goal_config(goal_name)
    
    handle = await temporal_client.start_workflow(
        "AgentGoalWorkflow",
        goal_config,
        id=workflow_id,
        task_queue="ai-agent-task-queue",
    )
    return {"workflow_id": handle.id, "status": "started"}

@app.post("/agent/{workflow_id}/message")
async def send_message(workflow_id: str, message: str):
    """Send a user message to a running agent workflow."""
    handle = temporal_client.get_workflow_handle(workflow_id)
    await handle.signal("receive_user_input", message)
    return {"status": "sent"}

@app.get("/agent/{workflow_id}/state")
async def get_state(workflow_id: str):
    """Query the current agent state."""
    handle = temporal_client.get_workflow_handle(workflow_id)
    state = await handle.query("get_state")
    return state
```

---

## Testing

Follow the temporal-ai-agent testing patterns:

```python
# tests/test_workflows.py
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from workflows.agent_goal_workflow import AgentGoalWorkflow
from activities.llm_activities import call_claude_with_tools, validate_user_input, summarize_conversation
from activities.tool_activities import execute_tool

@pytest.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env

@pytest.mark.asyncio
async def test_agent_workflow_starts(env):
    """Test that the agent workflow starts and accepts signals."""
    goal_config = {
        "name": "test_goal",
        "description": "A test goal",
        "system_prompt": "You are a test agent.",
        "tools": [],
        "required_info": ["name"],
        "show_confirm": False,
    }
    
    async with Worker(
        env.client,
        task_queue="test-queue",
        workflows=[AgentGoalWorkflow],
        activities=[call_claude_with_tools, validate_user_input, summarize_conversation, execute_tool],
    ):
        handle = await env.client.start_workflow(
            AgentGoalWorkflow.run,
            goal_config,
            id="test-workflow",
            task_queue="test-queue",
        )
        state = await handle.query(AgentGoalWorkflow.get_state)
        assert state["goal"] == "A test goal"

# Run: uv run pytest tests/ --workflow-environment=time-skipping
```

---

## Environment Configuration

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...

# Temporal (local dev)
TEMPORAL_ADDRESS=localhost:7233
TEMPORAL_NAMESPACE=default

# Temporal Cloud (production)
# TEMPORAL_ADDRESS=your-namespace.tmprl.cloud:7233
# TEMPORAL_NAMESPACE=your-namespace
# TEMPORAL_API_KEY=your-api-key

# Agent Configuration
AGENT_GOAL=goal_flight_booking     # Default goal
SHOW_CONFIRM=True                  # Require tool confirmation
MAX_CONVERSATION_TURNS=50          # Safety limit

# MCP (optional)
STRIPE_SECRET_KEY=sk-...
DATABASE_URL=postgresql://...
```

---

## Key Development Commands

```bash
# Setup
uv sync                              # Install dependencies

# Run services
temporal server start-dev             # Start Temporal dev server
uv run scripts/run_worker.py          # Start Temporal worker
uv run uvicorn api.main:app --reload  # Start API server

# Testing
uv run pytest                                           # Run all tests
uv run pytest --workflow-environment=time-skipping       # With time-skipping
uv run pytest tests/test_activities.py -v                # Specific test file

# Docker (alternative)
docker compose up -d                                    # Start all services
docker compose up -d --no-deps --build api worker       # Rebuild without infra
```

---

## Critical Rules for Claude Code

### Architecture Rules

- **ALL LLM calls MUST be Temporal activities** — never call Claude directly from workflow code. Workflows must be deterministic; LLM calls are non-deterministic and must run in activities.
- **ALL external I/O MUST be Temporal activities** — API calls, database queries, file operations, MCP tool calls — anything with side effects.
- **Workflow code is the orchestrator** — it contains the agent loop logic (if/else, loops, wait conditions) but delegates all real work to activities.
- **Use signals for human input** — the workflow waits for user messages via Temporal signals, not polling.
- **Use queries for state inspection** — the API reads agent state via Temporal queries, which are synchronous and don't change state.

### Claude SDK Rules

- **Always use `AsyncAnthropic`** in activities since Temporal activities are async.
- **Always handle `stop_reason`** — check for `"tool_use"` (Claude wants to call a tool), `"end_turn"` (Claude is done talking), and `"max_tokens"` (response was truncated).
- **Tool results MUST reference the `tool_use_id`** from Claude's response. Every `tool_use` block has a unique `id`that must be included in the corresponding `tool_result`.
- **Build tool definitions with detailed descriptions** — at least 3-4 sentences per tool. Include what the tool does, when to use it, what each parameter means, and any limitations.
- **Use `tool_choice: "auto"`** (default) to let Claude decide when to use tools. Use `tool_choice: {"type": "tool", "name": "..."}` only when you need to force a specific tool.
- **Handle parallel tool use** — Claude may return multiple `tool_use` blocks in a single response. Execute them all and return all results.

### Production Considerations

- **Payload size**: For long conversations, implement the claim-check pattern — store large payloads in S3 or a database, pass only references through Temporal.
- **Workflow IDs**: Use unique IDs (UUID or timestamp-based) for each agent conversation to support multiple concurrent agents.
- **Retry policy**: Configure appropriate retry policies for LLM activities — LLMs can return bad output, so retries with backoff are essential.
- **Context window management**: Monitor token usage. Summarize conversation history when it approaches the model's context limit. Claude Sonnet 4.5 supports 200K tokens input.
- **Cost control**: Use Haiku for validation and summarization tasks. Reserve Sonnet/Opus for the main agent reasoning.

---

## Reference Links

- **Reference Architecture**: https://github.com/temporal-community/temporal-ai-agent
- **Anthropic Python SDK**: https://github.com/anthropics/anthropic-sdk-python
- **Claude Agent SDK**: https://github.com/anthropics/claude-agent-sdk-python
- **Claude Tool Use Docs**: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- **Implementing Tool Use**: https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use
- **Agent SDK Overview**: https://platform.claude.com/docs/en/agent-sdk/overview
- **Temporal Python SDK**: https://docs.temporal.io/develop/python
- **MCP Specification**: https://modelcontextprotocol.io
- **Building Agents Blog**: https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk
