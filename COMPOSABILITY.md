# Composability — Using communis as a Building Block

communis is a standard Temporal workflow. Any system that can start a Temporal workflow can use it — Python, Go, TypeScript, Java, or the Temporal CLI. This document shows how.

## The Interface

**Input:** `CommunisConfig` dataclass

```python
@dataclass
class CommunisConfig:
    idea: str              # The prompt / task
    num_turns: int = 3     # How many iterative turns (1-10)
    model: str = "..."     # LLM model name
    auto: bool = False     # True = skip feedback pauses, run straight through
    provider: str = ""     # "anthropic" or "openai" (empty = env default)
    base_url: str = ""     # OpenAI-compatible base URL (empty = env default)
```

**Output:** `dict` with this shape

```python
{
    "idea": "the original prompt",
    "num_turns": 3,
    "current_turn": 3,
    "current_role": "Synthesizer",
    "status": "complete",           # "complete" | "cancelled" | "error"
    "workspace_dir": ".communis/communis-a1b2c3d4/",
    "latest_message": "All turns complete!",
    "turn_results": [
        {
            "turn_number": 1,
            "role": "Explorer",
            "key_insights": ["insight 1", "insight 2"],
            "token_usage": {"input_tokens": 500, "output_tokens": 1200},
            "truncated": false,
            "artifact_path": ".communis/communis-a1b2c3d4/turn-01-explorer.md"
        },
        # ... one per turn
    ]
}
```

**Signals** (for human-in-the-loop, when `auto=False`):
- `receive_user_feedback(feedback: str)` — inject steering feedback between turns
- `skip_feedback()` — skip the feedback pause and continue

**Queries** (read state without affecting the workflow):
- `get_state() -> dict` — full state snapshot
- `get_turn_result(turn_number: int) -> dict | None` — single turn result
- `get_all_results() -> list[dict]` — all completed turn results

**Task Queue:** `communis-task-queue`

---

## Pattern 1: Child Workflow (Python)

Call communis from another Temporal workflow. The parent orchestrates multiple communis sessions, chains results, or makes decisions based on outputs.

```python
# workflows/research_pipeline.py
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from models.data_types import CommunisConfig
    from workflows.communis_orchestrator import CommunisOrchestratorWorkflow


@workflow.defn
class ResearchPipelineWorkflow:
    """Run multiple communis sessions to research a topic from different angles."""

    @workflow.run
    async def run(self, topic: str) -> dict:
        # Phase 1: Broad exploration
        explore_result = await workflow.execute_child_workflow(
            CommunisOrchestratorWorkflow.run,
            CommunisConfig(
                idea=f"Research and explore: {topic}",
                num_turns=3,
                auto=True,
            ),
            id=f"{workflow.info().workflow_id}-explore",
        )

        # Extract insights from phase 1
        insights = []
        for turn in explore_result["turn_results"]:
            insights.extend(turn.get("key_insights", []))

        # Phase 2: Deep dive on the most interesting findings
        deep_dive_result = await workflow.execute_child_workflow(
            CommunisOrchestratorWorkflow.run,
            CommunisConfig(
                idea=(
                    f"Deep dive on: {topic}\n\n"
                    f"Prior research found these key insights:\n"
                    + "\n".join(f"- {i}" for i in insights)
                    + "\n\nGo deeper on the most promising findings. "
                    "Challenge assumptions. Find what was missed."
                ),
                num_turns=4,
                auto=True,
            ),
            id=f"{workflow.info().workflow_id}-deep-dive",
        )

        # Phase 3: Synthesize into a final deliverable
        final_result = await workflow.execute_child_workflow(
            CommunisOrchestratorWorkflow.run,
            CommunisConfig(
                idea=(
                    f"Create a comprehensive report on: {topic}\n\n"
                    f"Exploration insights:\n"
                    + "\n".join(f"- {i}" for i in insights)
                    + f"\n\nDeep dive insights:\n"
                    + "\n".join(
                        f"- {i}"
                        for turn in deep_dive_result["turn_results"]
                        for i in turn.get("key_insights", [])
                    )
                    + "\n\nSynthesize everything into a polished report."
                ),
                num_turns=3,
                auto=True,
            ),
            id=f"{workflow.info().workflow_id}-report",
        )

        return {
            "topic": topic,
            "phases": [
                {"name": "exploration", **explore_result},
                {"name": "deep_dive", **deep_dive_result},
                {"name": "final_report", **final_result},
            ],
            "total_turns": sum(
                len(r["turn_results"])
                for r in [explore_result, deep_dive_result, final_result]
            ),
        }
```

**Register the parent workflow with the same worker** (or a separate one on the same task queue):

```python
# scripts/run_pipeline_worker.py
from workflows.research_pipeline import ResearchPipelineWorkflow
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow
from workflows.communis_turn import CommunisTurnWorkflow

# ... (import all activities)

worker = Worker(
    client,
    task_queue="communis-task-queue",
    workflows=[ResearchPipelineWorkflow, CommunisOrchestratorWorkflow, CommunisTurnWorkflow],
    activities=[...],  # all 11 activities
)
```

---

## Pattern 2: Parallel Fan-Out

Run multiple communis sessions concurrently and combine results. Useful when you need different perspectives on the same problem.

```python
@workflow.defn
class ParallelAnalysisWorkflow:
    """Analyze a problem from multiple angles simultaneously."""

    @workflow.run
    async def run(self, problem: str) -> dict:
        import asyncio

        # Launch three communis sessions in parallel
        tasks = [
            workflow.execute_child_workflow(
                CommunisOrchestratorWorkflow.run,
                CommunisConfig(
                    idea=f"Analyze from a TECHNICAL perspective: {problem}",
                    num_turns=3,
                    auto=True,
                ),
                id=f"{workflow.info().workflow_id}-technical",
            ),
            workflow.execute_child_workflow(
                CommunisOrchestratorWorkflow.run,
                CommunisConfig(
                    idea=f"Analyze from a BUSINESS perspective: {problem}",
                    num_turns=3,
                    auto=True,
                ),
                id=f"{workflow.info().workflow_id}-business",
            ),
            workflow.execute_child_workflow(
                CommunisOrchestratorWorkflow.run,
                CommunisConfig(
                    idea=f"Analyze from a USER EXPERIENCE perspective: {problem}",
                    num_turns=3,
                    auto=True,
                ),
                id=f"{workflow.info().workflow_id}-ux",
            ),
        ]

        technical, business, ux = await asyncio.gather(*tasks)

        # Synthesize all perspectives
        all_insights = []
        for label, result in [("Technical", technical), ("Business", business), ("UX", ux)]:
            for turn in result["turn_results"]:
                for insight in turn.get("key_insights", []):
                    all_insights.append(f"[{label}] {insight}")

        synthesis = await workflow.execute_child_workflow(
            CommunisOrchestratorWorkflow.run,
            CommunisConfig(
                idea=(
                    f"Synthesize these multi-perspective findings on: {problem}\n\n"
                    + "\n".join(f"- {i}" for i in all_insights)
                    + "\n\nCreate a unified recommendation that balances all perspectives."
                ),
                num_turns=2,
                auto=True,
            ),
            id=f"{workflow.info().workflow_id}-synthesis",
        )

        return {
            "problem": problem,
            "perspectives": {
                "technical": technical,
                "business": business,
                "ux": ux,
            },
            "synthesis": synthesis,
        }
```

---

## Pattern 3: External Client (Python SDK)

Start and monitor a communis session from any Python service — a web app, a cron job, a Slack bot, etc. No need to be a Temporal workflow.

```python
# examples/external_client.py
"""Start an communis session from outside Temporal."""
import asyncio
from temporalio.client import Client
from models.data_types import CommunisConfig
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow


async def main():
    client = await Client.connect("localhost:7233")

    # Start the workflow
    handle = await client.start_workflow(
        CommunisOrchestratorWorkflow.run,
        CommunisConfig(
            idea="Design a notification system for a SaaS platform",
            num_turns=4,
            auto=True,
        ),
        id="communis-notifications-design",
        task_queue="communis-task-queue",
    )

    print(f"Started workflow: {handle.id}")

    # Poll for progress
    while True:
        state = await handle.query(CommunisOrchestratorWorkflow.get_state)
        print(f"  Turn {state['current_turn']}/{state['num_turns']} — {state['current_role']} — {state['status']}")

        if state["status"] in ("complete", "cancelled", "error"):
            break
        await asyncio.sleep(5)

    # Get final result
    result = await handle.result()
    print(f"\nDone! Status: {result['status']}")
    print(f"Workspace: {result['workspace_dir']}")
    for turn in result["turn_results"]:
        print(f"  Turn {turn['turn_number']}: {turn['role']} — {turn['key_insights']}")


asyncio.run(main())
```

---

## Pattern 4: External Client with Feedback

Same as above, but inject human feedback between turns instead of running in auto mode.

```python
# examples/external_client_feedback.py
"""Start an communis session and provide feedback between turns."""
import asyncio
from temporalio.client import Client
from models.data_types import CommunisConfig
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow


async def main():
    client = await Client.connect("localhost:7233")

    handle = await client.start_workflow(
        CommunisOrchestratorWorkflow.run,
        CommunisConfig(
            idea="Plan a company offsite for 50 people",
            num_turns=4,
            auto=False,  # Will pause for feedback between turns
        ),
        id="communis-offsite-planning",
        task_queue="communis-task-queue",
    )

    while True:
        state = await handle.query(CommunisOrchestratorWorkflow.get_state)

        if state["status"] == "waiting_for_feedback":
            # Show the latest turn results
            for turn in state["turn_results"]:
                print(f"\nTurn {turn['turn_number']}: {turn['role']}")
                for insight in turn["key_insights"]:
                    print(f"  - {insight}")

            # Get feedback (from a user, a Slack message, an API call, etc.)
            feedback = input("\nFeedback (Enter to skip): ").strip()
            if feedback:
                await handle.signal(CommunisOrchestratorWorkflow.receive_user_feedback, feedback)
            else:
                await handle.signal(CommunisOrchestratorWorkflow.skip_feedback)

        elif state["status"] in ("complete", "cancelled", "error"):
            break

        await asyncio.sleep(2)

    result = await handle.result()
    print(f"\nDone! Workspace: {result['workspace_dir']}")


asyncio.run(main())
```

---

## Pattern 5: From a Go Service

Any language with a Temporal SDK can start communis. The workflow ID, task queue, and config shape are all you need.

```go
// main.go
package main

import (
    "context"
    "log"

    "go.temporal.io/sdk/client"
)

// CommunisConfig mirrors the Python dataclass — Temporal serializes as JSON
type CommunisConfig struct {
    Idea     string `json:"idea"`
    NumTurns int    `json:"num_turns"`
    Model    string `json:"model"`
    Auto     bool   `json:"auto"`
    Provider string `json:"provider"`
    BaseURL  string `json:"base_url"`
}

func main() {
    c, err := client.Dial(client.Options{HostPort: "localhost:7233"})
    if err != nil {
        log.Fatal(err)
    }
    defer c.Close()

    run, err := c.ExecuteWorkflow(
        context.Background(),
        client.StartWorkflowOptions{
            ID:        "communis-from-go",
            TaskQueue: "communis-task-queue",
        },
        "CommunisOrchestratorWorkflow",  // workflow type name
        CommunisConfig{
            Idea:     "Design a caching strategy for a read-heavy API",
            NumTurns: 3,
            Auto:     true,
        },
    )
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Started workflow %s", run.GetID())

    var result map[string]interface{}
    if err := run.Get(context.Background(), &result); err != nil {
        log.Fatal(err)
    }

    log.Printf("Status: %s", result["status"])
}
```

---

## Pattern 6: From TypeScript

```typescript
// start-communis.ts
import { Client, Connection } from "@temporalio/client";

interface CommunisConfig {
  idea: string;
  num_turns: number;
  model?: string;
  auto?: boolean;
  provider?: string;
  base_url?: string;
}

async function main() {
  const connection = await Connection.connect({ address: "localhost:7233" });
  const client = new Client({ connection });

  const handle = await client.workflow.start("CommunisOrchestratorWorkflow", {
    args: [
      {
        idea: "Design a real-time collaboration feature for a document editor",
        num_turns: 4,
        auto: true,
      } satisfies CommunisConfig,
    ],
    taskQueue: "communis-task-queue",
    workflowId: "communis-from-typescript",
  });

  console.log(`Started: ${handle.workflowId}`);

  const result = await handle.result();
  console.log(`Status: ${result.status}`);
  console.log(`Turns: ${result.turn_results.length}`);

  for (const turn of result.turn_results) {
    console.log(`  ${turn.turn_number}. ${turn.role}: ${turn.key_insights.join(", ")}`);
  }
}

main().catch(console.error);
```

---

## Pattern 7: Temporal CLI (No Code)

Start a communis session directly from the command line. Useful for testing or one-off runs.

```bash
# Start a workflow
temporal workflow start \
  --type CommunisOrchestratorWorkflow \
  --task-queue communis-task-queue \
  --workflow-id "communis-cli-test" \
  --input '{"idea": "Compare microservices vs monolith for a startup MVP", "num_turns": 3, "auto": true}'

# Check status
temporal workflow query \
  --workflow-id "communis-cli-test" \
  --type get_state

# Send feedback (if not auto)
temporal workflow signal \
  --workflow-id "communis-cli-test" \
  --name receive_user_feedback \
  --input '"Focus on cost and speed to market"'

# Skip feedback
temporal workflow signal \
  --workflow-id "communis-cli-test" \
  --name skip_feedback

# Cancel
temporal workflow cancel --workflow-id "communis-cli-test"

# Get final result
temporal workflow show --workflow-id "communis-cli-test"
```

---

## Pattern 8: FastAPI Wrapper

Expose communis as a REST API so any HTTP client (frontend, mobile app, webhook) can use it.

```python
# examples/api_server.py
"""REST API wrapper around communis."""
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from temporalio.client import Client
from models.data_types import CommunisConfig
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow

app = FastAPI(title="communis API")
temporal: Client = None

TASK_QUEUE = "communis-task-queue"


class StartRequest(BaseModel):
    idea: str
    num_turns: int = 3
    model: str = ""
    auto: bool = True
    provider: str = ""
    base_url: str = ""


class FeedbackRequest(BaseModel):
    feedback: str


@app.on_event("startup")
async def startup():
    global temporal
    temporal = await Client.connect("localhost:7233")


@app.post("/communis")
async def start_communis(req: StartRequest):
    """Start a new communis session."""
    workflow_id = f"communis-{uuid.uuid4().hex[:8]}"
    handle = await temporal.start_workflow(
        CommunisOrchestratorWorkflow.run,
        CommunisConfig(
            idea=req.idea,
            num_turns=req.num_turns,
            model=req.model or "claude-sonnet-4-5-20250929",
            auto=req.auto,
            provider=req.provider,
            base_url=req.base_url,
        ),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    return {"workflow_id": handle.id}


@app.get("/communis/{workflow_id}")
async def get_status(workflow_id: str):
    """Get current state of a communis session."""
    handle = temporal.get_workflow_handle(workflow_id)
    return await handle.query(CommunisOrchestratorWorkflow.get_state)


@app.get("/communis/{workflow_id}/result")
async def get_result(workflow_id: str):
    """Get final result (blocks until complete)."""
    handle = temporal.get_workflow_handle(workflow_id)
    return await handle.result()


@app.post("/communis/{workflow_id}/feedback")
async def send_feedback(workflow_id: str, req: FeedbackRequest):
    """Send feedback to a paused communis session."""
    handle = temporal.get_workflow_handle(workflow_id)
    await handle.signal(CommunisOrchestratorWorkflow.receive_user_feedback, req.feedback)
    return {"status": "sent"}


@app.post("/communis/{workflow_id}/skip")
async def skip_feedback(workflow_id: str):
    """Skip the feedback pause and continue."""
    handle = temporal.get_workflow_handle(workflow_id)
    await handle.signal(CommunisOrchestratorWorkflow.skip_feedback)
    return {"status": "skipped"}


@app.post("/communis/{workflow_id}/cancel")
async def cancel_communis(workflow_id: str):
    """Cancel a running communis session."""
    handle = temporal.get_workflow_handle(workflow_id)
    await handle.cancel()
    return {"status": "cancelled"}
```

Run with:

```bash
pip install fastapi uvicorn
uvicorn examples.api_server:app --reload --port 8000
```

Then:

```bash
# Start a session
curl -X POST http://localhost:8000/communis \
  -H "Content-Type: application/json" \
  -d '{"idea": "Design an onboarding flow for a dev tools product", "num_turns": 3}'

# Check status
curl http://localhost:8000/communis/communis-a1b2c3d4

# Send feedback
curl -X POST http://localhost:8000/communis/communis-a1b2c3d4/feedback \
  -H "Content-Type: application/json" \
  -d '{"feedback": "Focus on time-to-first-value"}'
```

---

## Reading Workspace Artifacts

The workflow result contains `artifact_path` for each turn — these are markdown files on disk. To read the actual content programmatically:

```python
from pathlib import Path

result = await handle.result()

for turn in result["turn_results"]:
    path = Path(turn["artifact_path"])
    if path.exists():
        text = path.read_text()

        # Skip YAML frontmatter
        if text.startswith("---"):
            end = text.find("\n---\n", 3)
            if end != -1:
                text = text[end + 5:]

        print(f"Turn {turn['turn_number']} ({turn['role']}):")
        print(text[:500])
```

For distributed systems where the client isn't on the same machine as the worker, you'd need shared storage (S3, NFS, etc.) or add an activity that reads and returns the file content.

---

## Key Constraints

- **Task queue**: The communis worker must be running on `communis-task-queue` (or whatever you configure). Your parent workflow can be on any task queue — child workflows inherit the parent's task queue by default, so override it if needed.
- **Workspace files**: Turn artifacts are written to the local filesystem of the worker. For multi-machine setups, use shared storage or the claim-check pattern with S3.
- **Serialization**: `CommunisConfig` is a Python dataclass. Temporal serializes it as JSON. Cross-language clients just need to send the right JSON shape.
- **auto=True for child workflows**: Unless your parent workflow is going to signal feedback, always set `auto=True`. Otherwise the child will pause for 120s waiting for feedback that never comes.
- **Cancellation propagates**: If you cancel the parent workflow, Temporal cancels child workflows too. communis handles this gracefully and returns partial results.
