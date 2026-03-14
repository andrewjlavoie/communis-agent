"""Example: REST API wrapper around communis.

Exposes communis as an HTTP API so any client (frontend, mobile, webhook,
curl) can start sessions, check status, send feedback, and read results.

Usage:
    # Make sure the worker is running:
    #   uv run python scripts/run_worker.py

    pip install fastapi uvicorn
    uvicorn examples.api_server:app --reload --port 8000

    # Start a session:
    curl -X POST http://localhost:8000/communis \
      -H "Content-Type: application/json" \
      -d '{"idea": "design an onboarding flow", "max_turns": 3}'

    # Check status:
    curl http://localhost:8000/communis/<workflow_id>

    # Send feedback:
    curl -X POST http://localhost:8000/communis/<workflow_id>/feedback \
      -H "Content-Type: application/json" \
      -d '{"feedback": "focus on time-to-first-value"}'
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from temporalio.client import Client

from models.data_types import CommunisConfig
from shared.constants import TASK_QUEUE
from workflows.communis_orchestrator import CommunisOrchestratorWorkflow

app = FastAPI(title="communis API", description="Self-directing iterative work loop as a service")
temporal: Client | None = None


class StartRequest(BaseModel):
    idea: str
    max_turns: int = 3
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
    """Start a new communis session. Returns a workflow_id to track it."""
    workflow_id = f"communis-{uuid.uuid4().hex[:8]}"
    handle = await temporal.start_workflow(
        CommunisOrchestratorWorkflow.run,
        CommunisConfig(
            idea=req.idea,
            max_turns=req.max_turns,
            model=req.model,
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
    """Get current state of a running communis session."""
    try:
        handle = temporal.get_workflow_handle(workflow_id)
        return await handle.query(CommunisOrchestratorWorkflow.get_state)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/communis/{workflow_id}/result")
async def get_result(workflow_id: str):
    """Get final result. Blocks until the session completes."""
    handle = temporal.get_workflow_handle(workflow_id)
    return await handle.result()


@app.get("/communis/{workflow_id}/turns")
async def get_turns(workflow_id: str):
    """Get all completed turn results."""
    try:
        handle = temporal.get_workflow_handle(workflow_id)
        return await handle.query(CommunisOrchestratorWorkflow.get_all_results)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/communis/{workflow_id}/turns/{turn_number}")
async def get_turn(workflow_id: str, turn_number: int):
    """Get a specific turn result."""
    try:
        handle = temporal.get_workflow_handle(workflow_id)
        result = await handle.query(CommunisOrchestratorWorkflow.get_turn_result, turn_number)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Turn {turn_number} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/communis/{workflow_id}/feedback")
async def send_feedback(workflow_id: str, req: FeedbackRequest):
    """Send feedback to a paused session (when auto=False)."""
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
    """Cancel a running session. Preserves completed turns."""
    handle = temporal.get_workflow_handle(workflow_id)
    await handle.cancel()
    return {"status": "cancelled"}
