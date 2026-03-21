"""FastAPI server providing routine-centric REST API for OCSD.

Exposes endpoints for health checking, routine discovery, routine
detail inspection, and (stubbed) run management. Designed to run
as a daemon thread alongside the Qt event loop.

Usage:
    uvicorn api.server:app --port 8420 --reload
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for the API server.  "
        "Install with:  pip install 'ocsd[api]'  or  pip install fastapi uvicorn"
    )

from api.run_manager import (
    RunAlreadyActiveError,
    RunManager,
    RunNotFoundError,
    RunStatus,
)
from core.config import get_config
from routine.discovery import get_routine_dir, list_routines
from routine.format import Routine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    models_loaded: bool


class RoutineSummary(BaseModel):
    """Lightweight routine metadata for list endpoints."""

    id: str
    name: str
    version: str
    step_count: int
    tags: list[str]


class RoutineDetail(BaseModel):
    """Full routine metadata and steps."""

    id: str
    name: str
    version: str
    description: str
    tags: list[str]
    steps: list[dict[str, Any]]
    metadata: dict[str, Any]


class RunRequest(BaseModel):
    """Request body for starting a routine run."""

    params: dict[str, str] | None = None
    show_overlay: bool = True
    prompt_timeout_s: int | None = None


class RunResponse(BaseModel):
    """Response after starting a run."""

    run_id: str


class RunStatusResponse(BaseModel):
    """Status of an active or completed run."""

    run_id: str
    status: str
    current_step: int
    total_steps: int
    error: str | None = None
    prompt_text: str | None = None


class RespondRequest(BaseModel):
    """Request body for responding to a prompt."""

    response: str


class ErrorResponse(BaseModel):
    """Structured error response."""

    code: str
    message: str
    suggestion: str | None = None


# ---------------------------------------------------------------------------
# App and singletons
# ---------------------------------------------------------------------------

manager = RunManager()

app = FastAPI(
    title="OCSD API",
    version="1.0.0",
    description="OpenClaw Screen Driver -- routine execution API",
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RunAlreadyActiveError)
async def _handle_run_active(
    request: Any, exc: RunAlreadyActiveError
) -> JSONResponse:
    """Handle attempt to start a run when one is already active."""
    return JSONResponse(
        status_code=409,
        content={
            "code": "RUN_ALREADY_ACTIVE",
            "message": str(exc),
            "suggestion": "Wait for current run to finish or POST /runs/{run_id}/abort",
        },
    )


@app.exception_handler(RunNotFoundError)
async def _handle_run_not_found(
    request: Any, exc: RunNotFoundError
) -> JSONResponse:
    """Handle lookup of a nonexistent run."""
    return JSONResponse(
        status_code=404,
        content={
            "code": "RUN_NOT_FOUND",
            "message": str(exc),
            "suggestion": None,
        },
    )


def _routine_not_found(name: str) -> HTTPException:
    """Build a 404 HTTPException for a missing routine.

    Args:
        name: The routine ID that was not found.

    Returns:
        HTTPException with structured error body.
    """
    return HTTPException(
        status_code=404,
        detail={
            "code": "ROUTINE_NOT_FOUND",
            "message": f"Routine '{name}' not found",
            "suggestion": "Run 'ocsd list' to see available routines",
        },
    )


# ---------------------------------------------------------------------------
# GET endpoints (Plan 01)
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    operation_id="ocsd_health",
)
async def health() -> HealthResponse:
    """Health check -- returns server status, version, and model availability."""
    models_loaded = False
    try:
        from core.detection import OmniParser  # noqa: F401

        models_loaded = True
    except Exception:
        pass
    return HealthResponse(status="ok", version="1.0.0", models_loaded=models_loaded)


@app.get(
    "/routines",
    response_model=list[RoutineSummary],
    operation_id="ocsd_list_routines",
)
async def list_routines_endpoint(
    tag: str | None = Query(default=None, description="Filter routines by tag"),
) -> list[RoutineSummary]:
    """List all discovered routines with name, version, and step count."""
    discovered = list_routines()
    results: list[RoutineSummary] = []

    for info in discovered:
        try:
            routine = Routine.load(info.path)
            summary = RoutineSummary(
                id=info.name,
                name=routine.name,
                version=routine.version,
                step_count=len(routine.steps),
                tags=list(routine.tags),
            )
            if tag is not None and tag not in routine.tags:
                continue
            results.append(summary)
        except Exception as exc:
            logger.warning("Could not load routine '%s': %s", info.name, exc)

    return results


@app.get(
    "/routines/{routine_id}",
    response_model=RoutineDetail,
    operation_id="ocsd_get_routine",
)
async def get_routine(routine_id: str) -> RoutineDetail:
    """Return full routine metadata including steps."""
    routine_path = get_routine_dir() / routine_id
    if not routine_path.exists():
        raise _routine_not_found(routine_id)

    try:
        routine = Routine.load(routine_path)
    except (FileNotFoundError, ValueError) as exc:
        raise _routine_not_found(routine_id) from exc

    data = routine.to_dict()
    return RoutineDetail(
        id=routine_id,
        name=routine.name,
        version=routine.version,
        description=routine.description,
        tags=list(routine.tags),
        steps=data.get("steps", []),
        metadata={
            "author": data.get("author"),
            "platform": data.get("platform"),
            "resolution": data.get("resolution"),
            "category": data.get("category"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
        },
    )


# ---------------------------------------------------------------------------
# Stub endpoints (implemented in Plan 02)
# ---------------------------------------------------------------------------


@app.post(
    "/routines/{routine_id}/run",
    response_model=RunResponse,
    operation_id="ocsd_run_routine",
)
async def run_routine_endpoint(
    routine_id: str, req: RunRequest
) -> RunResponse:
    """Start a routine execution run (stub -- Plan 02)."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.get(
    "/runs/{run_id}/status",
    response_model=RunStatusResponse,
    operation_id="ocsd_get_run_status",
)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """Get the status of an active or recent run (stub -- Plan 02)."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.post(
    "/runs/{run_id}/respond",
    operation_id="ocsd_respond_to_prompt",
)
async def respond_to_prompt(run_id: str, req: RespondRequest) -> dict[str, str]:
    """Respond to a prompt during a running routine (stub -- Plan 02)."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.get(
    "/runs/{run_id}/screenshot",
    operation_id="ocsd_get_screenshot",
)
async def get_screenshot(run_id: str) -> bytes:
    """Get the latest screenshot from a run (stub -- Plan 02)."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.post(
    "/runs/{run_id}/abort",
    operation_id="ocsd_abort_run",
)
async def abort_run(run_id: str) -> dict[str, str]:
    """Abort an active run (stub -- Plan 02)."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
