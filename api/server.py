"""FastAPI server providing MCP-compatible tool call endpoints for OCSD.

Exposes REST endpoints for skill listing, execution, recording status,
and graph inspection.  Designed to be run as a sidecar or embedded
service that automation orchestrators (MCP hosts, custom agents, etc.)
can call.

Usage:
    uvicorn api.server:app --port 8420 --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for the API server.  "
        "Install with:  pip install 'ocsd[api]'  or  pip install fastapi uvicorn"
    )

from core.config import get_config
from mapper.export import load_skill, export_skill
from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCSD API",
    version="0.1.0",
    description="OpenClaw Screen Driver — skill execution and management API.",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    """Request body for skill execution."""
    skill_id: str
    start_node: str | None = None
    goal_node: str | None = None
    dry_run: bool = False
    params: dict[str, str] | None = None


class ExecuteResponse(BaseModel):
    """Response from a skill execution run."""
    skill_id: str
    success: bool
    steps_total: int
    steps_succeeded: int
    duration_ms: int
    errors: list[str]


class SkillSummary(BaseModel):
    """Lightweight skill metadata returned in list endpoints."""
    skill_id: str
    name: str
    node_count: int
    edge_count: int
    file_path: str


class NodeInfo(BaseModel):
    """Information about a single graph node."""
    node_id: str
    element_type: str
    label: str
    position: dict[str, float] | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skills_dir() -> Path:
    """Returns the configured skills directory."""
    config = get_config()
    return Path(config.get("paths", {}).get("skills_dir", "./skills"))


def _load_graph(skill_id: str) -> OCSDGraph:
    """Loads a skill graph from disk by ID."""
    skills_dir = _skills_dir()
    skill_path = skills_dir / f"{skill_id}.json"
    if not skill_path.exists():
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    try:
        return load_skill(skill_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading skill: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check — returns server status and version."""
    return HealthResponse(status="ok", version="0.1.0")


@app.get("/skills", response_model=list[SkillSummary])
async def list_skills() -> list[SkillSummary]:
    """Lists all available skills in the skills directory."""
    skills_dir = _skills_dir()
    if not skills_dir.exists():
        return []

    results: list[SkillSummary] = []
    for skill_file in sorted(skills_dir.glob("*.json")):
        try:
            graph = load_skill(skill_file)
            results.append(SkillSummary(
                skill_id=graph.skill_id,
                name=graph.name,
                node_count=len(graph.nodes),
                edge_count=graph.nx_graph.number_of_edges(),
                file_path=str(skill_file),
            ))
        except Exception as e:
            logger.warning("Skipping invalid skill file %s: %s", skill_file, e)
    return results


@app.get("/skills/{skill_id}", response_model=dict[str, Any])
async def get_skill(skill_id: str) -> dict[str, Any]:
    """Returns the full graph data for a skill."""
    graph = _load_graph(skill_id)
    return graph.to_dict()


@app.get("/skills/{skill_id}/nodes", response_model=list[NodeInfo])
async def list_nodes(skill_id: str) -> list[NodeInfo]:
    """Lists all nodes in a skill graph."""
    graph = _load_graph(skill_id)
    nodes: list[NodeInfo] = []
    for node_id in graph.nodes:
        data = graph.get_node(node_id)
        pos = data.get("relative_position")
        nodes.append(NodeInfo(
            node_id=node_id,
            element_type=data.get("element_type", "unknown"),
            label=data.get("label", ""),
            position=pos if pos else None,
        ))
    return nodes


@app.post("/skills/{skill_id}/execute", response_model=ExecuteResponse)
async def execute_skill(skill_id: str, req: ExecuteRequest) -> ExecuteResponse:
    """Executes a skill from start to goal node.

    If start_node / goal_node are omitted, uses the first and last
    nodes in the graph's recorded order.
    """
    graph = _load_graph(skill_id)
    nodes_list = list(graph.nodes)
    if len(nodes_list) < 2:
        raise HTTPException(
            status_code=400,
            detail="Skill must have at least 2 nodes to execute",
        )

    start_id = req.start_node or nodes_list[0]
    goal_id = req.goal_node or nodes_list[-1]

    try:
        from mapper.runner import run_skill

        replay_log = run_skill(
            graph, start_id, goal_id,
            dry_run=req.dry_run,
            execution_params=req.params,
        )

        errors = [
            step.error
            for step in replay_log.steps
            if step.error
        ]

        return ExecuteResponse(
            skill_id=skill_id,
            success=replay_log.overall_success,
            steps_total=len(replay_log.steps),
            steps_succeeded=sum(1 for s in replay_log.steps if s.success),
            duration_ms=replay_log.duration_ms,
            errors=errors,
        )
    except Exception as e:
        logger.error("Skill execution failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Execution error: {e}")


@app.get("/skills/{skill_id}/plan")
async def get_plan(
    skill_id: str,
    start: str | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    """Returns the execution plan (path, steps, reliability) without running it."""
    graph = _load_graph(skill_id)
    nodes_list = list(graph.nodes)
    if len(nodes_list) < 2:
        raise HTTPException(status_code=400, detail="Skill needs at least 2 nodes")

    start_id = start or nodes_list[0]
    goal_id = goal or nodes_list[-1]

    try:
        from mapper.pathfinder import get_execution_plan as build_plan
        plan = build_plan(graph, start_id, goal_id)
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planning error: {e}")


@app.delete("/skills/{skill_id}")
async def delete_skill(skill_id: str) -> dict[str, str]:
    """Deletes a skill file from disk."""
    skills_dir = _skills_dir()
    skill_path = skills_dir / f"{skill_id}.json"
    if not skill_path.exists():
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    skill_path.unlink()
    return {"status": "deleted", "skill_id": skill_id}
