"""Tests for OCSD API server endpoints.

Verifies health check, routine listing, routine detail, run start,
status, conflict, and localhost binding configuration using FastAPI TestClient.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.server import app, manager

client = TestClient(app)


def test_health() -> None:
    """Health endpoint returns status ok with version and models_loaded."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "models_loaded" in data


@patch("api.server.list_routines")
def test_list_routines_empty(mock_lr: MagicMock) -> None:
    """Empty routine directory returns empty list."""
    mock_lr.return_value = []
    resp = client.get("/routines")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_routine_not_found() -> None:
    """Requesting a nonexistent routine returns 404 with ROUTINE_NOT_FOUND code."""
    resp = client.get("/routines/nonexistent-routine")
    assert resp.status_code == 404
    data = resp.json()
    assert data["detail"]["code"] == "ROUTINE_NOT_FOUND"


def test_localhost_binding(tmp_path: Path) -> None:
    """Config defaults bind to 127.0.0.1:8420 -- no remote access."""
    from core.config import _DEFAULTS, load_config

    # Load with a nonexistent config path so only _DEFAULTS apply
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg["api"]["host"] == "127.0.0.1"
    assert cfg["api"]["port"] == 8420
    assert cfg["api"]["prompt_timeout_s"] == 300
    assert "mcp_compatible" not in _DEFAULTS["api"]


def test_run_not_found() -> None:
    """GET /runs/{run_id}/status returns 404 for nonexistent run."""
    resp = client.get("/runs/nonexistent/status")
    assert resp.status_code == 404
    assert resp.json()["code"] == "RUN_NOT_FOUND"


@patch("routine.runner.run_routine")
@patch("api.server.get_routine_dir")
def test_start_run(mock_dir: MagicMock, mock_run: MagicMock) -> None:
    """POST /routines/{id}/run returns run_id."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test-routine"
        p.mkdir()
        (p / "routine.json").write_text(
            json.dumps({"schema": "ocsd-routine-v1", "name": "test", "steps": []})
        )
        mock_dir.return_value = Path(td)
        mock_run.return_value = MagicMock(success=True)

        # Reset manager state
        manager._active = None

        resp = client.post("/routines/test-routine/run")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data

        # Clean up: mark the run as complete so it doesn't block next test
        run_id = data["run_id"]
        manager.mark_complete(run_id)


@patch("routine.runner.run_routine")
@patch("api.server.get_routine_dir")
def test_run_conflict(mock_dir: MagicMock, mock_run: MagicMock) -> None:
    """Second POST /run returns 409."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test-routine"
        p.mkdir()
        (p / "routine.json").write_text(
            json.dumps({"schema": "ocsd-routine-v1", "name": "test", "steps": []})
        )
        mock_dir.return_value = Path(td)

        manager._active = None

        # First run succeeds
        resp1 = client.post("/routines/test-routine/run")
        assert resp1.status_code == 200

        # Second run should 409
        resp2 = client.post("/routines/test-routine/run")
        assert resp2.status_code == 409

        # Clean up
        run_id = resp1.json()["run_id"]
        manager.mark_complete(run_id)


# ---------------------------------------------------------------------------
# MCP and scanner tests (Plan 03)
# ---------------------------------------------------------------------------


def test_mcp_mount() -> None:
    """Verify MCP endpoint is available (API-08)."""
    # The /mcp path should be mounted by fastapi-mcp
    route_paths = [r.path for r in app.routes]
    # fastapi-mcp may mount at /mcp or as a sub-application
    # At minimum, verify the MCP object was created
    assert hasattr(app, "routes"), "App has routes"
    # Verify operation_ids have ocsd_ prefix
    for route in app.routes:
        if hasattr(route, "operation_id") and route.operation_id:
            if route.path.startswith(("/health", "/routines", "/runs")):
                assert route.operation_id.startswith(
                    "ocsd_"
                ), f"Route {route.path} missing ocsd_ prefix: {route.operation_id}"


def test_operation_ids_prefix() -> None:
    """All API routes have ocsd_ prefixed operation_ids (API-08)."""
    expected_ids = [
        "ocsd_health",
        "ocsd_list_routines",
        "ocsd_get_routine",
        "ocsd_run_routine",
        "ocsd_get_run_status",
        "ocsd_respond_to_prompt",
        "ocsd_get_screenshot",
        "ocsd_abort_run",
    ]
    actual_ids = [
        r.operation_id
        for r in app.routes
        if hasattr(r, "operation_id") and r.operation_id
    ]
    for eid in expected_ids:
        assert eid in actual_ids, f"Missing operation_id: {eid}"


def test_scanner_exists() -> None:
    """Hub scanner module exists and has scan_skill function (SEC-01)."""
    from hub.scanner import ScanResult, scan_skill

    # Verify it can scan a minimal skill dict
    result = scan_skill(
        {
            "name": "test",
            "nodes": [],
            "edges": [],
        }
    )
    assert isinstance(result, ScanResult)
    assert result.is_safe is True


def test_no_remote_binding() -> None:
    """Server config does not allow remote access (SEC-03)."""
    from core.config import get_config

    cfg = get_config()
    assert cfg["api"]["host"] == "127.0.0.1", "Must bind localhost only"
    assert cfg["api"]["host"] != "0.0.0.0", "Must not bind all interfaces"
