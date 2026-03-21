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
