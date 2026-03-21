"""Tests for OCSD API server endpoints.

Verifies health check, routine listing, routine detail, and
localhost binding configuration using FastAPI TestClient.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.server import app

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


def test_stub_endpoints_return_501() -> None:
    """Stub endpoints return 501 Not Implemented."""
    resp = client.post("/routines/test-routine/run", json={})
    assert resp.status_code == 501

    resp = client.get("/runs/fakeid/status")
    assert resp.status_code == 501

    resp = client.post("/runs/fakeid/respond", json={"response": "yes"})
    assert resp.status_code == 501

    resp = client.post("/runs/fakeid/abort")
    assert resp.status_code == 501
