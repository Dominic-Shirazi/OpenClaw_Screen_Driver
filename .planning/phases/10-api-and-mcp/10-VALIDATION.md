---
phase: 10
slug: api-and-mcp
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/test_api_server.py -x -q` |
| **Full suite command** | `python -m pytest tests/test_api_server.py tests/test_api_integration.py -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_api_server.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_api_server.py tests/test_api_integration.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | API-09 | unit | `pytest tests/test_api_server.py::test_server_daemon_thread` | W0 | pending |
| 10-01-02 | 01 | 1 | API-01, API-02 | unit | `pytest tests/test_api_server.py::test_routine_endpoints` | W0 | pending |
| 10-02-01 | 02 | 1 | API-03, API-04 | unit | `pytest tests/test_api_server.py::test_run_endpoints` | W0 | pending |
| 10-02-02 | 02 | 1 | API-05 | unit | `pytest tests/test_api_server.py::test_respond_endpoint` | W0 | pending |
| 10-02-03 | 02 | 1 | API-06, API-07 | unit | `pytest tests/test_api_server.py::test_screenshot_abort` | W0 | pending |
| 10-03-01 | 03 | 2 | API-08 | unit | `pytest tests/test_api_server.py::test_mcp_tools` | W0 | pending |
| 10-03-02 | 03 | 2 | SEC-01, SEC-02, SEC-03 | unit | `pytest tests/test_api_server.py::test_security` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_api_server.py` — stubs for all API endpoint tests
- [ ] `tests/conftest.py` — FastAPI TestClient fixture, mock routine fixtures
- [ ] `httpx` dev dependency — required by FastAPI TestClient

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Pause overlay shows green shimmer + "Paused" badge | API-07 | Requires visual overlay on real display | 1. Start routine via API 2. POST /abort 3. Verify green shimmer + badge text 4. Press Enter to resume or Esc to cancel |
| MCP tools discoverable by Claude/agent | API-08 | Requires MCP client connection | 1. Start server 2. Connect MCP client to /mcp 3. List tools 4. Verify ocsd_ prefixed tools appear |
| Model preloading in `ocsd serve` | API-09 | Requires actual model files | 1. Run `ocsd serve` 2. Verify models load before "ready" message |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
