---
phase: 10-api-and-mcp
verified: 2026-03-20T12:00:00Z
status: passed
score: 17/17 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 16/17
  gaps_closed:
    - "URL path contract matches REQUIREMENTS.md spec for run-scoped endpoints"
    - "SEC-02 is marked Complete in REQUIREMENTS.md traceability table"
    - "prompt_timeout_s from API request body reaches prompt_user_blocking() during routine execution"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Start API server with 'ocsd serve' and confirm /mcp endpoint responds"
    expected: "HTTP GET http://127.0.0.1:8420/mcp returns an MCP-compatible response. MCP tools with ocsd_ prefix are discoverable by an MCP client."
    why_human: "fastapi-mcp mount_http() is confirmed in code and all 12 tests pass, but actual MCP protocol wire behavior requires a running server and MCP client connection."
  - test: "Trigger abort during a running routine and observe overlay"
    expected: "Overlay border shimmer changes from purple (REPLAYING) to green (READY) with a 'Paused' badge visible. Overlay stays open."
    why_human: "Code path is fully wired (runner emits RUN_PAUSED -> ReplayOverlayAdapter -> set_replay_mode(False) + set_replay_status('Paused')). Visual rendering requires a running Qt process."
  - test: "Run 'ocsd serve' then curl http://127.0.0.1:8420/health"
    expected: "Server starts and health endpoint returns {\"status\":\"ok\",\"version\":\"1.0.0\",\"models_loaded\":false}"
    why_human: "uvicorn.run() is a blocking call confirmed at code level. Actual server bind and HTTP response requires a live process."
---

# Phase 10: API and MCP Verification Report

**Phase Goal:** Agents and external tools can discover, run, and control routines via a local REST API with MCP-compatible tool names
**Verified:** 2026-03-20
**Status:** passed
**Re-verification:** Yes — after gap closure via 10-04-PLAN.md

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | GET /health returns status ok with version and models_loaded | VERIFIED | api/server.py operation_id ocsd_health; 12 tests pass |
| 2  | GET /routines lists all discovered routines with name and step count | VERIFIED | api/server.py operation_id ocsd_list_routines; test_list_routines_empty passes |
| 3  | GET /routines/{id} returns routine metadata and steps | VERIFIED | api/server.py operation_id ocsd_get_routine; test_get_routine_not_found passes |
| 4  | Server starts as daemon thread without conflicting with Qt signal handlers | VERIFIED | api/lifecycle.py _NoSignalServer.install_signal_handlers() is a no-op; daemon=True confirmed |
| 5  | Server binds to 127.0.0.1 only | VERIFIED | core/config.py "host": "127.0.0.1"; test_localhost_binding passes |
| 6  | Config defaults use port 8420 and host 127.0.0.1 | VERIFIED | core/config.py "port": 8420, "host": "127.0.0.1", "prompt_timeout_s": 300 |
| 7  | POST /routines/{id}/run starts a routine on a background thread and returns run_id | VERIFIED | api/server.py operation_id ocsd_run_routine; threading.Thread daemon=True; test_start_run passes |
| 8  | POST /routines/{id}/run returns 409 if a run is already active | VERIFIED | api/server.py raises 409 with RUN_ALREADY_ACTIVE; test_run_conflict passes |
| 9  | GET /runs/{run_id}/status returns current step, total steps, and status enum | VERIFIED | api/server.py operation_id ocsd_get_run_status; test_run_not_found confirms 404 path |
| 10 | POST /runs/{run_id}/respond unblocks a waiting prompt_user step | VERIFIED | api/server.py operation_id ocsd_respond_to_prompt; calls core.executor.respond_to_prompt |
| 11 | GET /runs/{run_id}/screenshot returns PNG binary with image/png content type | VERIFIED | api/server.py operation_id ocsd_get_screenshot; media_type="image/png"; cv2.imencode wired |
| 12 | POST /runs/{run_id}/abort sets abort flag | VERIFIED | api/server.py operation_id ocsd_abort_run; abort_event.set(); runner checks between steps |
| 13 | Abort triggers visual pause state (green shimmer, Paused badge) | VERIFIED | routine/replay_overlay.py case RunEvent.RUN_PAUSED: set_replay_mode(False); set_replay_status("Paused") |
| 14 | prompt_timeout_s from API request body reaches prompt_user_blocking() | VERIFIED | runner.py 7 occurrences (lines 415, 495, 535, 608, 643, 756, 811); server.py line 374 passes prompt_timeout_s=timeout_s; test_run_passes_prompt_timeout passes |
| 15 | All FastAPI endpoints exposed as MCP tools via fastapi-mcp | VERIFIED | api/server.py FastApiMCP(app); mcp.mount_http(); 8 operation_ids with ocsd_ prefix confirmed |
| 16 | ocsd serve command starts API server headless | VERIFIED | cli/app.py def serve(); uvicorn.run(api_app, host=host, port=port) |
| 17 | URL path contract matches REQUIREMENTS.md for run-scoped endpoints | VERIFIED | REQUIREMENTS.md lines 118-121 now read /runs/{run_id}/status, /respond, /screenshot, /abort matching implementation |

**Score:** 17/17 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `api/server.py` | FastAPI app with all 8 endpoints and MCP mount | VERIFIED | 8 operation_ids with ocsd_ prefix; FastApiMCP mounted; prompt_timeout_s=timeout_s passed to run_routine at line 374 |
| `api/run_manager.py` | Thread-safe RunManager | VERIFIED | RunManager, RunStatus, ActiveRun, RunAlreadyActiveError, RunNotFoundError; threading.Lock confirmed |
| `api/lifecycle.py` | Daemon thread server startup | VERIFIED | _NoSignalServer, daemon=True, install_signal_handlers no-op |
| `core/config.py` | API config defaults | VERIFIED | host 127.0.0.1, port 8420, prompt_timeout_s 300 |
| `tests/test_api_server.py` | TestClient tests for all endpoints | VERIFIED | 12 tests pass (11 original + test_run_passes_prompt_timeout) |
| `routine/runner.py` | prompt_timeout_s threaded through run_routine, _handle_loop_step, _dispatch_action | VERIFIED | 7 occurrences; reaches prompt_user_blocking(timeout=prompt_timeout_s) at line 495 |
| `routine/replay_overlay.py` | RUN_PAUSED handler | VERIFIED | case RunEvent.RUN_PAUSED: set_replay_mode(False); set_replay_status("Paused"); no close() call |
| `core/executor.py` | PromptTimeoutError + timeout param | VERIFIED | PromptTimeoutError class; prompt_user_blocking(timeout=...); _prompt_response_event.wait(timeout=timeout) |
| `cli/app.py` | ocsd serve subcommand | VERIFIED | def serve(); model preloading; uvicorn.run blocking call |
| `pyproject.toml` | fastapi-mcp in api extras | VERIFIED | fastapi-mcp>=0.4.0 in api extras; httpx>=0.27 in dev |
| `hub/scanner.py` | scan_skill + ScanResult | VERIFIED | scan_skill and ScanResult present; test_scanner_exists passes |
| `.planning/REQUIREMENTS.md` | API-04..07 flat paths; SEC-02 Complete | VERIFIED | Lines 118-121 updated; line 128 SEC-02 [x]; line 255 SEC-02 Phase 5 Complete |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| api/server.py | routine/discovery.py | list_routines() | WIRED | Direct import at module level |
| api/server.py | routine/format.py | Routine.load() | WIRED | Direct import at module level |
| api/lifecycle.py | api/server.py | uvicorn.Config(app) | WIRED | uvicorn.Config(app, host=h, port=p) in lifecycle.py |
| api/server.py | routine/runner.py | run_routine() on background thread | WIRED | Lazy import inside _run_in_thread; threading.Thread confirmed |
| api/server.py | routine/runner.py | prompt_timeout_s=timeout_s kwarg | WIRED | server.py line 374; confirmed by test_run_passes_prompt_timeout |
| routine/runner.py | core/executor.py | prompt_user_blocking(timeout=prompt_timeout_s) | WIRED | runner.py line 495: prompt_user_blocking(question, dry_run=dry_run, timeout=prompt_timeout_s) |
| api/server.py | core/executor.py | respond_to_prompt() | WIRED | Lazy import in respond_to_prompt_endpoint |
| api/server.py | core/capture.py | screenshot_full() | WIRED | Lazy import in get_screenshot; cv2.imencode result returned as PNG |
| api/server.py | api/run_manager.py | manager.start_run, get_run, mark_* | WIRED | manager = RunManager() singleton; all state transitions via _make_run_callback |
| routine/runner.py | routine/replay_overlay.py | RUN_PAUSED event | WIRED | _emit(callback, RunEvent.RUN_PAUSED, ...); ReplayOverlayAdapter handles it |
| api/server.py | fastapi_mcp | FastApiMCP(app) + mcp.mount_http() | WIRED | try/except ImportError; mount_http() confirmed |
| cli/app.py | api/lifecycle.py | uvicorn.run for serve command | WIRED | uvicorn.run(api_app, host=host, port=port) — direct blocking call |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| API-01 | 10-01 | GET /routines — list all routines | SATISFIED | api/server.py GET /routines; test_list_routines_empty passes |
| API-02 | 10-01 | GET /routines/{id} — routine metadata + steps | SATISFIED | api/server.py GET /routines/{routine_id}; test_get_routine_not_found passes |
| API-03 | 10-02 | POST /routines/{id}/run — start execution, return run_id | SATISFIED | api/server.py POST /routines/{routine_id}/run; test_start_run passes |
| API-04 | 10-02, 10-04 | GET /runs/{run_id}/status — execution status | SATISFIED | Flat path implemented; REQUIREMENTS.md line 118 updated |
| API-05 | 10-02, 10-04 | POST /runs/{run_id}/respond — respond to prompt_user | SATISFIED | Flat path implemented; REQUIREMENTS.md line 119 updated |
| API-06 | 10-02, 10-04 | GET /runs/{run_id}/screenshot — current screen state | SATISFIED | Flat path implemented; REQUIREMENTS.md line 120 updated |
| API-07 | 10-02, 10-04 | POST /runs/{run_id}/abort — cancel running routine | SATISFIED | Flat path implemented; REQUIREMENTS.md line 121 updated |
| API-08 | 10-03 | MCP-compatible tool names via fastapi-mcp | SATISFIED | FastApiMCP mounted; 8 ocsd_ operation_ids; test_operation_ids_prefix passes |
| API-09 | 10-01 | FastAPI server runs as daemon with install_signal_handlers=False | SATISFIED | api/lifecycle.py _NoSignalServer.install_signal_handlers() no-op; daemon=True |
| SEC-01 | 10-03 | Hub scanner flags suspicious patterns | SATISFIED | hub/scanner.py scan_skill + ScanResult; test_scanner_exists passes |
| SEC-02 | 10-04 (closure) | Routine files are auditable — every step human-readable JSON | SATISFIED | REQUIREMENTS.md line 128: [x]; traceability line 255: Phase 5, Complete |
| SEC-03 | 10-01, 10-03 | No cloud dependency at runtime — replay 100% local | SATISFIED | Localhost-only binding; test_no_remote_binding passes; no cloud endpoint in config |

All 12 requirement IDs accounted for. No orphaned requirements.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| api/server.py | 472-483 | try/except ImportError around FastApiMCP | INFO | Intentional graceful degradation when fastapi-mcp not installed. Design decision. Not a stub. |

No blocker or warning anti-patterns remain. The prompt_timeout_s warning from the initial verification is resolved.

---

## Human Verification Required

### 1. MCP Wire Protocol

**Test:** Start `ocsd serve`, then point an MCP client at `http://127.0.0.1:8420/mcp` (or run `curl -N http://127.0.0.1:8420/mcp`).
**Expected:** Server responds with an MCP tool list containing 8 tools with ocsd_ prefixed names. An MCP client can invoke `ocsd_health` and receive the health response.
**Why human:** fastapi-mcp mount_http() is confirmed in code and all 12 tests pass. The actual SSE/HTTP Streamable wire protocol behavior requires a live server and MCP client.

### 2. Abort Overlay Visual Behavior

**Test:** Record a multi-step routine, run it via `POST /routines/{id}/run`, then immediately `POST /runs/{run_id}/abort`. Observe the overlay.
**Expected:** Overlay border shimmer changes from purple (REPLAYING) to green (READY). A "Paused" badge appears. Overlay remains visible and does not auto-close.
**Why human:** Code path is fully wired. Visual rendering of shimmer color and badge text requires a running Qt process and screen observation.

### 3. ocsd serve End-to-End

**Test:** Run `ocsd serve`, then `curl http://127.0.0.1:8420/health`.
**Expected:** Server starts, displays "OCSD API ready at http://127.0.0.1:8420", health endpoint returns `{"status":"ok","version":"1.0.0","models_loaded":false}`.
**Why human:** uvicorn.run() is a blocking call confirmed at code level only. Actual server bind and HTTP response requires a live process.

---

## Re-Verification Summary

All three gaps from the initial verification are closed:

**Gap 1 — URL Path Shape (Closed):** REQUIREMENTS.md lines 118-121 now describe flat `/runs/{run_id}/...` paths (status, respond, screenshot, abort) matching the implemented design. Traceability table rows for API-04 through API-07 remain Complete.

**Gap 2 — SEC-02 Traceability (Closed):** REQUIREMENTS.md line 128 checkbox is now `[x]`. Traceability table row at line 255 reads `| SEC-02 | Phase 5 | Complete |`, correctly attributing the requirement to the Phase 5 ocsd-routine-v1 JSON format delivery.

**Gap 3 (Bonus) — Prompt Timeout Wiring (Closed):** `prompt_timeout_s` is now threaded from the API POST body (`body.prompt_timeout_s`) through `run_routine(prompt_timeout_s=timeout_s)` through `_handle_loop_step` and `_dispatch_action` to `prompt_user_blocking(timeout=prompt_timeout_s)`. The PromptTimeoutError path is now reachable via the normal API execution flow. Confirmed by test_run_passes_prompt_timeout (12th test, all passing).

No regressions detected. All 17 observable truths verified. All 12 requirements satisfied.

---

_Verified: 2026-03-20_
_Verifier: Claude (gsd-verifier)_
