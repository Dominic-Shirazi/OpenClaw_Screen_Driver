# Phase 10: API and MCP - Research

**Researched:** 2026-03-20
**Domain:** FastAPI REST API + MCP tool layer for routine automation
**Confidence:** HIGH

## Summary

Phase 10 replaces the existing skill-based `api/server.py` with a routine-centric REST API and adds MCP tool discovery via `fastapi-mcp`. The existing codebase provides all necessary building blocks: `run_routine()` for execution, `prompt_user_blocking()` / `respond_to_prompt()` for the prompt mechanism, `list_routines()` / `Routine.load()` for discovery, and `screenshot_full()` for screen capture. The old server is a clean 244-line file with no downstream dependents -- full replacement is safe.

The two key integration challenges are: (1) running uvicorn as a daemon thread alongside PyQt6 without signal handler conflicts, and (2) managing a single active run with abort/pause semantics that bridge the API layer and the runner's step loop. Both are well-understood patterns with straightforward implementations.

**Primary recommendation:** Build the API as a new `api/server.py` with FastAPI + Pydantic models, wire `fastapi-mcp` with explicit `operation_id` on each route for `ocsd_` prefixed tool names, run uvicorn via a custom Server subclass that overrides `install_signal_handlers`, and add an `ocsd serve` CLI command for headless usage.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Server lifecycle**: Two start modes: (1) daemon thread with Qt, (2) standalone `ocsd serve` headless
- **Daemon thread**: uvicorn.Server with `install_signal_handlers=False`, Qt owns signals
- **`ocsd serve`**: Preloads AI models, prints ready message, blocks, Ctrl+C shutdown
- **Single active run**: POST /run returns 409 CONFLICT with `RUN_ALREADY_ACTIVE` if busy
- **Port**: Default 8420, configurable via `api.port` in ocsd.yaml
- **Localhost only**: Bind 127.0.0.1. No auth, no API keys
- **Endpoints**: GET /health, GET /routines, GET /routines/{id}, POST /routines/{id}/run, GET /runs/{run_id}/status, POST /runs/{run_id}/respond, GET /runs/{run_id}/screenshot, POST /runs/{run_id}/abort
- **Routine IDs**: Folder name as ID (e.g., `/routines/google-search`)
- **Error codes**: Structured `{"code": "...", "message": "...", "suggestion": "..."}`
- **Prompt timeout**: Default 5 minutes, configurable per-run and globally
- **Abort = pause**: Shimmer green, badge "Paused", Enter resume, Esc cancel. Resume via keyboard only
- **Screenshot**: Raw PNG binary with `Content-Type: image/png`
- **Run status**: Summary only (status enum, current step, total steps, error/prompt text)
- **Old API**: Delete and rewrite entirely
- **fastapi-mcp**: Auto-generate MCP tools at `/mcp` path, `ocsd_` prefix via operation_id
- **MCP transport**: SSE via HTTP (same server, same port)
- **Scanner**: Stays as-is (skill-format), V1 routine scanning deferred to V2
- **SEC-01 scanning**: On install/download only, locally created routines trusted
- **SEC-02 auditability**: JSON is human-readable by design, no additional features
- **SEC-03 localhost**: Primary security mechanism, no remote access

### Claude's Discretion
- Exact uvicorn.Server configuration for daemon thread mode
- Pause/resume keyboard listener implementation during paused state
- Run ID generation scheme (UUID, timestamp, etc.)
- How fastapi-mcp operation_id mapping works for ocsd_ prefix
- Model preloading sequence in `ocsd serve`
- How to wire abort flag into runner's step loop
- CORS configuration (if any) for localhost

### Deferred Ideas (OUT OF SCOPE)
- V1 routine format scanning in hub/scanner.py
- Streaming run events via SSE/WebSocket
- Per-step detail in run status
- API authentication (API keys, OAuth)
- stdio MCP transport for local Claude Code integration
- Daemon mode for `ocsd serve` (backgrounding, PID file, --stop)
- Resume paused run via API (not just keyboard)

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| API-01 | GET /routines -- list all routines | `routine/discovery.py:list_routines()` returns `RoutineInfo` objects; map to Pydantic response model |
| API-02 | GET /routines/{id} -- routine metadata + steps | `routine/format.py:Routine.load()` + `Routine.to_dict()` for full serialization |
| API-03 | POST /routines/{id}/run -- start execution | `routine/runner.py:run_routine()` on background thread; return run_id from `RunManager` |
| API-04 | GET /runs/{run_id}/status -- execution status | `RunManager` tracks active run state via RunEvent callback |
| API-05 | POST /runs/{run_id}/respond -- answer prompt_user | `core/executor.py:respond_to_prompt()` sets text + signals threading.Event |
| API-06 | GET /runs/{run_id}/screenshot -- current screen | `core/capture.py:screenshot_full()` -> cv2.imencode PNG -> Response(content, media_type) |
| API-07 | POST /runs/{run_id}/abort -- cancel routine | Set abort threading.Event; runner checks between steps; triggers pause overlay state |
| API-08 | MCP-compatible tool names via fastapi-mcp | `fastapi-mcp` 0.4.0 maps operation_id to tool names; mount at `/mcp` |
| API-09 | FastAPI server as daemon thread | Custom uvicorn.Server subclass overriding install_signal_handlers |
| SEC-01 | Hub scanner flags suspicious patterns | `hub/scanner.py:scan_skill()` exists and works on skill dicts; kept as-is per decisions |
| SEC-02 | Routine files are auditable (human-readable JSON) | Already satisfied by `ocsd-routine-v1` JSON schema |
| SEC-03 | No cloud dependency at runtime | Enforce `host="127.0.0.1"` binding; no external HTTP calls in server code |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | >=0.100 | REST API framework | Already in pyproject.toml `[api]` extras; async-native, Pydantic integration |
| uvicorn[standard] | >=0.23 | ASGI server | Already in pyproject.toml; standard FastAPI deployment |
| fastapi-mcp | 0.4.0 | MCP tool generation from FastAPI routes | Locked decision; auto-generates MCP tools from operation_ids |
| Pydantic | (via FastAPI) | Request/response validation | Built into FastAPI; defines tool schemas for MCP |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cv2 (opencv-python) | >=4.8 | PNG encoding for /screenshot | Already installed; `cv2.imencode(".png", img)` |
| threading | stdlib | Daemon thread for uvicorn, abort events | Server lifecycle + run abort mechanism |
| uuid | stdlib | Run ID generation | `uuid.uuid4().hex[:12]` for short run IDs |
| Typer | >=0.12 | `ocsd serve` CLI command | Already the CLI framework |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| fastapi-mcp | Manual MCP server (fastmcp) | More control but requires maintaining two server definitions; fastapi-mcp is zero-config |
| cv2.imencode for PNG | Pillow Image.save to BytesIO | Either works; cv2 already imported in capture.py |
| threading.Event for abort | asyncio.Event | Runner is sync (runs on background thread); threading.Event is correct |

**Installation:**
```bash
pip install fastapi-mcp
```

Note: `fastapi` and `uvicorn[standard]` already in `pyproject.toml` `[api]` extras. Add `fastapi-mcp` to the same group.

## Architecture Patterns

### Recommended Project Structure
```
api/
  server.py          # Full rewrite -- FastAPI app, routes, Pydantic models
  run_manager.py     # RunManager class -- tracks active run, abort, status
  lifecycle.py       # UvicornDaemonServer, start_api_server(), ocsd_serve()
```

### Pattern 1: RunManager Singleton
**What:** A thread-safe manager class that tracks the single active run, holds its state, and provides methods for status, respond, abort, and screenshot.
**When to use:** Always -- this is the bridge between HTTP handlers and the runner thread.
**Example:**
```python
# Source: Project pattern extrapolated from runner.py + executor.py
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum

class RunStatus(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ActiveRun:
    run_id: str
    routine_id: str
    status: RunStatus = RunStatus.RUNNING
    current_step: int = 0
    total_steps: int = 0
    error: str | None = None
    prompt_text: str | None = None
    abort_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ActiveRun | None = None
        self._history: dict[str, ActiveRun] = {}

    def start_run(self, routine_id: str, ...) -> str:
        with self._lock:
            if self._active and self._active.status == RunStatus.RUNNING:
                raise RunAlreadyActiveError()
            run_id = uuid.uuid4().hex[:12]
            ...
            return run_id

    def get_status(self, run_id: str) -> ActiveRun | None: ...
    def abort(self, run_id: str) -> None: ...
```

### Pattern 2: Uvicorn Daemon Thread
**What:** Custom uvicorn.Server subclass that disables signal handlers, run in a daemon thread.
**When to use:** When starting API alongside Qt application.
**Example:**
```python
# Source: uvicorn docs + community pattern
import threading
import uvicorn

class _NoSignalServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        pass  # Qt owns signals

def start_api_daemon(app, host: str = "127.0.0.1", port: int = 8420) -> threading.Thread:
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = _NoSignalServer(config)
    thread = threading.Thread(target=server.run, daemon=True, name="ocsd-api")
    thread.start()
    return thread
```

### Pattern 3: RunEvent Callback Bridge
**What:** The API's RunManager receives RunEvent callbacks from the runner to update run status in real time.
**When to use:** POST /routines/{id}/run starts `run_routine()` on a background thread with a callback that updates RunManager state.
**Example:**
```python
def _make_callback(manager: RunManager, run_id: str) -> Callable:
    def callback(event: RunEvent, data: dict) -> None:
        if event == RunEvent.STEP_START:
            manager.update_step(run_id, data["step_index"], data["total_steps"])
        elif event == RunEvent.RUN_COMPLETE:
            manager.mark_complete(run_id)
        elif event == RunEvent.RUN_FAILED:
            manager.mark_failed(run_id, data.get("failure_reason"))
    return callback
```

### Pattern 4: Explicit operation_id for MCP Tool Names
**What:** Set `operation_id` on each FastAPI route to control the MCP tool name (fastapi-mcp uses operation_id directly as tool name).
**When to use:** All routes.
**Example:**
```python
@app.get("/routines", operation_id="ocsd_list_routines")
async def list_routines_endpoint(): ...

@app.post("/routines/{routine_id}/run", operation_id="ocsd_run_routine")
async def run_routine_endpoint(routine_id: str, body: RunRequest): ...
```

### Pattern 5: Abort Flag in Runner Step Loop
**What:** The runner checks an abort threading.Event between steps. If set, it breaks out of the step loop and returns a "paused" result.
**When to use:** Wire into `run_routine()` via an additional parameter or by checking a module-level event.
**Example:**
```python
# In runner step loop, between steps:
if abort_event and abort_event.is_set():
    logger.info("Abort requested, pausing run")
    # Emit paused event for overlay
    _emit(callback, RunEvent.RUN_PAUSED, {...})
    break
```

### Anti-Patterns to Avoid
- **Async run_routine:** The runner is deeply synchronous (PyAutoGUI, time.sleep, blocking locate). Do NOT try to make it async. Run it on a background thread.
- **Multiple active runs:** CONTEXT.md locks this to single-run. Do NOT build a run queue or pool.
- **Signal handlers in threads:** Never call `signal.signal()` from a non-main thread. The custom Server subclass pattern exists specifically for this.
- **Global mutable state for run tracking:** Use a proper RunManager class with a lock, not module-level variables (except for the prompt mechanism which already uses module-level events).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MCP tool schema generation | Manual MCP protocol implementation | `fastapi-mcp` 0.4.0 | Automatic schema from Pydantic models, zero config |
| PNG encoding | Custom image serialization | `cv2.imencode(".png", img)` | Already available, returns bytes buffer |
| Routine discovery | Custom filesystem walker | `routine.discovery.list_routines()` | Already built and tested |
| Routine loading | Custom JSON parser | `routine.format.Routine.load()` | Handles schema validation and checksum |
| Prompt blocking | Custom IPC/queue | `core.executor.prompt_user_blocking()` + `respond_to_prompt()` | Already built with threading.Event |
| ASGI server | Custom HTTP server | uvicorn | Production-grade, handles keep-alive, timeouts |

**Key insight:** The entire API layer is a thin HTTP interface over existing modules. Nearly zero business logic lives in the API -- it delegates to runner, discovery, format, executor, and capture.

## Common Pitfalls

### Pitfall 1: Signal Handler Conflict with Qt
**What goes wrong:** uvicorn installs SIGTERM/SIGINT handlers; Qt also needs them. Both crash or one silently overrides the other.
**Why it happens:** Python only allows signal handlers in the main thread.
**How to avoid:** Subclass `uvicorn.Server` and override `install_signal_handlers` with `pass`. Run in a daemon thread so it dies when the main process exits.
**Warning signs:** `ValueError: signal only works in main thread` at startup.

### Pitfall 2: Race Condition on Run State
**What goes wrong:** HTTP handler reads run status while callback thread writes it, producing torn reads.
**Why it happens:** RunManager accessed from both uvicorn async handlers and runner background thread.
**How to avoid:** All RunManager state mutations go through a `threading.Lock`. Status reads also take the lock.
**Warning signs:** Intermittent 500 errors or stale status responses.

### Pitfall 3: Prompt Timeout Not Wired
**What goes wrong:** `prompt_user_blocking()` currently blocks forever with no timeout. If no API client responds, the run hangs indefinitely.
**Why it happens:** The existing implementation uses `_prompt_response_event.wait()` with no timeout argument.
**How to avoid:** Pass `timeout=` to `Event.wait()` and handle the `False` return (timeout expired) by marking the run as failed.
**Warning signs:** Orphaned runs stuck in "waiting" status.

### Pitfall 4: CONTEXT.md URL Mismatch
**What goes wrong:** CONTEXT.md specifies simplified URLs like `GET /runs/{run_id}/status` but REQUIREMENTS.md has `GET /routines/{id}/runs/{run_id}/status` (with routine_id prefix).
**Why it happens:** Decisions evolved during discussion.
**How to avoid:** Follow CONTEXT.md (the later, more deliberate document). Use flat `/runs/{run_id}/...` URLs since run_id is globally unique. The routine_id prefix is redundant.
**Warning signs:** Confusion in planner about URL structure.

### Pitfall 5: Forgetting to Update Config Defaults
**What goes wrong:** `core/config.py` currently has `api.port: 8742` and `api.host: "0.0.0.0"`, but CONTEXT.md specifies port 8420 and host 127.0.0.1.
**Why it happens:** Config defaults were set in early project setup, before API decisions were locked.
**How to avoid:** Update `_DEFAULTS["api"]` in `config.py` to `{"host": "127.0.0.1", "port": 8420, "prompt_timeout_s": 300}`.
**Warning signs:** Server binding to wrong port/interface.

## Code Examples

### FastAPI App with MCP Integration
```python
# Source: fastapi-mcp docs + CONTEXT.md decisions
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI(
    title="OCSD API",
    version="1.0.0",
    description="OpenClaw Screen Driver -- routine execution API",
)

# ... define routes with explicit operation_id ...

mcp = FastApiMCP(app, name="ocsd", description="OCSD routine automation tools")
mcp.mount()  # Available at /mcp (HTTP Streamable transport, default)
```

### Screenshot Endpoint
```python
# Source: core/capture.py + cv2 imencode
from fastapi.responses import Response
import cv2
from core.capture import screenshot_full

@app.get("/runs/{run_id}/screenshot", operation_id="ocsd_get_screenshot")
async def get_screenshot(run_id: str) -> Response:
    """Get current screen state as PNG image."""
    manager.validate_run(run_id)
    img = screenshot_full()
    _, buf = cv2.imencode(".png", img)
    return Response(content=buf.tobytes(), media_type="image/png")
```

### Structured Error Response
```python
# Source: CONTEXT.md error response specification
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    code: str
    message: str
    suggestion: str | None = None

# Usage in exception handler:
@app.exception_handler(RoutineNotFoundError)
async def handle_not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            code="ROUTINE_NOT_FOUND",
            message=str(exc),
            suggestion="Run 'ocsd list' to see available routines",
        ).model_dump(),
    )
```

### Abort -> Pause Flow
```python
# Source: CONTEXT.md abort/pause behavior
@app.post("/runs/{run_id}/abort", operation_id="ocsd_abort_run")
async def abort_run(run_id: str):
    """Request pause of a running routine. Human at machine decides next step."""
    run = manager.get_run_or_404(run_id)
    run.abort_event.set()  # Runner checks between steps
    return {"status": "pause_requested", "run_id": run_id}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Skill-based API (graph nodes/edges) | Routine-centric API (steps, run status) | Phase 10 | Full server.py rewrite |
| `api.host: "0.0.0.0"`, port 8742 | `api.host: "127.0.0.1"`, port 8420 | Phase 10 decisions | Config defaults update |
| No MCP layer | fastapi-mcp auto-generation | Phase 10 | Agents can discover tools via MCP protocol |
| `prompt_user_blocking()` blocks forever | Configurable timeout (default 5min) | Phase 10 | Unattended runs don't hang |

**Deprecated/outdated:**
- `api/server.py` (current): Entire skill-based API is replaced. All endpoints reference `mapper.export`, `mapper.graph`, `mapper.runner` -- none of which are relevant to routine-centric V1.
- `_DEFAULTS["api"]["host"]` = `"0.0.0.0"`: Must change to `"127.0.0.1"` for security.

## Open Questions

1. **MCP Transport: SSE vs HTTP Streamable**
   - What we know: CONTEXT.md says "SSE via HTTP at `/mcp`". fastapi-mcp 0.4.0 offers `mount()` (HTTP Streamable, default) and `mount_sse()` (legacy SSE).
   - What's unclear: Whether the user specifically wants legacy SSE or is fine with the newer HTTP Streamable default.
   - Recommendation: Use `mcp.mount()` (HTTP Streamable, the modern default). If SSE is specifically needed for client compatibility, use `mcp.mount_sse()`. Both serve at `/mcp` or `/sse` respectively.

2. **Abort Flag Injection into Runner**
   - What we know: `run_routine()` takes `routine_dir`, `callback`, and `dry_run`. It has no `abort_event` parameter.
   - What's unclear: Whether to add an `abort_event` parameter to `run_routine()` or use a module-level event.
   - Recommendation: Add an `abort_event: threading.Event | None = None` parameter to `run_routine()`. Check it between steps in the loop. This is cleaner than module-level state and supports the single-run constraint.

3. **Prompt Timeout Implementation**
   - What we know: `_prompt_response_event.wait()` currently has no timeout.
   - What's unclear: Whether to modify `prompt_user_blocking()` to accept a timeout, or handle it in the RunManager.
   - Recommendation: Add `timeout: float | None = None` parameter to `prompt_user_blocking()`. If `Event.wait(timeout)` returns False, raise a `PromptTimeoutError` that the runner catches and reports as a failed step.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/test_api.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 | GET /routines lists routines | unit | `python -m pytest tests/test_api.py::test_list_routines -x` | Wave 0 |
| API-02 | GET /routines/{id} returns metadata | unit | `python -m pytest tests/test_api.py::test_get_routine -x` | Wave 0 |
| API-03 | POST /routines/{id}/run starts run | unit | `python -m pytest tests/test_api.py::test_start_run -x` | Wave 0 |
| API-04 | GET /runs/{run_id}/status returns status | unit | `python -m pytest tests/test_api.py::test_run_status -x` | Wave 0 |
| API-05 | POST /runs/{run_id}/respond unblocks prompt | unit | `python -m pytest tests/test_api.py::test_respond_prompt -x` | Wave 0 |
| API-06 | GET /runs/{run_id}/screenshot returns PNG | unit | `python -m pytest tests/test_api.py::test_screenshot -x` | Wave 0 |
| API-07 | POST /runs/{run_id}/abort triggers pause | unit | `python -m pytest tests/test_api.py::test_abort_run -x` | Wave 0 |
| API-08 | MCP tools generated with ocsd_ prefix | integration | `python -m pytest tests/test_api.py::test_mcp_tools -x` | Wave 0 |
| API-09 | Server starts as daemon thread | unit | `python -m pytest tests/test_api.py::test_daemon_server -x` | Wave 0 |
| SEC-01 | Scanner flags suspicious patterns | unit | `python -m pytest tests/test_api.py::test_scanner_exists -x` | Wave 0 |
| SEC-02 | Routine files are human-readable JSON | unit | Already validated by test_routine_format.py | Exists |
| SEC-03 | No cloud calls, localhost binding | unit | `python -m pytest tests/test_api.py::test_localhost_binding -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_api.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_api.py` -- covers API-01 through API-09, SEC-01, SEC-03
- [ ] FastAPI TestClient usage: `from fastapi.testclient import TestClient` (no additional install needed)
- [ ] Mock `run_routine`, `list_routines`, `Routine.load`, `screenshot_full` in tests

## Sources

### Primary (HIGH confidence)
- `api/server.py` -- existing server code being replaced (read directly)
- `routine/runner.py` -- run_routine(), RunEvent, RunResult, preflight_check (read directly)
- `core/executor.py` -- prompt_user_blocking(), respond_to_prompt(), threading.Event mechanism (read directly)
- `routine/discovery.py` -- list_routines(), RoutineInfo (read directly)
- `routine/format.py` -- Routine.load(), Routine.to_dict() (read directly)
- `core/config.py` -- get_config(), _DEFAULTS (read directly)
- `hub/scanner.py` -- scan_skill(), ScanResult (read directly)
- `cli/app.py` -- Typer app structure for adding serve command (read directly)

### Secondary (MEDIUM confidence)
- [fastapi-mcp PyPI](https://pypi.org/project/fastapi-mcp/) -- version 0.4.0, installation
- [fastapi-mcp GitHub](https://github.com/tadata-org/fastapi_mcp) -- basic usage pattern
- [fastapi-mcp docs: tool naming](https://fastapi-mcp.tadata.com/configurations/tool-naming.md) -- operation_id = tool name
- [fastapi-mcp docs: customization](https://fastapi-mcp.tadata.com/configurations/customization.md) -- constructor params, include/exclude
- [fastapi-mcp docs: transport](https://fastapi-mcp.tadata.com/advanced/transport.md) -- mount() vs mount_sse(), paths
- [uvicorn signal handler discussion](https://github.com/fastapi/fastapi/issues/650) -- daemon thread pattern

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in pyproject.toml except fastapi-mcp; fastapi-mcp verified on PyPI at 0.4.0
- Architecture: HIGH -- all integration points verified by reading existing source code directly
- Pitfalls: HIGH -- signal handler conflict, race conditions, and timeout issues are well-documented patterns

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable domain, low churn rate)
