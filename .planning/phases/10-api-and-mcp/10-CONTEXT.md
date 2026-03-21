# Phase 10: API and MCP - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Expose OCSD's routine operations via a local REST API and MCP-compatible tool layer. Agents and external tools can discover, run, monitor, pause, and abort routines via HTTP. The existing api/server.py (skill-based) is replaced entirely with routine-centric endpoints. fastapi-mcp auto-generates MCP tools from FastAPI routes. Security scanning stays as-is (V2 Hub concern). Does NOT build new action types, routine management features, or TUI changes.

</domain>

<decisions>
## Implementation Decisions

### Server lifecycle & threading (API-09)
- **Two start modes**: (1) Auto-start as daemon thread when QApplication launches (record/run via TUI/CLI). (2) Standalone `ocsd serve` command for headless/agent-only use
- **Daemon thread**: `uvicorn.Server` runs with `install_signal_handlers=False` in a daemon thread. Qt owns signal handling. Thread dies when main process exits
- **`ocsd serve` command**: Preloads AI models (OmniParser, CLIP, VLM connectivity check) before signaling ready. Prints `OCSD API ready at http://127.0.0.1:{port}` then blocks. Ctrl+C for graceful shutdown. No TUI loading screen
- **Single active run**: Only one routine can execute at a time. POST /run returns 409 CONFLICT with error code `RUN_ALREADY_ACTIVE` if a routine is already executing
- **Port**: Default 8420, configurable via `api.port` in ocsd.yaml
- **Localhost only**: Bind to 127.0.0.1 exclusively. No auth, no API keys. V1 is local-only

### Endpoint design (API-01 through API-07)
- **Routine IDs**: Use folder name as ID in URLs (e.g., `/routines/google-search`). Matches filesystem layout
- **Endpoints**:
  - `GET /health` — `{"status": "ok", "version": "1.0", "models_loaded": true/false}`
  - `GET /routines` — List all routines. Optional `?tag=X` filter
  - `GET /routines/{id}` — Routine metadata + steps
  - `POST /routines/{id}/run` — Start execution. Body: `{"params": {"key": "value"}, "show_overlay": true}`. Returns `{"run_id": "..."}`. Params work like CLI `--param` — skip interactive variable prompts
  - `GET /runs/{run_id}/status` — Summary: status (running/waiting/paused/completed/failed), current_step, total_steps, error message if failed, prompt question if waiting
  - `POST /runs/{run_id}/respond` — Answer a prompt_user step. Body: `{"response": "text"}`. Unblocks the threading.Event in executor
  - `GET /runs/{run_id}/screenshot` — Returns PNG binary (image/png content type). Current screen state
  - `POST /runs/{run_id}/abort` — Triggers pause (see abort behavior below)
- **Screenshot format**: Raw PNG binary with `Content-Type: image/png`
- **Run status**: Summary only — status enum, current step index, total steps, error/prompt text. No per-step result array

### Error responses
- **Structured error codes**: `{"code": "ROUTINE_NOT_FOUND", "message": "No routine named 'foo'", "suggestion": "Run ocsd list to see available routines"}`
- **Standard codes**: `ROUTINE_NOT_FOUND`, `RUN_NOT_FOUND`, `RUN_ALREADY_ACTIVE`, `INVALID_PARAMS`, `RUN_FAILED`, `SERVER_ERROR`

### Prompt timeout
- **Configurable**: Default 5 minutes for prompt_user steps. Configurable per-run via POST body `{"prompt_timeout_s": 300}` and globally via `api.prompt_timeout_s` in ocsd.yaml. Times out to 'failed' state if no response. Caller can always /abort

### Abort / pause behavior
- **Abort triggers pause**: POST /abort sets an abort flag. Runner checks flag between steps AND mid-action where possible
- **Pause UX**: Shimmer turns green (READY state), status badge shows "Paused". User presses Enter to resume execution, Esc to cancel and close the program entirely
- **Status during pause**: Run status returns `"paused"` state. Agent can poll and see it's paused
- **Resume via keyboard only**: Enter/Esc are local keyboard inputs, not API-driven. The API triggers the pause; the human at the machine decides what happens next

### Old API migration
- **Replace entirely**: Delete old skill-based endpoints from api/server.py. Rewrite with routine-centric endpoints. Old skill format is pre-routine era, no longer relevant

### MCP tool mapping (API-08)
- **fastapi-mcp**: Use fastapi-mcp package to auto-generate MCP tools from FastAPI routes. One-line setup mounting at `/mcp` path on the same uvicorn server
- **Tool name prefix**: `ocsd_` prefix on all tool names (e.g., `ocsd_list_routines`, `ocsd_run_routine`, `ocsd_get_run_status`). Set via FastAPI operation_id on routes
- **Tool descriptions**: Minimal effort — basic docstrings on FastAPI routes. Polish later based on agent feedback. fastapi-mcp generates tool schemas from Pydantic models automatically
- **Transport**: SSE via HTTP at `http://127.0.0.1:{port}/mcp`. Same server, same port. No separate stdio transport in V1

### Security (SEC-01, SEC-02, SEC-03)
- **Scanner stays as-is**: hub/scanner.py keeps its current skill-format scanning. V1 routine scanning deferred to V2 Hub
- **Scan timing**: Scanning happens on routine install/download from Hub. Locally created routines are trusted. No per-run scanning gate
- **Localhost binding**: Primary security mechanism. No remote access in V1
- **No auth**: No API keys, no tokens. If you can reach localhost, you're the user
- **Routine auditability**: JSON is human-readable by design (SEC-02). No additional auditability features needed

### Claude's Discretion
- Exact uvicorn.Server configuration for daemon thread mode
- How to implement the pause/resume keyboard listener during paused state
- Run ID generation scheme (UUID, timestamp, etc.)
- How fastapi-mcp operation_id mapping works for the ocsd_ prefix
- Model preloading sequence in `ocsd serve` (which models, timeout, retry)
- How to wire the abort flag into the runner's step loop
- CORS configuration (if any) for localhost

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### API requirements
- `.planning/REQUIREMENTS.md` — API-01 through API-09 and SEC-01 through SEC-03 define all requirements
- `.planning/ROADMAP.md` — Phase 10 success criteria (5 criteria that must be TRUE)

### Existing API server (being replaced)
- `api/server.py` — Old skill-based FastAPI server. Delete and rewrite with routine-centric endpoints

### Routine runner (wire to API)
- `routine/runner.py` — `run_routine()`, `RunEvent`, `RunResult`, `preflight_check()`, `PreflightError`. API /run endpoint calls `run_routine()` on a background thread
- `routine/replay_overlay.py` — `ReplayOverlayAdapter` for overlay during API-triggered replay

### Prompt user mechanism
- `core/executor.py` — `prompt_user_blocking()` uses module-level `threading.Event` (`_respond_event`). API /respond endpoint sets response text and signals the event

### Routine operations
- `routine/discovery.py` — `list_routines()`, `get_routine_dir()` for /routines endpoint
- `routine/format.py` — `Routine.load()` for /routines/{id} endpoint
- `routine/management.py` — Management functions (not directly needed for API but reference)

### Hub scanner (kept as-is)
- `hub/scanner.py` — `scan_skill()` scans skill dicts. Not adapted for v1 routines in this phase

### Overlay state (for pause UX)
- `recorder/overlay/state.py` — OverlayState enum (needs pause-related state or reuse READY)
- `recorder/overlay/shimmer_layer.py` — ShimmerLayer with `set_state()` for green shimmer during pause

### Core pipeline
- `core/config.py` — `get_config()` for ocsd.yaml settings (api.port, api.prompt_timeout_s)
- `core/capture.py` — `screenshot_full()` for /screenshot endpoint

### CLI entry point
- `cli/app.py` or equivalent — Add `ocsd serve` subcommand alongside existing commands

### Prior phase context
- `.planning/phases/06-action-types/06-CONTEXT.md` — prompt_user mechanism, threading.Event for /respond
- `.planning/phases/07-run-flow/07-CONTEXT.md` — Overlay optional, run logs, failure cascade, pre-flight validation
- `.planning/phases/09-tui-and-cli/09-CONTEXT.md` — Typer CLI framework, --param flag, TUI-to-Qt handoff

### Project constraints
- `.planning/PROJECT.md` — "FastAPI uvicorn runs as daemon thread with install_signal_handlers=False", localhost-only, no cloud at runtime
- `CLAUDE.md` — Type hints, logging, thread safety, venv isolation, cross-platform awareness

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `routine/runner.py:run_routine()`: Core replay engine. API /run endpoint wraps this in a background thread
- `routine/runner.py:preflight_check()`: Validates routine assets before execution. API calls this before starting run
- `core/executor.py:prompt_user_blocking()`: threading.Event mechanism ready for /respond. API sets `_prompt_response_text` and signals `_respond_event`
- `routine/discovery.py:list_routines()`: Returns `RoutineInfo` objects. Direct mapping to GET /routines response
- `routine/format.py:Routine.load()`: Load routine for GET /routines/{id}
- `core/capture.py:screenshot_full()`: Returns numpy array. Convert to PNG bytes for /screenshot
- `recorder/overlay/shimmer_layer.py`: Green shimmer for READY state — reuse for pause UX

### Established Patterns
- PipelineBridge signals for thread-safe background-to-main delivery
- `run_routine(routine, callback, params)` with RunEvent callback for progress
- `prompt_user_blocking()` blocks on threading.Event until response arrives
- `get_config()` for YAML config access with sensible defaults
- Typer app with lazy imports for fast CLI startup

### Integration Points
- `api/server.py` — Full rewrite from skill-based to routine-centric endpoints
- `pyproject.toml` — `fastapi-mcp` added to `[api]` optional dependency group
- `core/config.py` — Add `api` config section (port, prompt_timeout_s)
- CLI app — Add `ocsd serve` subcommand
- `routine/runner.py` — Add abort flag check mechanism (threading.Event or similar)
- `recorder/overlay/controller.py` — Pause/resume handling for abort-triggered pause

</code_context>

<specifics>
## Specific Ideas

- Abort = pause, not kill. Shimmer goes green, badge says "Paused", Enter to resume, Esc to cancel. This gives the human at the machine final say — the agent can request a stop, but the user decides whether to actually terminate or continue
- `ocsd serve` preloads models so first routine run is fast. This is the agent-oriented entry point — no TUI, just API
- Both modes (auto-start with Qt + standalone serve) cover all use cases: interactive users get API alongside overlay, agents get headless API
- Run status is summary-only (not per-step detail). Agents poll for completion, not for play-by-play. Keeps the API surface small
- Error responses include suggestion field — helps agents self-correct ("Run ocsd list to see available routines")
- Scanner stays as-is — V1 Hub is local-only with bundled demos. Real scanning happens at install/download time when V2 Hub lands

</specifics>

<deferred>
## Deferred Ideas

- V1 routine format scanning in hub/scanner.py — V2, when Hub enables routine sharing
- Streaming run events via SSE/WebSocket — V2, polling is sufficient for V1
- Per-step detail in run status — V2, summary status covers agent needs
- API authentication (API keys, OAuth) — V2, when remote access is enabled
- stdio MCP transport for local Claude Code integration — V2
- Daemon mode for `ocsd serve` (backgrounding, PID file, --stop) — V2
- Resume paused run via API (not just keyboard) — V2

</deferred>

---

*Phase: 10-api-and-mcp*
*Context gathered: 2026-03-20*
