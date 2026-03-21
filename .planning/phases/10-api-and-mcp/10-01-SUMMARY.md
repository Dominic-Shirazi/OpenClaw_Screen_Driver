---
phase: 10-api-and-mcp
plan: 01
subsystem: api
tags: [fastapi, uvicorn, pydantic, rest-api, daemon-thread]

# Dependency graph
requires:
  - phase: 05-routine-format
    provides: Routine dataclass, discovery, format loading
  - phase: 06-action-types
    provides: prompt_user_blocking for API prompt flow
provides:
  - FastAPI app with routine-centric GET endpoints
  - RunManager for thread-safe run state tracking
  - Daemon thread lifecycle for uvicorn alongside Qt
  - Pydantic request/response models for all endpoints
affects: [10-02-run-endpoints, 10-03-mcp-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [daemon-thread-uvicorn, no-signal-server, run-manager-singleton]

key-files:
  created:
    - api/run_manager.py
    - api/lifecycle.py
    - api/__init__.py
    - tests/test_api_server.py
  modified:
    - api/server.py
    - core/config.py

key-decisions:
  - "RunManager uses threading.Lock with single active run constraint"
  - "_NoSignalServer subclass prevents uvicorn from stealing Qt signal handlers"
  - "Config defaults updated to 127.0.0.1:8420 -- localhost-only binding for security"
  - "Stub endpoints return 501 for Plan 02 implementation"

patterns-established:
  - "RunManager singleton pattern: module-level manager = RunManager() in server.py"
  - "Exception handler pattern: custom exceptions mapped to structured JSON error responses"
  - "Operation IDs on all endpoints for MCP tool discovery (ocsd_health, ocsd_list_routines, etc.)"

requirements-completed: [API-01, API-02, API-09, SEC-03]

# Metrics
duration: 3min
completed: 2026-03-21
---

# Phase 10 Plan 01: API Foundation Summary

**FastAPI routine-centric REST API with RunManager, daemon lifecycle, and localhost-only binding on port 8420**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-21T00:44:22Z
- **Completed:** 2026-03-21T00:47:04Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Rewrote api/server.py from skill-based to routine-centric endpoints (GET /health, /routines, /routines/{id})
- Created thread-safe RunManager with full lifecycle tracking (start/step/wait/pause/complete/fail)
- Built daemon thread lifecycle with _NoSignalServer to coexist with Qt event loop
- Updated config defaults from 0.0.0.0:8742 to 127.0.0.1:8420 with prompt_timeout_s
- All stub endpoints (run, status, respond, screenshot, abort) return 501 for Plan 02

## Task Commits

Each task was committed atomically:

1. **Task 1: RunManager, Pydantic models, config update, and daemon lifecycle** - `d3211c2` (feat)
2. **Task 2: Rewrite api/server.py with routine-centric endpoints and tests** - `27aa4e1` (feat)

## Files Created/Modified
- `api/run_manager.py` - Thread-safe RunManager with RunStatus enum, ActiveRun dataclass, and error classes
- `api/lifecycle.py` - _NoSignalServer and start_api_daemon for daemon thread uvicorn
- `api/__init__.py` - Package init
- `api/server.py` - Full rewrite: FastAPI app with routine-centric endpoints and Pydantic models
- `core/config.py` - API defaults updated to 127.0.0.1:8420 with prompt_timeout_s
- `tests/test_api_server.py` - TestClient tests for health, routines, 404s, localhost binding, and 501 stubs

## Decisions Made
- RunManager uses threading.Lock with single active run constraint (no concurrent runs)
- _NoSignalServer subclass prevents uvicorn from stealing Qt signal handlers
- Config defaults updated to 127.0.0.1:8420 for localhost-only security
- Stub endpoints return 501 (not 404) to indicate planned but not yet implemented
- Test for localhost binding uses tmp_path to avoid config.yaml override interference

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed localhost binding test using tmp_path**
- **Found during:** Task 2 (test execution)
- **Issue:** config.yaml on disk overrides _DEFAULTS, causing test to see old 0.0.0.0 binding
- **Fix:** Test uses load_config with nonexistent tmp_path to verify pure defaults
- **Files modified:** tests/test_api_server.py
- **Verification:** All 5 tests pass
- **Committed in:** 27aa4e1 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test fix ensures defaults are verified correctly regardless of local config.yaml.

## Issues Encountered
None beyond the test fix documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- API foundation complete with all GET endpoints working
- RunManager ready for Plan 02 to wire up POST /routines/{id}/run and run status endpoints
- Pydantic models for run requests/responses already defined
- Stub endpoints provide clear 501 placeholder for Plan 02 implementation

---
*Phase: 10-api-and-mcp*
*Completed: 2026-03-21*
