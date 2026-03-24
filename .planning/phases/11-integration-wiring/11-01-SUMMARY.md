---
phase: 11-integration-wiring
plan: 01
subsystem: security
tags: [scanner, preflight, routine-runner, SEC-01]

requires:
  - phase: 08-hub-scanner
    provides: scan_skill() malicious pattern detection engine
  - phase: 07-replay-engine
    provides: run_routine() with preflight_check pipeline
provides:
  - scan_routine() v1 format adapter in hub/scanner.py
  - Security scan wired into run_routine() preflight (all entry points)
  - 8 tests covering adapter correctness and runner integration
affects: [routine-runner, cli, tui, api]

tech-stack:
  added: []
  patterns: [lazy-import-with-ImportError-fallback, format-adapter-delegation]

key-files:
  created:
    - tests/test_scanner.py
  modified:
    - hub/scanner.py
    - routine/runner.py
    - tests/test_routine_runner.py

key-decisions:
  - "scan_routine() is a thin adapter delegating to scan_skill() rather than duplicating detection logic"
  - "Scanner uses lazy import with ImportError fallback so runner works without hub module"
  - "Security scan runs after asset preflight but before execution start"

patterns-established:
  - "Format adapter pattern: new format -> legacy format -> existing engine"

requirements-completed: [SEC-01]

duration: 3min
completed: 2026-03-24
---

# Phase 11 Plan 01: Scanner-to-Runner Wiring Summary

**scan_routine() adapter maps v1 routine steps to scanner format; wired into run_routine() preflight so all entry points (CLI, TUI, API) block unsafe routines**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-24T22:42:45Z
- **Completed:** 2026-03-24T22:45:15Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- scan_routine() adapter in hub/scanner.py converts v1 steps to nodes/edges format and delegates to scan_skill()
- Security scanner wired into run_routine() preflight -- blocks execution with PREFLIGHT_FAILED when risk >= 0.5
- 6 unit tests for adapter (empty, URL, system path, combined risk, field mapping, keystroke)
- 2 integration tests for runner (blocks unsafe, allows safe)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add scan_routine() adapter and scanner tests** - `79404ed` (feat, TDD)
2. **Task 2: Wire scanner into run_routine() preflight and add runner scan tests** - `e8872fd` (feat)

## Files Created/Modified
- `hub/scanner.py` - Added scan_routine() adapter function after scan_skill()
- `routine/runner.py` - Added security scan block in run_routine() after asset preflight
- `tests/test_scanner.py` - 6 tests for scan_routine() adapter correctness
- `tests/test_routine_runner.py` - 2 tests for scanner integration in runner preflight

## Decisions Made
- scan_routine() is a thin adapter delegating to scan_skill() -- no duplicated detection logic
- Scanner uses lazy import with ImportError fallback so runner works without hub module
- Security scan runs after asset preflight (PREFLIGHT_OK) but before execution start
- save_run_result receives a dict (not RunResult dataclass) matching existing pattern

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed save_run_result call with dict instead of RunResult**
- **Found during:** Task 2 (Wire scanner into runner)
- **Issue:** Plan code passed RunResult dataclass to save_run_result which expects a dict
- **Fix:** Changed to pass dict matching the existing preflight failure pattern in runner.py
- **Files modified:** routine/runner.py
- **Verification:** test_run_routine_blocks_unsafe_routine passes
- **Committed in:** e8872fd (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential correctness fix. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SEC-01 requirement closed: all routine execution paths scan for malicious patterns before running
- Ready for 11-02 (RUN-09 integration wiring)

## Self-Check: PASSED

All 4 files verified on disk. Both commit hashes (79404ed, e8872fd) found in git log.

---
*Phase: 11-integration-wiring*
*Completed: 2026-03-24*
