---
phase: 09-tui-and-cli
plan: 03
subsystem: cli
tags: [typer, rich, tui, variable-collection, sequential-gate, qt-handoff]

requires:
  - phase: 09-tui-and-cli
    provides: "CLI app.py with Typer commands, tui.py with loading screen and menu, output.py helpers"
  - phase: 07-run-flow
    provides: "run_routine, RunResult"
  - phase: 05-routine-format
    provides: "Routine.load, Routine.save"
provides:
  - "Pre-run variable collection with --param injection"
  - "TUI-to-Qt sequential gate: Rich exits before QApplication created"
  - "Terminal minimize/restore around Qt overlay operations"
  - "main.py backward-compat redirect to cli.app:main"
affects: [10-api-and-mcp]

tech-stack:
  added: []
  patterns: [sequential-gate-pattern, variable-injection-via-temp-copy, minimize-restore-bracket]

key-files:
  created: []
  modified:
    - cli/output.py
    - cli/app.py
    - cli/tui.py
    - main.py
    - tests/test_cli.py
    - tests/test_tui.py

key-decisions:
  - "Variable injection creates a temporary routine copy rather than mutating the original"
  - "collect_variables scans input_spec.type=='variable' steps and prompts only for missing params"
  - "Qt-launching TUI commands (record, run, update) bracket with minimize/restore in finally blocks"

patterns-established:
  - "Sequential gate: TUI Rich Live exits completely before any Qt import"
  - "Variable collection: scan steps for input_spec variables, merge with --param flags"
  - "Temp routine copy: prepare_run creates disposable copy for variable injection"

requirements-completed: [TUI-06]

duration: 5min
completed: 2026-03-20
---

# Phase 09 Plan 03: TUI-to-Qt Handoff Summary

**Sequential TUI-to-Qt gate with pre-run variable collection, --param injection via temp routine copies, and main.py backward-compat redirect**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-20T22:41:32Z
- **Completed:** 2026-03-20T22:46:37Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Complete TUI-to-Qt sequential gate: Rich event loop exits before QApplication is created
- Pre-run variable collection scans routine steps and prompts only for missing params
- Terminal minimizes during overlay operations and restores after via finally blocks
- main.py replaced with thin 20-line redirect to cli.app:main
- 41 tests passing (8 new integration tests for gate, variables, and menu loop)

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire TUI-to-Qt handoff, variable collection, and main.py redirect** - `0bd675b` (feat)
2. **Task 2: Integration tests for sequential gate and variable collection** - `9ba70fb` (test)

## Files Created/Modified
- `cli/output.py` - Added collect_variables and prepare_run for variable injection
- `cli/app.py` - Wired collect_variables/prepare_run into run command, added minimize/restore to record and update
- `cli/tui.py` - Rewrote run_tui with proper sequential gate pattern, minimize/restore around Qt commands
- `main.py` - Replaced 348-line legacy entry point with 20-line redirect to cli.app:main
- `tests/test_cli.py` - Added 6 new tests: variable collection, prepare_run, main redirect
- `tests/test_tui.py` - Added 3 new tests: sequential gate, quit exits, menu loop back

## Decisions Made
- Variable injection via temp routine copy (prepare_run) rather than in-place mutation -- preserves original routine
- collect_variables strips {} from input_spec.value to get var_name, matching the template format
- TUI commands use finally blocks for restore_terminal to ensure cleanup even on error
- Fixed existing tests to mock collect_variables/prepare_run since run_command now calls them

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated existing test mocks for collect_variables**
- **Found during:** Task 1
- **Issue:** Existing run tests failed because run_command now calls collect_variables which tries to load routine.json from disk
- **Fix:** Added patch("cli.app.collect_variables") and patch("cli.app.prepare_run") to existing run tests
- **Files modified:** tests/test_cli.py
- **Committed in:** 0bd675b

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary fix to keep existing tests passing after behavioral change. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TUI-to-Qt handoff complete, sequential gate verified by test
- All CLI commands properly wired with variable collection and Qt lifecycle management
- Ready for Phase 10 (API and MCP)

---
*Phase: 09-tui-and-cli*
*Completed: 2026-03-20*
