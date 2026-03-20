---
phase: 07-run-flow
plan: 03
subsystem: routine-replay
tags: [runner, cascade, loop, run-log, screenshot-annotation, event-driven]

requires:
  - phase: 07-01
    provides: "locate_element_from_step cascade, ConditionChecker"
provides:
  - "run_routine() step-sequential replay engine"
  - "5-stage failure cascade (retry, region, full-screen, LiteLLM, abort)"
  - "Run log directory management with self-cleaning"
  - "Screenshot annotation for post-mortem debugging"
  - "RunEvent enum for overlay integration"
affects: [07-04, 08-api, 09-terminal]

tech-stack:
  added: []
  patterns: [event-callback protocol, 5-stage failure cascade, never-blind-click guarantee]

key-files:
  created:
    - routine/runner.py
    - routine/run_log.py
    - tests/test_routine_runner.py
  modified: []

key-decisions:
  - "n_iterations loop exit handled directly in loop counter (not ConditionChecker) to avoid counter reset per iteration"
  - "validate_action imported lazily inside step loop to avoid hard dependency on mapper.validator at module level"
  - "Loop body steps that fail locate are skipped with warning rather than aborting the entire loop"

patterns-established:
  - "RunEvent enum + RunCallback protocol for decoupled overlay/UI integration"
  - "Never-blind-click: abort with annotated failure screenshot when all cascade stages fail"
  - "Self-cleaning run dirs: keep 5 success + 10 failed, prune oldest"

requirements-completed: [RUN-01, RUN-03, RUN-04, RUN-05, RUN-06, RUN-07, RUN-08]

duration: 8min
completed: 2026-03-20
---

# Phase 7 Plan 3: Routine Runner Summary

**Step-sequential replay engine with 5-stage failure cascade, loop inline replay, and run logging with annotated screenshots**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-20T02:47:13Z
- **Completed:** 2026-03-20T02:55:02Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Built routine/runner.py (500+ lines) -- the core replay engine that loads routines, validates pre-flight, iterates steps with locate/execute/validate
- Implemented 5-stage failure cascade (retry at position, 25% region scan, full-screen, LiteLLM AI fallback, abort with annotated screenshot) -- never blind-clicks
- Created routine/run_log.py for run directory management, screenshot annotation, result saving, and self-cleaning (prune old runs)
- 20 passing tests covering preflight, dispatch, failure cascade, loops, events, screenshot events, and never-blind-click guarantee

## Task Commits

1. **Task 1: Create routine/run_log.py** - `f9595f1` (feat)
2. **Task 2: Create routine/runner.py** - `b18f44a` (feat)
3. **Task 3: Create tests for routine runner** - `676bbfe` (test)

## Files Created/Modified
- `routine/run_log.py` - Run directory creation, screenshot annotation, result saving, self-cleaning pruning
- `routine/runner.py` - Step-sequential replay engine with run_routine(), failure cascade, dispatch, loop handling
- `tests/test_routine_runner.py` - 20 comprehensive tests with full mocking

## Decisions Made
- n_iterations loop exit handled directly in loop counter rather than creating a new ConditionChecker per iteration (which would reset the internal counter)
- validate_action imported lazily inside the step loop to avoid hard module-level dependency on mapper.validator
- Loop body steps that fail to locate are skipped with a warning rather than aborting the entire loop

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed n_iterations counter reset in loop handler**
- **Found during:** Task 3 (test_loop_step_execution)
- **Issue:** Creating a new ConditionChecker per iteration reset the n_iterations counter, causing loops to exit after 1 iteration instead of N
- **Fix:** Handle n_iterations counting directly in the while loop without ConditionChecker
- **Files modified:** routine/runner.py
- **Verification:** test_loop_step_execution passes with correct iteration count
- **Committed in:** 676bbfe (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential correctness fix for loop iteration counting. No scope creep.

## Issues Encountered
- Test parameter naming conflicted with pytest fixtures when using @patch decorators with replacement values -- resolved by switching to context-manager style patching

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Runner is ready for Plan 04 (ReplayOverlayAdapter) to wire RunEvent callbacks to overlay widgets
- run_routine() provides the callback protocol for camera flash, target highlight, and status badge integration

---
*Phase: 07-run-flow*
*Completed: 2026-03-20*
