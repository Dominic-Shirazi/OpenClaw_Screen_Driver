---
phase: 06-action-types
plan: 02
subsystem: core
tags: [condition-engine, clipboard, pyperclip, opencv, polling, vlm-fallback]

requires:
  - phase: 01-foundation
    provides: "core/capture.py screenshot_full(), core/vision.py analyze_crop_array()"
provides:
  - "ConditionChecker class with 6 condition types for wait/loop actions"
  - "select_all_extract executor function with clipboard + VLM fallback"
  - "pyperclip dependency for cross-platform clipboard access"
affects: [06-03, 06-04, 06-05, 07-locate-cascade]

tech-stack:
  added: [pyperclip]
  patterns: [condition-polling-engine, clipboard-with-vlm-fallback]

key-files:
  created: [core/conditions.py, tests/test_condition_engine.py]
  modified: [core/executor.py, pyproject.toml]

key-decisions:
  - "n_iterations uses internal counter (_iteration_count) instead of relying on max_iterations to return met=True"
  - "ConditionChecker is synchronous blocking on caller thread (expected background thread)"
  - "element_appears and text_matches are stubs pending Phase 7 locate cascade"

patterns-established:
  - "ConditionChecker pattern: construct with condition_type + params, call poll_until() on background thread"
  - "Clipboard-first with VLM fallback for text extraction"

requirements-completed: [ACT-08, ACT-10]

duration: 2min
completed: 2026-03-19
---

# Phase 6 Plan 2: Condition Engine and Select-All-Extract Summary

**Shared ConditionChecker engine with 6 condition types (fixed_timer, screen_change, n_iterations, element_appears, vlm_check, text_matches) plus select_all_extract clipboard reader with VLM fallback**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-19T21:47:57Z
- **Completed:** 2026-03-19T21:50:12Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- ConditionChecker class supports all 6 condition types (4 working, 2 stubbed for Phase 7)
- poll_until() loop with timeout, max_iterations, and cancel() safety mechanisms
- select_all_extract sends Ctrl+A/Ctrl+C, reads clipboard via pyperclip, falls back to VLM
- All 10 tests pass covering timer, screen change, iteration count, timeout, and clipboard flows

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `c5ec1aa` (test)
2. **Task 1 GREEN: Implementation** - `6b4ea87` (feat)

## Files Created/Modified
- `core/conditions.py` - ConditionChecker class with poll_until() and 6 condition type handlers
- `core/executor.py` - Added select_all_extract() function
- `pyproject.toml` - Added pyperclip>=1.8 to dependencies
- `tests/test_condition_engine.py` - 10 tests covering all behaviors

## Decisions Made
- n_iterations tracks an internal `_iteration_count` and returns `met=True` on Nth call, rather than relying solely on max_iterations overflow
- element_appears and text_matches are stubs returning False (Phase 7 provides the locate cascade)
- fixed_timer is exempt from positive-timeout requirement (it uses params.seconds for sleep)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ConditionChecker ready for wait action (Plan 03/04) and loop action (Plan 05)
- select_all_extract ready for dry-run dispatch wiring in Plan 01
- element_appears and text_matches stubs will be implemented when Phase 7 locate cascade is available

## Self-Check: PASSED

All files exist, all commits verified, all content claims confirmed.

---
*Phase: 06-action-types*
*Completed: 2026-03-19*
