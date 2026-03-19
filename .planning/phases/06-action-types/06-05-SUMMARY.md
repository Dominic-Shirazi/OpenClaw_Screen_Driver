---
phase: 06-action-types
plan: 05
subsystem: recording
tags: [loop, mini-dialog, pyqt6, step-range, exit-condition, node-id-resolution]

requires:
  - phase: 06-action-types plan 02
    provides: ConditionChecker shared engine with n_iterations, element_appears
  - phase: 06-action-types plan 03
    provides: RecordSession with toolbar action handlers, step list, save routine
  - phase: 06-action-types plan 04
    provides: WaitDialog/PromptDialog patterns, controller show/hide methods
provides:
  - LoopDialog QGraphicsObject for loop definition with step range selector and exit conditions
  - RecordSession loop handler creating loop steps with body_step_indices
  - resolve_loop_node_ids for converting step indices to stable node_id references at save time
  - Controller show_loop_dialog/hide_loop_dialog lifecycle management
affects: [07-replay-engine, 08-routine-management]

tech-stack:
  added: []
  patterns: [loop-body-as-node-id-references, save-time-resolution-of-indices]

key-files:
  created: []
  modified:
    - recorder/overlay/mini_dialogs.py
    - recorder/record_session.py
    - recorder/overlay/controller.py
    - routine/format.py
    - tests/test_condition_engine.py
    - tests/test_action_types.py

key-decisions:
  - "Loop body stored as body_step_indices temporarily, resolved to body_step_node_ids at save time via resolve_loop_node_ids"
  - "LoopDialog instantiated directly in controller (not via generic show_mini_dialog) due to extra steps parameter"

patterns-established:
  - "Save-time resolution: store temporary indices during recording, convert to stable IDs at save"
  - "LoopDialog follows same frosted glass visual pattern as WaitDialog/PromptDialog"

requirements-completed: [ACT-11]

duration: 5min
completed: 2026-03-19
---

# Phase 6 Plan 5: Add Loop Action Summary

**LoopDialog with step range selector, 4 exit conditions, node_id body references resolved at save time via resolve_loop_node_ids**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-19T22:08:01Z
- **Completed:** 2026-03-19T22:13:13Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- LoopDialog mini-dialog with from/to step range selector and 4 exit condition types (n_iterations, element_appears, text_matches, prompt_user)
- Mandatory max_iterations safety limit on all non-N-iterations conditions
- Loop body uses stable node_id references via save-time resolution (resolve_loop_node_ids)
- Controller show_loop_dialog/hide_loop_dialog with direct LoopDialog instantiation for extra steps parameter
- Nested loop serialization verified without circular references

## Task Commits

Each task was committed atomically:

1. **Task 1: LoopDialog mini-dialog widget** - `a086bcd` (feat)
2. **Task 2: Wire loop handler, controller, format, tests** - `f9ebee6` (feat)

## Files Created/Modified
- `recorder/overlay/mini_dialogs.py` - Added LoopDialog with step range selector, exit conditions, validation
- `recorder/record_session.py` - Replaced _handle_add_loop stub with full loop flow, added _on_loop_configured/_on_loop_dismissed
- `recorder/overlay/controller.py` - Added show_loop_dialog/hide_loop_dialog methods
- `routine/format.py` - Added resolve_loop_node_ids for save-time index-to-node_id conversion
- `tests/test_condition_engine.py` - Added LoopDialog signal tests, loop definition validation
- `tests/test_action_types.py` - Updated loop step format test, added resolve_loop_node_ids and nested loop tests

## Decisions Made
- Loop body stored as body_step_indices during recording, resolved to body_step_node_ids at save time -- ensures stability for Phase 8 editing
- LoopDialog instantiated directly in controller rather than via generic show_mini_dialog, because it requires the extra steps parameter
- No loop-level dry-run per established user decision (individual steps already tested)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 12 action types now have capture and format support
- Phase 6 complete -- ready for Phase 7 replay engine
- Loop step format with body_step_node_ids is ready for graph traversal during replay

## Self-Check: PASSED

All 7 files verified present. Both commits (a086bcd, f9ebee6) confirmed in git log. All 15 acceptance criteria grep checks returned positive counts.

---
*Phase: 06-action-types*
*Completed: 2026-03-19*
