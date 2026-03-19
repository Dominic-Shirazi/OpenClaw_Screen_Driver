---
phase: 06-action-types
plan: 03
subsystem: recording
tags: [toolbar, record-phase, step-format, click-drag, look-here, pipeline-bridge]

# Dependency graph
requires:
  - phase: 06-01
    provides: "Action type step format extensions (type, scroll)"
  - phase: 06-02
    provides: "Executor functions (select_all_extract, press_enter)"
provides:
  - "6-button RECORDING toolbar (Look Here, Add Wait, Add Loop, Add Prompt)"
  - "5 new RecordPhase states for non-click action flows"
  - "Look Here region-drag capture pipeline"
  - "click_drag two-bbox capture flow (source + drag target)"
  - "Step format extensions for all 12 action types"
  - "drag_target_ready signal on PipelineBridge"
affects: [06-04, 06-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dynamic toolbar width per mode (button count * 70 + 40)"
    - "Look Here flow: AWAITING_REGION_DRAG -> CAPTURING -> VLM_ANALYZING (skip detection)"
    - "click_drag two-pass capture: tag confirm -> AWAITING_DRAG_TARGET -> second selection -> COUNTDOWN"

key-files:
  created: []
  modified:
    - recorder/overlay/toolbar_panel.py
    - recorder/overlay/record_phase.py
    - recorder/record_session.py
    - routine/format.py
    - recorder/overlay/pipeline_bridge.py
    - tests/test_action_types.py
    - tests/test_record_phase.py
    - tests/test_toolbar_panel.py

key-decisions:
  - "Dynamic toolbar width computed per mode to accommodate variable button counts"
  - "Look Here skips detection entirely (no AI bbox needed for user-drawn regions)"
  - "click_drag transitions to AWAITING_DRAG_TARGET after tag confirm, before countdown"
  - "Wait/loop/prompt handlers are stubs only in Plan 03 (Plans 04/05 own implementation)"

patterns-established:
  - "Non-click actions enter pipeline via toolbar quick-add buttons, not click/drag gestures"
  - "Two-pass capture pattern for click_drag: source bbox from normal flow, target bbox from second selection"

requirements-completed: [ACT-04, ACT-06, ACT-07]

# Metrics
duration: 4min
completed: 2026-03-19
---

# Phase 6 Plan 3: Toolbar Quick-Add Buttons and Action Flow Extensions Summary

**6-button RECORDING toolbar with Look Here region-drag flow, click_drag two-bbox capture, and step format extensions for all 12 action types**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-19T21:52:55Z
- **Completed:** 2026-03-19T21:57:07Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Extended RECORDING toolbar from 2 to 6 buttons with dynamic width computation
- Implemented Look Here flow: button -> AWAITING_REGION_DRAG -> region drag -> skip detection -> VLM -> tag dialog
- Implemented click_drag two-bbox capture: normal capture -> tag confirm -> AWAITING_DRAG_TARGET -> second selection -> countdown
- Extended build_v1_step for click_drag, read, snip_and_search, select_all_extract, prompt_user, wait, and loop action types
- Added 5 new RecordPhase states and drag_target_ready PipelineBridge signal

## Task Commits

Each task was committed atomically:

1. **Task 1: Toolbar buttons, new RecordPhases, and step format extensions** - `f0c3c9a` (feat)
2. **Task 2: RecordSession handlers for Look Here, click_drag target, and toolbar dispatch stubs** - `c75cf58` (feat)

## Files Created/Modified
- `recorder/overlay/toolbar_panel.py` - Added 4 quick-add buttons, dynamic width per mode
- `recorder/overlay/record_phase.py` - Added 5 new RecordPhase enum members
- `routine/format.py` - Extended build_v1_step for 7 more action types
- `recorder/overlay/pipeline_bridge.py` - Added drag_target_ready signal
- `recorder/record_session.py` - Added Look Here, click_drag target, and stub handlers
- `tests/test_action_types.py` - Added 7 new tests for action type step formats
- `tests/test_record_phase.py` - Updated for 15 phases and 7 signals
- `tests/test_toolbar_panel.py` - Updated for dynamic toolbar width

## Decisions Made
- Dynamic toolbar width computed per mode (max(250, 70 * btn_count + 40)) instead of fixed width
- Look Here skips detection entirely -- user-drawn regions don't need AI bbox refinement
- click_drag routes to AWAITING_DRAG_TARGET after tag confirm, separate from normal countdown path
- Wait/loop/prompt handlers are stubs only (log messages), full implementation deferred to Plans 04/05

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated existing tests for new member/signal counts**
- **Found during:** Task 1
- **Issue:** test_record_phase.py hardcoded 10 members and 6 signals, test_toolbar_panel.py hardcoded width 250.0
- **Fix:** Updated member count to 15, signal count to 7, width to 460.0 (dynamic)
- **Files modified:** tests/test_record_phase.py, tests/test_toolbar_panel.py
- **Verification:** All 71 tests pass
- **Committed in:** f0c3c9a (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Necessary test updates for new enum members and dynamic width. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Toolbar buttons wired and ready for Plan 04 (wait/prompt mini-dialogs) and Plan 05 (loop definition)
- click_drag flow complete end-to-end (capture, tag, target, countdown, dry-run)
- Look Here flow complete for observation actions (read, snip_and_search)

---
*Phase: 06-action-types*
*Completed: 2026-03-19*
