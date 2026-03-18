---
phase: 04-record-flow
plan: 02
subsystem: ui
tags: [pyqt6, overlay, bbox, countdown, abort-panel, card-glow, click-through]

# Dependency graph
requires:
  - phase: 04-record-flow/01
    provides: "RecordPhase enum, CountdownWidget, AbortPanel, PipelineBridge contracts"
  - phase: 03-overlay-hud-panels
    provides: "TagDialogPanel, ToolbarPanel, card glow painting"
provides:
  - "8 new controller methods for recording pipeline visual control"
  - "Interactive 8-handle bbox editing with accept/reject"
  - "Countdown, abort, flash, click-through, card glow pulse on view"
  - "TagDialogPanel.set_glow_pulsing for loading indicator"
affects: [04-record-flow/03, 04-record-flow/04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Controller delegates new methods to view (existing pattern extended)"
    - "TagDialogPanel glow pulsing via sine-wave brightness oscillation"
    - "BboxLayer 8-handle editing with accept/reject_edit lifecycle"

key-files:
  created:
    - tests/test_overlay_extensions.py
  modified:
    - recorder/overlay/controller.py
    - recorder/overlay/view.py
    - recorder/overlay/bbox_layer.py
    - recorder/overlay/tag_dialog_panel.py

key-decisions:
  - "BboxLayer handles start non-movable; enable_editing() activates them"
  - "_handle_close no longer auto-closes on save; RecordSession decides when to close"
  - "Card glow pulse uses sine oscillation (brightness 0.5-2.0) on TagDialogPanel glow_phase"
  - "_corner_positions aliased to _handle_positions for backward compat with morph code"

patterns-established:
  - "Interactive editing lifecycle: enable_editing -> get_edited_rect -> accept_edit/reject_edit"
  - "Card glow pulsing as loading indicator pattern"

requirements-completed: [REC-04, REC-05, REC-06, REC-09]

# Metrics
duration: 10min
completed: 2026-03-18
---

# Phase 4 Plan 02: Overlay Extensions Summary

**Extended overlay controller with 8 recording pipeline methods, 8-handle bbox editing, and card glow pulse loading indicator**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-18T21:22:42Z
- **Completed:** 2026-03-18T21:33:03Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Extended OverlayView with countdown, abort confirm, success flash, click-through toggle, and card glow pulse methods
- Extended BboxLayer from 4 corner handles to 8 handles (corners + edge midpoints) with interactive editing lifecycle
- Extended OverlayController with 8 new public methods wired to view
- Updated _handle_close to not auto-close overlay on save (lets RecordSession decide)
- Added set_glow_pulsing to TagDialogPanel for loading indicator during detection/VLM analysis
- Created 13 passing tests for all new functionality

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend OverlayView and BboxLayer** - `64432b8` (feat)
2. **Task 2: Extend Controller with API methods and tests** - `bc58a87` (feat)

## Files Created/Modified
- `recorder/overlay/view.py` - Added countdown, abort, flash, click-through, card glow pulse methods
- `recorder/overlay/bbox_layer.py` - Extended to 8 handles with interactive editing lifecycle
- `recorder/overlay/controller.py` - 8 new public methods, updated _handle_close
- `recorder/overlay/tag_dialog_panel.py` - Added set_glow_pulsing and pulse animation
- `tests/test_overlay_extensions.py` - 13 tests for all extensions

## Decisions Made
- BboxLayer handles start non-movable; enable_editing() activates them (avoids accidental drags during display)
- _handle_close no longer auto-closes on save; RecordSession decides when to close (separation of concerns)
- Card glow pulse uses sine-wave brightness oscillation between 0.5 and 2.0 for smooth loading indicator
- _corner_positions aliased to _handle_positions for backward compatibility with existing morph animation code

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added set_glow_pulsing to TagDialogPanel**
- **Found during:** Task 1 (View extension)
- **Issue:** Plan referenced `self._tag_dialog.set_glow_pulsing()` but the method didn't exist on TagDialogPanel
- **Fix:** Added `set_glow_pulsing(enabled: bool)` method and `_glow_pulsing` instance variable to TagDialogPanel, with sine-wave brightness oscillation in `_tick()`
- **Files modified:** recorder/overlay/tag_dialog_panel.py
- **Verification:** Tests pass, import succeeds
- **Committed in:** 64432b8 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Essential method needed for view to delegate card glow pulsing to tag dialog. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All overlay visual capabilities needed by RecordSession are now available
- Controller can toggle click-through, show countdown/abort, flash success, pulse card glow
- BboxLayer supports full interactive editing with 8 drag handles and accept/reject lifecycle
- Ready for Plan 03 (RecordSession state machine) and Plan 04 (pipeline wiring)

---
*Phase: 04-record-flow*
*Completed: 2026-03-18*
