---
phase: 01-overlay-foundation
plan: 02
subsystem: overlay
tags: [pyqt6, qgraphicsview, qgraphicsitem, layers, state-machine, click-through, hotkeys]

# Dependency graph
requires:
  - phase: 01-overlay-foundation/01
    provides: "OverlayState enum, DPI utils, platform helpers, capture_guard"
provides:
  - "OverlayView: QGraphicsView-based fullscreen overlay shell"
  - "BorderLayer: screen-edge coloured border"
  - "ClickCatcherLayer: invisible full-screen rect for mouse hit-testing"
  - "ModeIndicatorLayer: HUD text label with mode and hotkeys"
  - "BboxLayer: bounding box with corner handles"
  - "OverlayController: public API for overlay lifecycle and state management"
affects: [01-overlay-foundation/03, 02-recording-core]

# Tech tracking
tech-stack:
  added: []
  patterns: [layer-based-composition, lazy-import-for-circular-deps, callback-driven-controller]

key-files:
  created:
    - recorder/overlay/view.py
    - recorder/overlay/border_layer.py
    - recorder/overlay/click_catcher_layer.py
    - recorder/overlay/mode_indicator_layer.py
    - recorder/overlay/bbox_layer.py
    - recorder/overlay/controller.py
  modified:
    - recorder/overlay/__init__.py

key-decisions:
  - "Each layer is an independent QGraphicsItem subclass in its own file for fault isolation"
  - "Controller lazy-imports OverlayView inside show() to avoid circular dependencies"
  - "Ctrl+Q while RECORDING fires on_save; while READY/PAUSED fires on_abort"

patterns-established:
  - "Layer composition: each visual concern is a QGraphicsItemGroup subclass added/removed from scene"
  - "Controller-view separation: controller owns state, view is a thin rendering shell"
  - "Safe callback firing: try/except with logger.error for all callback invocations"

requirements-completed: [OVLY-01, OVLY-04, OVLY-05]

# Metrics
duration: 3min
completed: 2026-03-16
---

# Phase 1 Plan 02: Overlay View, Layers, and Controller Summary

**Layer-based QGraphicsView overlay with 4 independent visual layers, drag-to-draw bbox selection, and OverlayController managing F2 state toggle and hotkey lifecycle**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T18:14:54Z
- **Completed:** 2026-03-16T18:18:04Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- OverlayView: transparent fullscreen QGraphicsView with WA_TranslucentBackground, FramelessWindowHint, click-through toggle, drag-to-draw rubber-band selection
- 4 independent layer items: BorderLayer (4-rect screen border), ClickCatcherLayer (invisible hit-test rect), ModeIndicatorLayer (mode text + hotkey hints), BboxLayer (border + corner handles + label)
- OverlayController: public API with show/close lifecycle, F2 state transitions, save/abort distinction on Ctrl+Q, bbox management delegation

## Task Commits

Each task was committed atomically:

1. **Task 1: Overlay view shell and 4 layer items** - `de0edb1` (feat)
2. **Task 2: OverlayController with state machine and hotkey lifecycle** - `a132d8d` (feat)

## Files Created/Modified
- `recorder/overlay/border_layer.py` - Screen-edge border with set_color() for state-driven updates
- `recorder/overlay/click_catcher_layer.py` - Invisible full-screen rect for mouse capture in RECORDING state
- `recorder/overlay/mode_indicator_layer.py` - HUD text label showing current mode and hotkey shortcuts
- `recorder/overlay/bbox_layer.py` - Bounding box with 2px border, corner handles, optional label
- `recorder/overlay/view.py` - QGraphicsView shell with apply_state(), render_bboxes(), drag-to-draw
- `recorder/overlay/controller.py` - OverlayController with state machine, hotkey wiring, safe callbacks
- `recorder/overlay/__init__.py` - Added OverlayController export

## Decisions Made
- Each layer is an independent QGraphicsItem subclass in its own file for fault isolation and easy testing
- Controller lazy-imports OverlayView inside show() to avoid circular dependencies (view imports state, not controller)
- Ctrl+Q while RECORDING fires on_save callback; while READY/PAUSED fires on_abort callback -- clear semantic distinction

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All overlay visual and lifecycle components are in place
- Plan 03 (integration tests / manual verification) can wire everything together
- 31 Plan 01 tests still pass -- no regressions
- OverlayController provides clean public API for Phase 4 recording integration

## Self-Check: PASSED

- All 7 files verified on disk
- Commits de0edb1 and a132d8d verified in git log
- 31 Plan 01 tests pass with no regressions

---
*Phase: 01-overlay-foundation*
*Completed: 2026-03-16*
