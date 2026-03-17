---
phase: 03-overlay-hud-panels
plan: 03
subsystem: ui
tags: [pyqt6, qgraphics, toolbar, hud, overlay, drag]

requires:
  - phase: 03-overlay-hud-panels/01
    provides: "HUD common constants, card glow helper, typewriter engine"
  - phase: 03-overlay-hud-panels/02
    provides: "TagDialogPanel with form fields and typewriter fill"
provides:
  - "ToolbarPanel: draggable floating pill with 3 context modes"
  - "OverlayView HUD panel management (show/dismiss/hide for capture)"
  - "OverlayController public HUD API (6 methods)"
  - "Shimmer avoidance rect integration for both HUD panels"
affects: [04-recording-engine, 05-replay-engine]

tech-stack:
  added: []
  patterns:
    - "Context-sensitive toolbar via ToolbarMode enum + button proxy swapping"
    - "View manages HUD panel lifecycle (lazy create, scene add, avoidance rects)"
    - "Controller delegates all HUD operations to view (consistent with existing pattern)"

key-files:
  created:
    - recorder/overlay/toolbar_panel.py
    - tests/test_toolbar_panel.py
    - tests/test_hud_integration.py
  modified:
    - recorder/overlay/view.py
    - recorder/overlay/controller.py
    - recorder/overlay/__init__.py

key-decisions:
  - "ToolbarPanel uses simple visibility toggle for mode switching (no QPropertyAnimation fade on proxies) for reliability in offscreen mode"
  - "View lazily creates HUD panels on first use to avoid unnecessary scene items"

patterns-established:
  - "HUD panel lifecycle: lazy-create in view, add to scene, manage avoidance rects"
  - "Controller HUD methods: null-check _view, delegate to view method"

requirements-completed: [HUD-05, HUD-06]

duration: 3min
completed: 2026-03-17
---

# Phase 3 Plan 3: Toolbar + HUD Integration Summary

**Draggable pill toolbar with 3 context modes (RECORDING/TAG_OPEN/DRY_RUN) integrated with tag dialog into overlay view/controller with capture hiding and shimmer avoidance**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-17T01:02:33Z
- **Completed:** 2026-03-17T01:06:05Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- ToolbarPanel renders as draggable pill with card border glow and 3 context-sensitive button sets
- Both HUD panels (TagDialogPanel + ToolbarPanel) fully hide before screenshot capture and register shimmer avoidance rects
- OverlayController exposes complete 6-method HUD public API (show_tag_dialog, dismiss_tag_dialog, get_tag_data, show_toolbar, hide_toolbar, set_toolbar_mode)
- Full test suite green (338 tests passing)

## Task Commits

Each task was committed atomically:

1. **Task 1: Build ToolbarPanel and integrate HUD panels** - `1866dac` (feat)
2. **Task 2: Create toolbar and integration tests** - `c5f3504` (test)

## Files Created/Modified
- `recorder/overlay/toolbar_panel.py` - Draggable floating toolbar with ToolbarMode enum and 3 button sets
- `recorder/overlay/view.py` - Extended with HUD panel management, avoidance rects, capture hiding
- `recorder/overlay/controller.py` - Extended with 6 HUD API methods delegating to view
- `recorder/overlay/__init__.py` - Added TagDialogPanel, ToolbarPanel, TypewriterEngine exports
- `tests/test_toolbar_panel.py` - Toolbar tests: instantiation, mode switching, avoidance rect, visibility
- `tests/test_hud_integration.py` - Integration tests: both panels, controller API, capture hide/restore

## Decisions Made
- ToolbarPanel uses simple visibility toggle for mode switching rather than QPropertyAnimation fade on proxy widgets, for reliability in offscreen test mode
- View lazily creates HUD panels on first use to avoid scene clutter when panels are not needed

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 (Overlay HUD Panels) is fully complete
- All overlay visual layers, HUD panels, and controller API are ready for recording engine integration
- Ready for Phase 4 (Recording Engine) to wire toolbar button signals to recording state machine

---
*Phase: 03-overlay-hud-panels*
*Completed: 2026-03-17*
