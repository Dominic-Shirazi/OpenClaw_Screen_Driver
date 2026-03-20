---
phase: 07-run-flow
plan: 04
subsystem: overlay
tags: [pyqt6, pyqtSignal, thread-safety, replay, adapter]

requires:
  - phase: 07-02
    provides: StatusBadge, TargetHighlight, CameraFlash overlay widgets
  - phase: 07-03
    provides: RunEvent enum, RunCallback protocol, runner step loop

provides:
  - ReplayOverlayAdapter bridging runner events to overlay controller
  - Replay mode API on OverlayController (set_replay_mode, camera_flash, etc.)
  - View replay widget lifecycle with hide_for_capture support

affects: [08-terminal-ui, 09-api-mcp]

tech-stack:
  added: []
  patterns: [pyqtSignal adapter for cross-thread event delivery]

key-files:
  created:
    - routine/replay_overlay.py
  modified:
    - recorder/overlay/controller.py
    - recorder/overlay/view.py

key-decisions:
  - "Adapter is callable (__call__) so it can be passed directly as RunCallback"
  - "pyqtSignal(int, object) used because pyqtSignal doesn't support custom Enum types"
  - "RUN_FAILED shows message for 2s via QTimer.singleShot then cleans up"
  - "TargetHighlight not registered with clock in view (self-registers in constructor)"

patterns-established:
  - "QObject adapter pattern: __call__ emits signal from any thread, slot handles on main thread"

requirements-completed: [RUN-09]

duration: 2min
completed: 2026-03-20
---

# Phase 7 Plan 4: Replay Overlay Adapter Summary

**Thread-safe ReplayOverlayAdapter bridges runner events to overlay controller via pyqtSignal for replay progress UX**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T02:56:50Z
- **Completed:** 2026-03-20T02:58:44Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- OverlayController extended with replay mode API (set_replay_mode, set_replay_status, show_target_highlight, camera_flash)
- OverlayView manages StatusBadge, TargetHighlight, CameraFlash lifecycle including hide_for_capture
- ReplayOverlayAdapter uses pyqtSignal for thread-safe cross-thread event delivery from runner to overlay
- Headless mode (controller=None or show_overlay=False) gracefully no-ops

## Task Commits

Each task was committed atomically:

1. **Task 1: Add replay methods to OverlayController and OverlayView** - `a59f725` (feat)
2. **Task 2: Create thread-safe ReplayOverlayAdapter with pyqtSignal** - `ef3d624` (feat)

## Files Created/Modified
- `routine/replay_overlay.py` - Thread-safe adapter translating RunEvent to overlay controller calls
- `recorder/overlay/controller.py` - Replay mode API section (set_replay_mode, camera_flash, etc.)
- `recorder/overlay/view.py` - Replay widget management (StatusBadge, TargetHighlight, CameraFlash lifecycle)

## Decisions Made
- Adapter is callable (__call__) so it can be passed directly as RunCallback -- no wrapping needed
- pyqtSignal(int, object) because pyqtSignal doesn't support custom Enum types directly
- RUN_FAILED shows failure message for 2 seconds via QTimer.singleShot before cleanup
- TargetHighlight self-registers with AnimationClock in its constructor, so view skips clock.register

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Runner events now flow to overlay for visual feedback during replay
- All Phase 7 plans complete -- ready for Phase 8 (Terminal UI)

---
*Phase: 07-run-flow*
*Completed: 2026-03-20*
