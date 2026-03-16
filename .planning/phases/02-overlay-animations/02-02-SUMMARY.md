---
phase: 02-overlay-animations
plan: 02
subsystem: ui
tags: [pyqt6, animation, qgraphicsobject, qpropertyanimation, state-machine, easing]

# Dependency graph
requires:
  - phase: 01-overlay-foundation
    provides: "BboxLayer with corner handles, OverlayView scene, state machine"
provides:
  - "ScanLayer with 9-phase animation state machine (idle through done)"
  - "ScanPhase enum for scan animation sequencing"
  - "BboxLayer.morph_to() with InOutCubic easing over 500ms"
  - "_MorphHelper QObject bridge for QPropertyAnimation on QGraphicsItemGroup"
affects: [02-overlay-animations plan 03 (wiring), 03-capture-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [QGraphicsObject for animated items, _MorphHelper QObject bridge for QPropertyAnimation on non-QObject items, instant-transition phase in state machine]

key-files:
  created:
    - recorder/overlay/scan_layer.py
    - tests/test_scan_layer.py
    - tests/test_bbox_morph.py
  modified:
    - recorder/overlay/bbox_layer.py

key-decisions:
  - "ScanLayer uses QGraphicsObject base; WAITING_AI is an instant-transition phase (no duration)"
  - "_MorphHelper QObject bridge pattern to enable QPropertyAnimation on QGraphicsItemGroup-based BboxLayer"
  - "Post-morph styling: thin red outline (1px) with faint fill per CONTEXT.md decisions"

patterns-established:
  - "Instant-transition phase: state machine phases without duration advance immediately on next tick"
  - "_MorphHelper bridge: QObject helper owns animated property, drives non-QObject parent via callback"

requirements-completed: [ANIM-02, ANIM-03, ANIM-06]

# Metrics
duration: 5min
completed: 2026-03-16
---

# Phase 2 Plan 02: Scan Layer & Bbox Morph Summary

**ScanLayer 9-phase state machine with laser loop and BboxLayer morph_to() using QPropertyAnimation InOutCubic easing**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-16T21:28:31Z
- **Completed:** 2026-03-16T21:33:06Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- ScanLayer sequences through corner glow, CCW line draw, fill inward, vertical laser, horizontal laser phases with per-phase durations
- Laser sweeps loop via WAITING_AI until receive_fitted_bbox() delivers AI result, then snap-to-fitted with cubic ease-in-out
- BboxLayer gains morph_to() that smoothly animates rect, handles, and label to new positions over 500ms
- 24 unit tests covering all phase transitions, durations, morph interpolation, and original API regression

## Task Commits

Each task was committed atomically:

1. **Task 1: ScanLayer with multi-phase animation state machine** - `2a61c0f` (test+feat via TDD)
2. **Task 2: Extend BboxLayer with smooth morph_to() animation** - `3da6ea7` (feat via TDD)

_Note: TDD tasks combined RED+GREEN into single commits for atomic task delivery._

## Files Created/Modified
- `recorder/overlay/scan_layer.py` - ScanPhase enum and ScanLayer QGraphicsObject with 9-phase state machine, custom paint for each phase
- `recorder/overlay/bbox_layer.py` - Extended with _MorphHelper QObject, morph_to(), _apply_morph_progress(), is_morphing property, post-morph styling
- `tests/test_scan_layer.py` - 14 tests: construction, phase transitions, durations, laser loop, AI bbox reception, reset
- `tests/test_bbox_morph.py` - 10 tests: original API regression, morph target storage, progress interpolation, duration, is_morphing flag

## Decisions Made
- ScanLayer WAITING_AI phase has no duration and advances immediately on next tick (instant transition), avoiding artificial delay when AI result is already available
- Used _MorphHelper QObject bridge pattern rather than converting BboxLayer to QGraphicsObject, preserving all existing API and QGraphicsItemGroup child management
- Post-morph styling applies thin 1px red outline with faint 20-alpha red fill per CONTEXT.md "thin red outline, glow retreats" decision

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed WAITING_AI phase stuck without duration**
- **Found during:** Task 1 (ScanLayer phase transitions)
- **Issue:** WAITING_AI phase had no entry in _PHASE_DURATIONS, defaulting to 1.0s delay before looping. Test expected immediate transition.
- **Fix:** Added explicit instant-transition handling in tick() for WAITING_AI phase (advance immediately without duration check)
- **Files modified:** recorder/overlay/scan_layer.py
- **Verification:** test_waiting_ai_loops_back_to_laser_vertical passes
- **Committed in:** 2a61c0f (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary for correct behavior -- WAITING_AI should loop immediately, not wait 1 second.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ScanLayer and BboxLayer morph are ready for wiring in Plan 03
- ScanLayer.tick() is designed to be registered with AnimationClock (Plan 01)
- receive_fitted_bbox() provides the bridge from AI pipeline to scan completion

## Self-Check: PASSED

- FOUND: recorder/overlay/scan_layer.py
- FOUND: recorder/overlay/bbox_layer.py
- FOUND: tests/test_scan_layer.py
- FOUND: tests/test_bbox_morph.py
- FOUND: .planning/phases/02-overlay-animations/02-02-SUMMARY.md
- COMMIT: 2a61c0f (Task 1)
- COMMIT: 3da6ea7 (Task 2)
- TESTS: 24 passed, 0 failed

---
*Phase: 02-overlay-animations*
*Completed: 2026-03-16*
