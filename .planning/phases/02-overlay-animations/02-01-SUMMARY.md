---
phase: 02-overlay-animations
plan: 01
subsystem: ui
tags: [pyqt6, animation, qgraphicsobject, qtimer, qelapsedtimer, conical-gradient, smoothstep]

# Dependency graph
requires:
  - phase: 01-overlay-foundation
    provides: "OverlayState enum, STATE_COLORS map, BorderLayer pattern, QGraphicsView scene"
provides:
  - "AnimationClock: shared 16ms PreciseTimer with QElapsedTimer delta-time"
  - "ShimmerLayer: rotating gradient border glow with state colors and mouse retreat"
affects: [02-overlay-animations, 03-scan-layer, overlay-view-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: ["QGraphicsObject base for animated items", "Shared AnimationClock with register/unregister callback API", "Smoothstep distance falloff for organic mouse retreat"]

key-files:
  created:
    - recorder/overlay/animation_clock.py
    - recorder/overlay/shimmer_layer.py
    - tests/test_animation_clock.py
    - tests/test_shimmer_layer.py
  modified: []

key-decisions:
  - "QGraphicsObject (not QGraphicsItemGroup) as ShimmerLayer base to avoid MRO issues and enable custom paint"
  - "Segment-based mouse retreat: divide border into 32 segments with per-segment intensity modulation"
  - "Avoidance rects API on ShimmerLayer for UI element retreat (future integration point)"

patterns-established:
  - "AnimationClock.register(callback) pattern: all animated layers register a tick(dt) method"
  - "self.update() only in tick(), never in paint() -- prevents repaint loops"
  - "Smoothstep (t*t*(3-2t)) for organic distance falloff"

requirements-completed: [ANIM-05, ANIM-01, ANIM-06]

# Metrics
duration: 3min
completed: 2026-03-16
---

# Phase 2 Plan 01: Animation Clock + Shimmer Layer Summary

**AnimationClock with 16ms delta-time tick infrastructure and ShimmerLayer with rotating QConicalGradient border, state-driven color/speed/width, and smoothstep mouse retreat**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T21:28:11Z
- **Completed:** 2026-03-16T21:31:01Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- AnimationClock delivers capped delta-time (max 0.1s) to registered callbacks via 16ms PreciseTimer
- ShimmerLayer renders rotating conical gradient shimmer along screen border strip
- Shimmer changes color (green/red), speed (5s/2s), and width (16px/12px) per overlay state
- Mouse retreat with smoothstep falloff attenuates shimmer within 200px radius
- 25 unit tests all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: AnimationClock with delta-time and callback registration** - `721540e` (feat)
2. **Task 2: ShimmerLayer with rotating gradient, state colors, and mouse retreat** - `ace0186` (feat)

_Both tasks used TDD: RED (failing tests) -> GREEN (implementation passes)_

## Files Created/Modified
- `recorder/overlay/animation_clock.py` - Central 16ms timer with QElapsedTimer delta-time and callback registration
- `recorder/overlay/shimmer_layer.py` - Animated border shimmer glow with QConicalGradient, state colors, mouse retreat
- `tests/test_animation_clock.py` - 8 tests: instantiation, register/unregister, start/stop, delta cap, multi-callback
- `tests/test_shimmer_layer.py` - 17 tests: instantiation, bounding rect, state colors, tick phase, intensity, border width

## Decisions Made
- Used QGraphicsObject as ShimmerLayer base class (avoids MRO issues flagged in STATE.md blockers)
- 32-segment border sampling for mouse retreat modulation (balances quality vs performance)
- Avoidance rects use closest-point-on-rect distance calculation for accurate UI element retreat
- Border width smooth transition uses exponential decay (dt * 4.0 factor) for cinematic feel

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- AnimationClock ready for all subsequent animation layers to register tick callbacks
- ShimmerLayer ready for view integration (Plan 03 wiring)
- BorderLayer remains in place until Plan 03 replaces it with ShimmerLayer in the scene

## Self-Check: PASSED

- [x] recorder/overlay/animation_clock.py exists
- [x] recorder/overlay/shimmer_layer.py exists
- [x] tests/test_animation_clock.py exists
- [x] tests/test_shimmer_layer.py exists
- [x] Commit 721540e exists (Task 1)
- [x] Commit ace0186 exists (Task 2)
- [x] 25/25 tests pass

---
*Phase: 02-overlay-animations*
*Completed: 2026-03-16*
