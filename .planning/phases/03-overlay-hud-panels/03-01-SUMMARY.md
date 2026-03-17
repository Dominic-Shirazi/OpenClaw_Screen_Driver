---
phase: 03-overlay-hud-panels
plan: 01
subsystem: ui
tags: [pyqt6, overlay, hud, typewriter, animation, qcolor, radial-gradient]

requires:
  - phase: 02-overlay-animations
    provides: AnimationClock register/unregister API, ShimmerLayer CompositionMode_Plus technique
provides:
  - Shared HUD constants module (colors, spacing, typography, timing, z-values)
  - Card border glow painting helper (additive-blended radial gradients)
  - TypewriterEngine for simultaneous QLineEdit field animation
affects: [03-02-tag-dialog, 03-03-toolbar]

tech-stack:
  added: []
  patterns: [SimpleNamespace for spacing tokens, standalone painting helpers, clock-registered tick engines]

key-files:
  created:
    - recorder/overlay/hud_common.py
    - recorder/overlay/card_glow.py
    - recorder/overlay/typewriter_engine.py
    - tests/test_hud_common.py
    - tests/test_typewriter_engine.py
  modified: []

key-decisions:
  - "SimpleNamespace for SPACING tokens (lightweight, dot-access, no dataclass overhead)"
  - "Card glow uses same _ACTIVE_SPAN=0.35 as ShimmerLayer for visual family consistency"
  - "TypewriterEngine emits per-character signals for glow synchronization"

patterns-established:
  - "Standalone paint helpers: stateless functions accepting QPainter + params, save/restore painter state"
  - "Clock-registered engines: QObject subclass with tick(dt) method, register/unregister lifecycle"
  - "HUD constants centralized in hud_common.py, imported by all panel modules"

requirements-completed: [HUD-02]

duration: 3min
completed: 2026-03-17
---

# Phase 3 Plan 01: HUD Foundation Summary

**Shared HUD constants, card border glow painting helper, and TypewriterEngine for simultaneous VLM field animation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-17T00:50:24Z
- **Completed:** 2026-03-17T00:53:33Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- hud_common.py with all color, spacing, typography, timing, and z-value constants matching UI-SPEC
- card_glow.py painting helper using same additive blending technique as ShimmerLayer
- TypewriterEngine filling multiple QLineEdit fields simultaneously at varied speeds with signal-based glow sync
- 20 new tests (13 hud_common + 7 typewriter), 302 total suite passes

## Task Commits

Each task was committed atomically:

1. **Task 1: Create hud_common.py and card_glow.py** - `ee7337b` (feat)
2. **Task 2: Create TypewriterEngine with tests** - `e15471f` (feat)

_TDD workflow: tests written first (RED), implementation passes (GREEN)._

## Files Created/Modified
- `recorder/overlay/hud_common.py` - All shared HUD constants (colors, spacing, typography, timing, z-values, field stylesheet)
- `recorder/overlay/card_glow.py` - Standalone painting helper for additive-blended radial gradient card border glow
- `recorder/overlay/typewriter_engine.py` - Clock-registered engine driving simultaneous QLineEdit typewriter fill
- `tests/test_hud_common.py` - 13 tests verifying constant values and card_glow API
- `tests/test_typewriter_engine.py` - 7 tests covering simultaneous fill, interruption, signals, edge cases

## Decisions Made
- Used SimpleNamespace for SPACING tokens (lightweight, dot-access, no dataclass overhead)
- Card glow uses same _ACTIVE_SPAN=0.35 as ShimmerLayer for visual family consistency
- TypewriterEngine emits per-character char_inserted signals enabling glow pulse synchronization

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- hud_common.py ready for import by tag dialog (Plan 02) and toolbar (Plan 03)
- card_glow.py ready for use in both panel paint methods
- TypewriterEngine ready for integration with tag dialog VLM auto-fill

## Self-Check: PASSED

All 5 created files exist. Both task commits (ee7337b, e15471f) verified in git log.

---
*Phase: 03-overlay-hud-panels*
*Completed: 2026-03-17*
