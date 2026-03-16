---
phase: 02-overlay-animations
verified: 2026-03-16T22:00:00Z
status: passed
score: 15/15 must-haves verified
re_verification: false
---

# Phase 2: Overlay Animations Verification Report

**Phase Goal:** Cinematic animation items run at 60fps without CPU spike and give users clear visual feedback during element capture
**Verified:** 2026-03-16
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

All truths derived from REQUIREMENTS.md success criteria and plan must_haves.

| #  | Truth                                                                                               | Status     | Evidence                                                                                          |
|----|-----------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------|
| 1  | AnimationClock ticks at ~60fps and delivers frame-independent delta-time to registered callbacks    | VERIFIED   | `animation_clock.py` line 33: `setInterval(16)`, line 32: `PreciseTimer`, line 77: `dt = min(dt, 0.1)` |
| 2  | ShimmerLayer renders a rotating gradient sweep along the screen border that changes color with state | VERIFIED   | `shimmer_layer.py` uses `QConicalGradient`, `set_state()` updates color + speed per OverlayState  |
| 3  | Shimmer retreats from the mouse cursor with smooth organic falloff                                  | VERIFIED   | `_shimmer_intensity_at()` implements smoothstep `t * t * (3.0 - 2.0 * t)` with 200px radius      |
| 4  | Scan layer sequences through all 8 phases (CORNER_GLOW through DONE)                               | VERIFIED   | `scan_layer.py` defines all phases in `ScanPhase` enum + `_advance_phase()` transition map        |
| 5  | Scan repeats laser sweeps until receive_fitted_bbox() is called                                     | VERIFIED   | `WAITING_AI` is instant-transition, loops back to `LASER_VERTICAL` until `_ai_ready`              |
| 6  | Bbox corners morph smoothly from rough to AI-fitted positions over 500ms with InOutCubic easing     | VERIFIED   | `bbox_layer.py` `morph_to()` uses `QPropertyAnimation` + `QEasingCurve.Type.InOutCubic`, 500ms   |
| 7  | Donut cloud renders Gaussian heat map blob that fades in from center outward                        | VERIFIED   | `donut_cloud_layer.py` uses `QRadialGradient` with `_fade_in_progress` scaling ellipse radius     |
| 8  | Donut cloud transitions from red to green on accept()                                               | VERIFIED   | `accept()` sets `_target_color = QColor(50, 200, 50)`, tick() lerps `_color` toward target        |
| 9  | Raindrop ripple effect with 3-5 concurrent expanding/fading circles                                 | VERIFIED   | `_Raindrop` class + spawn logic in `tick()` with `_max_raindrops: int = 5`, Gaussian positioning  |
| 10 | OverlayView.apply_state() starts/stops shimmer animation and updates shimmer color per state        | VERIFIED   | `view.py` line 139: `self._shimmer.set_state(state)` in `apply_state()`                           |
| 11 | AnimationClock stops during hide_for_capture() and restarts in show_after_capture()                 | VERIFIED   | `view.py` line 186: `self._clock.stop()`, line 192: `self._clock.start()`                         |
| 12 | Mouse tracking feeds cursor position to shimmer layer each animation tick                           | VERIFIED   | `mouseMoveEvent()` calls `self._shimmer.set_mouse_pos(pos.x(), pos.y())`, `setMouseTracking(True)` |
| 13 | finish_scan() triggers bbox morph from rough to AI-fitted coordinates                               | VERIFIED   | `view.py` lines 333-336: `self._active_bbox.morph_to(fitted_x, fitted_y, fitted_w, fitted_h)`     |
| 14 | All overlay transitions use fluid animation — nothing appears or disappears instantly               | VERIFIED   | Every layer uses tick-driven interpolation; no direct setVisible/setOpacity without animation     |
| 15 | No repaint loops — self.update() never called inside paint() methods                                | VERIFIED   | All `self.update()` calls verified in `tick()` methods only; paint() methods contain explicit "NO self.update()" comments |

**Score:** 15/15 truths verified

---

### Required Artifacts

| Artifact                                            | Provides                                          | Status     | Details                                                               |
|-----------------------------------------------------|---------------------------------------------------|------------|-----------------------------------------------------------------------|
| `recorder/overlay/animation_clock.py`               | Central 16ms PreciseTimer with delta-time         | VERIFIED   | 80 lines, `AnimationClock(QObject)` with register/unregister/start/stop |
| `recorder/overlay/shimmer_layer.py`                 | Animated border shimmer glow (replaces BorderLayer) | VERIFIED | 262 lines, `ShimmerLayer(QGraphicsObject)`, QConicalGradient, smoothstep |
| `recorder/overlay/scan_layer.py`                    | Multi-phase element scan animation (ScanPhase SM) | VERIFIED   | 412 lines, `ScanLayer(QGraphicsObject)`, 9 phases, all paint helpers   |
| `recorder/overlay/bbox_layer.py`                    | Extended with morph_to() InOutCubic animation     | VERIFIED   | `_MorphHelper`, `morph_to()`, `_apply_morph_progress()`, `is_morphing` |
| `recorder/overlay/donut_cloud_layer.py`             | Gaussian heat map with raindrop effect            | VERIFIED   | 241 lines, `DonutCloudLayer(QGraphicsObject)`, `_Raindrop`, Gaussian spawn |
| `recorder/overlay/view.py`                          | AnimationClock, ShimmerLayer, mouse tracking wired | VERIFIED  | No BorderLayer import; AnimationClock + ShimmerLayer created in `__init__` |
| `recorder/overlay/controller.py`                    | start_scan, finish_scan, show_donut_cloud, accept_donut_cloud | VERIFIED | All 4 methods present, delegate to view |
| `recorder/overlay/__init__.py`                      | Exports AnimationClock, ShimmerLayer, ScanLayer, DonutCloudLayer | VERIFIED | Lines 5-11 export all required symbols |
| `tests/test_animation_clock.py`                     | 8 unit tests for clock                            | VERIFIED   | 8 test functions, all pass                                             |
| `tests/test_shimmer_layer.py`                       | 17 unit tests for shimmer                         | VERIFIED   | 17 test functions, all pass                                            |
| `tests/test_scan_layer.py`                          | 14 unit tests for scan phases                     | VERIFIED   | 14 test functions, all pass                                            |
| `tests/test_bbox_morph.py`                          | 10 unit tests for bbox morph                      | VERIFIED   | 10 test functions, all pass                                            |
| `tests/test_donut_cloud.py`                         | 10 unit tests for donut cloud                     | VERIFIED   | 10 test functions, all pass                                            |
| `tests/test_overlay_animations_integration.py`      | 14 integration tests including bbox morph wiring  | VERIFIED   | 14 test functions including `morph_to` mock assertion test             |

---

### Key Link Verification

| From                            | To                              | Via                                    | Status   | Details                                                        |
|---------------------------------|---------------------------------|----------------------------------------|----------|----------------------------------------------------------------|
| `animation_clock.py`            | `shimmer_layer.py`              | `clock.register(shimmer.tick)`         | WIRED    | `view.py:94` — `self._clock.register(self._shimmer.tick)`      |
| `shimmer_layer.py`              | `state.py`                      | `from recorder.overlay.state import STATE_COLORS` | WIRED | `shimmer_layer.py:18` — explicit import verified               |
| `scan_layer.py`                 | `animation_clock.py`            | `def tick(dt: float)` registered       | WIRED    | `view.py:310` — `self._clock.register(self._scan_layer.tick)`  |
| `view.py`                       | `animation_clock.py`            | View creates and owns AnimationClock   | WIRED    | `view.py:89` — `self._clock = AnimationClock()`                |
| `view.py`                       | `shimmer_layer.py`              | View creates ShimmerLayer, registers tick | WIRED  | `view.py:92-94` — create + register in `__init__`              |
| `view.py`                       | `bbox_layer.py`                 | `finish_scan()` calls `morph_to()`     | WIRED    | `view.py:334` — `self._active_bbox.morph_to(fitted_x, ...)`    |
| `controller.py`                 | `scan_layer.py`                 | `start_scan/receive_fitted_bbox` chain | WIRED    | `controller.py:172` delegates to `view.start_scan()`           |
| `donut_cloud_layer.py`          | `animation_clock.py`            | `clock.register(donut.tick)`           | WIRED    | `view.py:365` — `self._clock.register(self._donut_cloud.tick)` |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                                                   | Status    | Evidence                                                                  |
|-------------|------------|--------------------------------------------------------------------------------|-----------|---------------------------------------------------------------------------|
| ANIM-01     | 02-01      | Border shimmer glow — faded moving pulse, green=ready, red=recording           | SATISFIED | `ShimmerLayer` QConicalGradient, `set_state()` maps to green/red via STATE_COLORS |
| ANIM-02     | 02-02      | Element scan animation — glowing perimeter trace, scan line sweep              | SATISFIED | `ScanLayer` with CORNER_GLOW, LINE_DRAW, FILL_INWARD, LASER phases        |
| ANIM-03     | 02-02      | Bbox animation — smooth morph from rough to AI-fitted bbox                     | SATISFIED | `BboxLayer.morph_to()` with InOutCubic easing, triggered by `finish_scan()` |
| ANIM-04     | 02-03      | Donut cloud — probability density visualizer showing simulated click landing    | SATISFIED | `DonutCloudLayer` with QRadialGradient + raindrop ripples, Gaussian spawn  |
| ANIM-05     | 02-01      | All animations run at 60fps via QTimer (16ms interval), never trigger repaint loops | SATISFIED | `AnimationClock` 16ms PreciseTimer; `self.update()` only in `tick()` never in `paint()` |
| ANIM-06     | 02-01/02/03 | All overlay elements use fluid transitions — slides, fades, scales, eases     | SATISFIED | Every layer interpolates; no instant state changes; clock-driven throughout |

All 6 ANIM requirements satisfied. No orphaned requirements.

---

### Anti-Patterns Found

| File                        | Line | Pattern              | Severity | Impact  |
|-----------------------------|------|----------------------|----------|---------|
| None detected               | -    | -                    | -        | -       |

Checked for:
- `self.update()` inside `paint()` — CLEAR across all 3 animated layers
- TODO/FIXME/PLACEHOLDER comments — none found in Phase 2 files
- Empty return implementations — none; all paint methods dispatch to phase helpers
- Static/hardcoded API responses — not applicable

The legacy `recorder/overlay/border_layer.py` remains on disk (visible in git status as untracked modified file from before Phase 2). It is not imported anywhere in the Phase 2 code path. This is an orphaned legacy file, not a blocker.

---

### Human Verification Required

The following items cannot be verified programmatically:

#### 1. Shimmer Visual Quality at 60fps

**Test:** Launch the overlay (`python -m recorder` or via the test harness), let it idle for 30 seconds.
**Expected:** Green border shimmer rotates smoothly at ~60fps with no visible stutter or CPU spike above 10%.
**Why human:** CPU profiling and visual smoothness require runtime measurement.

#### 2. Mouse Retreat Organic Feel

**Test:** Move the mouse slowly along the border area of the overlay window.
**Expected:** The shimmer visibly attenuates (darkens) within ~200px of the cursor, retreating with a smooth (non-linear) falloff — not an abrupt cut.
**Why human:** Perceptual quality of the smoothstep falloff cannot be verified by grep.

#### 3. Scan Animation Cinematic Sequence

**Test:** Trigger a capture (drag-to-draw in RECORDING mode). Observe the full scan animation from CORNER_GLOW through LASER sweeps to SNAP_TO_FITTED.
**Expected:** Each phase appears visually distinct and cinematic; the laser line has a visible white core with red glow halo; the bbox corners snap with a smooth ease-in-out.
**Why human:** Multi-phase visual quality and timing feel require live observation.

#### 4. Donut Cloud Red-to-Green Transition

**Test:** Show the donut cloud (red), then call `accept()`.
**Expected:** The cloud gradually transitions from red to green over ~0.3s; raindrop ripples adapt to the new color; no visual discontinuity.
**Why human:** Color lerp quality and raindrop behavior during transition require visual inspection.

---

### Test Suite Summary

| Test File                                  | Tests | Result    |
|--------------------------------------------|-------|-----------|
| `tests/test_animation_clock.py`            | 8     | PASS      |
| `tests/test_shimmer_layer.py`              | 17    | PASS      |
| `tests/test_scan_layer.py`                 | 14    | PASS      |
| `tests/test_bbox_morph.py`                 | 10    | PASS      |
| `tests/test_donut_cloud.py`                | 10    | PASS      |
| `tests/test_overlay_animations_integration.py` | 14 | PASS     |
| **Phase 2 subtotal**                       | **73** | **ALL PASS** |
| **Full suite (all phases)**                | **282** | **ALL PASS** |

No regressions against Phase 1 tests.

---

### Commit Verification

All 7 commits referenced in SUMMARY files confirmed present in git log:

| Commit    | Description                                             |
|-----------|---------------------------------------------------------|
| `721540e` | feat(02-01): AnimationClock with delta-time             |
| `ace0186` | feat(02-01): ShimmerLayer rotating gradient + mouse retreat |
| `2a61c0f` | test(02-02): ScanLayer multi-phase state machine        |
| `3da6ea7` | feat(02-02): BboxLayer morph_to() animation             |
| `843f8e0` | test(02-03): DonutCloudLayer failing tests              |
| `17c856a` | feat(02-03): DonutCloudLayer implementation             |
| `2c62cf5` | feat(02-03): Wire all animation layers into view/controller |

---

### Gaps Summary

No gaps. All must-haves verified. Phase goal achieved.

---

_Verified: 2026-03-16_
_Verifier: Claude (gsd-verifier)_
