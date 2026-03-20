---
phase: 07-run-flow
verified: 2026-03-19T00:00:00Z
status: passed
score: 9/9 must-haves verified
gaps: []
human_verification:
  - test: "Run a real saved routine end-to-end with overlay visible"
    expected: "Purple shimmer border shows during execution, StatusBadge shows step progress, TargetHighlight briefly glows over located element, CameraFlash 15px border fires on each screenshot capture"
    why_human: "Visual animation quality and timing cannot be verified programmatically. Requires a running Qt app against a real screen."
  - test: "Trigger failure cascade manually (remove a snippet file, run routine)"
    expected: "Runner retries twice, widens search to 25% region, then full screen, then LiteLLM prompt, then aborts with annotated failure screenshot saved to runs/ directory. Never executes a click without confirmation."
    why_human: "End-to-end failure path requires a real screen environment and LiteLLM connectivity."
---

# Phase 7: Run Flow Verification Report

**Phase Goal:** A saved routine executes reliably with human-like motion, visible step progress, and graceful failure handling
**Verified:** 2026-03-19
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can trigger routine execution via run_routine() | VERIFIED | `routine/runner.py` — `run_routine(routine_dir, callback, dry_run)` entry point exists and is importable |
| 2 | Elements are located using 5-stage cascade (pixel/CLIP/OCR/VLM/position) | VERIFIED | `core/locate.py` — `locate_element_from_step()` implements all 5 stages for v1 step dicts; stage 4 skippable via `skip_vlm`; stage 5 skippable via `skip_position_fallback` |
| 3 | Human-like mouse motion via Bézier curves with overshoot-and-correct | VERIFIED | `core/executor.py` — `_bezier_move()` generates cubic Bézier paths with random offset radius; `click()` and `drag()` call this; human_delay multiplier scales all timing |
| 4 | Human-like typing with per-letter delays, variance, typo+correction | VERIFIED | `core/executor.py` — `type_text()` uses `type_interval` config + typo_chance; pre-existing from Phase 4 |
| 5 | Configurable execution speed via human_delay multiplier | VERIFIED | `core/config.py` execution section — `human_delay` float driven by `OCSD_HUMAN_DELAY` env var; used in `_human_sleep()` throughout executor |
| 6 | Post-action validation via pixel diff + optional VLM | VERIFIED | `routine/runner.py` — after each non-wait/non-prompt action, takes before/after screenshots and calls `validate_action()` lazily; VALIDATION_PASSED/FAILED events emitted |
| 7 | 5-stage failure cascade — never blind-clicks when unsure | VERIFIED | `routine/runner.py` `_failure_cascade()` — Stage 1: retry 2x, Stage 2: 25% region scan, Stage 3: full-screen, Stage 4: LiteLLM AI fallback, Stage 5: abort with annotated screenshot + return None; if None, abort with log, never execute click |
| 8 | Uncertainty always surfaced to user/agent — never silent fail | VERIFIED | `routine/runner.py` — `RunEvent.STEP_FAILED` emitted on cascade abort; `failure_reason` set in `RunResult`; annotated failure screenshot saved to run_dir |
| 9 | Step-by-step replay status visible to user | VERIFIED | `routine/runner.py` emits `STEP_START` per step; `ReplayOverlayAdapter` translates to `StatusBadge.set_text()` "Step N/M: Action Label"; `TargetHighlight` shows located element; `CameraFlash` fires on each screenshot |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `core/locate.py` | `locate_element_from_step()` for v1 step dicts | VERIFIED | 440 lines; full 5-stage cascade; skip_vlm, skip_position_fallback params |
| `core/conditions.py` | Complete `element_appears` and `text_matches` | VERIFIED | `_check_element_appears()` uses adaptive VLM escalation (skip VLM first 3 polls); `_check_text_matches()` uses rapidfuzz Levenshtein with threshold formula |
| `pyproject.toml` | `rapidfuzz>=3.0` dependency | VERIFIED | Line 19: `"rapidfuzz>=3.0"` |
| `recorder/overlay/state.py` | `REPLAYING` enum value with purple color | VERIFIED | `REPLAYING = auto()` in enum; `STATE_COLORS[OverlayState.REPLAYING] = (160, 80, 220, 180)` |
| `recorder/overlay/shimmer_layer.py` | Purple shimmer for REPLAYING state | VERIFIED | `elif state == OverlayState.REPLAYING:` branch with `_loop_duration = 3.0`, `_alpha_mult = 0.45` |
| `recorder/overlay/status_badge.py` | Frosted-glass pill showing step progress | VERIFIED | `class StatusBadge(QGraphicsObject)` with `set_text()`, `set_visible_animated()`; top-center positioning |
| `recorder/overlay/target_highlight.py` | 300ms purple glow around located element | VERIFIED | `class TargetHighlight(QGraphicsObject)` with `highlight()`, `_duration = 0.3`, linear fade in `tick()` |
| `recorder/overlay/camera_flash.py` | 15px border camera flash after screenshots | VERIFIED | `class CameraFlash(QGraphicsObject)` with `flash()`, `_border_width = 15`, `_duration = 0.2` |
| `core/config.py` | `replay` config section with defaults | VERIFIED | `_DEFAULTS["replay"]` contains `show_overlay: True`, `auto_minimize: True`, `poll_interval: 2.0`, `keep_successful_runs: 5`, `keep_failed_runs: 10` |
| `routine/runner.py` | Step-sequential replay engine | VERIFIED | 898 lines; `run_routine()`, `preflight_check()`, `_failure_cascade()`, `_dispatch_action()`, `_handle_loop_step()`, `_build_ai_fallback_prompt()`, `_compute_search_region()`; `RunEvent.SCREENSHOT_TAKEN` present |
| `routine/run_log.py` | Run directory management and annotation | VERIFIED | `create_run_dir()`, `save_run_result()`, `annotate_screenshot()`, `save_annotated_screenshot()`, `prune_old_runs()`, `setup_run_logger()`; `shutil.rmtree` for pruning; `cv2.rectangle` for annotation |
| `routine/replay_overlay.py` | Thread-safe adapter connecting runner to overlay | VERIFIED | `class ReplayOverlayAdapter(QObject)`; `_event_signal = pyqtSignal(int, object)`; `__call__` emits signal; `_on_event` dispatches on main thread; handles RUN_START, STEP_START, ELEMENT_LOCATED, SCREENSHOT_TAKEN, STEP_COMPLETE, RUN_COMPLETE, RUN_FAILED |
| `recorder/overlay/controller.py` | Replay mode API methods | VERIFIED | `set_replay_mode()`, `set_replay_status()`, `show_target_highlight()`, `camera_flash()` all present in "Replay mode API" section |
| `recorder/overlay/view.py` | View methods for replay widget lifecycle | VERIFIED | Imports StatusBadge, TargetHighlight, CameraFlash; `show_replay_badge()`, `show_target_highlight()`, `camera_flash()`, `hide_replay_badge()`, `hide_target_highlight()`, `hide_camera_flash()`; `hide_for_capture()` hides all three replay widgets; `show_after_capture()` restores status_badge if text is set |
| `tests/test_locate_adapter.py` | 5 tests for locate adapter | VERIFIED | `test_locate_from_step_ocr_hit`, `test_locate_from_step_position_fallback`, `test_locate_from_step_skip_position`, `test_locate_from_step_skip_vlm`, `test_locate_from_step_snippet_resolve` — all pass |
| `tests/test_condition_engine.py` | 7+ tests for condition completions | VERIFIED | `test_element_appears_with_locate_success/failure/adaptive_vlm`, `test_text_matches_exact_hit/fuzzy_hit/no_match/empty_target` — all pass |
| `tests/test_replay_overlay.py` | 11 tests for overlay widgets | VERIFIED | `test_replaying_state_exists`, `test_shimmer_replaying_state`, `test_status_badge_set_text/position`, `test_target_highlight_lifecycle/partial_fade/hide`, `test_camera_flash_lifecycle/border_width/hide`, `test_replay_config_defaults` — all pass |
| `tests/test_routine_runner.py` | 20+ comprehensive runner tests | VERIFIED | 20 named test functions; covers preflight, dispatch (click/type/scroll), failure cascade abort, never-blind-click, loop execution, run logging, callback events, screenshot_taken events, validation, self-cleaning prune |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `core/conditions.py` | `core/locate.py` | `locate_element_from_step` call in `_check_element_appears` | WIRED | Line 190: `from core.locate import locate_element_from_step`; called with adaptive `skip_vlm` |
| `core/conditions.py` | `rapidfuzz` | `Levenshtein.distance` in `_check_text_matches` | WIRED | Line 248: `from rapidfuzz.distance import Levenshtein`; threshold formula applied |
| `recorder/overlay/shimmer_layer.py` | `recorder/overlay/state.py` | `STATE_COLORS[OverlayState.REPLAYING]` | WIRED | `elif state == OverlayState.REPLAYING:` branch in `set_state()` |
| `recorder/overlay/camera_flash.py` | `recorder/overlay/animation_clock.py` | `clock.register(self.tick)` in constructor | WIRED | CameraFlash self-registers with AnimationClock in constructor |
| `routine/runner.py` | `core/locate.py` | `locate_element_from_step` per step | WIRED | Line 42: `from core.locate import locate_element_from_step`; called in step loop and `_failure_cascade` |
| `routine/runner.py` | `core/executor.py` | click, type_text, scroll, drag, etc. | WIRED | Lines 30-41: `from core.executor import click, double_click, drag, hotkey, press_enter, prompt_user_blocking, right_click, scroll, select_all_extract, type_text` |
| `routine/runner.py` | `mapper/validator.py` | `validate_action` for post-action verification | WIRED | Lazy import inside step loop: `from mapper.validator import validate_action`; called for non-exempt actions |
| `routine/runner.py` | `routine/run_log.py` | `create_run_dir, prune_old_runs, annotate_screenshot` | WIRED | Lines 45-52: `from routine.run_log import annotate_screenshot, create_run_dir, prune_old_runs, save_annotated_screenshot, save_run_result, setup_run_logger` |
| `routine/replay_overlay.py` | `recorder/overlay/controller.py` | `controller.set_replay_mode`, `camera_flash`, etc. | WIRED | `self._controller.set_replay_mode()`, `self._controller.camera_flash()` in `_on_event` |
| `recorder/overlay/controller.py` | `recorder/overlay/view.py` | `view.set_replay_mode`, `view.camera_flash`, etc. | WIRED | `self._view.camera_flash()`, `self._view.show_target_highlight()`, etc. in controller replay API methods |
| `routine/replay_overlay.py` | `routine/runner.py` | `RunEvent` enum used for dispatch, including `SCREENSHOT_TAKEN` | WIRED | Line 17: `from routine.runner import RunEvent`; `case RunEvent.SCREENSHOT_TAKEN: self._controller.camera_flash()` |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| RUN-01 | 07-03 | User selects routine from TUI or triggers via CLI/API | SATISFIED | `run_routine(routine_dir)` callable entry point in `routine/runner.py`; accepts path and returns `RunResult` |
| RUN-02 | 07-01 | Per-step element location using 5-stage cascade | SATISFIED | `locate_element_from_step()` in `core/locate.py` implements all 5 stages for v1 step dicts |
| RUN-03 | 07-03 | Human-like mouse via Bézier curves with overshoot-and-correct | SATISFIED | `core/executor.py` `_bezier_move()` with cubic Bézier + random overshoot radius; runner calls executor for all click/drag actions |
| RUN-04 | 07-03 | Human-like typing with per-letter delays, variance, occasional typo + correction | SATISFIED | `core/executor.py` `type_text()` with `type_interval` delays and `typo_chance` simulation |
| RUN-05 | 07-03 | Configurable execution speed via human_delay multiplier | SATISFIED | `core/config.py` `execution.human_delay` driven by `OCSD_HUMAN_DELAY`; scales all executor timing |
| RUN-06 | 07-03 | Post-action validation via pixel-diff + optional VLM confirmation | SATISFIED | `routine/runner.py` — before/after screenshots, `validate_action()` called; `VALIDATION_PASSED/FAILED` events emitted |
| RUN-07 | 07-03 | Failure cascade: auto-retry (2x) → widen search → prompt user/agent → log | SATISFIED | `_failure_cascade()` in `routine/runner.py` — Stage 1: 2x retry, Stage 2: 25% region, Stage 3: full-screen, Stage 4: LiteLLM, Stage 5: abort + annotated screenshot |
| RUN-08 | 07-03 | Never silently fail or blind-click when unsure | SATISFIED | `_failure_cascade()` returns None on Stage 5; step loop checks `if locate_result is None: break` — never calls `_dispatch_action` without confirmed location |
| RUN-09 | 07-02, 07-04 | Step-by-step replay status visible to user | SATISFIED | `RunEvent.STEP_START` emitted per step; `ReplayOverlayAdapter` updates `StatusBadge`; `TargetHighlight` shows located element; `CameraFlash` confirms screenshot capture |

All 9 requirements (RUN-01 through RUN-09) are accounted for and satisfied. No orphaned requirements found.

---

### Anti-Patterns Found

No anti-patterns detected. Scan of all phase-modified files found:
- No TODO/FIXME/PLACEHOLDER comments
- No stub return values (`return null`, `return {}`, `return []`)
- No empty handlers
- No console.log-only implementations
- SCREENSHOT_TAKEN events properly emitted on all `screenshot_full()` calls in step loop and cascade stages

---

### Human Verification Required

#### 1. Replay visual feedback end-to-end

**Test:** Run a saved routine with `run_routine(routine_dir, callback=ReplayOverlayAdapter(controller))` against a live screen
**Expected:** Purple shimmer border visible at all times during replay; StatusBadge pill shows "Step N/M: Action Label" updating per step; TargetHighlight briefly glows purple around each located element before the click; CameraFlash 15px white border fires on every before/after screenshot
**Why human:** Qt animation quality, timing, and z-ordering cannot be verified programmatically. Requires a running QApplication and real screen capture.

#### 2. Failure cascade end-to-end

**Test:** Remove a snippet PNG from a routine, then run it. Observe cascade stages.
**Expected:** Two retries at position; 25% region scan attempt; full-screen attempt; LiteLLM prompt (if available); abort with annotated failure screenshot in `runs/{run_id}/stepN_failed.png`. No click executed. `RunResult.success = False`, `failure_reason` populated.
**Why human:** Requires real screen, real file system, and optionally LiteLLM connectivity. Stage transitions need live observation.

#### 3. Self-cleaning run directory pruning

**Test:** Execute the same routine more than 5 times successfully. Check `runs/` directory.
**Expected:** Only the 5 most recent successful run directories remain; older ones pruned automatically after each run.
**Why human:** Requires multiple actual runs; although tested with tmp_path fixtures, real-world path interaction warrants human confirmation.

---

### Gaps Summary

No gaps. All 9 requirements verified, all artifacts substantive and wired, all key links confirmed, all tests passing.

The one deviation from plan documented in 07-03-SUMMARY.md (n_iterations loop counter fix) was caught and corrected during implementation — the fix is verified in `test_loop_step_execution` which passes.

---

_Verified: 2026-03-19_
_Verifier: Claude (gsd-verifier)_
