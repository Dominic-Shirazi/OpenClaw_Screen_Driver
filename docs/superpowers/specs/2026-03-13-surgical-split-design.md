# Surgical Split — Architecture Refactor

**Date:** 2026-03-13
**Goal:** Split 3 bloated files into focused modules so overlay rendering, recording logic, and execution can be changed independently without breaking each other.

## Problem

- `overlay.py` (1,140 lines) mixes Qt graphics primitives, window management, hotkey handling, and controller logic. Touching rendering breaks the state machine.
- `main.py` (1,273 lines) mixes argparse, recording callbacks, auto-snip, smart detect, review flow, and execution. It's a god module.
- `mapper/runner.py` (690 lines) mixes the 5-stage locate cascade with the replay loop and event system.

Additionally: no GPU memory coordination, stale Docker/YOLOE artifacts, and an outdated README.

## Approach

Extract, don't rewrite. Move code into focused files. Keep all public APIs stable so nothing downstream breaks.

---

## 1. Overlay Split

### `recorder/hotkeys.py` (~170 lines)
Extracted verbatim from overlay.py:
- `_Win32PollingHotkeyListener` — GetAsyncKeyState polling on QTimer
- `_PynputHotkeyListener` — pynput fallback for macOS/Linux
- `_create_hotkey_listener(on_toggle, on_close)` — factory function

### `recorder/overlay_items.py` (~200 lines)
Pure Qt graphics primitives, no business logic:
- `_TYPE_COLORS`, `_DEFAULT_COLOR`, `_HANDLE_RADIUS`, `_BORDER_WIDTH` constants
- `_HandleItem(QGraphicsEllipseItem)` — draggable corner handle
- `_ElementBoxGroup` — rect + 4 handles + label + donut gradient + center dot

### `recorder/overlay_view.py` (~350 lines)
The Qt window:
- `_OverlayView(QGraphicsView)` — window setup, Win32 layered flags, scene management
- `render_candidates()`, `clear_scene()`, `highlight_candidate()`, `get_candidate_rect()`
- Mouse events: press/move/release with handle detection
- `refresh_overlay()`, `_draw_border()`, `_update_mode_indicator()`, `_update_click_catcher()`
- Imports `_HandleItem`, `_ElementBoxGroup` from `overlay_items`

### `recorder/overlay.py` (~250 lines, slimmed)
Controller only:
- `OverlayMode` enum
- `OverlayController` — mode state machine, candidate list management, review flow, callbacks
- `set_candidates()`, `start_review()`, `highlight_candidate()`, `close()`
- Imports `_OverlayView` from `overlay_view`, listener from `hotkeys`
- **Public API unchanged** — `from recorder.overlay import OverlayController` still works

---

## 2. main.py Split

### `recorder/record_controller.py` (~500 lines)
All recording logic extracted from main.py:
- `_auto_snip(x, y, radius)` — snip region around click, run detection
- `_try_refine_bbox(...)` — bbox refinement with OmniParser
- `_trigger_smart_detect(...)` — async detection trigger
- `_start_review(...)` — one-by-one element review with TagDialog
- `on_element_clicked(...)` — click handler callback
- `cmd_record(args)` — overlay setup, callback wiring, skill saving
- `cmd_diagram(args)` — diagram mode variant

### `mapper/execute_controller.py` (~200 lines)
All execution logic:
- `cmd_execute(args)` — load skill, plan path, run replay
- `cmd_compose(args)` — compose mode if separate from record
- Dry-run support

### `main.py` (~150 lines, thin router)
- `_setup_logging()`, `_setup_dpi_awareness()`
- `build_parser()` — argument definitions
- `main()` — parse args, lazy-import and dispatch to controllers
- No business logic

---

## 3. Runner Split

### `core/locate.py` (~300 lines)
The 5-stage locate cascade as a standalone module:
- `locate_element(graph, node_id, screen) -> LocateResult | None`
- Private stage functions: `_try_omniparser()`, `_try_clip()`, `_try_ocr()`, `_try_vlm()`, `_try_position()`
- Each stage is self-contained, returns `LocateResult` or `None`

### `mapper/runner.py` (~350 lines, slimmed)
Replay loop only:
- `replay_skill()` — step through nodes, call `locate_element()`, execute actions
- Event emission, step logging, ReplayLog generation
- Imports `locate_element` from `core/locate`

---

## 4. GPU Coordination

### `core/gpu.py` (~50 lines)
Centralized GPU memory management:
- `cleanup()` — `torch.cuda.empty_cache()` with ImportError guard
- `get_device(role: str) -> str` — returns configured device for "vlm" (cuda:0), "embeddings" (cuda:1), etc.
- Replaces scattered try/except torch blocks in florence.py, smart_detect.py, main.py

---

## 5. Repo Cleanup

### Delete
- `Dockerfile` — premature, not needed for V1
- `docker-entrypoint.sh` — premature
- `scripts/benchmark_yoloe_ui.py` — references dead YOLOE code
- `Claude_CLI_Guide.md` — internal tooling doc, not part of the product
- `Gemini_CLI_Guide.md` — internal tooling doc, not part of the product
- `PROGRESS.md` — stale build log, superseded by git history

### Update
- `pyproject.toml` — remove dead `yoloe` dependency group
- `README.md` — rewrite to reflect current architecture (OmniParser not YOLOE, no executor/ dir, accurate file list, correct locate cascade stages)
- `.gitignore` — verify Docker artifacts excluded

---

## 6. What Does NOT Change

These files are untouched:
- `core/types.py`, `core/config.py`, `core/capture.py`, `core/executor.py`
- `core/omniparser.py`, `core/florence.py`, `core/embeddings.py` (except using `gpu.cleanup()`)
- `core/ocr.py`, `core/vision.py`, `core/watcher.py`, `core/accessibility.py`, `core/model_cache.py`
- `core/detection.py`
- `mapper/graph.py`, `mapper/export.py`, `mapper/pathfinder.py`, `mapper/validator.py`
- `mapper/orchestrator.py`, `mapper/diff.py`, `mapper/layers.py`
- `recorder/dialog.py`, `recorder/refine_dialog.py`, `recorder/smart_detect.py`
- `recorder/session.py`, `recorder/element_types.py`, `recorder/step_ui.py`, `recorder/tui.py`
- `hub/manifest.py`, `hub/scanner.py`, `hub/schema.py`
- `api/server.py`

---

## 7. File Layout After

```
main.py                          (~150 lines, argparse + dispatch)
core/
    gpu.py                       (~50 lines, GPU coordination)
    locate.py                    (~300 lines, 5-stage cascade)
    types.py                     (unchanged)
    config.py                    (unchanged)
    capture.py                   (unchanged)
    detection.py                 (unchanged)
    executor.py                  (unchanged)
    omniparser.py                (unchanged)
    florence.py                   (unchanged, uses gpu.cleanup())
    embeddings.py                (unchanged)
    ocr.py                       (unchanged)
    vision.py                    (unchanged)
    watcher.py                   (unchanged)
    accessibility.py             (unchanged)
    model_cache.py               (unchanged)
recorder/
    hotkeys.py                   (~170 lines, keyboard backends)
    overlay.py                   (~250 lines, OverlayController only)
    overlay_view.py              (~350 lines, Qt window)
    overlay_items.py             (~200 lines, graphics primitives)
    record_controller.py         (~500 lines, recording flow)
    dialog.py                    (unchanged)
    refine_dialog.py             (unchanged)
    smart_detect.py              (unchanged)
    session.py                   (unchanged)
    element_types.py             (unchanged)
    step_ui.py                   (unchanged)
    tui.py                       (unchanged)
mapper/
    execute_controller.py        (~200 lines, execution flow)
    runner.py                    (~350 lines, replay loop only)
    graph.py                     (unchanged)
    export.py                    (unchanged)
    pathfinder.py                (unchanged)
    validator.py                 (unchanged)
    orchestrator.py              (unchanged)
    diff.py                      (unchanged)
    layers.py                    (unchanged)
hub/                             (unchanged)
api/                             (unchanged)
tests/                           (unchanged — imports stay stable)
```

---

## 8. Testing Strategy

- All existing tests must pass without modification after the split
- New imports (`from recorder.hotkeys import ...`) are internal — no test changes needed
- Run full suite after each extraction step to catch breakage immediately
- If any test imports a moved function, update that import (unlikely — tests import public APIs)

---

## 9. Implementation Order

1. Create `core/gpu.py` (no dependencies, foundational)
2. Extract `recorder/hotkeys.py` (self-contained, no coupling)
3. Extract `recorder/overlay_items.py` (pure data, no coupling)
4. Extract `recorder/overlay_view.py` (depends on overlay_items)
5. Slim `recorder/overlay.py` (depends on overlay_view + hotkeys)
6. Extract `core/locate.py` from `mapper/runner.py`
7. Slim `mapper/runner.py`
8. Extract `recorder/record_controller.py` from `main.py`
9. Extract `mapper/execute_controller.py` from `main.py`
10. Slim `main.py`
11. Delete stale files (Dockerfile, docker-entrypoint.sh, scripts/benchmark_yoloe_ui.py, CLI guides, PROGRESS.md)
12. Update pyproject.toml (remove yoloe group)
13. Rewrite README.md
14. Run full test suite, verify all pass
