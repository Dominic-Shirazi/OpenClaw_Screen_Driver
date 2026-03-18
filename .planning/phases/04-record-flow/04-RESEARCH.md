# Phase 4: Record Flow - Research

**Researched:** 2026-03-18
**Domain:** Recording session orchestration, state machine wiring, detection pipeline, dry-run execution
**Confidence:** HIGH

## Summary

Phase 4 wires together all Phase 1-3 overlay infrastructure (shimmer, scan animation, tag dialog, toolbar, donut cloud) with the core pipeline modules (capture, detection, vision, embeddings, executor) into a complete recording session. The legacy `record_controller.py` contains substantial reusable logic for bbox refinement, VLM labeling, snippet saving, and graph export -- but its architecture is a flat procedural flow with modal Qt dialogs (TagDialog.exec()) that must be replaced with the new overlay-integrated pipeline using non-blocking HUD panels.

The primary architectural challenge is sequencing: the recording flow is a complex state machine with async detection/VLM calls, user interactions through overlay panels, and a dry-run execution loop -- all running on the Qt main thread. The existing `_SignalBridge` pattern for background-to-main-thread communication is the established pattern and should be extended.

**Primary recommendation:** Build a `RecordSession` class that owns the recording state machine, step list, and pipeline orchestration. It receives callbacks from `OverlayController` (selection, toolbar actions, tag dialog events) and delegates to background threads for detection/VLM work via `_SignalBridge`. The `OverlayController` gains new API methods for the countdown widget and dry-run mode, but remains a thin delegation layer.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Pre-recording naming**: Rich TUI prompt in the terminal before overlay launches. No Qt UI for naming
- **Window minimize**: Auto-minimize all windows (Win+D equivalent) to establish clean desktop baseline
- **Minimize failure**: Warn user, offer Continue/Cancel -- never silently proceed or block forever
- **Start state**: Ask user "desktop or already-open app?" per routine -- stored in routine metadata
- **Bbox detection**: Cascade -- click-local crop first (~240px centered on click), run OmniParser on that region. If no element found, fall back to full-screen detection. Click location sent to OmniParser as context. Crop coordinates adjusted to screen space
- **Drag-highlight capture**: AI proposes tighter bbox from user's drag region. User can reject and keep original. Shown as morph animation user can undo
- **Bbox editing**: Both click AND drag captures allow user to edit/adjust bbox before accepting AI proposal. Bbox overlay is editable (drag corners) in both cases
- **VLM timing**: Sequential -- after bbox is accepted, scan animation plays, VLM runs, tag dialog opens with results. One element at a time. Card glow pulses as loading indicator
- **VLM failure handling**: Retry VLM once with shorter timeout. If still fails, open tag dialog with partial data from detection pipeline (type_guess, florence_caption). User completes manually
- **Countdown UX**: Cursor-following countdown spinner -- mouse changes to spinning "thinking" logo with countdown that follows cursor. Mouse stays free to move
- **Dry-run mandatory**: Every step must be dry-run tested before saved. No skip option
- **Overlay during execution**: Overlay switches to click-through mode during dry-run execution (stays visible, shimmer running, but fully transparent to input). Does NOT fully hide
- **Validation**: User confirms result -- toolbar shows Yes / No (edit step) / Retry after action executes
- **Edit on failure**: "No (edit step)" gives full redo -- user can edit tag dialog fields OR go back and re-click/re-drag
- **Retry behavior**: Full 3-2-1 countdown plays again before retry
- **Loop-back**: After user confirms "Yes", brief green success flash (~500ms), then back to recording mode
- **No mid-recording undo**: Once step confirmed via dry-run, it's locked. Edit/delete happens in Update flow (Phase 8)
- **Save format**: Full routine.json + snippet PNGs + CLIP embeddings
- **Save location**: `~/.ocsd/routines/{name}/` -- routine.json + snippets/ + embeddings/
- **Save error handling**: Show error on overlay, let user fix issue, retry. Don't lose session data
- **ESC abort**: Confirmation prompt -- "Are you sure? All recorded steps will be lost." If confirmed, discard everything. If no steps captured yet, close silently

### Claude's Discretion
- Exact cursor-following countdown spinner implementation (Qt custom cursor, small overlay widget, etc.)
- How to structure the interim routine.json before Phase 5 finalizes the format
- Detection pipeline threading and async orchestration
- How to implement "start from desktop vs already-open app" choice in routine metadata
- Green success flash implementation (which overlay element flashes)
- Error retry UX details (overlay panel vs toast vs toolbar message)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REC-01 | User names routine before recording starts | Rich TUI prompt before overlay launch; pathlib for `~/.ocsd/routines/{name}/` validation |
| REC-02 | System auto-minimizes all windows to desktop as reproducible baseline | Win32 `keybd_event(VK_LWIN + D)` or `ShowWindow(SW_MINIMIZE)` enumeration; platform guard |
| REC-03 | F2 toggles recording on/off, Ctrl+Q saves, ESC aborts | Existing hotkey infrastructure in `recorder/hotkeys.py`; state machine needs new transitions for sub-states |
| REC-04 | Click capture -- overlay hides, clean screenshot, AI bbox fitting, scan animation | `hide_for_capture()` + `screenshot_full()` + cascade detection + `start_scan()`/`finish_scan()` |
| REC-05 | Drag-highlight capture -- user drags region, AI tightens bbox | `mouseReleaseEvent` + `_try_refine_bbox` IoU logic + `BboxLayer.morph_to()` |
| REC-06 | Smart crop -- final bbox + 30% padding, VLM analysis, tag dialog | `_save_snippets_and_embeddings` crop logic + `analyze_crop_array()` + `show_tag_dialog()` |
| REC-07 | VLM auto-fill with manual fallback | `analyze_crop_array()` with timeout + tag dialog `show_dialog(vlm_data=None)` fallback path |
| REC-08 | Dry run per step -- Enter, 3-2-1 countdown, execute, validate | New countdown widget + `core.executor.click()` + toolbar Yes/No/Retry mode |
| REC-09 | Dry run does NOT block/lock the mouse | Cursor-following countdown widget; overlay click-through during execution |
| REC-10 | After dry run, loop back for next element until Ctrl+Q | State machine loop: CAPTURING -> TAGGING -> DRY_RUN -> CAPTURING |
| REC-11 | Save routine on Ctrl+Q -- routine.json + snippets/ + embeddings/ | Adapted `_save_recording` + `_save_snippets_and_embeddings` with `~/.ocsd/routines/` path |

</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyQt6 | >=6.5 | Overlay UI, countdown widget, timer events | Project UI framework |
| mss | >=9.0 | Screenshot capture | Fast, cross-platform screen grab |
| opencv-python | >=4.8 | Image crop/resize, BGR conversion | Standard CV library |
| pyautogui | >=0.9.54 | Mouse/keyboard execution for dry-run | Already used by executor |
| networkx | >=3.0 | Routine graph construction | Project graph library |
| rich | >=13.0 | TUI prompt for routine naming | Project TUI library (optional dep) |

### Supporting (already available)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| openai | >=1.0 | VLM calls via LiteLLM proxy | Element analysis after capture |
| transformers + torch | >=4.36 / >=2.1 | CLIP embeddings | Snippet embedding generation |
| faiss-cpu | >=1.7 | Embedding storage | Per-element save to index |

### No New Dependencies
This phase wires existing modules together. No new pip packages needed.

## Architecture Patterns

### Recommended Project Structure
```
recorder/
  record_session.py       # NEW: RecordSession class (pipeline orchestrator)
  record_flow.py          # NEW: cmd_record_v2() entry point (replaces legacy cmd_record)
  overlay/
    controller.py         # EXTEND: add countdown, click-through toggle, abort confirm
    view.py               # EXTEND: add countdown widget, click-through toggle
    countdown_widget.py   # NEW: cursor-following countdown spinner
    state.py              # EXTEND: add sub-states for recording pipeline phases
```

### Pattern 1: RecordSession State Machine
**What:** A class that manages the recording session lifecycle with explicit sub-states beyond the existing READY/RECORDING/PAUSED overlay states.
**When to use:** Always -- the recording pipeline has many intermediate states (waiting for click, detecting, tagging, dry-running, validating).

```python
class RecordPhase(Enum):
    """Sub-states within the RECORDING overlay state."""
    AWAITING_CLICK = auto()     # Red shimmer, crosshair cursor, waiting for user click/drag
    CAPTURING = auto()          # Overlay hidden, screenshot in progress
    DETECTING = auto()          # Background OmniParser running, scan animation visible
    BBOX_EDITING = auto()       # User editing/accepting AI bbox proposal
    VLM_ANALYZING = auto()      # VLM running, card glow pulsing
    TAG_DIALOG = auto()         # Tag dialog open, user reviewing/editing
    COUNTDOWN = auto()          # 3-2-1 countdown before dry-run
    EXECUTING = auto()          # Dry-run action in progress, overlay click-through
    VALIDATING = auto()         # Post-execution, toolbar shows Yes/No/Retry
    SUCCESS_FLASH = auto()      # Brief green flash, then back to AWAITING_CLICK
```

### Pattern 2: Signal Bridge for Background Work
**What:** Use QObject signals (existing `_SignalBridge` pattern) for detection and VLM results to cross from background threads to the Qt main thread.
**When to use:** Any time detection or VLM work runs in a background thread.

```python
class _PipelineBridge(QObject):
    """Thread-safe signal bridge for pipeline results."""
    detection_ready = pyqtSignal(dict)    # {bbox, candidates, type_guess}
    vlm_ready = pyqtSignal(dict)          # {element_type, label_guess, confidence, ocr_text}
    vlm_failed = pyqtSignal(str)          # error message
    save_complete = pyqtSignal(str)       # routine path
    save_failed = pyqtSignal(str)         # error message
```

### Pattern 3: Overlay Controller API Extension
**What:** Extend OverlayController with new methods for recording flow needs without bloating it -- keep it as a thin delegation layer.
**When to use:** For new visual capabilities needed by the recording flow.

New methods needed:
- `set_click_through(enabled: bool)` -- toggle click-through during dry-run execution
- `show_countdown(seconds: int)` -- start cursor-following countdown
- `hide_countdown()` -- remove countdown widget
- `show_abort_confirm() -> bool` -- ESC confirmation dialog
- `flash_success(bbox: QRectF)` -- brief green flash on confirmed step

### Pattern 4: Routine Save Format (Interim)
**What:** Save enough data for Phase 5 to refine, using the existing `export_skill` format with additional recording metadata.
**When to use:** On Ctrl+Q save.

```python
routine_data = {
    "$schema": "ocsd-routine-v0",  # interim, Phase 5 upgrades to v1
    "name": routine_name,
    "description": f"Recorded routine: {routine_name}",
    "start_from": "desktop" | "app",  # user choice
    "created_at": iso_timestamp,
    "resolution": [screen_w, screen_h],
    "steps": [
        {
            "step_index": 0,
            "node_id": uuid,
            "element_type": "button",
            "label": "Login Button",
            "caption": "Blue login button in top right",
            "action": "click",
            "bbox": {"x": 100, "y": 200, "w": 150, "h": 40},
            "bbox_pct": {"x_pct": 0.052, "y_pct": 0.185, ...},
            "region_hint": "top_right",
            "snippet_path": "snippets/step_00.png",
            "embedding_path": "embeddings/step_00.npy",
            "confidence": 0.92,
            "dry_run_passed": True,
        }
    ],
    # Also include the full graph for Phase 5
    "graph": { ... },  # OCSDGraph.to_dict()
}
```

### Anti-Patterns to Avoid
- **Modal dialogs (.exec()):** Never use `QDialog.exec()` in the recording flow. The legacy `TagDialog` uses `.exec()` which blocks the event loop. Use overlay-integrated panels (TagDialogPanel) which are non-blocking.
- **Synchronous VLM calls on main thread:** VLM calls take 2-10 seconds. Always run in a background thread with signal bridge.
- **Hiding overlay completely during dry-run:** Decision says overlay stays visible in click-through mode. Don't call `hide()` -- call `set_click_through(True)`.
- **Auto-saving partial routines:** On ESC abort, discard everything. No partial saves to disk.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Window minimize to desktop | Custom window enumeration | `keybd_event` for Win+D shortcut | Win+D is the OS standard; enumerating all windows misses edge cases |
| Bbox refinement IoU | New IoU logic | Adapt `_try_refine_bbox()` from record_controller | Already handles drawn bbox (IoU) and point click (smallest containing) |
| Click-local detection | New crop/detect flow | Adapt `_auto_snip()` from record_controller | Already handles radius crop + OmniParser + coordinate offset |
| Snippet saving with padding | New crop/save pipeline | Reuse `_save_snippets_and_embeddings()` logic | Handles 30% buffer, CLIP embedding, FAISS save |
| Graph building | New serialization | Use `OCSDGraph` + `export_skill()` | Full schema validation, checksum, node/edge CRUD |
| Human-like click execution | New mouse automation | `core.executor.click()` | Already has Bezier curves, doughnut offset, human delay |
| VLM element analysis | New prompt engineering | `core.vision.analyze_crop_array()` | Already handles prompt, JSON extraction, element type sanitization |
| Screenshot with overlay hidden | New hide/show cycle | `OverlayController.hide_for_capture()` / `show_after_capture()` | Already handles clock stop/start, HUD panel visibility, DWM flush |

**Key insight:** ~80% of the recording pipeline logic already exists in either the legacy `record_controller.py` or the core modules. Phase 4 is primarily an integration/wiring task, not a greenfield implementation.

## Common Pitfalls

### Pitfall 1: Main Thread Blocking During Detection
**What goes wrong:** OmniParser detection takes 1-5 seconds. Running it on the main thread freezes the UI (no animations, no cursor updates).
**Why it happens:** Easy to call `detector.detect()` synchronously after screenshot.
**How to avoid:** Always use `threading.Thread(target=..., daemon=True)` for detection, deliver results via `_SignalBridge.candidates_ready.emit()`.
**Warning signs:** Shimmer animation stutters or freezes during detection.

### Pitfall 2: Screenshot Contains Overlay Artifacts
**What goes wrong:** The overlay is visible in the screenshot, polluting the image for detection.
**Why it happens:** `hide_for_capture()` needs 80ms+ DWM flush time before mss captures. Skipping or shortening this delay causes partial overlay artifacts.
**How to avoid:** Always call `hide_for_capture()`, wait `capture_delay_ms` (from config, default 80ms), then capture. Already handled by the existing infrastructure.
**Warning signs:** OmniParser detects overlay border elements.

### Pitfall 3: Click-Through State Not Restored After Dry-Run
**What goes wrong:** After dry-run execution, overlay stays in click-through mode, so user clicks pass through to the desktop instead of being captured.
**Why it happens:** Forgetting to restore click-through=False after execution completes.
**How to avoid:** Use try/finally or a context manager for the click-through toggle.
**Warning signs:** User clicks after dry-run validation don't register on the overlay.

### Pitfall 4: Coordinate System Mismatch Between Crop and Screen
**What goes wrong:** Detection on a cropped region returns coordinates relative to the crop, but overlay renders in screen coordinates.
**Why it happens:** Forgetting to add the crop origin offset back to detected bboxes.
**How to avoid:** The existing `_auto_snip()` already handles this correctly (adds `x1`, `y1` back to detected `rect`). Follow the same pattern for the cascade detection.
**Warning signs:** Bbox appears in wrong location on screen (offset by the crop origin).

### Pitfall 5: Race Condition Between User Clicks and Pipeline State
**What goes wrong:** User clicks while detection is still running from previous click, creating a second pipeline instance.
**Why it happens:** Click events are not gated on pipeline state.
**How to avoid:** Disable click capture (set phase to DETECTING/TAGGING/etc.) immediately after first click. Only re-enable when back in AWAITING_CLICK.
**Warning signs:** Multiple tag dialogs open simultaneously, or overlapping scan animations.

### Pitfall 6: VLM Timeout Causes Silent Hang
**What goes wrong:** VLM call hangs indefinitely if LiteLLM proxy is slow or down.
**Why it happens:** No timeout on the HTTP request to LiteLLM.
**How to avoid:** Use timeout parameter on the OpenAI client call. First attempt: 15s. Retry: 8s. If both fail, proceed with partial data.
**Warning signs:** Recording session appears frozen after bbox acceptance -- card glow pulsing indefinitely.

### Pitfall 7: Routine Name Collision
**What goes wrong:** User enters a name that already exists in `~/.ocsd/routines/`.
**Why it happens:** No existence check before starting recording.
**How to avoid:** Check `~/.ocsd/routines/{name}/` existence in the TUI prompt. Offer rename or overwrite.
**Warning signs:** Previous routine silently overwritten.

## Code Examples

### Window Minimize (Win+D Simulation)
```python
# Source: Windows API documentation + existing platform pattern in the codebase
import ctypes
import sys
import time

def minimize_all_windows() -> bool:
    """Minimizes all windows to show desktop (Win+D equivalent).

    Returns:
        True if successful, False if platform not supported.
    """
    if sys.platform != "win32":
        logger.warning("Window minimize only supported on Windows")
        return False

    # Simulate Win+D keypress
    VK_LWIN = 0x5B
    VK_D = 0x44
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32

    user32.keybd_event(VK_LWIN, 0, 0, 0)
    user32.keybd_event(VK_D, 0, 0, 0)
    user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)

    time.sleep(0.5)  # Wait for animation
    return True
```

### Cascade Detection (Click-Local then Full-Screen)
```python
# Source: Adapted from record_controller._auto_snip() + CONTEXT.md cascade decision
def cascade_detect(
    screenshot: np.ndarray,
    click_x: int,
    click_y: int,
    local_radius: int = 120,
) -> dict | None:
    """Runs cascade detection: local crop first, full-screen fallback.

    Args:
        screenshot: Full clean screenshot (BGR).
        click_x: Click X in screen coordinates.
        click_y: Click Y in screen coordinates.
        local_radius: Pixel radius for local crop.

    Returns:
        Best candidate dict with screen-space rect, or None.
    """
    detector = get_detector()

    # Stage 1: Click-local crop
    sh, sw = screenshot.shape[:2]
    x1 = max(0, click_x - local_radius)
    y1 = max(0, click_y - local_radius)
    x2 = min(sw, click_x + local_radius)
    y2 = min(sh, click_y + local_radius)
    crop = screenshot[y1:y2, x1:x2]

    if crop.size > 0:
        candidates = detector.detect(crop)
        if candidates:
            # Offset back to screen space
            best = _find_nearest(candidates, click_x - x1, click_y - y1)
            if best:
                r = best["rect"]
                best["rect"] = {
                    "x": r["x"] + x1, "y": r["y"] + y1,
                    "w": r["w"], "h": r["h"],
                }
                return best

    # Stage 2: Full-screen fallback
    candidates = detector.detect(screenshot)
    if candidates:
        return _find_nearest(candidates, click_x, click_y)

    return None
```

### Countdown Widget (Cursor-Following)
```python
# Source: Architecture decision -- small QGraphicsObject overlay item
class CountdownWidget(QGraphicsObject):
    """Cursor-following countdown spinner.

    Renders a circular countdown (3, 2, 1) that tracks the mouse
    position. The number shrinks and fades each tick.
    """

    countdown_finished = pyqtSignal()

    def __init__(self, seconds: int = 3) -> None:
        super().__init__()
        self._total = seconds
        self._remaining = seconds
        self._timer = QTimer()
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self.setZValue(300)  # Above everything

    def start(self) -> None:
        self._remaining = self._total
        self._timer.start()
        self.setVisible(True)

    def _tick(self) -> None:
        self._remaining -= 1
        self.update()
        if self._remaining <= 0:
            self._timer.stop()
            self.setVisible(False)
            self.countdown_finished.emit()

    def set_position(self, x: float, y: float) -> None:
        """Called from mouseMoveEvent to track cursor."""
        self.setPos(x + 20, y + 20)  # Offset from cursor
```

### Click-Through Toggle During Dry-Run
```python
# Source: Existing pattern in view.py apply_state()
def set_click_through(self, enabled: bool) -> None:
    """Toggle click-through mode without changing overlay state."""
    if sys.platform == "win32":
        from recorder.overlay.platform_win32 import set_click_through_win32
        try:
            set_click_through_win32(int(self.winId()), enabled)
        except RuntimeError:
            logger.debug("Window not realised for click-through toggle")
    else:
        from recorder.overlay.platform_linux import set_click_through_linux
        set_click_through_linux(self, enabled)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Modal TagDialog (.exec()) | Non-blocking TagDialogPanel | Phase 3 | Dialog no longer blocks event loop |
| Legacy OverlayMode enum | OverlayState + RecordPhase | Phase 1 + Phase 4 | Finer-grained state control |
| Border gradient layer | ShimmerLayer with mouse retreat | Phase 2 | Better visual indicator |
| Flat `cmd_record()` function | RecordSession class | Phase 4 (planned) | Testable, state-machine driven |

**Deprecated/outdated:**
- `recorder.dialog.TagDialog`: Legacy modal dialog. Replaced by `recorder.overlay.tag_dialog_panel.TagDialogPanel`
- `recorder.overlay.OverlayMode`: Legacy enum from old overlay. Replaced by `recorder.overlay.state.OverlayState`
- `record_controller.cmd_record()`: Legacy entry point. Will be replaced by new `record_flow.cmd_record()`

## Open Questions

1. **How should the countdown widget track the mouse across the full screen?**
   - What we know: The overlay view already has `mouseMoveEvent` which tracks mouse position for shimmer. During COUNTDOWN phase the overlay is in recording mode (not click-through), so it receives mouse events.
   - What's unclear: Whether to use a small separate always-on-top widget, or an item in the QGraphicsScene.
   - Recommendation: Use a QGraphicsObject in the scene (consistent with all other overlay elements). Update its position from `mouseMoveEvent`. Z-value 300 (above everything).

2. **How to handle "Edit step" after failed dry-run?**
   - What we know: User can either edit tag dialog fields OR go back to re-click/re-drag.
   - What's unclear: Exact UX flow -- does a sub-menu appear? Does toolbar get new buttons?
   - Recommendation: Toolbar shows "Edit Tags" and "Re-capture" buttons. "Edit Tags" reopens tag dialog with existing data (edit_mode=True). "Re-capture" returns to AWAITING_CLICK and discards current step.

3. **Should the interim routine.json include the full graph, or just a step list?**
   - What we know: Phase 5 will finalize the format. The graph contains more metadata than a flat step list.
   - What's unclear: Whether Phase 5 will want graph data or rebuild from steps.
   - Recommendation: Save both -- a flat `steps` array (simple, human-readable) and a `graph` dict (OCSDGraph.to_dict()). Phase 5 can choose which to use.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=7.0 + pytest-mock >=3.0 |
| Config file | pyproject.toml [tool.pytest] (implicit) |
| Quick run command | `python -m pytest tests/test_record_session.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REC-01 | Routine naming via TUI prompt | unit | `python -m pytest tests/test_record_session.py::test_naming -x` | No -- Wave 0 |
| REC-02 | Window minimize to desktop | unit | `python -m pytest tests/test_record_session.py::test_minimize_windows -x` | No -- Wave 0 |
| REC-03 | F2/Ctrl+Q/ESC hotkey routing | unit | `python -m pytest tests/test_record_session.py::test_hotkey_routing -x` | No -- Wave 0 |
| REC-04 | Click capture pipeline | unit | `python -m pytest tests/test_record_session.py::test_click_capture -x` | No -- Wave 0 |
| REC-05 | Drag-highlight capture + AI tighten | unit | `python -m pytest tests/test_record_session.py::test_drag_capture -x` | No -- Wave 0 |
| REC-06 | Smart crop + VLM + tag dialog | unit | `python -m pytest tests/test_record_session.py::test_smart_crop_vlm -x` | No -- Wave 0 |
| REC-07 | VLM fallback on failure | unit | `python -m pytest tests/test_record_session.py::test_vlm_fallback -x` | No -- Wave 0 |
| REC-08 | Dry-run countdown + execute | unit | `python -m pytest tests/test_record_session.py::test_dry_run -x` | No -- Wave 0 |
| REC-09 | Mouse free during countdown | unit | `python -m pytest tests/test_record_session.py::test_mouse_free_countdown -x` | No -- Wave 0 |
| REC-10 | Loop back after dry-run confirm | unit | `python -m pytest tests/test_record_session.py::test_loop_back -x` | No -- Wave 0 |
| REC-11 | Save routine.json + snippets + embeddings | unit | `python -m pytest tests/test_record_session.py::test_save_routine -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_record_session.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_record_session.py` -- covers REC-01 through REC-11 (mock all core modules)
- [ ] `tests/test_countdown_widget.py` -- covers countdown rendering and signal emission
- [ ] `tests/test_record_flow.py` -- integration test for cmd_record entry point

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `recorder/record_controller.py` (legacy flow, reusable logic)
- Codebase analysis: `recorder/overlay/controller.py` (current overlay API)
- Codebase analysis: `recorder/overlay/view.py` (capture lifecycle, HUD panels)
- Codebase analysis: `recorder/overlay/state.py` (state machine)
- Codebase analysis: `core/capture.py`, `core/detection.py`, `core/vision.py`, `core/embeddings.py`, `core/executor.py`
- Codebase analysis: `recorder/hotkeys.py` (F2/Ctrl+Q/ESC handling)

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions (user-locked choices from discussion session)
- Phase 1-3 implementation patterns (established by previous successful phases)

### Tertiary (LOW confidence)
- Countdown widget UX: Based on standard Qt patterns, not verified with specific library docs. Implementation details are Claude's discretion per CONTEXT.md.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, no new deps
- Architecture: HIGH -- follows established Phase 1-3 patterns (layer-based, signal bridge, controller delegation)
- Pitfalls: HIGH -- derived from reading actual codebase issues and patterns
- Countdown widget: MEDIUM -- implementation approach is sound but specific Qt rendering details need validation during implementation

**Research date:** 2026-03-18
**Valid until:** 2026-04-18 (stable -- all dependencies already locked in project)
