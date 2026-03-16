# Architecture Research

**Domain:** Screen automation tool with cinematic PyQt6 overlay, Rich TUI, and FastAPI server
**Researched:** 2026-03-16
**Confidence:** HIGH (based on existing codebase inspection + verified Qt/FastAPI documentation)

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Entry Layer                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  Rich TUI        │  │  CLI (argparse)  │  │  FastAPI (uvicorn        │  │
│  │  (blocking       │  │  (main.py)       │  │  daemon thread)          │  │
│  │   pre-launch)    │  │                  │  │                          │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────────┬─────────────┘  │
│           │                     │                          │                │
│           └─────────────────────┴──────────────────────────┘                │
│                                 │                                            │
├─────────────────────────────────┼────────────────────────────────────────────┤
│                    Qt Application Layer                                      │
│                    (QApplication, main thread)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    OverlayController                                  │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  CinematicOverlayView (QGraphicsView, fullscreen transparent) │   │   │
│  │  │  ┌─────────────────┐  ┌────────────────┐  ┌──────────────┐  │   │   │
│  │  │  │ BorderGlowItem  │  │  ScanAnimItems │  │ HUDTagPanel  │  │   │   │
│  │  │  │ (state-driven   │  │  (perimeter    │  │ (frosted     │  │   │   │
│  │  │  │  shimmer)       │  │   trace, sweep)│  │  glass QFrame│  │   │   │
│  │  │  └─────────────────┘  └────────────────┘  └──────────────┘  │   │   │
│  │  │  ┌─────────────────┐  ┌────────────────┐  ┌──────────────┐  │   │   │
│  │  │  │ ElementBoxGroup │  │  DonutProbItem │  │ FloatingTool │  │   │   │
│  │  │  │ (bbox + handles)│  │  (click cloud) │  │  bar (drag.) │  │   │   │
│  │  │  └─────────────────┘  └────────────────┘  └──────────────┘  │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                 │                                            │
├─────────────────────────────────┼────────────────────────────────────────────┤
│                    Recording / Execution Layer                               │
│  ┌───────────────────────┐  ┌───────────────────────────────────────────┐   │
│  │  RecordController     │  │  ExecuteController / Orchestrator         │   │
│  │  (session, capture,   │  │  (runner, pathfinder, validator)          │   │
│  │   bbox fitting, VLM,  │  │                                           │   │
│  │   routine save)       │  │                                           │   │
│  └────────────┬──────────┘  └───────────────────────────────────────────┘   │
│               │                                                              │
├───────────────┼──────────────────────────────────────────────────────────────┤
│               │               Core Services Layer                            │
│  ┌────────────▼────────────────────────────────────────────────────────┐    │
│  │  capture  │  locate  │  detection  │  embeddings  │  vision  │  executor │
│  │  (mss)    │ (cascade)│ (OmniParser)│ (CLIP+FAISS) │ (LiteLLM)│(PyAutoGUI)│
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                    Persistence Layer                                         │
│  ┌──────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  │
│  │  routine.json        │  │  snippets/         │  │  embeddings/       │  │
│  │  (NetworkX → JSON)   │  │  (PNG crops)       │  │  (FAISS vectors)   │  │
│  └──────────────────────┘  └────────────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| CinematicOverlayView | Transparent fullscreen window; all animations, HUD panels, bbox items | OverlayController (via signals/callbacks), QGraphicsScene items |
| OverlayController | State machine (PASSTHROUGH/RECORD/SCANNING/IDLE); coordinates hide-capture-show cycle | CinematicOverlayView, RecordController, HotkeyListener |
| BorderGlowItem | Animated perimeter border (green=ready, red=recording); QPropertyAnimation on opacity + blur radius | CinematicOverlayView scene |
| ScanAnimItems | Element-scan animations: perimeter trace + sweep line; driven by QTimeLine | CinematicOverlayView scene, triggered by OverlayController |
| DonutProbItem | Probability cloud donut ring around click target; draws N concentric arcs with opacity falloff | CinematicOverlayView scene |
| HUDTagPanel | Frosted-glass dialog for element tagging; typewriter-fill for VLM suggestions | OverlayController, RecordController (VLM result callback) |
| FloatingToolbar | Draggable context-sensitive action bar | OverlayController |
| RecordController | Orchestrates full F2→click→hide→screenshot→AI bbox→tag→dry-run→save flow | OverlayController, core.capture, core.detection, core.vision, mapper.graph |
| ExecuteController | Loads routine.json, runs step-by-step, fires progress events | mapper.runner, mapper.orchestrator, mapper.validator |
| Rich TUI | Pre-launch blocking menu; routine browser; loading screen with community routines | main.py (returns command+kwargs, then exits before Qt starts) |
| FastAPI server | REST/MCP endpoints for programmatic routine control; runs in daemon thread | ExecuteController, mapper.graph, core.config |
| HotkeyListener | Background thread; global keyboard hooks (pynput) | OverlayController (via thread-safe Qt signal emission) |

## Recommended Project Structure

```
recorder/
├── overlay.py               # OverlayController: state machine, hide/capture/show cycle
├── overlay_view.py          # CinematicOverlayView: QGraphicsView window setup
├── overlay_items.py         # All QGraphicsItem subclasses
│   ├── BorderGlowItem       # Animated perimeter glow
│   ├── ScanAnimItems        # Perimeter trace + sweep line
│   ├── DonutProbItem        # Probability cloud rings
│   └── ElementBoxGroup      # Bbox + corner handles (existing, may need animation)
├── hud/
│   ├── tag_panel.py         # HUDTagPanel: frosted glass + typewriter dialog
│   └── toolbar.py           # FloatingToolbar: draggable context bar
├── record_controller.py     # RecordController: orchestrates capture→AI→save
├── session.py               # RecordSession: in-memory routine state
├── hotkeys.py               # HotkeyListener: pynput global hooks
└── tui.py                   # Rich TUI: pre-launch blocking menu

api/
└── server.py                # FastAPI app: REST/MCP endpoints

mapper/
├── graph.py                 # OCSDGraph: NetworkX DiGraph wrapper
├── export.py                # routine.json serialization (evolving format)
├── runner.py                # Step-by-step replay with wait loop
├── orchestrator.py          # Pre-flight + recovery coordination
├── validator.py             # Pixel-diff action validation
└── pathfinder.py            # Graph path computation

core/
├── capture.py               # mss screenshot, smart crop, snippet save
├── locate.py                # 5-stage cascade element location
├── detection.py             # OmniParser/YOLOv8 candidates
├── embeddings.py            # CLIP + FAISS
├── ocr.py                   # Tesseract
├── vision.py                # LiteLLM VLM calls
├── executor.py              # PyAutoGUI mouse/keyboard (thread-safe)
├── types.py                 # Shared dataclasses
└── config.py                # YAML config loader
```

### Structure Rationale

- **recorder/hud/**: The frosted-glass HUD components (tag dialog, toolbar) are visually and logically distinct from the animation items that live directly on the QGraphicsScene. Separating them prevents overlay_items.py from growing unmanageable.
- **recorder/overlay_items.py**: All QGraphicsItem subclasses stay in one file to keep the custom paint/animation patterns co-located and reviewable together.
- **api/server.py**: FastAPI stays isolated; it does not import from `recorder/` (recording is not API-driven in V1).

## Architectural Patterns

### Pattern 1: Hide-Flush-Capture-Show Cycle

**What:** Before any mss screenshot, the overlay hides ALL visible items atomically, flushes Qt's paint buffer, captures the clean desktop, then restores items.

**When to use:** Every screenshot during record flow (click capture, drag capture, dry-run validation).

**Trade-offs:** Introduces ~50-100ms latency per capture step. Acceptable for recording; non-issue for replay (no overlay during replay). The alternative — painting a black mask over the overlay — leaves Qt window chrome artifacts on some compositors.

**Implementation (MEDIUM confidence — based on Qt docs and forum patterns):**

```python
def hide_for_capture(self) -> None:
    """Hides all overlay items, flushes paint, for clean screenshot."""
    self._view.hide()
    QApplication.processEvents()  # flush pending repaints
    # On Windows: additionally call user32.UpdateWindow(hwnd) for certainty

def show_after_capture(self) -> None:
    """Restores overlay visibility after screenshot is taken."""
    self._view.show()
    self._view.raise_()
```

Key detail: `QApplication.processEvents()` after `hide()` is necessary to force the compositor to repaint the desktop before mss grabs the screen. On Windows with DWM, an additional `time.sleep(0.02)` guard is often required because DWM composites asynchronously.

### Pattern 2: QGraphicsItem + QPropertyAnimation for Cinematic Effects

**What:** Custom QGraphicsItem subclasses (or QObject + QGraphicsItem multiple inheritance) expose custom Qt properties. QPropertyAnimation drives those properties on a curve. QGraphicsScene.update() is called by the animation framework automatically.

**When to use:** Border glow shimmer, donut probability fade-in, scan-line sweep. Any animation that maps cleanly to a single animatable numeric value.

**Trade-offs:** QPropertyAnimation is the right tool for property-driven transitions. For frame-by-frame canvas animations (scan perimeter trace), a QTimer at 16ms (≈60fps) driving manual `update()` calls is more direct and avoids the overhead of Qt's animation group machinery.

```python
class BorderGlowItem(QObject, QGraphicsRectItem):
    """Screen-edge glow with animatable intensity."""

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        QGraphicsRectItem.__init__(self)
        self._intensity = 0.0

    @pyqtProperty(float)
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        self._intensity = value
        self.update()  # triggers paint

    def paint(self, painter, option, widget=None):
        # Draw glow with self._intensity as alpha multiplier
        ...

# Animation usage:
anim = QPropertyAnimation(border_item, b"intensity")
anim.setDuration(1200)
anim.setStartValue(0.3)
anim.setEndValue(1.0)
anim.setEasingCurve(QEasingCurve.Type.SineCurve)
anim.setLoopCount(-1)  # infinite
anim.start()
```

**Note:** `QObject` must come before `QGraphicsItem` in the MRO for `pyqtProperty` to work correctly with PyQt6.

### Pattern 3: Frosted Glass HUD via QWidget Compositing

**What:** True frosted glass (blur of background pixels) requires platform-specific APIs. On Windows, the DWM acrylic brush is accessible via ctypes `DwmSetWindowAttribute` with `DWMWA_USE_IMMERSIVE_DARK_MODE` and accent policy. On Linux with a compositor, `_NET_WM_WINDOW_TYPE_DIALOG` + compositor blur hints work but are unreliable.

**Practical approach for V1:** Fake frosted glass using a semi-transparent QFrame (rgba 30-40 alpha white) positioned over the scene, with `QGraphicsBlurEffect` applied to a screenshot of the region captured *before* the HUD opens. The blur effect is applied to a QLabel holding the captured region image, then the dialog floats over it.

**When to use:** HUD tag dialog only — not for the border or scan items.

**Trade-offs:** The real-blur approach requires a region screenshot before showing the dialog, adding a small latency. The fake semi-transparent approach is instant but less visually impressive. For V1, fake semi-transparent is recommended given the platform complexity.

```python
class HUDTagPanel(QFrame):
    """Frosted-glass appearance via semi-transparent background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: rgba(20, 20, 30, 180);
                border: 1px solid rgba(120, 200, 255, 120);
                border-radius: 12px;
            }
        """)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
```

### Pattern 4: Uvicorn in a Daemon Thread (FastAPI Coexistence)

**What:** FastAPI/uvicorn runs in a dedicated daemon thread alongside the Qt event loop on the main thread. The Qt main loop and uvicorn asyncio event loop are completely separate.

**When to use:** When `ocsd api` mode is requested, or when `--serve` flag is passed at startup.

**Trade-offs:** Uvicorn's `install_signal_handlers=False` is required when running in a non-main thread (uvicorn raises RuntimeError otherwise). Communication between the FastAPI thread and Qt/recording state uses thread-safe queues or shared state protected by `threading.Lock` — never asyncio primitives crossing the boundary.

```python
import threading
import uvicorn
from api.server import app

def start_api_server(host: str = "127.0.0.1", port: int = 8420) -> None:
    """Starts uvicorn in a daemon thread. Call before QApplication.exec()."""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # must not install in non-main thread
    thread = threading.Thread(target=server.run, daemon=True, name="ocsd-api")
    thread.start()
    return server  # caller can set server.should_exit = True to stop
```

**Important:** `install_signal_handlers = lambda: None` (monkey-patch) is the established workaround per uvicorn issue #650 and community patterns. Confirmed as the approach in production use as of uvicorn 0.34.x.

### Pattern 5: Rich TUI as Pre-Launch Gate (Not Concurrent)

**What:** Rich TUI runs synchronously *before* QApplication is created. It returns a `(command, kwargs)` tuple, then exits. The Qt application starts after the TUI completes. The two never run concurrently.

**When to use:** Interactive human-driven launch via `ocsd` CLI with no arguments.

**Trade-offs:** This is simpler than running Rich Live alongside Qt. The architectural constraint is that Rich's `Live` context manager (for animated spinners etc.) must complete before `QApplication()` is instantiated. If a loading screen with scrolling text is wanted after the TUI, it can be a Rich `Live` display in the same pre-Qt window — then Qt launches.

**Not concurrent — sequential:**
```
user runs `ocsd`
    → Rich TUI menu (blocking, no Qt) [pre-launch]
    → user selects mode
    → Rich loading screen with bundled demo routines (blocking) [pre-launch]
    → QApplication created
    → Qt overlay launched
    → (if --serve flag) uvicorn daemon thread started
    → QApplication.exec() blocks main thread
```

## Data Flow

### Record Flow: Click → Capture → AI → Tag → Save

```
User presses F2 (hotkey thread)
    → OverlayController.toggle_mode() [Qt signal, main thread]
    → border glow transitions green → red (QPropertyAnimation)
    → overlay mode = RECORD

User clicks element (mouse event, main thread)
    → OverlayController._handle_selection(x, y, w, h)
    → OverlayController.hide_for_capture()
        → _view.hide() + QApplication.processEvents() [+ optional sleep on Windows]
    → core.capture.screenshot_full() → numpy array (mss, no overlay visible)
    → core.capture.smart_crop(x, y, w, h, padding=0.3) → cropped PIL image
    → OverlayController.show_after_capture() → _view.show()

    Simultaneously:
    → core.detection.get_candidates(full_screenshot) → list[CandidateElement]
        → OmniParser YOLOv8 boxes → CLIP embeddings → sorted by confidence
    → core.vision.analyze_element(crop) → VLM label suggestion
        (may fail/timeout → fallback = None)

    → OverlayController.set_candidates(candidates) → render bbox items on scene
    → ScanAnimItems triggered: perimeter trace + sweep on best candidate
    → DonutProbItem triggered: probability rings rendered at (x, y)

    → HUDTagPanel.open(vlm_suggestion=label_or_none)
        → typewriter animation fills suggestion text (QTimer 40ms tick)
        → user confirms or overrides label

    → RecordController.accept_step(x, y, w, h, label, candidate)
        → OCSDGraph.add_node(...)
        → core.embeddings.embed_image(crop) → vector
        → core.capture.save_snippet(crop, snippet_path)
        → optional: dry-run step (hide overlay → execute → validate → show)

User presses Ctrl+Q
    → RecordController.save()
        → mapper.export.export_routine(graph, routine_path)
        → save snippets/ and embeddings/ to ~/.ocsd/routines/{name}/
    → QApplication.quit()
```

### Replay Flow: Load → Locate → Execute → Validate

```
ExecuteController.run(routine_path)
    → mapper.export.load_routine() → OCSDGraph
    → mapper.orchestrator.preflight_check()

For each step:
    → mapper.runner.execute_step(node)
        → core.locate.locate_element() [5-stage cascade]
            → pixel match → CLIP search → OCR → VLM → position fallback
        → visual wait loop: overlay (if any) shows locate progress
        → core.executor.click/type/scroll() [PyAutoGUI, mutex-locked]
        → mapper.validator.validate_action() [pixel diff]
        → if fail: mapper.orchestrator.recover()
```

### API Flow: HTTP Request → Shared State → Response

```
HTTP POST /routines/{id}/run
    → FastAPI endpoint (uvicorn thread, asyncio)
    → asyncio.to_thread(run_skill, ...) [offloads blocking call to thread pool]
    → mapper.runner.run_skill() [runs in thread pool]
    → ReplayLog returned
    → JSON response
```

### Key State Boundaries

```
Qt main thread owns:
    - QApplication, QGraphicsView, QGraphicsScene, all QGraphicsItems
    - OverlayController._mode, ._candidates
    - All animation objects (QPropertyAnimation, QTimeLine)

Background threads communicate TO Qt via:
    - Qt signals (cross-thread signal connection is thread-safe in Qt6)
    - QMetaObject.invokeMethod(..., Qt.ConnectionType.QueuedConnection)

Uvicorn thread owns:
    - asyncio event loop
    - FastAPI request handling

Shared across threads (requires threading.Lock):
    - ExecuteController state (running, current step)
    - Any shared routine metadata

HotkeyListener thread:
    - pynput listener loop
    - Emits Qt signal only — does NOT modify overlay state directly
```

## Scaling Considerations

This is a local desktop tool — scale means "adding more capabilities" not "handling more users."

| Concern | Current | With New Milestone |
|---------|---------|-------------------|
| Qt main thread load | Light (static overlay) | Animation timers + scan effects add ~5ms/frame; acceptable at 60fps |
| Hide-capture-show latency | N/A (not yet implemented) | 50-150ms per capture; 100ms sleep budget on Windows DWM |
| Memory (overlay items) | ~10-30 bbox items | Add ~5 animation objects per scan; cleaned up after each step |
| API thread vs Qt thread | No shared state today | Need threading.Lock around ExecuteController when API can trigger runs |

## Anti-Patterns

### Anti-Pattern 1: Modifying Qt Objects from Non-Main Thread

**What people do:** Call `view.hide()`, `scene.addItem()`, or animation `.start()` from the hotkey listener thread or uvicorn thread directly.

**Why it's wrong:** Qt's GUI objects are not thread-safe. Crashes or silent corruption. PyQt6 will sometimes raise a RuntimeError; sometimes it silently corrupts rendering state.

**Do this instead:** Emit a Qt signal from the background thread. Qt's cross-thread signal delivery (QueuedConnection) marshals the call to the main thread safely.

```python
# In HotkeyListener (background thread):
self.toggle_requested.emit()  # Qt signal — safe cross-thread

# In OverlayController (connected on main thread):
# toggle_requested connected to self.toggle_mode
```

### Anti-Pattern 2: QApplication.processEvents() Inside Animation Callbacks

**What people do:** Call `processEvents()` inside a QPropertyAnimation `valueChanged` handler or inside `paintEvent` to "force" a repaint.

**Why it's wrong:** Creates reentrant event processing, which can cause infinite loops and hard-to-reproduce visual glitches. `processEvents()` is only safe to call at controlled sync points (hide-for-capture, review loop).

**Do this instead:** Trust Qt's paint scheduler. Call `item.update()` to mark the item dirty; Qt will batch and render at the next frame.

### Anti-Pattern 3: Running Both Rich TUI and Qt Concurrently

**What people do:** Start `rich.Live` in a thread while QApplication is running, to show a status panel alongside the overlay.

**Why it's wrong:** Rich writes directly to stdout/stderr using control sequences. When Qt is also running, the terminal is typically hidden or redirected. Rich Live in a thread also uses `sys.stdout` locking that can interfere with logging handlers.

**Do this instead:** Rich TUI is pre-launch only. During recording/replay, status goes to the overlay HUD or to the log file. The terminal can show minimal log output via the standard logging handler.

### Anti-Pattern 4: Forgetting `install_signal_handlers = False` for Uvicorn in Thread

**What people do:** Call `uvicorn.run(app, ...)` in a threading.Thread without disabling signal handler installation.

**Why it's wrong:** Uvicorn tries to install SIGINT/SIGTERM handlers, which Python only allows on the main thread. Raises `ValueError: signal only works in main thread`.

**Do this instead:** Use `uvicorn.Server(config)` with `server.install_signal_handlers = lambda: None` before calling `server.run()` in the daemon thread (see Pattern 4 above).

### Anti-Pattern 5: Screenshotting While Overlay Is Still Visible

**What people do:** Call mss screenshot immediately after hide() without waiting for a repaint cycle.

**Why it's wrong:** On Windows with DWM (Desktop Window Manager), the compositor is asynchronous. The window compositing pipeline may not have repainted the desktop behind the overlay by the time mss grabs the framebuffer. Result: overlay artifacts (colored border, bbox outlines) appear in the captured screenshot.

**Do this instead:** Always use the hide-flush-sleep-capture sequence:
```python
self._view.hide()
QApplication.processEvents()
if sys.platform == "win32":
    time.sleep(0.03)  # 30ms DWM compositor flush budget
screenshot = mss_capture()
self._view.show()
```

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| LiteLLM proxy (VLM) | HTTP via litellm client in `core.vision`; timeout + graceful fallback to None | Must not block Qt main thread; run in QThread or background thread with Qt signal result |
| mss (screen capture) | Synchronous call after overlay hidden | Fast (~5ms); safe on main thread within hide/show cycle |
| pynput (global hotkeys) | Background listener thread; Qt signal for cross-thread callback | pynput listener cannot call Qt functions directly |
| Tesseract | Subprocess via pytesseract; blocking | Offload to QThread or thread pool if called during recording flow |
| FastAPI/uvicorn | Daemon thread; asyncio event loop entirely separate from Qt | No shared mutable objects without threading.Lock |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| HotkeyListener → OverlayController | Qt cross-thread signal (QueuedConnection) | Safe; no locks needed for signal delivery |
| RecordController → OverlayController | Direct method calls on main thread | Both run on main thread; no locking needed |
| VLM worker → HUDTagPanel | Qt signal from QThread worker | LiteLLM call in QThread, result emitted as signal |
| FastAPI thread → ExecuteController | Shared state + threading.Lock | Lock required around run-state mutation |
| OverlayController → core.capture | Direct call during hide/show cycle | Both on main thread; mss is stateless |

## Build Order Implications

The component dependencies suggest this build order:

1. **CinematicOverlayView base window** (transparent fullscreen, proper Win32 flags, hide/show API) — everything else attaches to this
2. **BorderGlowItem** (simplest animation; validates QPropertyAnimation pattern before building more complex items)
3. **ScanAnimItems** (perimeter trace + sweep; validates QTimer 60fps pattern)
4. **ElementBoxGroup animation extensions** (extend existing boxes with scan animation trigger hooks)
5. **DonutProbItem** (probability cloud; self-contained QGraphicsItem)
6. **HUDTagPanel** (frosted glass dialog + typewriter; requires VLM integration point)
7. **FloatingToolbar** (draggable widget; build last as it requires knowing what actions exist)
8. **RecordController full flow** (wires all overlay components into F2→capture→AI→tag→save)
9. **Routine.json format evolution** (NetworkX → new format; depends on RecordController knowing the full step shape)
10. **Rich TUI** (pre-launch; no Qt dependency; can build independently)
11. **FastAPI daemon thread** (wires uvicorn to existing ExecuteController; can build independently)

The overlay components (1-7) are the critical path. Rich TUI and FastAPI can be built in parallel with the record flow (8-9).

## Sources

- Qt for Python official docs: `doc.qt.io/qtforpython-6` — QPropertyAnimation, QGraphicsItem, WA_TranslucentBackground
- Qt Forum: frosted glass acrylic discussion — `forum.qt.io/topic/110293` (DWM acrylic confirmed Windows-specific)
- Qt Forum: Qt6 main window hide/show issue — `forum.qt.io/topic/136295` (confirmed processEvents pattern)
- Medium: Seamless desktop widgets with PyQt6 — `medium.com/@hudbeard` (WA_TranslucentBackground confirmed pattern)
- FastAPI issue #650 / Discussion #7957: uvicorn in thread + `install_signal_handlers` workaround
- Uvicorn Discussion #1103: programmatic shutdown via `server.should_exit = True`
- FastAPI docs: `fastapi.tiangolo.com/async` — asyncio.to_thread for blocking calls
- BugFactory: starting/stopping uvicorn in background — `bugfactory.io/articles/starting-and-stopping-uvicorn-in-the-background`
- Existing OCSD codebase: `recorder/overlay.py`, `recorder/overlay_view.py`, `recorder/overlay_items.py`, `api/server.py`

---
*Architecture research for: OCSD cinematic overlay + TUI + API milestone*
*Researched: 2026-03-16*
