# Pitfalls Research

**Domain:** PyQt6 transparent overlay + Rich TUI + screen automation tool
**Researched:** 2026-03-16
**Confidence:** HIGH (most pitfalls verified against Qt docs, official issues, and codebase inspection)

---

## Critical Pitfalls

### Pitfall 1: Screenshot Contains Overlay Elements

**What goes wrong:**
`hide()` is called on the overlay but the screenshot capture fires before the compositor has actually removed the window from the framebuffer. The captured PNG includes the red/green border glow, scan lines, or HUD panel burned into it. The AI then analyses its own UI artifacts as "elements to click."

**Why it happens:**
`QWidget.hide()` schedules a repaint but does not block until the OS compositor has composited the result. On Windows, DWM compositing is asynchronous. `mss` grabs the display framebuffer immediately — it does not wait for the event loop to drain. Even a `QTimer.singleShot(0, ...)` callback fires after Python yields to the event loop but *before* DWM has flushed the window removal to the screen buffer.

**How to avoid:**
Use a two-step hide+delay pattern with a tested minimum delay, not a zero-shot timer.

```python
def hide_and_capture(self, callback):
    self.overlay.hide()
    # process_events() drains pending Qt paints
    QApplication.processEvents()
    # Sleep gives DWM time to composite the removal.
    # 80ms is empirically safe on Windows; 50ms is marginal.
    QTimer.singleShot(80, callback)
```

Do NOT use `QApplication.processEvents()` alone — it flushes Qt's paint queue but DWM composites on its own schedule. The sleep is mandatory. Test the delay on the slowest target machine.

**Warning signs:**
- Screenshots during recording have colored borders or scan-line artifacts
- VLM misidentifies overlay elements as clickable targets
- Bounding boxes are systematically misaligned (elements drawn at logical-DPI coords, screenshot at physical pixels)

**Phase to address:** Cinematic overlay phase (Phase 1). The hide-before-capture contract must be established before any animation work is added on top of it.

---

### Pitfall 2: DPI Scaling Makes Overlay Coordinates Wrong on High-DPI Screens

**What goes wrong:**
The overlay window renders at logical (96 DPI) coordinates. `mss` captures at physical pixel coordinates. On a 150% scaled 4K display, an overlay bounding box at `(100, 100, 200, 200)` logical pixels corresponds to `(150, 150, 300, 300)` physical pixels. Every recorded bbox is wrong by the scale factor. Click replays miss targets. VLM crops wrong regions.

**Why it happens:**
`main.py` correctly calls `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)` at startup, which tells Windows to give the process true physical pixel values. BUT: Qt internally uses device-pixel-ratio scaling and `QGraphicsView` scene coordinates are logical unless explicitly scaled. `QApplication.primaryScreen().geometry()` returns logical geometry when DPI awareness is active. `mss` always returns physical pixels. The two coordinate systems diverge.

**How to avoid:**
When constructing the overlay's scene rect and interpreting mouse click positions, always multiply by `QApplication.primaryScreen().devicePixelRatio()` to convert to physical pixels before saving to the routine. Verify with a test: draw a box, save the coords, mss-capture and confirm the crop matches the box.

```python
ratio = QApplication.primaryScreen().devicePixelRatio()
physical_x = int(logical_x * ratio)
physical_y = int(logical_y * ratio)
```

The existing `overlay_view.py` reads `screen_geom.width()` (logical) and uses it as both the scene rect AND the full-screen window size. This is likely already wrong on 150% scaled displays — verify early.

**Warning signs:**
- Clicks land noticeably above-left of targets on HiDPI displays
- Bounding boxes look correct visually but VLM crops miss the element
- The overlay covers less than full screen on scaled displays (DPI mismatch shrinks the window)

**Phase to address:** Cinematic overlay phase. Fix the coordinate system before any bounding-box recording logic is wired up.

---

### Pitfall 3: Click-Through Breaks on Wayland (Ubuntu)

**What goes wrong:**
`WindowTransparentForInput` and `WS_EX_TRANSPARENT` are Windows/X11 mechanisms. On Wayland (Ubuntu 22.04+ default, Ubuntu 24.04 ships Wayland by default), these flags are silently ignored. The overlay eats all mouse clicks even when in passthrough mode. The user cannot click through to the application under the overlay, making recording impossible.

**Why it happens:**
Wayland's security model forbids client windows from declaring "give my clicks to another window" — that is a compositor decision, not an app decision. Qt maps `WindowTransparentForInput` to an X11 hint that has no Wayland equivalent. The flag is accepted without error but has no effect.

**How to avoid:**
For Ubuntu support, detect the session type at startup and force X11 backend if Wayland is detected:

```python
import os
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    # must be set BEFORE QApplication()
```

Document this as a known requirement. Alternatively, detect at runtime and abort with a clear error message instead of silently failing.

**Warning signs:**
- On Ubuntu, passthrough mode still intercepts clicks
- `echo $XDG_SESSION_TYPE` returns `wayland` on the target machine
- Qt logs "This plugin does not support setting window opacity" (confirms Wayland backend is active)

**Phase to address:** Cinematic overlay phase. The Ubuntu click-through test must be part of the overlay acceptance criteria.

---

### Pitfall 4: Rich `Live` Display Conflicts with PyQt6 Main Thread

**What goes wrong:**
`rich.Live` and `rich.Console` detect whether they are running in the main thread to decide whether to enable ANSI escape codes and terminal control. When Rich is initialized in a background thread (e.g., spawned to run the TUI while PyQt6 occupies the main thread), `isatty` returns `False` or `None`, Rich falls back to plain text output, and all TUI styling disappears. Worse: if `Live.start()` is called in a thread while Qt is also polling the event loop from the main thread, the terminal cursor control and Qt's stdin handling can fight, producing garbled output.

**Why it happens:**
PyQt6's `QApplication.exec()` must run on the main thread. Rich's `Live` display assumes it owns the terminal from the main thread. When you put Rich in a background thread, Python's `threading.current_thread() is threading.main_thread()` returns `False`, which triggers Rich's degraded mode.

**How to avoid:**
Run Rich TUI in the main thread, move the PyQt6 overlay into a separate QThread. Since QApplication must be created in the main thread (Qt hard requirement), the correct architecture is:

```
main thread:   QApplication.exec() → overlay lives here
               Rich used only BEFORE QApplication.exec() starts (loading screen)
               OR Rich used AFTER overlay exits (post-run summary)

background thread: FastAPI/uvicorn server (separate process is better)
```

If you need a live TUI while the overlay is running, use a separate subprocess for the TUI, communicating via a queue or named pipe. Do not attempt to run both event loops in the same process on the same thread.

**Warning signs:**
- Rich output loses color/formatting at runtime
- `Console.is_terminal` returns `False` unexpectedly
- Terminal cursor artifacts appear when overlay is visible

**Phase to address:** TUI + CLI phase. Architecture decision must be made before implementing the loading screen.

---

### Pitfall 5: FastAPI/uvicorn Signal Handler Conflict with QApplication

**What goes wrong:**
Uvicorn installs its own signal handlers (SIGINT, SIGTERM) when started. PyQt6's `QApplication` also installs signal handlers. When both run in the same process, one overwrites the other. The result: Ctrl+C either kills the process without cleanup (uvicorn wins) or is swallowed silently (Qt wins). More critically, uvicorn's signal handler setup **requires the main thread** — running `uvicorn.run()` in a background thread raises `ValueError: signal only works in main thread`.

**Why it happens:**
Python's `signal` module only allows signal handlers to be set from the main thread. Both uvicorn and Qt assume they are the primary controller of the process. Neither was designed for cohabitation.

**How to avoid:**
Run FastAPI in a **separate process**, not a thread:

```python
import multiprocessing
api_proc = multiprocessing.Process(
    target=uvicorn.run,
    args=(app,),
    kwargs={"host": "127.0.0.1", "port": 8000},
    daemon=True,
)
api_proc.start()
```

If a thread is truly required (e.g., to share in-process state), use `uvicorn.Server` with `server.install_signal_handlers = False` to suppress uvicorn's signal setup, and handle shutdown manually via a threading Event. This is documented but obscure.

**Warning signs:**
- `ValueError: signal only works in main thread` at startup
- Ctrl+C doesn't cleanly shut down both systems
- FastAPI endpoints become unresponsive after overlay focus events

**Phase to address:** API endpoints phase. Settle the process architecture before wiring any endpoint to overlay state.

---

### Pitfall 6: Animation `self.update()` Inside `paintEvent` Creates Infinite Repaint Loop

**What goes wrong:**
A shimmer/glow animation implemented as a custom `paintEvent` that calls `self.update()` at the end creates an infinite loop: paint → update → schedule repaint → paint → update → .... CPU hits 100% on the animation thread. The overlay becomes unresponsive. The machine heats up during routine recording.

**Why it happens:**
`self.update()` schedules a repaint. When called inside `paintEvent`, it schedules another repaint before the current one completes. Qt will coalesce these but the effective result is painting as fast as the CPU allows, not at 60fps.

**How to avoid:**
Drive all animations from a single `QTimer` at 16ms (60fps) that calls `self.update()`. Never call `self.update()` inside `paintEvent`. Use `QPropertyAnimation` for property interpolation (opacity, geometry) — it drives updates correctly via Qt's animation timer.

```python
self._anim_timer = QTimer()
self._anim_timer.setInterval(16)  # ~60fps
self._anim_timer.timeout.connect(self._tick_animation)
self._anim_timer.start()

def _tick_animation(self):
    self._glow_phase = (self._glow_phase + 0.05) % (2 * math.pi)
    self.update()  # safe: called from timer, not from paintEvent
```

**Warning signs:**
- CPU usage stays at 30-100% while overlay is visible but idle
- `top` or Task Manager shows Python pegged during animation
- Any movement on screen causes lag due to paint pressure

**Phase to address:** Cinematic overlay phase. Establish the animation tick pattern before implementing any animated effects.

---

### Pitfall 7: Multi-Monitor Overlay Covers Wrong Screen

**What goes wrong:**
`QApplication.primaryScreen().geometry()` returns the geometry of the primary monitor, not necessarily the monitor where the application being recorded is running. The overlay appears on monitor 1 while the user is working on monitor 2. The bounding box coordinates are also wrong because they're relative to the wrong screen origin.

**Why it happens:**
The existing `overlay_view.py` hardcodes `QApplication.primaryScreen()` for both the scene rect and the window geometry. On multi-monitor setups, `QGuiApplication.screens()` returns all monitors. The "correct" screen is whichever screen the target application is on, not necessarily primary.

**How to avoid:**
At overlay startup, either:
1. Let the user choose which monitor to overlay (simplest for V1), or
2. Detect the foreground window's screen using Win32 `MonitorFromWindow` and map the overlay to that screen

For V1, showing the overlay on the primary screen and documenting the limitation is acceptable. But the screen selection must be explicit, not implicit via `primaryScreen()`.

```python
screens = QGuiApplication.screens()
target_screen = screens[0]  # explicit, documented
geo = target_screen.geometry()
self.setGeometry(geo)
self.setSceneRect(0, 0, geo.width(), geo.height())
```

**Warning signs:**
- Overlay appears on the wrong monitor
- Bounding boxes recorded on secondary monitor have wrong origin coordinates
- `QGuiApplication.screens()` returns more than one screen and the code doesn't handle it

**Phase to address:** Cinematic overlay phase. Multi-monitor coordinate handling must be correct before recording is wired up.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `QTimer.singleShot(0, capture)` for hide-then-capture | Simpler code | Screenshot contains overlay artifacts on any machine with compositing | Never — use minimum 80ms delay |
| Hardcode 1920x1080 as fallback screen size | No crash at startup | Overlay clips off on larger screens, coords wrong on smaller | Never in production code |
| `QApplication.primaryScreen()` for overlay geometry | Works on single-monitor dev machine | Wrong screen on multi-monitor setups | Only if V1 officially supports single-monitor only |
| Drive shimmer animation from `paintEvent` via `self.update()` | Fewer moving parts | Infinite repaint loop, 100% CPU | Never |
| Run Rich TUI and PyQt6 in same thread | Simpler startup | Rich degrades to no-color, terminal artifacts | Never |
| Run uvicorn in a thread with default config | One-process deployment | Signal handler conflict, Ctrl+C broken | Never — use daemon process or `install_signal_handlers=False` |
| Skip `devicePixelRatio` multiplication for bbox coords | Simpler coordinate math | All bboxes wrong on HiDPI displays (affects most modern laptops) | Never — must account for DPI from day one |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `mss` screenshot after overlay hide | Call `hide()` then capture immediately | `hide()` → `processEvents()` → 80ms sleep → capture |
| Win32 `SetWindowLong` for layered flags | Apply flags before `show()` | Apply via `QTimer.singleShot(0, ...)` — HWND not valid until after `show()` |
| `QPropertyAnimation` on window opacity | Animate opacity on Wayland Ubuntu | Wayland returns "plugin does not support setting window opacity" — use `setWindowOpacity` sparingly and only after confirming X11 backend |
| Rich `Live` with PyQt6 | Start `Live` in same thread as `QApplication.exec()` | Use Rich only before or after Qt event loop; never concurrent in same thread |
| FastAPI + uvicorn + QApplication | `uvicorn.run()` in a Thread | Spawn as daemon Process, or use `Server(install_signal_handlers=False)` |
| Human mouse movement (PyAutoGUI Bezier) | Use constant-speed linear interpolation | Vary acceleration: slow start, peak at midpoint, slow end; add micro-jitter (±1-2px) at destination; randomise total duration ±20% |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Redrawing entire scene on every animation tick | CPU stays high during idle overlay | Only `update()` dirty regions; use `QGraphicsItem.setPos()` instead of recreating items | Immediately on any animation |
| `scene().clear()` + rebuild on every `render_candidates()` call | Visible flash when candidates update | Reuse existing `_ElementBoxGroup` items; update position/color in-place | Every candidate refresh during recording |
| `QApplication.processEvents()` in a loop | Event loop starvation, GUI freeze | Use QThread workers for any blocking work; never spin-wait in main thread | Any blocking operation >50ms |
| Full-resolution 4K screenshots stored in `ReplayStep.before_screenshot` | Memory exhaustion on long routines | Already noted in CONCERNS.md — implement log rotation before enabling cinematic replay logging | After ~50 steps on 4K display |
| `QGraphicsScene` with hundreds of animation items | Scene graph traversal becomes slow | Keep overlay items sparse (<50 items); batch shimmer effects into a single custom `paintEvent` item | >100 active items in scene |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Overlay window intercepts keyboard input during RECORD mode | Key logger surface: overlay receives all keystrokes while recording | Use `WA_ShowWithoutActivating` (already done); ensure overlay never grabs keyboard focus; `keyPressEvent` should only handle overlay hotkeys |
| Routine JSON includes full absolute paths to embeddings | Routines not portable across machines/users | Store paths relative to `~/.ocsd/routines/{name}/`; resolve at load time |
| FastAPI endpoint `/routines/{id}/run` with no auth | Any local process can trigger automation | Bind to 127.0.0.1 only (not 0.0.0.0); add optional API key header for agent clients; document security model |
| Screenshot crops sent to LiteLLM proxy over HTTP | Screenshot data in cleartext on local network | Default proxy is `http://localhost` which is acceptable; warn loudly if configured to a non-localhost URL without HTTPS |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Overlay border glow is the only recording indicator | User unsure if recording is active when looking at a different part of screen | Add a floating persistent status chip (draggable) showing mode + step count, always visible regardless of screen region |
| No escape route when VLM analysis hangs | User sees spinner with no way out; must kill process | Every VLM call must have an explicit timeout (already partially done); show "VLM timeout — enter label manually" after N seconds |
| Dry-run countdown covers the element being validated | User can't see if the click landed correctly | Show countdown in overlay corner, not over the target element |
| Overlay hides for screenshot then reappears — visible flicker | Distracting during recording; signals "something happened" | Minimize hide duration; use opacity fade instead of hard hide where possible (where compositor timing allows) |
| Rich TUI loading screen blocks until models load | Non-technical users think the app is frozen | Show a progress indicator with estimated time; never block with a silent spinner |

---

## "Looks Done But Isn't" Checklist

- [ ] **Overlay hide-before-screenshot:** Visually hides in dev, but test with screen recording software to confirm no artifacts in captured PNG at full speed
- [ ] **Click-through passthrough mode:** Works on Windows; explicitly test on Ubuntu under X11 (`QT_QPA_PLATFORM=xcb`); document Wayland not supported
- [ ] **DPI correctness:** Draw a bbox, save the routine, replay and confirm click lands in the center of the recorded element — test on 100%, 125%, 150%, 200% scale
- [ ] **Multi-monitor:** Test recording on the non-primary monitor; confirm overlay appears on correct screen and coords are correct
- [ ] **Animation CPU:** Profile with `py-spy` or Task Manager during shimmer animation idle; should be <5% CPU on modern hardware
- [ ] **FastAPI startup:** Confirm Ctrl+C cleanly shuts down both PyQt6 and FastAPI; no zombie uvicorn processes
- [ ] **Rich TUI styling:** Confirm colors and panels render correctly in Windows Terminal, cmd.exe, and Ubuntu terminal (some terminals strip ANSI)
- [ ] **VLM fallback during recording:** Disconnect LiteLLM proxy mid-recording; confirm graceful degradation to manual label entry with visible warning

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Screenshots contain overlay artifacts (bug found post-implementation) | MEDIUM | Increase hide delay; add visual diff test comparing capture with known-clean reference; re-record affected routines |
| DPI scaling wrong in recorded routines | HIGH | All recorded bboxes invalid; must re-record; add migration script to multiply legacy coords by display scale factor if scale factor is known |
| Wayland click-through broken in production | MEDIUM | Add startup check for `XDG_SESSION_TYPE`; auto-set `QT_QPA_PLATFORM=xcb`; ship as hotfix |
| Animation infinite repaint loop discovered in production | LOW | Single-line fix: remove `self.update()` from `paintEvent`; add timer-driven tick pattern |
| FastAPI + Qt signal conflict at startup | LOW | Spawn API as daemon process; one-day fix |
| Rich TUI degraded in background thread | LOW | Move Rich display to pre-Qt phase or separate process; two-day fix |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Screenshot overlay contamination | Phase 1: Cinematic overlay | Automated test: capture during recording, pixel-diff against clean reference |
| DPI coordinate mismatch | Phase 1: Cinematic overlay | Manual test: record on 150% scale display, replay, confirm click accuracy |
| Wayland click-through failure | Phase 1: Cinematic overlay | Run test suite on Ubuntu with `QT_QPA_PLATFORM=wayland` — expect documented failure; confirm `xcb` workaround works |
| Multi-monitor wrong screen | Phase 1: Cinematic overlay | Manual test: connect second monitor, verify overlay on correct screen |
| Animation CPU from `paintEvent` loop | Phase 1: Cinematic overlay | Profile animation idle CPU before merging |
| Rich + PyQt6 event loop conflict | Phase 2: TUI + CLI | Architecture review before implementation; confirm Rich runs only pre/post Qt loop |
| FastAPI signal conflict | Phase 3: API endpoints | Integration test: start full stack, send Ctrl+C, verify clean shutdown |
| Human mouse identical timing (anti-bot) | Phase 1 (executor already exists) | Not blocking for V1 — OCSD targets local automation, not anti-bot bypassing; validate human_mouse library provides sufficient variance for target apps |

---

## Sources

- [Qt High DPI documentation](https://doc.qt.io/qt-6/highdpi.html) — DPI scaling behavior and devicePixelRatio
- [Qt Wayland and Qt](https://doc.qt.io/qt-6/wayland-and-qt.html) — Wayland limitations for overlay applications
- [Qt Forum: WindowTransparentForInput not worked on Wayland](https://forum.qt.io/topic/154266/windowtransparentforinput-not-worked-on-wayland) — confirmed Wayland click-through failure
- [Qt Forum: Qt6 QPropertyAnimation — "This plugin does not support setting window opacity"](https://forum.qt.io/topic/158586/qt6-qpropertyanimation-this-plugin-does-not-support-setting-window-opacity) — Wayland opacity limitation
- [Qt Forum: Click-through window blink due setWindowFlags](https://forum.qt.io/topic/156799/click-through-window-will-blink-due-setwindowflags) — timing issues with Win32 flag application
- [Rich GitHub Issue #1530: live displays and console printing are not thread safe](https://github.com/willmcgugan/rich/issues/1530) — Rich threading limitations
- [Rich GitHub Issue #2665: Console appears not to be a terminal in background thread](https://github.com/Textualize/rich/issues/2665) — isatty failure in threads
- [FastAPI GitHub Issue #650: starting uvicorn in background thread signal conflict](https://github.com/fastapi/fastapi/issues/650) — signal handler conflict
- [pythonguis.com: Creating a very heavy paintEvent](https://www.pythonguis.com/faq/creating-a-new-widget-very-heavy-paintevent/) — paintEvent/update() infinite loop warning
- [Qt Forum: How to get correct screen size in PyQt6 (multi-monitor)](https://forum.qt.io/topic/139569/how-to-get-correct-screen-size-in-pyqt6) — multi-monitor geometry
- `recorder/overlay_view.py` — existing codebase: DPI hardcoding at line 79-85, Win32 flag timing via `QTimer.singleShot(0, ...)` at line 89
- `.planning/codebase/CONCERNS.md` — `recorder/overlay.py` and `overlay_view.py` fragile area, no platform-specific tests

---
*Pitfalls research for: PyQt6 cinematic overlay + Rich TUI + FastAPI screen automation tool (OCSD)*
*Researched: 2026-03-16*
