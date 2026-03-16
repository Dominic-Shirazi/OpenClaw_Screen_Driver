# Phase 1: Overlay Foundation - Research

**Researched:** 2026-03-16
**Domain:** PyQt6 transparent overlay window, DPI scaling, compositor-aware screenshot capture, cross-platform (Windows + Ubuntu X11)
**Confidence:** HIGH

## Summary

Phase 1 builds a transparent fullscreen PyQt6 overlay that serves as the visual shell for all subsequent recording and replay phases. The overlay must handle three deceptively hard problems: (1) click-through passthrough toggling on both Windows (Win32 extended styles) and Linux (X11 input shapes or `WA_TransparentForMouseEvents`), (2) DPI-correct coordinate mapping between Qt's device-independent pixel system and mss's physical pixel output, and (3) reliable hide-before-screenshot with compositor flush to prevent overlay artifacts in captured images.

The existing codebase already has a working overlay (`recorder/overlay.py`, `recorder/overlay_view.py`, `recorder/overlay_items.py`) with Win32 click-through, basic border drawing, and candidate rendering. Per PROJECT.md, the new overlay is built fresh -- the existing code is reference only. The new overlay needs a proper state machine (ready/recording/paused), smooth transitions, and the hide-capture-show lifecycle baked in from the start.

**Primary recommendation:** Build the overlay as a `QGraphicsView`-based fullscreen window with an explicit state machine (Python enum + transition methods, not `QStateMachine` which adds complexity without benefit at this stage). Use `DwmFlush()` on Windows and `QApplication.processEvents()` + configurable sleep on Linux for compositor flush before screenshots. Establish the DPI conversion layer as a standalone utility module tested independently.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| OVLY-01 | Overlay renders as transparent fullscreen window with click-through passthrough when not interacting | Win32 `WS_EX_TRANSPARENT` flag (existing pattern works), Linux `WA_TransparentForMouseEvents` + `WindowTransparentForInput` flag; see Architecture Patterns |
| OVLY-02 | Overlay hides ALL visual elements before any screenshot capture (80ms+ DWM flush) | `DwmFlush()` via ctypes on Windows, `processEvents()` + configurable sleep on Linux; see Common Pitfalls |
| OVLY-03 | Overlay handles DPI scaling correctly -- all coordinates use physical pixels matching mss output | `devicePixelRatio()` conversion layer; mss always captures physical pixels; see DPI section |
| OVLY-04 | Overlay state machine: ready (green) <-> recording (red) <-> paused (green) with smooth transitions | Python enum state machine with `QPropertyAnimation` for color transitions; see Architecture Patterns |
| OVLY-05 | Overlay works on Windows (primary) and Ubuntu (X11/XCB fallback for Wayland) | `QT_QPA_PLATFORM=xcb` env var for Wayland sessions; platform-guarded code paths; see Cross-Platform section |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyQt6 | >=6.5 | Overlay window, graphics scene, animations | Already in pyproject.toml; project decision |
| mss | >=9.0 | Screenshot capture (physical pixels) | Already in pyproject.toml; fast, cross-platform |
| pynput | >=1.7 | Global hotkeys on Linux | Already in pyproject.toml; fallback for non-Win32 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ctypes (stdlib) | - | Win32 API calls (DPI, WS_EX_*, DwmFlush) | Windows platform code |
| PyQt6.QtStateMachine | >=6.5 | Formal state machine | NOT recommended -- use Python enum instead for simplicity |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| QStateMachine | Python Enum + methods | QStateMachine is heavyweight for 3 states; Python enum is debuggable and testable without Qt event loop |
| QGraphicsView | QWidget + QPainter | QGraphicsView gives scene graph, z-ordering, item management for free -- needed for later phases |

**Installation:**
```bash
# All deps already in pyproject.toml -- no new packages needed for Phase 1
pip install -e ".[dev]"
```

## Architecture Patterns

### Recommended Project Structure
```
recorder/
    overlay/                  # NEW package (replaces old overlay.py/overlay_view.py)
        __init__.py           # Re-exports OverlayController
        controller.py         # OverlayController -- owns state machine, hide/show lifecycle
        view.py               # _OverlayView (QGraphicsView) -- window setup, rendering
        state.py              # OverlayState enum, transition table
        platform_win32.py     # Win32-specific: WS_EX_* flags, DwmFlush, DPI awareness
        platform_linux.py     # Linux-specific: X11 input passthrough, XCB detection
        dpi.py                # DPI conversion utilities (logical <-> physical pixels)
        capture_guard.py      # Hide-flush-capture-show lifecycle
```

### Pattern 1: State Machine via Python Enum
**What:** Three states (READY, RECORDING, PAUSED) with explicit transition methods and color properties.
**When to use:** Always -- this is the overlay's core behavior.
**Example:**
```python
from enum import Enum, auto

class OverlayState(Enum):
    READY = auto()       # Green shimmer, click-through ON
    RECORDING = auto()   # Red shimmer, click-through OFF (captures clicks)
    PAUSED = auto()      # Green shimmer, click-through ON (same visual as READY)

# Transition table: (current_state, trigger) -> new_state
TRANSITIONS: dict[tuple[OverlayState, str], OverlayState] = {
    (OverlayState.READY, "f2"):       OverlayState.RECORDING,
    (OverlayState.RECORDING, "f2"):   OverlayState.READY,
    (OverlayState.PAUSED, "f2"):      OverlayState.RECORDING,
    # Ctrl+Q and ESC handled separately (close/save)
}

STATE_COLORS: dict[OverlayState, tuple[int, int, int, int]] = {
    OverlayState.READY:     (50, 200, 50, 150),    # Green
    OverlayState.RECORDING: (255, 50, 50, 200),    # Red
    OverlayState.PAUSED:    (50, 200, 50, 150),    # Green (same as READY)
}
```

### Pattern 2: DPI Conversion Layer
**What:** Centralized conversion between Qt logical coordinates and mss physical pixels.
**When to use:** Every coordinate exchange between Qt and mss/screenshot systems.
**Example:**
```python
# Source: Qt 6 High DPI docs (https://doc.qt.io/qt-6/highdpi.html)
from PyQt6.QtWidgets import QApplication

def logical_to_physical(x: int, y: int) -> tuple[int, int]:
    """Convert Qt logical coordinates to mss physical pixels."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return x, y
    dpr = screen.devicePixelRatio()
    return int(x * dpr), int(y * dpr)

def physical_to_logical(x: int, y: int) -> tuple[int, int]:
    """Convert mss physical pixels to Qt logical coordinates."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return x, y
    dpr = screen.devicePixelRatio()
    return int(x / dpr), int(y / dpr)

def get_physical_screen_size() -> tuple[int, int]:
    """Get screen size in physical pixels (what mss sees)."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return 1920, 1080
    size = screen.size()  # logical
    dpr = screen.devicePixelRatio()
    return int(size.width() * dpr), int(size.height() * dpr)
```

### Pattern 3: Capture Guard (Hide-Flush-Capture-Show)
**What:** Context manager that hides overlay, waits for compositor flush, returns control for screenshot, then restores overlay.
**When to use:** Every screenshot capture operation.
**Example:**
```python
import sys
import time
from contextlib import contextmanager
from typing import Generator

from PyQt6.QtWidgets import QApplication

@contextmanager
def capture_guard(
    view: QGraphicsView,
    flush_ms: int = 100,
) -> Generator[None, None, None]:
    """Hide overlay, flush compositor, yield for capture, restore."""
    # Hide all overlay elements
    view.hide()
    QApplication.processEvents()

    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.dwmapi.DwmFlush()
        except (AttributeError, OSError):
            time.sleep(flush_ms / 1000.0)
    else:
        time.sleep(flush_ms / 1000.0)

    try:
        yield  # Caller takes screenshot here
    finally:
        view.show()
        QApplication.processEvents()
```

### Pattern 4: Platform-Guarded Click-Through
**What:** Platform-specific click-through toggling.
**When to use:** Every state transition that changes passthrough behavior.
**Example:**
```python
import sys
from PyQt6.QtCore import Qt

def set_click_through(view: QGraphicsView, passthrough: bool) -> None:
    """Enable or disable click-through for the overlay."""
    if sys.platform == "win32":
        import ctypes
        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x00000020
        hwnd = int(view.winId())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if passthrough:
            style |= WS_EX_TRANSPARENT
        else:
            style &= ~WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    else:
        # Linux: WA_TransparentForMouseEvents works on X11/XCB
        view.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, passthrough
        )
```

### Anti-Patterns to Avoid
- **Using QStateMachine for 3 states:** Adds XML-like verbosity, harder to debug, harder to test without event loop.
- **Coordinates without DPI conversion:** Qt's `geometry()` returns logical pixels; mss captures physical pixels. Mixing them produces off-by-2x errors on HiDPI displays.
- **Sleeping a fixed duration instead of DwmFlush:** `DwmFlush()` blocks until the current frame is composited -- much more reliable than guessing a sleep duration. Use sleep only as fallback on Linux.
- **Toggling window flags with `setWindowFlags()`:** This destroys and recreates the window on some platforms. Use `setAttribute()` for Linux and `SetWindowLongW` for Windows instead.
- **Calling `hide()`/`show()` instead of hiding scene items:** For the capture guard, hiding the *entire window* is safest since even invisible scene items can sometimes leave compositor artifacts. However, if flicker is unacceptable, hiding all scene items + `DwmFlush()` is the alternative.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DPI detection | Custom registry reading | `QScreen.devicePixelRatio()` | Qt already queries the OS correctly; manual approaches miss per-monitor changes |
| Window transparency | Custom compositor protocols | `WA_TranslucentBackground` + `FramelessWindowHint` + `WindowStaysOnTopHint` | Qt handles the platform-specific transparency setup |
| Global hotkeys (Linux) | Raw X11 key grabbing | pynput `GlobalHotKeys` | Already in project deps; handles keyboard layouts |
| Global hotkeys (Windows) | RegisterHotKey / pynput | `GetAsyncKeyState` polling on QTimer | Existing pattern proven to work alongside PyQt6 event loop (see `recorder/hotkeys.py`) |
| Animation easing | Manual interpolation | `QPropertyAnimation` with `QEasingCurve` | Built into Qt; handles timing, easing curves, and property updates |
| Scene graph management | Manual draw/erase lists | `QGraphicsScene` + `QGraphicsView` | Z-ordering, hit testing, item management all handled |

**Key insight:** The existing `recorder/hotkeys.py` already solves the global hotkey problem correctly with platform-specific approaches. The new overlay should reuse this pattern (or the module directly).

## Common Pitfalls

### Pitfall 1: DWM Flush Timing on Windows
**What goes wrong:** Overlay elements appear in screenshots even after `hide()` is called.
**Why it happens:** Windows DWM composites at VSync intervals (~16ms at 60Hz). Calling `hide()` queues the change but it may not be painted until the next VSync. Meanwhile, `mss.grab()` captures what's currently on screen.
**How to avoid:** Call `QApplication.processEvents()` to flush Qt's paint queue, then call `DwmFlush()` (from `dwmapi.dll`) which blocks until DWM has composited the current frame. Add configurable `capture_delay_ms` (default 100ms) as safety margin.
**Warning signs:** Ghostly overlay borders appearing in saved screenshots.

### Pitfall 2: Qt Logical vs Physical Pixels
**What goes wrong:** Click coordinates from the overlay are at half the actual screen position on 2x DPI displays, or overlay doesn't fill the full screen.
**Why it happens:** Qt 6 operates in logical (device-independent) pixels by default. On a 4K display at 200% scaling, `QScreen.geometry()` returns 1920x1080 while mss captures 3840x2160.
**How to avoid:** Always multiply Qt coordinates by `devicePixelRatio()` before passing to mss, and divide mss coordinates by `devicePixelRatio()` before passing to Qt. Centralize this in `dpi.py`.
**Warning signs:** Overlay covers only top-left quarter of screen; clicks register at wrong positions.

### Pitfall 3: mss DPI Awareness Import Order
**What goes wrong:** mss returns incorrect monitor dimensions.
**Why it happens:** Per mss issue #184, some packages (pyautogui, mouseinfo, pyscreeze) call `SetProcessDpiAware()` during import, which conflicts with the `SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)` call in `main.py`. The first call wins.
**How to avoid:** Import mss before pyautogui. In the overlay module, the DPI awareness should be set in `main.py` before any other imports, which is already the case in the existing code.
**Warning signs:** Monitor coordinates from `sct.monitors` don't match actual resolution.

### Pitfall 4: WA_TransparentForMouseEvents on Linux
**What goes wrong:** Click-through stops working or never works on Linux X11.
**Why it happens:** Qt 6.2.3+ had a regression where `WA_TransparentForMouseEvents` stopped passing events through on Linux/macOS. The fix requires using `WindowTransparentForInput` window flag directly, but `setWindowFlags()` recreates the window.
**How to avoid:** On Linux, set `Qt.WindowType.WindowTransparentForInput` as part of the initial window flags (before `show()`). Toggle by adding/removing the flag and calling `show()` again -- or use `setAttribute(WA_TransparentForMouseEvents)` which on X11/XCB still modifies the X input shape correctly in recent PyQt6.
**Warning signs:** Overlay captures all mouse events even in passthrough mode on Linux.

### Pitfall 5: Wayland Session Detection
**What goes wrong:** Overlay fails to launch or has no transparency on Wayland.
**Why it happens:** Wayland doesn't support `WindowStaysOnTopHint` or arbitrary window positioning in the same way X11 does. The Wayland plugin also warns "This plugin does not support setting window opacity."
**How to avoid:** Detect Wayland session (`XDG_SESSION_TYPE=wayland` or `WAYLAND_DISPLAY` set) and force `QT_QPA_PLATFORM=xcb` before `QApplication` is created. Most Ubuntu Wayland sessions also run XWayland, so xcb works.
**Warning signs:** "Could not load platform plugin" errors, or "This plugin does not support setting window opacity" warnings.

### Pitfall 6: QGraphicsView Scroll Bars and Margins
**What goes wrong:** Scene doesn't cover full screen; scroll bars appear.
**Why it happens:** QGraphicsView defaults to showing scroll bars and has default margins.
**How to avoid:** Set `ScrollBarAlwaysOff` for both axes, `setFrameShape(NoFrame)`, `setContentsMargins(0,0,0,0)`, and `setViewportMargins(0,0,0,0)`. The existing code already does this correctly -- carry it forward.
**Warning signs:** 1-2 pixel border around overlay, scroll bars flickering.

## Code Examples

### Window Setup (Verified Pattern from Existing Code)
```python
# Source: recorder/overlay_view.py (existing codebase)
self.setWindowFlags(
    Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.Tool
)
self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
self.setStyleSheet("background: transparent;")
self.setFrameShape(QFrame.Shape.NoFrame)
self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
```

### Win32 Layered Window Flags (Verified Pattern)
```python
# Source: recorder/overlay_view.py (existing codebase)
import ctypes
WS_EX_LAYERED = 0x00080000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

hwnd = int(self.winId())
style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
new_style = style | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
```

### DwmFlush for Compositor Synchronization
```python
# Source: Microsoft Learn (https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/nf-dwmapi-dwmflush)
import ctypes

def dwm_flush() -> None:
    """Block until DWM has composited the current frame."""
    try:
        ctypes.windll.dwmapi.DwmFlush()
    except (AttributeError, OSError):
        # DWM not available (pre-Vista or server core)
        pass
```

### Forcing XCB on Wayland
```python
import os
import sys

def ensure_xcb_platform() -> None:
    """Force XCB backend if running under Wayland.

    Must be called BEFORE QApplication is created.
    """
    if sys.platform == "win32":
        return

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "")

    if session_type == "wayland" or wayland_display:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
```

### Smooth Color Transition with QPropertyAnimation
```python
# Source: Qt 6 Animation Framework docs
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QColor

# Animate border color from green to red
animation = QPropertyAnimation(border_widget, b"border_color")
animation.setDuration(300)  # 300ms transition
animation.setStartValue(QColor(50, 200, 50, 150))
animation.setEndValue(QColor(255, 50, 50, 200))
animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
animation.start()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `SetProcessDPIAware()` | `SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)` | Win10 1703 | Per-monitor DPI; required for multi-monitor setups |
| Disabling DWM composition | DWM always-on (Win8+) | Windows 8 | Must work with compositor; can't bypass it |
| `WA_TransparentForMouseEvents` alone | `WindowTransparentForInput` flag + `WA_TransparentForMouseEvents` | Qt 6.2.3 | Both needed for reliable cross-platform click-through |
| pynput for Windows hotkeys | `GetAsyncKeyState` polling | Existing codebase | pynput hooks fail alongside PyQt6 event loop on Windows |

**Deprecated/outdated:**
- `DwmEnableComposition()`: No-op since Windows 8. DWM cannot be disabled.
- `Qt.WindowType.X11BypassWindowManagerHint`: Makes window unmanageable; avoid.

## Open Questions

1. **Multi-monitor DPI mismatch**
   - What we know: `devicePixelRatio()` is per-screen; mss monitors have independent coordinates.
   - What's unclear: How to handle overlay spanning two monitors with different DPI (e.g., 1x laptop + 2x external).
   - Recommendation: Phase 1 targets primary monitor only. Multi-monitor is a natural extension but not in OVLY-01 through OVLY-05.

2. **Linux compositor flush reliability**
   - What we know: There's no `DwmFlush` equivalent on Linux X11. `processEvents()` + sleep is the only option.
   - What's unclear: Exact minimum sleep needed on various Ubuntu compositors (Mutter, KWin).
   - Recommendation: Make `capture_delay_ms` configurable in YAML config (default 100ms). Test on Ubuntu with both Mutter and KWin.

3. **Window hide vs. scene item hide for capture**
   - What we know: `window.hide()` is most reliable but causes flicker. Hiding scene items avoids flicker but may leave compositor artifacts.
   - What's unclear: Whether scene-item-only hiding + `DwmFlush()` is sufficient to eliminate artifacts.
   - Recommendation: Start with `window.hide()` approach. If flicker is unacceptable, try scene-item hiding + DwmFlush as optimization.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=7.0 (already in pyproject.toml dev deps) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/test_overlay.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OVLY-01 | Transparent fullscreen window with click-through toggle | unit (mock Qt) | `python -m pytest tests/test_overlay_state.py::test_passthrough_toggle -x` | -- Wave 0 |
| OVLY-02 | All elements hidden before screenshot capture | unit (mock DwmFlush) | `python -m pytest tests/test_capture_guard.py::test_hide_before_capture -x` | -- Wave 0 |
| OVLY-03 | DPI coordinate conversion matches mss output | unit | `python -m pytest tests/test_dpi.py -x` | -- Wave 0 |
| OVLY-04 | State machine transitions ready<->recording<->paused | unit | `python -m pytest tests/test_overlay_state.py::test_state_transitions -x` | -- Wave 0 |
| OVLY-05 | Platform detection and XCB fallback | unit | `python -m pytest tests/test_platform.py -x` | -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_overlay_state.py tests/test_dpi.py tests/test_capture_guard.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_overlay_state.py` -- covers OVLY-01, OVLY-04 (state machine, passthrough toggle)
- [ ] `tests/test_dpi.py` -- covers OVLY-03 (logical/physical pixel conversion)
- [ ] `tests/test_capture_guard.py` -- covers OVLY-02 (hide-flush-capture lifecycle)
- [ ] `tests/test_platform.py` -- covers OVLY-05 (Wayland detection, XCB fallback)

Note: Qt widget tests require either a display or `QT_QPA_PLATFORM=offscreen`. Tests should mock `QScreen.devicePixelRatio()` and platform calls rather than requiring a live display. Add `conftest.py` fixture for `QT_QPA_PLATFORM=offscreen` when running in CI.

## Sources

### Primary (HIGH confidence)
- [Qt 6 High DPI Documentation](https://doc.qt.io/qt-6/highdpi.html) - DPI scaling model, devicePixelRatio behavior
- [Qt 6 QScreen API](https://doc.qt.io/qt-6/qscreen.html) - Physical vs logical DPI, geometry methods
- [Microsoft DwmFlush](https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/nf-dwmapi-dwmflush) - Compositor synchronization
- [Qt 6 Wayland Integration](https://doc.qt.io/qt-6/wayland-and-qt.html) - Wayland limitations, XCB fallback
- Existing codebase: `recorder/overlay.py`, `recorder/overlay_view.py`, `recorder/hotkeys.py` - Proven Win32 patterns

### Secondary (MEDIUM confidence)
- [mss GitHub Issue #184](https://github.com/BoboTiG/python-mss/issues/184) - DPI awareness import order problem
- [Qt Forum: Click through windows](https://forum.qt.io/topic/83161/click-through-windows) - WA_TransparentForMouseEvents behavior
- [Qt Forum: Window opacity on Wayland](https://forum.qt.io/topic/158586/qt6-qpropertyanimation-this-plugin-does-not-support-setting-window-opacity) - Wayland opacity limitation
- [Shallow Thoughts: X11 Input Shapes](https://shallowsky.com/blog/programming/translucent-window-click-thru.html) - X11 SHAPE extension for click-through

### Tertiary (LOW confidence)
- Linux compositor flush timing estimates (100ms default) - based on general knowledge, not empirical testing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in pyproject.toml, existing code proves they work together
- Architecture: HIGH - patterns derived from existing working code + official Qt docs
- Pitfalls: HIGH - DWM flush timing and DPI issues are well-documented; import order issue confirmed by mss maintainer
- Cross-platform (Linux): MEDIUM - X11/XCB click-through approach verified in docs, but Wayland fallback untested
- Compositor flush timing (Linux): LOW - no definitive source for minimum sleep duration; needs empirical testing

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (stable domain; Qt6 and Win32 APIs unlikely to change)
