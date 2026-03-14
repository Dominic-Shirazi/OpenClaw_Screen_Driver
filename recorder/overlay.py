"""Transparent fullscreen overlay for recording sessions.

Provides a PyQt6 overlay window that sits on top of all other windows.
Two modes:
- PASSTHROUGH: Clicks go through to the app beneath (WS_EX_TRANSPARENT)
- RECORD: Captures mouse clicks for element tagging

Ctrl+R / F2 toggles between modes. Ctrl+Q / ESC closes the overlay.

Windows: Polls keyboard via GetAsyncKeyState on a QTimer — the only
approach that works alongside PyQt6's event loop (RegisterHotKey and
pynput hooks both fail to receive events). Falls back to pynput on
non-Windows platforms.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum, auto
from typing import Any, Callable

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from core.types import Rect
from recorder.element_types import ElementType

logger = logging.getLogger(__name__)

# Win32 constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

# Color map for element types (RGBA)
_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    # Interactive
    "textbox": (0, 150, 255, 140),         # Blue
    "button": (0, 200, 0, 140),            # Green
    "button_nav": (0, 255, 100, 140),      # Bright green
    "toggle": (255, 165, 0, 140),          # Orange
    "tab": (128, 0, 128, 140),             # Purple
    "dropdown": (255, 200, 0, 140),        # Gold
    "scrollbar": (128, 128, 128, 100),     # Gray
    "link": (0, 180, 230, 140),            # Light blue
    "icon": (180, 180, 0, 140),            # Olive
    "drag_source": (0, 255, 255, 140),     # Cyan
    "drag_target": (0, 200, 200, 140),     # Dark cyan
    # Structural regions (distinct muted tones)
    "region_chrome": (180, 120, 60, 80),   # Brown
    "region_menu": (160, 80, 160, 80),     # Mauve
    "region_sidebar": (60, 140, 130, 80),  # Teal
    "region_content": (100, 140, 200, 60), # Soft blue
    "region_form": (200, 160, 80, 80),     # Warm tan
    "region_header": (140, 100, 180, 80),  # Lavender
    "region_footer": (100, 120, 100, 80),  # Sage
    "region_toolbar": (160, 140, 100, 80), # Khaki
    "region_modal": (200, 80, 80, 80),     # Muted red
    "region_custom": (120, 120, 180, 80),  # Slate blue
    "landmark": (255, 200, 0, 120),        # Bright gold
    # Static / read-only
    "read_here": (255, 0, 0, 140),         # Red
    "image": (200, 200, 200, 80),          # Light gray
    "modal": (255, 100, 100, 100),         # Light red
    "notification": (255, 255, 0, 140),    # Yellow
    # Meta
    "unknown": (100, 100, 100, 100),       # Dark gray
}

_DEFAULT_COLOR = (100, 100, 100, 100)

# Border width for the screen-edge indicator
_BORDER_WIDTH = 3


class OverlayMode(Enum):
    """Overlay interaction modes."""
    PASSTHROUGH = auto()  # Clicks go through to underlying app
    RECORD = auto()       # Overlay captures clicks for tagging


# ---------------------------------------------------------------------------
# Global hotkey backends
# ---------------------------------------------------------------------------

class _Win32PollingHotkeyListener:
    """Polls keyboard state via GetAsyncKeyState on a QTimer.

    Both RegisterHotKey (hooks on a background thread) and pynput
    (SetWindowsHookEx) fail to receive key events when PyQt6's event
    loop is running. GetAsyncKeyState reads raw key state from the OS
    regardless of focus or hooks — and since QTimer fires on the Qt
    main thread, no cross-thread marshaling is needed.

    Supported shortcuts:
        Ctrl+R / F2  → toggle mode
        Ctrl+Q / ESC → close overlay
    """

    # Virtual key codes
    VK_CONTROL = 0x11
    VK_R = 0x52
    VK_Q = 0x51
    VK_F2 = 0x71
    VK_ESCAPE = 0x1B

    _POLL_MS = 80  # ~12 Hz — responsive without burning CPU

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_close: Callable[[], None],
    ) -> None:
        self._on_toggle = on_toggle
        self._on_close = on_close
        self._timer: QTimer | None = None
        # Track previous key-down state to fire only on press (not repeat)
        self._prev_r = False
        self._prev_q = False
        self._prev_f2 = False
        self._prev_esc = False

    def start(self) -> None:
        """Starts the polling timer."""
        import ctypes  # noqa: F811 — verify ctypes.windll available
        ctypes.windll.user32.GetAsyncKeyState  # quick sanity check

        self._timer = QTimer()
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        logger.info(
            "Win32 polling hotkeys started (%d ms): Ctrl+R, Ctrl+Q, F2, ESC",
            self._POLL_MS,
        )

    def stop(self) -> None:
        """Stops the polling timer."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
            logger.debug("Win32 polling hotkeys stopped")

    def _poll(self) -> None:
        """Called every _POLL_MS to check key states."""
        import ctypes
        user32 = ctypes.windll.user32
        get = user32.GetAsyncKeyState

        ctrl = bool(get(self.VK_CONTROL) & 0x8000)
        r_down = bool(get(self.VK_R) & 0x8000)
        q_down = bool(get(self.VK_Q) & 0x8000)
        f2_down = bool(get(self.VK_F2) & 0x8000)
        esc_down = bool(get(self.VK_ESCAPE) & 0x8000)

        # Ctrl+R or F2 → toggle (fire on leading edge only)
        if ctrl and r_down and not self._prev_r:
            self._on_toggle()
        if f2_down and not self._prev_f2:
            self._on_toggle()

        # Ctrl+Q or ESC → close (fire on leading edge only)
        if ctrl and q_down and not self._prev_q:
            self._on_close()
        if esc_down and not self._prev_esc:
            self._on_close()

        self._prev_r = ctrl and r_down
        self._prev_q = ctrl and q_down
        self._prev_f2 = f2_down
        self._prev_esc = esc_down


class _PynputHotkeyListener:
    """Global hotkey listener using pynput (macOS / Linux fallback).

    Note: pynput does NOT work alongside PyQt6 on Windows — the
    low-level keyboard hooks receive zero events when Qt's event loop
    is running. Use _Win32PollingHotkeyListener on Windows instead.
    """

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_close: Callable[[], None],
    ) -> None:
        self._on_toggle = on_toggle
        self._on_close = on_close
        self._listener: Any = None
        self._key_listener: Any = None

    def start(self) -> None:
        """Starts the pynput key listener."""
        try:
            from pynput import keyboard

            def on_activate_toggle() -> None:
                QTimer.singleShot(0, self._on_toggle)

            def on_activate_close() -> None:
                QTimer.singleShot(0, self._on_close)

            hotkeys = keyboard.GlobalHotKeys({
                "<ctrl>+r": on_activate_toggle,
                "<ctrl>+q": on_activate_close,
            })
            hotkeys.start()
            self._listener = hotkeys

            # F2 / ESC as single-key shortcuts via a regular Listener
            def _on_press(key: Any) -> None:
                try:
                    if key == keyboard.Key.f2:
                        QTimer.singleShot(0, self._on_toggle)
                    elif key == keyboard.Key.esc:
                        QTimer.singleShot(0, self._on_close)
                except Exception:
                    pass

            self._key_listener = keyboard.Listener(on_press=_on_press)
            self._key_listener.start()

            logger.info("pynput global hotkeys registered: Ctrl+R, Ctrl+Q, F2, ESC")
        except ImportError:
            logger.warning(
                "pynput not installed — global hotkeys unavailable. "
                "Ctrl+R / Ctrl+Q will only work when the overlay has focus."
            )
        except Exception as e:
            logger.warning("Failed to start pynput hotkeys: %s", e)

    def stop(self) -> None:
        """Stops the pynput listeners."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._key_listener is not None:
            self._key_listener.stop()
            self._key_listener = None


def _create_hotkey_listener(
    on_toggle: Callable[[], None],
    on_close: Callable[[], None],
) -> _Win32PollingHotkeyListener | _PynputHotkeyListener:
    """Creates the appropriate global hotkey listener for the platform.

    Windows: GetAsyncKeyState polling (only approach that works with PyQt6).
    macOS/Linux: pynput GlobalHotKeys + Listener.
    """
    if sys.platform == "win32":
        return _Win32PollingHotkeyListener(on_toggle, on_close)
    return _PynputHotkeyListener(on_toggle, on_close)


# ---------------------------------------------------------------------------
# Overlay controller
# ---------------------------------------------------------------------------

class OverlayController:
    """Controls the transparent fullscreen overlay.

    Manages mode switching, candidate rendering, click capture, and
    integration with the recording session pipeline.
    """

    # Minimum drag distance (pixels) to count as a bounding box vs a click
    _MIN_DRAG_PX = 8

    def __init__(
        self,
        on_element_clicked: Callable[[int, int, int, int, dict[str, Any] | None], None] | None = None,
        on_mode_changed: Callable[[OverlayMode], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Initializes the overlay controller.

        Args:
            on_element_clicked: Callback when user selects in RECORD mode.
                               Args: (x, y, w, h, matched_candidate_dict_or_None).
                               For point clicks, w=0, h=0.
                               For bounding boxes, (x, y) is the top-left corner.
            on_mode_changed: Callback when overlay mode changes.
            on_close: Callback when overlay is closed via Ctrl+Q / ESC.
        """
        self._on_element_clicked = on_element_clicked
        self._on_mode_changed = on_mode_changed
        self._on_close = on_close

        self._mode = OverlayMode.PASSTHROUGH
        self._candidates: list[dict[str, Any]] = []
        self._view: _OverlayView | None = None
        self._is_active = False
        self._hotkey_listener: _Win32PollingHotkeyListener | _PynputHotkeyListener | None = None

    @property
    def mode(self) -> OverlayMode:
        """Returns the current overlay mode."""
        return self._mode

    @property
    def is_active(self) -> bool:
        """Returns whether the overlay is currently shown."""
        return self._is_active

    def show(self, *, start_mode: OverlayMode = OverlayMode.PASSTHROUGH) -> None:
        """Shows the overlay in the specified mode.

        Creates the overlay window and registers global hotkeys.

        Args:
            start_mode: Initial mode (PASSTHROUGH or RECORD).
        """
        if self._view is not None and self._is_active:
            logger.debug("Overlay already active")
            return

        self._view = _OverlayView(controller=self)
        # Use show() + setGeometry instead of showFullScreen() to avoid
        # exclusive fullscreen mode that fights with Windows focus management
        # and causes other windows to freeze when clicked.
        if QApplication.primaryScreen():
            self._view.setGeometry(QApplication.primaryScreen().geometry())
        self._view.show()
        self._view.raise_()
        # Don't call activateWindow() — with WS_EX_NOACTIVATE the overlay
        # must never take focus.  Hotkeys come from pynput globally.
        self._is_active = True
        self._set_mode(start_mode)

        # Start global hotkeys (work even when overlay loses focus)
        self._hotkey_listener = _create_hotkey_listener(
            on_toggle=self.toggle_mode,
            on_close=self.close,
        )
        self._hotkey_listener.start()

        logger.info("Overlay shown in %s mode", start_mode.name)

    def close(self) -> None:
        """Closes and cleans up the overlay and hotkeys."""
        # Stop global hotkeys first
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
            self._hotkey_listener = None

        if self._view is not None:
            self._view.close()
            self._view = None
        self._is_active = False
        self._candidates = []

        if self._on_close:
            try:
                self._on_close()
            except Exception as e:
                logger.error("on_close callback error: %s", e)

        logger.info("Overlay closed")

    def toggle_mode(self) -> None:
        """Toggles between PASSTHROUGH and RECORD modes."""
        if self._mode == OverlayMode.PASSTHROUGH:
            self._set_mode(OverlayMode.RECORD)
        else:
            self._set_mode(OverlayMode.PASSTHROUGH)

    def set_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Updates the list of candidate elements to render.

        Args:
            candidates: List of dicts with rect, type_guess, label_guess, confidence.
        """
        self._candidates = candidates
        if self._view is not None:
            self._view.render_candidates(candidates)
        for c in candidates:
            r = c.get("rect", {})
            logger.info(
                "  candidate: %s '%s' at (%d,%d) %dx%d conf=%.2f",
                c.get("type_guess", "?"), c.get("label_guess", ""),
                r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0),
                c.get("confidence", 0),
            )
        logger.info("Rendered %d candidates on overlay", len(candidates))

    def clear_candidates(self) -> None:
        """Removes all candidate renderings."""
        self._candidates = []
        if self._view is not None:
            self._view.clear_scene()

    def _set_mode(self, mode: OverlayMode) -> None:
        """Sets the overlay mode and updates Win32 flags accordingly."""
        self._mode = mode

        if self._view is not None:
            if sys.platform == "win32":
                self._apply_win32_flags(mode)
            else:
                # Non-Windows: toggle mouse tracking attribute
                if mode == OverlayMode.PASSTHROUGH:
                    self._view.setAttribute(
                        Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
                    )
                else:
                    self._view.setAttribute(
                        Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
                    )

            # Update cursor
            if mode == OverlayMode.RECORD:
                self._view.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._view.setCursor(Qt.CursorShape.ArrowCursor)

            # Redraw the border and mode indicator
            self._view.refresh_overlay()

        if self._on_mode_changed:
            try:
                self._on_mode_changed(mode)
            except Exception as e:
                logger.error("on_mode_changed callback error: %s", e)

        logger.info("Overlay mode: %s", mode.name)

    def _apply_win32_flags(self, mode: OverlayMode) -> None:
        """Applies Win32 extended window style flags for click-through.

        In PASSTHROUGH mode, adds WS_EX_TRANSPARENT so clicks go through.
        In RECORD mode, removes WS_EX_TRANSPARENT so overlay captures clicks.
        """
        if self._view is None:
            return

        import ctypes
        user32 = ctypes.windll.user32

        hwnd = int(self._view.winId())
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)

        if mode == OverlayMode.PASSTHROUGH:
            # Add transparent flag — clicks pass through
            new_style = style | WS_EX_TRANSPARENT
        else:
            # Remove transparent flag — overlay captures clicks
            new_style = style & ~WS_EX_TRANSPARENT

        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    def _handle_selection(self, x: int, y: int, w: int, h: int) -> bool:
        """Handles a completed selection (point click or bounding box drag).

        For point clicks (w=0, h=0), finds the candidate at the click point.
        For bounding boxes, (x, y) is the top-left and (w, h) are dimensions.

        Args:
            x: Top-left X (or click X for point clicks).
            y: Top-left Y (or click Y for point clicks).
            w: Width of bounding box (0 for point click).
            h: Height of bounding box (0 for point click).

        Returns:
            True if the element was accepted (recorded), False if skipped.
        """
        if w > 0 and h > 0:
            # Bounding box — find candidate at center
            cx, cy = x + w // 2, y + h // 2
            matched = self._find_candidate_at(cx, cy)
            logger.info("Bbox selection: (%d, %d) %dx%d", x, y, w, h)
        else:
            matched = self._find_candidate_at(x, y)
            logger.info("Point click: (%d, %d)", x, y)

        accepted = False
        if self._on_element_clicked:
            try:
                # Callback returns True if element was recorded, False if skipped
                result = self._on_element_clicked(x, y, w, h, matched)
                accepted = bool(result)
            except Exception as e:
                logger.error("on_element_clicked callback error: %s", e)

        return accepted

    def _find_candidate_at(self, x: int, y: int) -> dict[str, Any] | None:
        """Finds the candidate element closest to a screen coordinate.

        Args:
            x: Screen X.
            y: Screen Y.

        Returns:
            The matching candidate dict, or None if no candidate is near.
        """
        best: dict[str, Any] | None = None
        best_area = float("inf")

        for candidate in self._candidates:
            rect = candidate.get("rect", {})
            rx = rect.get("x", 0)
            ry = rect.get("y", 0)
            rw = rect.get("w", 0)
            rh = rect.get("h", 0)

            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                area = rw * rh
                # Prefer smallest containing element (most specific)
                if area < best_area:
                    best = candidate
                    best_area = area

        return best


# ---------------------------------------------------------------------------
# Overlay view (Qt widget)
# ---------------------------------------------------------------------------

class _OverlayView(QGraphicsView):
    """The actual PyQt6 overlay window.

    Renders as a transparent fullscreen window with bounding box
    overlays for candidate elements. A colored border around the screen
    edge indicates the overlay is active and which mode it's in.
    """

    def __init__(self, controller: OverlayController) -> None:
        """Initializes the overlay view.

        Args:
            controller: The OverlayController managing this view.
        """
        scene = QGraphicsScene()
        super().__init__(scene)
        self._controller = controller

        # Window configuration for transparent overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # Don't show in taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setStyleSheet("background: transparent;")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Make scene cover the full screen
        self._screen_w = 1920
        self._screen_h = 1080
        if QApplication.primaryScreen():
            screen_geom = QApplication.primaryScreen().geometry()
            self._screen_w = screen_geom.width()
            self._screen_h = screen_geom.height()
        self.setSceneRect(0, 0, self._screen_w, self._screen_h)

        # Set up Win32 layered flags on Windows
        if sys.platform == "win32":
            QTimer.singleShot(0, self._setup_win32_layered)

        # Track scene items for cleanup
        self._mode_label: QGraphicsSimpleTextItem | None = None
        self._mode_bg: QGraphicsRectItem | None = None
        self._border_items: list[QGraphicsRectItem] = []
        self._click_catcher: QGraphicsRectItem | None = None

        # Drag-to-draw bounding box state
        self._drag_start: QPointF | None = None
        self._rubber_band: QGraphicsRectItem | None = None

        # Draw initial overlay indicators after event loop starts
        QTimer.singleShot(50, self.refresh_overlay)

    def _setup_win32_layered(self) -> None:
        """Applies WS_EX_LAYERED and WS_EX_TOOLWINDOW flags after window creation."""
        try:
            import ctypes
            user32 = ctypes.windll.user32

            hwnd = int(self.winId())
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            # WS_EX_LAYERED: enables per-pixel alpha (transparent pixels pass clicks)
            # WS_EX_TOOLWINDOW: hides from taskbar/alt-tab
            # WS_EX_NOACTIVATE: prevents overlay from stealing focus when clicked
            #   This is safe because we use pynput for global hotkeys (no need
            #   for keyboard focus). Without this flag, clicking any other window
            #   in PASSTHROUGH mode triggers a focus war with WindowStaysOnTopHint
            #   that freezes the Python event loop.
            new_style = style | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            logger.debug("Win32 layered flags applied to overlay (hwnd=%d)", hwnd)
        except Exception as e:
            logger.warning("Failed to apply Win32 flags: %s", e)

    def refresh_overlay(self) -> None:
        """Redraws border, mode indicator, and click catcher for the current mode."""
        self._update_click_catcher()
        self._draw_border()
        self._update_mode_indicator()

    def _update_click_catcher(self) -> None:
        """Adds/removes a nearly-invisible full-screen rect for mouse hit-testing.

        With WS_EX_LAYERED, Windows does per-pixel alpha hit-testing.
        Fully transparent pixels don't receive mouse events. In RECORD
        mode we need the entire screen to capture clicks, so we add a
        rect with alpha=1 (invisible to the eye but enough for hit-test).
        In PASSTHROUGH mode we remove it so clicks pass through.
        """
        if self._click_catcher is not None:
            self.scene().removeItem(self._click_catcher)
            self._click_catcher = None

        if self._controller.mode == OverlayMode.RECORD:
            self._click_catcher = self.scene().addRect(
                QRectF(0, 0, self._screen_w, self._screen_h),
                QPen(Qt.PenStyle.NoPen),
                QBrush(QColor(0, 0, 0, 1)),  # alpha=1: invisible but hittable
            )
            self._click_catcher.setZValue(-100)  # Behind everything

    def render_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Renders bounding boxes for candidate elements on the overlay.

        Args:
            candidates: List of candidate element dicts with rect and type_guess.
        """
        self.clear_scene()

        for candidate in candidates:
            rect = candidate.get("rect", {})
            x = rect.get("x", 0)
            y = rect.get("y", 0)
            w = rect.get("w", 0)
            h = rect.get("h", 0)

            if w <= 0 or h <= 0:
                continue

            type_guess = candidate.get("type_guess", "unknown")
            label = candidate.get("label_guess", "")
            confidence = candidate.get("confidence", 0.0)

            # Get color for element type
            r, g, b, a = _TYPE_COLORS.get(type_guess, _DEFAULT_COLOR)

            # Draw bounding box
            pen = QPen(QColor(r, g, b, min(a + 60, 255)))
            pen.setWidth(2)
            brush = QBrush(QColor(r, g, b, a // 3))  # Lighter fill

            rect_item = self.scene().addRect(QRectF(x, y, w, h), pen, brush)

            # Add label text above the box
            if label:
                text_str = f"{label} ({confidence:.0%})"
                text_item = self.scene().addSimpleText(text_str)
                text_item.setPos(x, max(0, y - 16))
                text_item.setBrush(QBrush(QColor(r, g, b, 230)))
                font = QFont("Segoe UI", 9)
                font.setBold(True)
                text_item.setFont(font)

        self.refresh_overlay()
        # Force viewport repaint — WA_TranslucentBackground on Windows
        # may not auto-repaint when scene changes arrive via QTimer
        self.viewport().update()

    def clear_scene(self) -> None:
        """Removes all items from the scene."""
        self.scene().clear()
        self._mode_label = None
        self._mode_bg = None
        self._border_items = []
        self._click_catcher = None

    def _draw_border(self) -> None:
        """Draws a colored border around screen edges to show overlay is active.

        Green border = PASSTHROUGH (clicks go through).
        Red border = RECORD (clicks captured by overlay).
        """
        # Remove old border items
        for item in self._border_items:
            self.scene().removeItem(item)
        self._border_items = []

        mode = self._controller.mode
        if mode == OverlayMode.RECORD:
            color = QColor(255, 50, 50, 200)  # Red
        else:
            color = QColor(50, 200, 50, 150)  # Green

        bw = _BORDER_WIDTH
        w = self._screen_w
        h = self._screen_h

        pen = QPen(Qt.PenStyle.NoPen)
        brush = QBrush(color)

        # Top
        self._border_items.append(
            self.scene().addRect(QRectF(0, 0, w, bw), pen, brush)
        )
        # Bottom
        self._border_items.append(
            self.scene().addRect(QRectF(0, h - bw, w, bw), pen, brush)
        )
        # Left
        self._border_items.append(
            self.scene().addRect(QRectF(0, 0, bw, h), pen, brush)
        )
        # Right
        self._border_items.append(
            self.scene().addRect(QRectF(w - bw, 0, bw, h), pen, brush)
        )

    def _update_mode_indicator(self) -> None:
        """Shows the current mode as a label in the top-left corner.

        Renders with a dark semi-transparent background so it's always
        visible regardless of what's underneath.
        """
        mode = self._controller.mode
        if mode == OverlayMode.RECORD:
            mode_text = "[RECORD] Click or drag-to-box elements  |  Ctrl+R = passthrough  |  Ctrl+Q = save & quit"
        else:
            mode_text = "[PASSTHROUGH] Clicks go through  |  Ctrl+R = record  |  Ctrl+Q = save & quit"

        font = QFont("Segoe UI", 10)
        font.setBold(True)

        if mode == OverlayMode.RECORD:
            text_color = QColor(255, 100, 100, 240)
        else:
            text_color = QColor(100, 255, 100, 240)

        if self._mode_label is not None:
            self._mode_label.setText(mode_text)
            self._mode_label.setFont(font)
            self._mode_label.setBrush(QBrush(text_color))
            # Resize background to fit new text
            if self._mode_bg is not None:
                br = self._mode_label.boundingRect()
                self._mode_bg.setRect(QRectF(
                    6, 6, br.width() + 18, br.height() + 8,
                ))
            return

        # Background: dark semi-transparent box
        self._mode_label = self.scene().addSimpleText(mode_text)
        self._mode_label.setFont(font)
        self._mode_label.setBrush(QBrush(text_color))

        br = self._mode_label.boundingRect()
        self._mode_bg = self.scene().addRect(
            QRectF(6, 6, br.width() + 18, br.height() + 8),
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(0, 0, 0, 180)),
        )
        # Ensure background is behind the text
        self._mode_bg.setZValue(100)
        self._mode_label.setZValue(101)
        self._mode_label.setPos(15, 10)

    def keyPressEvent(self, event: Any) -> None:
        """Handles keyboard input when overlay has focus (fallback).

        Global hotkeys (Ctrl+R, Ctrl+Q) work even without focus.
        F2 / ESC only work when the overlay has keyboard focus.
        """
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if key == Qt.Key.Key_F2 or (ctrl and key == Qt.Key.Key_R):
            self._controller.toggle_mode()
        elif key == Qt.Key.Key_Escape or (ctrl and key == Qt.Key.Key_Q):
            self._controller.close()
        else:
            # Eat all other keys so they don't cause confusion
            event.accept()

    def mousePressEvent(self, event: Any) -> None:
        """Starts a drag-to-draw bounding box or point click in RECORD mode.

        Records the start point and creates a rubber-band rectangle.
        In PASSTHROUGH mode, events are eaten silently.
        """
        if self._controller.mode == OverlayMode.RECORD:
            self._drag_start = self.mapToScene(event.pos())
            # Create the rubber-band rectangle (initially zero-size)
            pen = QPen(QColor(255, 255, 0, 220))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            brush = QBrush(QColor(255, 255, 0, 30))
            self._rubber_band = self.scene().addRect(
                QRectF(self._drag_start, self._drag_start), pen, brush,
            )
            self._rubber_band.setZValue(200)  # On top of everything
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        """Updates the rubber-band rectangle as the user drags.

        Only active in RECORD mode while a drag is in progress.
        """
        if (
            self._controller.mode == OverlayMode.RECORD
            and self._drag_start is not None
            and self._rubber_band is not None
        ):
            current = self.mapToScene(event.pos())
            # Build a normalized rect (handles dragging in any direction)
            x1 = min(self._drag_start.x(), current.x())
            y1 = min(self._drag_start.y(), current.y())
            x2 = max(self._drag_start.x(), current.x())
            y2 = max(self._drag_start.y(), current.y())
            self._rubber_band.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:
        """Completes the bounding box or point click on mouse release.

        If the drag distance is below the threshold, treats it as a
        point click (w=0, h=0). Otherwise sends the bounding box.
        """
        if (
            self._controller.mode == OverlayMode.RECORD
            and self._drag_start is not None
        ):
            end = self.mapToScene(event.pos())
            # Compute normalized rectangle
            x1 = min(self._drag_start.x(), end.x())
            y1 = min(self._drag_start.y(), end.y())
            x2 = max(self._drag_start.x(), end.x())
            y2 = max(self._drag_start.y(), end.y())
            w = x2 - x1
            h = y2 - y1

            # Clean up rubber band
            if self._rubber_band is not None:
                self.scene().removeItem(self._rubber_band)
                self._rubber_band = None

            min_drag = self._controller._MIN_DRAG_PX
            if w >= min_drag and h >= min_drag:
                # Bounding box — show pending rect, finalize after dialog
                accepted = self._controller._handle_selection(
                    int(x1), int(y1), int(w), int(h),
                )
                if accepted:
                    # Draw persistent confirmed rect (cyan solid)
                    pen = QPen(QColor(0, 255, 255, 200))
                    pen.setWidth(2)
                    brush = QBrush(QColor(0, 255, 255, 40))
                    self.scene().addRect(QRectF(x1, y1, w, h), pen, brush)
                # If skipped, no persistent rect — clean exit
            else:
                # Point click — use the original press position
                sx = int(self._drag_start.x())
                sy = int(self._drag_start.y())
                self._controller._handle_selection(sx, sy, 0, 0)

            self._drag_start = None
        event.accept()
