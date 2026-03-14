"""Transparent fullscreen overlay for recording sessions.

Provides a PyQt6 overlay window that sits on top of all other windows.
Two modes:
- PASSTHROUGH: Clicks go through to the app beneath (WS_EX_TRANSPARENT)
- RECORD: Captures mouse clicks for element tagging

Features:
- Interactive bounding boxes with draggable corner handles
- Click probability donut visualization (radial gradient)
- One-by-one element review flow after batch detection

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
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
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
    "textbox": (0, 150, 255, 140),
    "button": (0, 200, 0, 140),
    "button_nav": (0, 255, 100, 140),
    "toggle": (255, 165, 0, 140),
    "tab": (128, 0, 128, 140),
    "dropdown": (255, 200, 0, 140),
    "scrollbar": (128, 128, 128, 100),
    "link": (0, 180, 230, 140),
    "icon": (180, 180, 0, 140),
    "drag_source": (0, 255, 255, 140),
    "drag_target": (0, 200, 200, 140),
    # Structural regions
    "region_chrome": (180, 120, 60, 80),
    "region_menu": (160, 80, 160, 80),
    "region_sidebar": (60, 140, 130, 80),
    "region_content": (100, 140, 200, 60),
    "region_form": (200, 160, 80, 80),
    "region_header": (140, 100, 180, 80),
    "region_footer": (100, 120, 100, 80),
    "region_toolbar": (160, 140, 100, 80),
    "region_modal": (200, 80, 80, 80),
    "region_custom": (120, 120, 180, 80),
    "landmark": (255, 200, 0, 120),
    # Static / read-only
    "read_here": (255, 0, 0, 140),
    "image": (200, 200, 200, 80),
    "modal": (255, 100, 100, 100),
    "notification": (255, 255, 0, 140),
    # Meta
    "unknown": (100, 100, 100, 100),
}

_DEFAULT_COLOR = (100, 100, 100, 100)

# Border width for the screen-edge indicator
_BORDER_WIDTH = 3

# Corner handle radius in pixels
_HANDLE_RADIUS = 5


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
        self._prev_r = False
        self._prev_q = False
        self._prev_f2 = False
        self._prev_esc = False

    def start(self) -> None:
        """Starts the polling timer."""
        import ctypes
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

        if ctrl and r_down and not self._prev_r:
            self._on_toggle()
        if f2_down and not self._prev_f2:
            self._on_toggle()

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
    """Creates the appropriate global hotkey listener for the platform."""
    if sys.platform == "win32":
        return _Win32PollingHotkeyListener(on_toggle, on_close)
    return _PynputHotkeyListener(on_toggle, on_close)


# ---------------------------------------------------------------------------
# Interactive element box components
# ---------------------------------------------------------------------------

class _HandleItem(QGraphicsEllipseItem):
    """Draggable corner handle for resizing element bounding boxes.

    Small circle at a bbox corner. Drag to resize the parent box.
    The cursor changes on hover to indicate resize direction.
    """

    def __init__(
        self,
        corner: str,
        box_group: _ElementBoxGroup,
        color: QColor,
    ) -> None:
        r = _HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self._corner = corner
        self._box_group = box_group

        self.setPen(QPen(color, 1.5))
        self.setBrush(QBrush(QColor(255, 255, 255, 200)))
        self.setZValue(50)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        if corner in ("tl", "br"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        self.setAcceptHoverEvents(True)

    def itemChange(
        self, change: QGraphicsItem.GraphicsItemChange, value: Any,
    ) -> Any:
        """Notifies the parent box group when this handle moves."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._box_group.handle_moved(self._corner)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event: Any) -> None:
        """Enlarges handle on hover for easier grabbing."""
        self.setBrush(QBrush(QColor(255, 255, 100, 240)))
        self.setScale(1.4)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: Any) -> None:
        """Restores handle size when mouse leaves."""
        self.setBrush(QBrush(QColor(255, 255, 255, 200)))
        self.setScale(1.0)
        super().hoverLeaveEvent(event)


class _ElementBoxGroup:
    """Manages a bounding box with corner handles, label, and click donut.

    Contains:
    - Main rect item (colored border + light fill)
    - 4 corner handles (draggable to resize)
    - Label text above the box
    - Radial gradient "donut" showing click probability distribution
    - Center dot marking the precise click target
    """

    def __init__(
        self,
        scene: QGraphicsScene,
        x: float,
        y: float,
        w: float,
        h: float,
        color_rgba: tuple[int, int, int, int],
        label: str,
        confidence: float,
    ) -> None:
        self._scene = scene
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        r, g, b, a = color_rgba
        self._color = QColor(r, g, b)
        self._alpha = a

        # Main rect
        pen = QPen(QColor(r, g, b, min(a + 60, 255)))
        pen.setWidth(2)
        brush = QBrush(QColor(r, g, b, a // 4))
        self._rect_item = scene.addRect(QRectF(x, y, w, h), pen, brush)
        self._rect_item.setZValue(10)

        # Donut gradient (click probability visualization)
        self._donut_item = self._create_donut(x, y, w, h)

        # Center dot — precise click target indicator
        dot_r = 3.0
        self._center_dot = scene.addEllipse(
            x + w / 2 - dot_r, y + h / 2 - dot_r, dot_r * 2, dot_r * 2,
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(255, 255, 255, 180)),
        )
        self._center_dot.setZValue(15)

        # Label text
        self._label_item: QGraphicsSimpleTextItem | None = None
        if label:
            text_str = f"{label} ({confidence:.0%})"
            self._label_item = scene.addSimpleText(text_str)
            self._label_item.setPos(x, max(0, y - 16))
            self._label_item.setBrush(QBrush(QColor(r, g, b, 230)))
            font = QFont("Segoe UI", 9)
            font.setBold(True)
            self._label_item.setFont(font)
            self._label_item.setZValue(20)

        # Corner handles
        handle_color = QColor(r, g, b, 220)
        self._handles: dict[str, _HandleItem] = {}
        for corner in ("tl", "tr", "bl", "br"):
            handle = _HandleItem(corner, self, handle_color)
            scene.addItem(handle)
            self._handles[corner] = handle
        self._position_handles()

    def _create_donut(
        self, x: float, y: float, w: float, h: float,
    ) -> QGraphicsEllipseItem:
        """Creates the radial gradient donut showing click probability."""
        cx = x + w / 2
        cy = y + h / 2
        radius = max(w, h) / 2
        if radius < 1:
            radius = 1.0

        gradient = QRadialGradient(cx, cy, radius)
        gradient.setColorAt(0.0, QColor(80, 220, 80, 100))
        gradient.setColorAt(0.4, QColor(80, 220, 80, 60))
        gradient.setColorAt(0.8, QColor(80, 220, 80, 20))
        gradient.setColorAt(1.0, QColor(80, 220, 80, 0))

        donut = self._scene.addEllipse(
            x, y, w, h,
            QPen(Qt.PenStyle.NoPen),
            QBrush(gradient),
        )
        donut.setZValue(5)
        return donut

    def _position_handles(self) -> None:
        """Sets handle positions to match current rect corners."""
        x, y, w, h = self._x, self._y, self._w, self._h
        positions = {
            "tl": (x, y),
            "tr": (x + w, y),
            "bl": (x, y + h),
            "br": (x + w, y + h),
        }
        for corner, (hx, hy) in positions.items():
            handle = self._handles[corner]
            # Disable geometry signals to prevent recursion
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
                False,
            )
            handle.setPos(hx, hy)
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
                True,
            )

    def handle_moved(self, corner: str) -> None:
        """Called when a corner handle is dragged. Updates rect and siblings.

        The opposite corner stays fixed; the moved corner defines the new
        edge positions. Adjacent corners are repositioned to match.
        """
        moved = self._handles[corner].pos()

        if corner == "tl":
            fixed = self._handles["br"].pos()
            new_x, new_y = moved.x(), moved.y()
            new_w = fixed.x() - moved.x()
            new_h = fixed.y() - moved.y()
        elif corner == "tr":
            fixed = self._handles["bl"].pos()
            new_x = fixed.x()
            new_y = moved.y()
            new_w = moved.x() - fixed.x()
            new_h = fixed.y() - moved.y()
        elif corner == "bl":
            fixed = self._handles["tr"].pos()
            new_x = moved.x()
            new_y = fixed.y()
            new_w = fixed.x() - moved.x()
            new_h = moved.y() - fixed.y()
        elif corner == "br":
            fixed = self._handles["tl"].pos()
            new_x = fixed.x()
            new_y = fixed.y()
            new_w = moved.x() - fixed.x()
            new_h = moved.y() - fixed.y()
        else:
            return

        # Enforce minimum size — don't update if too small
        if new_w < 10 or new_h < 10:
            return

        self._x = new_x
        self._y = new_y
        self._w = new_w
        self._h = new_h

        # Update rect
        self._rect_item.setRect(QRectF(new_x, new_y, new_w, new_h))

        # Rebuild donut
        self._scene.removeItem(self._donut_item)
        self._donut_item = self._create_donut(new_x, new_y, new_w, new_h)

        # Update center dot
        dot_r = 3.0
        self._center_dot.setRect(QRectF(
            new_x + new_w / 2 - dot_r, new_y + new_h / 2 - dot_r,
            dot_r * 2, dot_r * 2,
        ))

        # Update label position
        if self._label_item is not None:
            self._label_item.setPos(new_x, max(0, new_y - 16))

        # Reposition sibling handles
        self._position_handles()

    def _all_items(self) -> list[Any]:
        """Returns all scene items owned by this group."""
        items: list[Any] = [
            self._rect_item, self._donut_item, self._center_dot,
        ]
        if self._label_item:
            items.append(self._label_item)
        items.extend(self._handles.values())
        return items

    def highlight(self, active: bool) -> None:
        """Highlights (active review) or dims this element box."""
        if active:
            pen = QPen(QColor(255, 255, 0, 255))
            pen.setWidth(3)
            self._rect_item.setPen(pen)
            for item in self._all_items():
                item.setOpacity(1.0)
        else:
            for item in self._all_items():
                item.setOpacity(0.3)

    def reset_highlight(self) -> None:
        """Restores normal appearance after review."""
        r = self._color.red()
        g = self._color.green()
        b = self._color.blue()
        pen = QPen(QColor(r, g, b, min(self._alpha + 60, 255)))
        pen.setWidth(2)
        self._rect_item.setPen(pen)
        for item in self._all_items():
            item.setOpacity(1.0)

    def get_rect(self) -> tuple[int, int, int, int]:
        """Returns the current bbox as (x, y, w, h) after any resizing."""
        return (int(self._x), int(self._y), int(self._w), int(self._h))


# ---------------------------------------------------------------------------
# Overlay controller
# ---------------------------------------------------------------------------

class OverlayController:
    """Controls the transparent fullscreen overlay.

    Manages mode switching, candidate rendering, click capture,
    review flow, and integration with the recording session pipeline.
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
        """Shows the overlay in the specified mode."""
        if self._view is not None and self._is_active:
            logger.debug("Overlay already active")
            return

        self._view = _OverlayView(controller=self)
        if QApplication.primaryScreen():
            self._view.setGeometry(QApplication.primaryScreen().geometry())
        self._view.show()
        self._view.raise_()
        self._is_active = True
        self._set_mode(start_mode)

        self._hotkey_listener = _create_hotkey_listener(
            on_toggle=self.toggle_mode,
            on_close=self.close,
        )
        self._hotkey_listener.start()

        logger.info("Overlay shown in %s mode", start_mode.name)

    def close(self) -> None:
        """Closes and cleans up the overlay and hotkeys."""
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

    def start_review(
        self,
        on_review_element: Callable[[int, dict[str, Any]], bool],
    ) -> None:
        """Starts one-by-one review of detected candidates.

        Iterates through each candidate, highlighting it on the overlay,
        and calling the callback which should open a TagDialog.

        Args:
            on_review_element: Called for each candidate with (index, candidate).
                              Should return True if accepted, False if skipped.
                              The candidate dict's rect may have been updated
                              by handle dragging.
        """
        if not self._candidates or self._view is None:
            return

        for i, candidate in enumerate(self._candidates):
            # Highlight current candidate, dim others
            self._view.highlight_candidate(i)
            QApplication.processEvents()

            # Update candidate rect from the (possibly resized) box
            rect_tuple = self._view.get_candidate_rect(i)
            if rect_tuple:
                x, y, w, h = rect_tuple
                candidate["rect"] = {"x": x, "y": y, "w": w, "h": h}

            on_review_element(i, candidate)

        # Restore all highlights after review
        self._view.reset_highlights()
        logger.info("Review complete — %d candidates reviewed", len(self._candidates))

    def highlight_candidate(self, index: int) -> None:
        """Highlights a single candidate on the overlay."""
        if self._view is not None:
            self._view.highlight_candidate(index)

    def get_candidate_rect(self, index: int) -> tuple[int, int, int, int] | None:
        """Returns the (possibly resized) rect for a candidate."""
        if self._view is not None:
            return self._view.get_candidate_rect(index)
        return None

    def _set_mode(self, mode: OverlayMode) -> None:
        """Sets the overlay mode and updates Win32 flags accordingly."""
        self._mode = mode

        if self._view is not None:
            if sys.platform == "win32":
                self._apply_win32_flags(mode)
            else:
                if mode == OverlayMode.PASSTHROUGH:
                    self._view.setAttribute(
                        Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
                    )
                else:
                    self._view.setAttribute(
                        Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
                    )

            if mode == OverlayMode.RECORD:
                self._view.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._view.setCursor(Qt.CursorShape.ArrowCursor)

            self._view.refresh_overlay()

        if self._on_mode_changed:
            try:
                self._on_mode_changed(mode)
            except Exception as e:
                logger.error("on_mode_changed callback error: %s", e)

        logger.info("Overlay mode: %s", mode.name)

    def _apply_win32_flags(self, mode: OverlayMode) -> None:
        """Applies Win32 extended window style flags for click-through."""
        if self._view is None:
            return

        import ctypes
        user32 = ctypes.windll.user32

        hwnd = int(self._view.winId())
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)

        if mode == OverlayMode.PASSTHROUGH:
            new_style = style | WS_EX_TRANSPARENT
        else:
            new_style = style & ~WS_EX_TRANSPARENT

        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    def _handle_selection(self, x: int, y: int, w: int, h: int) -> bool:
        """Handles a completed selection (point click or bounding box drag).

        Args:
            x: Top-left X (or click X for point clicks).
            y: Top-left Y (or click Y for point clicks).
            w: Width of bounding box (0 for point click).
            h: Height of bounding box (0 for point click).

        Returns:
            True if the element was accepted (recorded), False if skipped.
        """
        if w > 0 and h > 0:
            cx, cy = x + w // 2, y + h // 2
            matched = self._find_candidate_at(cx, cy)
            logger.info("Bbox selection: (%d, %d) %dx%d", x, y, w, h)
        else:
            matched = self._find_candidate_at(x, y)
            logger.info("Point click: (%d, %d)", x, y)

        accepted = False
        if self._on_element_clicked:
            try:
                result = self._on_element_clicked(x, y, w, h, matched)
                accepted = bool(result)
            except Exception as e:
                logger.error("on_element_clicked callback error: %s", e)

        return accepted

    def _find_candidate_at(self, x: int, y: int) -> dict[str, Any] | None:
        """Finds the candidate element closest to a screen coordinate."""
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
                if area < best_area:
                    best = candidate
                    best_area = area

        return best


# ---------------------------------------------------------------------------
# Overlay view (Qt widget)
# ---------------------------------------------------------------------------

class _OverlayView(QGraphicsView):
    """The actual PyQt6 overlay window.

    Renders as a transparent fullscreen window with interactive bounding
    box overlays for candidate elements. Corner handles allow resizing.
    A colored border around the screen edge indicates mode.
    """

    def __init__(self, controller: OverlayController) -> None:
        """Initializes the overlay view."""
        scene = QGraphicsScene()
        super().__init__(scene)
        self._controller = controller

        # Window configuration for transparent overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
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

        # Interactive element boxes
        self._element_boxes: list[_ElementBoxGroup] = []

        # Drag-to-draw bounding box state
        self._drag_start: QPointF | None = None
        self._rubber_band: QGraphicsRectItem | None = None

        # Handle dragging state (when user drags a corner handle)
        self._handle_dragging = False

        # Draw initial overlay indicators after event loop starts
        QTimer.singleShot(50, self.refresh_overlay)

    def _setup_win32_layered(self) -> None:
        """Applies WS_EX_LAYERED and WS_EX_TOOLWINDOW flags after window creation."""
        try:
            import ctypes
            user32 = ctypes.windll.user32

            hwnd = int(self.winId())
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
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
        """Adds/removes a nearly-invisible full-screen rect for mouse hit-testing."""
        if self._click_catcher is not None:
            self.scene().removeItem(self._click_catcher)
            self._click_catcher = None

        if self._controller.mode == OverlayMode.RECORD:
            self._click_catcher = self.scene().addRect(
                QRectF(0, 0, self._screen_w, self._screen_h),
                QPen(Qt.PenStyle.NoPen),
                QBrush(QColor(0, 0, 0, 1)),
            )
            self._click_catcher.setZValue(-100)

    def render_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Renders interactive bounding boxes with handles and donut overlays.

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
            color_rgba = _TYPE_COLORS.get(type_guess, _DEFAULT_COLOR)

            box = _ElementBoxGroup(
                self.scene(), x, y, w, h,
                color_rgba, label, confidence,
            )
            self._element_boxes.append(box)

        self.refresh_overlay()
        self.viewport().update()

    def clear_scene(self) -> None:
        """Removes all items from the scene."""
        self.scene().clear()
        self._mode_label = None
        self._mode_bg = None
        self._border_items = []
        self._click_catcher = None
        self._element_boxes = []

    def highlight_candidate(self, index: int) -> None:
        """Highlights candidate at index, dims all others for review."""
        for i, box in enumerate(self._element_boxes):
            box.highlight(i == index)
        self.viewport().update()

    def reset_highlights(self) -> None:
        """Restores all candidates to normal appearance after review."""
        for box in self._element_boxes:
            box.reset_highlight()
        self.viewport().update()

    def get_candidate_rect(self, index: int) -> tuple[int, int, int, int] | None:
        """Returns the (possibly resized) rect for candidate at index."""
        if 0 <= index < len(self._element_boxes):
            return self._element_boxes[index].get_rect()
        return None

    def _draw_border(self) -> None:
        """Draws a colored border around screen edges to show overlay is active."""
        for item in self._border_items:
            self.scene().removeItem(item)
        self._border_items = []

        mode = self._controller.mode
        if mode == OverlayMode.RECORD:
            color = QColor(255, 50, 50, 200)
        else:
            color = QColor(50, 200, 50, 150)

        bw = _BORDER_WIDTH
        w = self._screen_w
        h = self._screen_h

        pen = QPen(Qt.PenStyle.NoPen)
        brush = QBrush(color)

        self._border_items.append(
            self.scene().addRect(QRectF(0, 0, w, bw), pen, brush)
        )
        self._border_items.append(
            self.scene().addRect(QRectF(0, h - bw, w, bw), pen, brush)
        )
        self._border_items.append(
            self.scene().addRect(QRectF(0, 0, bw, h), pen, brush)
        )
        self._border_items.append(
            self.scene().addRect(QRectF(w - bw, 0, bw, h), pen, brush)
        )

    def _update_mode_indicator(self) -> None:
        """Shows the current mode as a label in the top-left corner."""
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
            if self._mode_bg is not None:
                br = self._mode_label.boundingRect()
                self._mode_bg.setRect(QRectF(
                    6, 6, br.width() + 18, br.height() + 8,
                ))
            return

        self._mode_label = self.scene().addSimpleText(mode_text)
        self._mode_label.setFont(font)
        self._mode_label.setBrush(QBrush(text_color))

        br = self._mode_label.boundingRect()
        self._mode_bg = self.scene().addRect(
            QRectF(6, 6, br.width() + 18, br.height() + 8),
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(0, 0, 0, 180)),
        )
        self._mode_bg.setZValue(100)
        self._mode_label.setZValue(101)
        self._mode_label.setPos(15, 10)

    def keyPressEvent(self, event: Any) -> None:
        """Handles keyboard input when overlay has focus (fallback)."""
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if key == Qt.Key.Key_F2 or (ctrl and key == Qt.Key.Key_R):
            self._controller.toggle_mode()
        elif key == Qt.Key.Key_Escape or (ctrl and key == Qt.Key.Key_Q):
            self._controller.close()
        else:
            event.accept()

    def mousePressEvent(self, event: Any) -> None:
        """Starts a drag-to-draw bounding box, point click, or handle drag.

        In RECORD mode, checks if the click is on a corner handle first.
        If so, delegates to the scene for handle dragging. Otherwise
        starts the rubber-band selection.
        """
        if self._controller.mode == OverlayMode.RECORD:
            # Check if clicking on a corner handle
            item = self.itemAt(event.pos())
            if isinstance(item, _HandleItem):
                self._handle_dragging = True
                super().mousePressEvent(event)
                return

            self._drag_start = self.mapToScene(event.pos())
            pen = QPen(QColor(255, 255, 0, 220))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            brush = QBrush(QColor(255, 255, 0, 30))
            self._rubber_band = self.scene().addRect(
                QRectF(self._drag_start, self._drag_start), pen, brush,
            )
            self._rubber_band.setZValue(200)
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        """Updates the rubber-band rectangle or delegates handle drag."""
        if self._handle_dragging:
            super().mouseMoveEvent(event)
            return

        if (
            self._controller.mode == OverlayMode.RECORD
            and self._drag_start is not None
            and self._rubber_band is not None
        ):
            current = self.mapToScene(event.pos())
            x1 = min(self._drag_start.x(), current.x())
            y1 = min(self._drag_start.y(), current.y())
            x2 = max(self._drag_start.x(), current.x())
            y2 = max(self._drag_start.y(), current.y())
            self._rubber_band.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:
        """Completes the bounding box, point click, or handle drag."""
        if self._handle_dragging:
            super().mouseReleaseEvent(event)
            self._handle_dragging = False
            return

        if (
            self._controller.mode == OverlayMode.RECORD
            and self._drag_start is not None
        ):
            end = self.mapToScene(event.pos())
            x1 = min(self._drag_start.x(), end.x())
            y1 = min(self._drag_start.y(), end.y())
            x2 = max(self._drag_start.x(), end.x())
            y2 = max(self._drag_start.y(), end.y())
            w = x2 - x1
            h = y2 - y1

            if self._rubber_band is not None:
                self.scene().removeItem(self._rubber_band)
                self._rubber_band = None

            min_drag = self._controller._MIN_DRAG_PX
            if w >= min_drag and h >= min_drag:
                accepted = self._controller._handle_selection(
                    int(x1), int(y1), int(w), int(h),
                )
                if accepted:
                    pen = QPen(QColor(0, 255, 255, 200))
                    pen.setWidth(2)
                    brush = QBrush(QColor(0, 255, 255, 40))
                    self.scene().addRect(QRectF(x1, y1, w, h), pen, brush)
            else:
                sx = int(self._drag_start.x())
                sy = int(self._drag_start.y())
                self._controller._handle_selection(sx, sy, 0, 0)

            self._drag_start = None
        event.accept()
