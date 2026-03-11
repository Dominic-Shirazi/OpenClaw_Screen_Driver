"""Transparent fullscreen overlay for recording sessions.

Provides a PyQt6 overlay window that sits on top of all other windows.
Two modes:
- PASSTHROUGH: Clicks go through to the app beneath (WS_EX_TRANSPARENT)
- RECORD: Captures mouse clicks for element tagging

F2 toggles between modes. ESC closes the overlay.
Bounding boxes are rendered for candidate elements, color-coded by type.

Windows-specific: Uses ctypes for WS_EX_TRANSPARENT/WS_EX_LAYERED flags.
On non-Windows platforms, the overlay will render but click-through
passthrough mode is not available.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from PyQt6.QtCore import QPoint, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)

from core.types import CandidateElement, Rect
from recorder.element_types import ElementType

logger = logging.getLogger(__name__)

# Win32 constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000

# Color map for element types (RGBA)
_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "textbox": (0, 150, 255, 140),       # Blue
    "button": (0, 200, 0, 140),          # Green
    "button_nav": (0, 255, 100, 140),    # Bright green
    "toggle": (255, 165, 0, 140),        # Orange
    "tab": (128, 0, 128, 140),           # Purple
    "dropdown": (255, 200, 0, 140),      # Gold
    "scrollbar": (128, 128, 128, 100),   # Gray
    "read_here": (255, 0, 0, 140),       # Red
    "drag_source": (0, 255, 255, 140),   # Cyan
    "drag_target": (0, 200, 200, 140),   # Dark cyan
    "image": (200, 200, 200, 80),        # Light gray
    "modal": (255, 100, 100, 100),       # Light red
    "notification": (255, 255, 0, 140),  # Yellow
    "unknown": (100, 100, 100, 100),     # Dark gray
}

_DEFAULT_COLOR = (100, 100, 100, 100)


class OverlayMode(Enum):
    """Overlay interaction modes."""
    PASSTHROUGH = auto()  # Clicks go through to underlying app
    RECORD = auto()       # Overlay captures clicks for tagging


class OverlayController:
    """Controls the transparent fullscreen overlay.

    Manages mode switching, candidate rendering, click capture, and
    integration with the recording session pipeline.
    """

    def __init__(
        self,
        on_element_clicked: Callable[[int, int, dict[str, Any] | None], None] | None = None,
        on_mode_changed: Callable[[OverlayMode], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Initializes the overlay controller.

        Args:
            on_element_clicked: Callback when user clicks in RECORD mode.
                               Args: (x, y, matched_candidate_dict_or_None).
            on_mode_changed: Callback when overlay mode changes.
            on_close: Callback when overlay is closed via ESC.
        """
        self._on_element_clicked = on_element_clicked
        self._on_mode_changed = on_mode_changed
        self._on_close = on_close

        self._mode = OverlayMode.PASSTHROUGH
        self._candidates: list[dict[str, Any]] = []
        self._view: _OverlayView | None = None
        self._is_active = False

    @property
    def mode(self) -> OverlayMode:
        """Returns the current overlay mode."""
        return self._mode

    @property
    def is_active(self) -> bool:
        """Returns whether the overlay is currently shown."""
        return self._is_active

    def show(self) -> None:
        """Shows the overlay in PASSTHROUGH mode.

        Creates the overlay window if it doesn't exist.
        """
        if self._view is not None and self._is_active:
            logger.debug("Overlay already active")
            return

        self._view = _OverlayView(controller=self)
        self._view.showFullScreen()
        self._is_active = True
        self._set_mode(OverlayMode.PASSTHROUGH)

        logger.info("Overlay shown in PASSTHROUGH mode")

    def close(self) -> None:
        """Closes and cleans up the overlay."""
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
        logger.debug("Updated %d candidates", len(candidates))

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

    def _handle_click(self, x: int, y: int) -> None:
        """Handles a mouse click in RECORD mode.

        Finds the candidate element at (x, y) and invokes the callback.

        Args:
            x: Screen X coordinate of the click.
            y: Screen Y coordinate of the click.
        """
        matched = self._find_candidate_at(x, y)

        if self._on_element_clicked:
            try:
                self._on_element_clicked(x, y, matched)
            except Exception as e:
                logger.error("on_element_clicked callback error: %s", e)

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


class _OverlayView(QGraphicsView):
    """The actual PyQt6 overlay window.

    Renders as a transparent fullscreen window with bounding box
    overlays for candidate elements.
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
        if QApplication.primaryScreen():
            screen_geom = QApplication.primaryScreen().geometry()
            self.setSceneRect(
                0, 0, screen_geom.width(), screen_geom.height()
            )

        # Set up Win32 layered flags on Windows
        if sys.platform == "win32":
            QTimer.singleShot(0, self._setup_win32_layered)

        # Mode indicator label
        self._mode_label: QGraphicsSimpleTextItem | None = None

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

        self._update_mode_indicator()

    def clear_scene(self) -> None:
        """Removes all items from the scene."""
        self.scene().clear()
        self._mode_label = None

    def _update_mode_indicator(self) -> None:
        """Shows the current mode as a small label in the corner."""
        mode = self._controller.mode
        mode_text = f"[{mode.name}] F2=toggle  ESC=close"

        if self._mode_label is not None:
            self._mode_label.setText(mode_text)
            return

        self._mode_label = self.scene().addSimpleText(mode_text)
        self._mode_label.setPos(10, 10)
        font = QFont("Segoe UI", 11)
        font.setBold(True)
        self._mode_label.setFont(font)

        if mode == OverlayMode.RECORD:
            self._mode_label.setBrush(QBrush(QColor(255, 80, 80, 200)))
        else:
            self._mode_label.setBrush(QBrush(QColor(80, 200, 80, 200)))

    def keyPressEvent(self, event: Any) -> None:
        """Handles keyboard input for mode toggle and close.

        F2 toggles between PASSTHROUGH and RECORD.
        ESC closes the overlay.
        """
        if event.key() == Qt.Key.Key_F2:
            self._controller.toggle_mode()
            self._update_mode_indicator()
            # Re-render candidates so the mode label refreshes
            self.render_candidates(self._controller._candidates)
        elif event.key() == Qt.Key.Key_Escape:
            self._controller.close()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        """Captures mouse clicks in RECORD mode.

        In RECORD mode, maps the click position to screen coordinates
        and forwards to the controller.
        """
        if self._controller.mode == OverlayMode.RECORD:
            # Map widget coordinates to scene (screen) coordinates
            scene_pos = self.mapToScene(event.pos())
            self._controller._handle_click(int(scene_pos.x()), int(scene_pos.y()))
        else:
            super().mousePressEvent(event)
