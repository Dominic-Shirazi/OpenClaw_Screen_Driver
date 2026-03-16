"""QGraphicsView-based fullscreen overlay shell.

Thin container that manages a ``QGraphicsScene`` and delegates visual
concerns to independent layer items (border, click catcher, mode
indicator, bounding boxes).  Platform-specific window flags are applied
at creation time.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from recorder.overlay.bbox_layer import BboxLayer
from recorder.overlay.border_layer import BorderLayer
from recorder.overlay.click_catcher_layer import ClickCatcherLayer
from recorder.overlay.mode_indicator_layer import ModeIndicatorLayer
from recorder.overlay.state import STATE_COLORS, OverlayState

logger = logging.getLogger(__name__)


class OverlayView(QGraphicsView):
    """Transparent fullscreen overlay window.

    Creates a frameless, always-on-top, translucent ``QGraphicsView``
    that covers the primary monitor.  All visual elements are
    independent ``QGraphicsItem`` layers managed through the scene.

    Args:
        on_selection: Optional callback invoked with ``(x, y, w, h)``
            when the user completes a drag-to-draw bounding box in
            RECORDING mode.
    """

    def __init__(
        self,
        *,
        on_selection: Callable[[int, int, int, int], None] | None = None,
    ) -> None:
        scene = QGraphicsScene()
        super().__init__(scene)

        # ---- Window configuration ----
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setStyleSheet("background: transparent;")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # ---- Screen geometry ----
        self._screen_w: int = 1920
        self._screen_h: int = 1080
        primary = QApplication.primaryScreen()
        if primary is not None:
            geom = primary.geometry()
            self._screen_w = geom.width()
            self._screen_h = geom.height()
        self.setSceneRect(0, 0, self._screen_w, self._screen_h)

        # ---- Layers ----
        self._border = BorderLayer(self._screen_w, self._screen_h)
        scene.addItem(self._border)

        self._mode_indicator = ModeIndicatorLayer()
        scene.addItem(self._mode_indicator)

        self._click_catcher: ClickCatcherLayer | None = None
        self._bbox_layers: list[BboxLayer] = []

        # ---- Drag-to-draw state ----
        self._on_selection = on_selection
        self._drag_start: QPointF | None = None
        self._rubber_band: QGraphicsRectItem | None = None
        self._min_drag_px: int = 8

        # ---- Win32 layered flags (deferred until window handle exists) ----
        if sys.platform == "win32":
            from recorder.overlay.platform_win32 import setup_win32_layered

            QTimer.singleShot(
                0,
                lambda: setup_win32_layered(int(self.winId())),
            )

    # ------------------------------------------------------------------
    # State-driven updates
    # ------------------------------------------------------------------

    def apply_state(self, state: OverlayState) -> None:
        """Update all layers for a new overlay state.

        Args:
            state: The overlay state to apply.
        """
        # Border colour
        r, g, b, a = STATE_COLORS[state]
        self._border.set_color(r, g, b, a)

        # Mode indicator text
        self._mode_indicator.update_mode(state)

        # Click catcher: add for RECORDING, remove otherwise
        if state == OverlayState.RECORDING:
            if self._click_catcher is None:
                self._click_catcher = ClickCatcherLayer(
                    self._screen_w, self._screen_h
                )
                self.scene().addItem(self._click_catcher)
        else:
            if self._click_catcher is not None:
                self.scene().removeItem(self._click_catcher)
                self._click_catcher = None

        # Click-through toggle
        passthrough = state != OverlayState.RECORDING
        if sys.platform == "win32":
            from recorder.overlay.platform_win32 import (
                set_click_through_win32,
            )

            try:
                set_click_through_win32(int(self.winId()), passthrough)
            except RuntimeError:
                logger.debug("Window not yet realised; skipping click-through")
        else:
            from recorder.overlay.platform_linux import (
                set_click_through_linux,
            )

            set_click_through_linux(self, passthrough)

        # Cursor
        if state == OverlayState.RECORDING:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # Capture lifecycle
    # ------------------------------------------------------------------

    def hide_for_capture(self) -> None:
        """Hide the overlay window before a screenshot capture."""
        self.hide()

    def show_after_capture(self) -> None:
        """Restore the overlay window after a screenshot capture."""
        self.show()

    # ------------------------------------------------------------------
    # Bounding box management
    # ------------------------------------------------------------------

    def render_bboxes(self, bboxes: list[dict[str, Any]]) -> None:
        """Replace all bounding box layers with the supplied list.

        Args:
            bboxes: List of dicts, each with keys ``x``, ``y``, ``w``,
                ``h``, ``color`` (RGBA tuple), and optionally ``label``
                and ``confidence``.
        """
        self.clear_bboxes()
        for bbox in bboxes:
            layer = BboxLayer(
                x=bbox.get("x", 0),
                y=bbox.get("y", 0),
                w=bbox.get("w", 0),
                h=bbox.get("h", 0),
                color_rgba=bbox.get("color", (100, 100, 100, 100)),
                label=bbox.get("label", ""),
                confidence=bbox.get("confidence", 0.0),
            )
            self.scene().addItem(layer)
            self._bbox_layers.append(layer)

    def clear_bboxes(self) -> None:
        """Remove all bounding box layers from the scene."""
        for layer in self._bbox_layers:
            self.scene().removeItem(layer)
        self._bbox_layers.clear()

    # ------------------------------------------------------------------
    # Mouse event handlers (drag-to-draw)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: Any) -> None:
        """Start a rubber-band drag in RECORDING mode."""
        # Only react to left button in recording mode
        if self._click_catcher is not None:
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
        """Update the rubber-band rectangle during drag."""
        if (
            self._drag_start is not None
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
        """Complete the rubber-band selection and fire the callback."""
        if self._drag_start is not None:
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

            if (
                w >= self._min_drag_px
                and h >= self._min_drag_px
                and self._on_selection is not None
            ):
                self._on_selection(int(x1), int(y1), int(w), int(h))

            self._drag_start = None
        event.accept()

    # ------------------------------------------------------------------
    # Keyboard fallback
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: Any) -> None:
        """Accept key events as fallback (controller handles via hotkeys)."""
        event.accept()
