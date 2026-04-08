"""PyQt6 overlay window for recording sessions.

Transparent fullscreen window with interactive bounding box overlays.
Handles mouse events for click/drag selection and corner handle resizing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from recorder.overlay_items import (
    _BORDER_WIDTH,
    _DEFAULT_COLOR,
    _TYPE_COLORS,
    _ElementBoxGroup,
    _HandleItem,
)

# OverlayMode is a simple enum — safe to import directly (no circular dep)
from recorder.overlay import OverlayMode

if TYPE_CHECKING:
    from recorder.overlay import OverlayController

logger = logging.getLogger(__name__)

# Win32 constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000


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
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)
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

        logger.debug(
            "render_candidates: %d boxes created, scene has %d items, "
            "sceneRect=(%.0f,%.0f %.0fx%.0f), viewport=%dx%d",
            len(self._element_boxes), len(self.scene().items()),
            self.sceneRect().x(), self.sceneRect().y(),
            self.sceneRect().width(), self.sceneRect().height(),
            self.viewport().width(), self.viewport().height(),
        )
        if self._element_boxes:
            first = self._element_boxes[0]
            rx, ry, rw, rh = first.get_rect()
            logger.debug(
                "render_candidates: first box rect=(%d,%d %dx%d)",
                rx, ry, rw, rh,
            )
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

    def remove_candidate(self, index: int) -> None:
        """Removes a candidate's visual elements from the scene."""
        if 0 <= index < len(self._element_boxes):
            self._element_boxes[index].remove_from_scene()
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
