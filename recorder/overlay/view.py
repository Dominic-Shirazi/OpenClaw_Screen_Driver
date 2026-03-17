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

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.bbox_layer import BboxLayer
from recorder.overlay.click_catcher_layer import ClickCatcherLayer
from recorder.overlay.donut_cloud_layer import DonutCloudLayer
from recorder.overlay.mode_indicator_layer import ModeIndicatorLayer
from recorder.overlay.scan_layer import ScanLayer
from recorder.overlay.shimmer_layer import ShimmerLayer
from recorder.overlay.state import STATE_COLORS, OverlayState
from recorder.overlay.tag_dialog_panel import TagDialogPanel
from recorder.overlay.toolbar_panel import ToolbarMode, ToolbarPanel

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

        # ---- Animation clock ----
        self._clock = AnimationClock()

        # ---- Layers ----
        self._shimmer = ShimmerLayer(self._screen_w, self._screen_h)
        scene.addItem(self._shimmer)
        self._clock.register(self._shimmer.tick)

        self._mode_indicator = ModeIndicatorLayer()
        scene.addItem(self._mode_indicator)

        self._click_catcher: ClickCatcherLayer | None = None
        self._bbox_layers: list[BboxLayer] = []

        # ---- Animation layer tracking ----
        self._scan_layer: ScanLayer | None = None
        self._donut_cloud: DonutCloudLayer | None = None
        self._active_bbox: BboxLayer | None = None

        # ---- HUD panels ----
        self._tag_dialog: TagDialogPanel | None = None
        self._toolbar: ToolbarPanel | None = None

        # ---- Mouse tracking ----
        self.setMouseTracking(True)

        # ---- Start animation clock ----
        self._clock.start()

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
        # Shimmer state
        self._shimmer.set_state(state)

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
        # Force-hide HUD panels (don't rely on animation)
        if self._tag_dialog is not None:
            self._tag_dialog.setVisible(False)
        if self._toolbar is not None:
            self._toolbar.setVisible(False)
        self._clock.stop()
        self.hide()

    def show_after_capture(self) -> None:
        """Restore the overlay window after a screenshot capture."""
        self.show()
        # Restore HUD panels visibility
        if self._tag_dialog is not None and self._tag_dialog._opacity > 0:
            self._tag_dialog.setVisible(True)
        if self._toolbar is not None and self._toolbar._opacity > 0:
            self._toolbar.setVisible(True)
        self._clock.start()

    # ------------------------------------------------------------------
    # HUD panel management
    # ------------------------------------------------------------------

    def show_tag_dialog(
        self,
        element_rect: QRectF,
        vlm_data: dict | None = None,
        edit_mode: bool = False,
    ) -> None:
        """Show the tag dialog panel near the captured element.

        Args:
            element_rect: Bounding rect of the captured element in scene coords.
            vlm_data: Optional VLM analysis results for auto-fill.
            edit_mode: If True, pre-fill fields without typewriter animation.
        """
        if self._tag_dialog is None:
            self._tag_dialog = TagDialogPanel(self._clock)
            self.scene().addItem(self._tag_dialog)
        self._tag_dialog.show_dialog(
            element_rect, vlm_data=vlm_data, edit_mode=edit_mode,
        )
        self._update_avoidance_rects()

    def dismiss_tag_dialog(self) -> None:
        """Dismiss the tag dialog with fade-out animation."""
        if self._tag_dialog is not None:
            self._tag_dialog.dismiss()
            self._update_avoidance_rects()

    def get_tag_data(self) -> dict | None:
        """Return current tag dialog form data, or None if not showing.

        Returns:
            Dict of form field values, or None.
        """
        if self._tag_dialog is not None:
            return self._tag_dialog.get_form_data()
        return None

    def show_toolbar(self) -> None:
        """Show the floating toolbar."""
        if self._toolbar is None:
            self._toolbar = ToolbarPanel(
                self._clock, self._screen_w, self._screen_h,
            )
            self.scene().addItem(self._toolbar)
        self._toolbar.show_toolbar()
        self._update_avoidance_rects()

    def hide_toolbar(self) -> None:
        """Hide the floating toolbar."""
        if self._toolbar is not None:
            self._toolbar.hide_toolbar()
            self._update_avoidance_rects()

    def set_toolbar_mode(self, mode: ToolbarMode) -> None:
        """Switch toolbar button set for the current context.

        Args:
            mode: The toolbar mode to display.
        """
        if self._toolbar is not None:
            self._toolbar.set_mode(mode)

    def _update_avoidance_rects(self) -> None:
        """Collect avoidance rects from all HUD panels and update shimmer."""
        rects: list[QRectF] = []
        if self._tag_dialog is not None and self._tag_dialog.isVisible():
            rects.append(self._tag_dialog.get_avoidance_rect())
        if self._toolbar is not None and self._toolbar.isVisible():
            rects.append(self._toolbar.get_avoidance_rect())
        self._shimmer.set_avoidance_rects(rects)

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
        """Update the rubber-band rectangle during drag and feed mouse position to shimmer."""
        pos = self.mapToScene(event.pos())
        self._shimmer.set_mouse_pos(pos.x(), pos.y())

        if (
            self._drag_start is not None
            and self._rubber_band is not None
        ):
            x1 = min(self._drag_start.x(), pos.x())
            y1 = min(self._drag_start.y(), pos.y())
            x2 = max(self._drag_start.x(), pos.x())
            y2 = max(self._drag_start.y(), pos.y())
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
    # Scan layer lifecycle
    # ------------------------------------------------------------------

    def start_scan(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        bbox: BboxLayer | None = None,
    ) -> None:
        """Create and start a scan animation at the given coordinates.

        Args:
            x: Left edge of rough snip boundary.
            y: Top edge of rough snip boundary.
            w: Width of rough snip boundary.
            h: Height of rough snip boundary.
            bbox: Optional BboxLayer whose corners will morph when
                AI result arrives.
        """
        self._scan_layer = ScanLayer(x, y, w, h)
        self.scene().addItem(self._scan_layer)
        self._clock.register(self._scan_layer.tick)
        self._scan_layer.start_scan()
        self._active_bbox = bbox

    def finish_scan(
        self,
        fitted_x: int,
        fitted_y: int,
        fitted_w: int,
        fitted_h: int,
    ) -> None:
        """Deliver AI-fitted bbox to the scan layer and trigger bbox morph.

        Args:
            fitted_x: Left edge of AI-fitted bbox.
            fitted_y: Top edge of AI-fitted bbox.
            fitted_w: Width of AI-fitted bbox.
            fitted_h: Height of AI-fitted bbox.
        """
        if self._scan_layer is not None:
            self._scan_layer.receive_fitted_bbox(
                fitted_x, fitted_y, fitted_w, fitted_h,
            )
        if self._active_bbox is not None:
            self._active_bbox.morph_to(
                fitted_x, fitted_y, fitted_w, fitted_h,
            )

    def remove_scan(self) -> None:
        """Remove the scan layer from the scene and unregister its tick."""
        if self._scan_layer is not None:
            self._clock.unregister(self._scan_layer.tick)
            self.scene().removeItem(self._scan_layer)
            self._scan_layer = None
        self._active_bbox = None

    # ------------------------------------------------------------------
    # Donut cloud lifecycle
    # ------------------------------------------------------------------

    def show_donut_cloud(
        self,
        center_x: float,
        center_y: float,
        radius: float = 60.0,
    ) -> None:
        """Create and display a donut cloud probability visualizer.

        Args:
            center_x: Cloud center X coordinate.
            center_y: Cloud center Y coordinate.
            radius: Cloud radius (used for both rx and ry).
        """
        self._donut_cloud = DonutCloudLayer(center_x, center_y, radius, radius)
        self.scene().addItem(self._donut_cloud)
        self._clock.register(self._donut_cloud.tick)

    def accept_donut_cloud(self) -> None:
        """Transition the donut cloud color from red to green (accepted)."""
        if self._donut_cloud is not None:
            self._donut_cloud.accept()

    def remove_donut_cloud(self) -> None:
        """Remove the donut cloud from the scene and unregister its tick."""
        if self._donut_cloud is not None:
            self._clock.unregister(self._donut_cloud.tick)
            self.scene().removeItem(self._donut_cloud)
            self._donut_cloud = None

    # ------------------------------------------------------------------
    # Keyboard fallback
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: Any) -> None:
        """Accept key events as fallback (controller handles via hotkeys)."""
        event.accept()
