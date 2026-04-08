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

from recorder.overlay.abort_panel import AbortPanel
from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.bbox_layer import BboxLayer
from recorder.overlay.click_catcher_layer import ClickCatcherLayer
from recorder.overlay.countdown_widget import CountdownWidget
from recorder.overlay.donut_cloud_layer import DonutCloudLayer
from recorder.overlay.mode_indicator_layer import ModeIndicatorLayer
from recorder.overlay.scan_layer import ScanLayer
from recorder.overlay.shimmer_layer import ShimmerLayer
from recorder.overlay.state import STATE_COLORS, OverlayState
from recorder.overlay.tag_dialog_panel import TagDialogPanel
from recorder.overlay.camera_flash import CameraFlash
from recorder.overlay.status_badge import StatusBadge
from recorder.overlay.target_highlight import TargetHighlight
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
        self._countdown: CountdownWidget | None = None
        self._abort_panel: AbortPanel | None = None
        self._flash_timer: QTimer | None = None
        self._flash_bbox_rect: QGraphicsRectItem | None = None
        self._card_glow_pulsing: bool = False
        self._mini_dialog: Any = None  # WaitDialog or PromptDialog
        self._status_badge: StatusBadge | None = None
        self._target_highlight: TargetHighlight | None = None
        self._camera_flash: CameraFlash | None = None

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
        if self._countdown is not None:
            self._countdown.setVisible(False)
        if self._abort_panel is not None:
            self._abort_panel.setVisible(False)
        if self._status_badge is not None:
            self._status_badge.setVisible(False)
        if self._target_highlight is not None:
            self._target_highlight.setVisible(False)
        if self._camera_flash is not None:
            self._camera_flash.setVisible(False)
        self.clear_rubber_band()
        # Ensure non-activating flag is set before hiding
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._clock.stop()
        self.hide()

    def show_after_capture(self) -> None:
        """Restore the overlay window after a screenshot capture."""
        # Always clear non-activating flag when restoring — overlay must be
        # activatable whenever visible.  The flag is only needed during the
        # hidden capture period (set in hide_for_capture).
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self.show()
        # Restore HUD panels visibility (check _target_opacity to avoid
        # re-showing panels that were mid-fade-out when capture started)
        if self._tag_dialog is not None and self._tag_dialog._target_opacity > 0:
            self._tag_dialog.setVisible(True)
            # Re-activate window for keyboard input if tag dialog is visible
            self.activateWindow()
            self.raise_()
        if self._toolbar is not None and self._toolbar._target_opacity > 0:
            self._toolbar.setVisible(True)
        if self._countdown is not None and self._countdown._remaining > 0:
            self._countdown.setVisible(True)
        # abort panel stays hidden after capture (user must re-trigger)
        if self._status_badge is not None and self._status_badge._text:
            self._status_badge.setVisible(True)
        # target_highlight and camera_flash intentionally NOT restored (ephemeral)
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
        # Allow OS keyboard input by removing non-activating flag
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self.activateWindow()
        self.raise_()
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

    def show_toolbar(
        self,
        on_action: Callable[[str], None] | None = None,
    ) -> None:
        """Show the floating toolbar.

        Args:
            on_action: Optional callback for toolbar button clicks.
        """
        if self._toolbar is None:
            self._toolbar = ToolbarPanel(
                self._clock, self._screen_w, self._screen_h,
            )
            self.scene().addItem(self._toolbar)
        if on_action is not None:
            try:
                self._toolbar.button_clicked.disconnect()
            except TypeError:
                pass  # No existing connections — safe to ignore
            self._toolbar.button_clicked.connect(on_action)
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

    def _is_hud_item(self, item: Any) -> bool:
        """Check if a scene item belongs to a HUD panel (toolbar, dialog, etc.)."""
        from recorder.overlay.toolbar_panel import ToolbarPanel
        from recorder.overlay.tag_dialog_panel import TagDialogPanel
        from recorder.overlay.abort_panel import AbortPanel
        from recorder.overlay.countdown_widget import CountdownWidget
        from recorder.overlay.status_badge import StatusBadge

        # Walk up the parent chain — proxy widgets are children of panels
        current = item
        while current is not None:
            if isinstance(current, (
                ToolbarPanel, TagDialogPanel, AbortPanel,
                CountdownWidget, StatusBadge,
            )):
                return True
            # Also check for mini dialogs (WaitDialog, PromptDialog) stored dynamically
            if self._mini_dialog is not None and current is self._mini_dialog:
                return True
            current = current.parentItem()
        return False

    def mousePressEvent(self, event: Any) -> None:
        """Start a rubber-band drag in RECORDING mode.

        If the click lands on a HUD widget (toolbar, dialog, etc.),
        delegate to the default handler so buttons receive the event.
        In non-RECORDING states, ignore the event so it passes through.
        """
        if self._click_catcher is not None:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene().itemAt(scene_pos, self.viewportTransform())
            # Let HUD widgets handle their own clicks
            if item is not None and self._is_hud_item(item):
                logger.debug("HUD click detected on %s, delegating", type(item).__name__)
                super().mousePressEvent(event)
                return
            logger.debug("Non-HUD click at (%.0f,%.0f) item=%s", scene_pos.x(), scene_pos.y(), type(item).__name__ if item else "None")
            self._drag_start = scene_pos
            pen = QPen(QColor(255, 255, 0, 220))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            brush = QBrush(QColor(255, 255, 0, 30))
            self._rubber_band = self.scene().addRect(
                QRectF(self._drag_start, self._drag_start), pen, brush,
            )
            self._rubber_band.setZValue(200)
            event.accept()
            return
        # Not recording — ignore so clicks pass through  # Updated: ignore mouse press in non-RECORDING states — fixes replay click passthrough — 2026-04-03
        event.ignore()

    def mouseMoveEvent(self, event: Any) -> None:
        """Update the rubber-band rectangle during drag and feed mouse position to shimmer."""
        pos = self.mapToScene(event.pos())
        self._shimmer.set_mouse_pos(pos.x(), pos.y())

        # Update countdown widget position when visible
        if self._countdown is not None and self._countdown.isVisible():
            self._countdown.set_position(pos.x(), pos.y())

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
            return
        # Not dragging — ignore so moves pass through  # Updated: ignore mouse move when not dragging — fixes replay click passthrough — 2026-04-03
        if self._click_catcher is None:
            event.ignore()
        else:
            event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:
        """Complete the rubber-band selection and fire the callback."""
        # If no drag in progress, delegate or ignore  # Updated: ignore release in non-RECORDING states — fixes replay click passthrough — 2026-04-03
        if self._drag_start is None:
            if self._click_catcher is not None:
                super().mouseReleaseEvent(event)
            else:
                event.ignore()
            return
        if self._drag_start is not None:
            end = self.mapToScene(event.pos())
            x1 = min(self._drag_start.x(), end.x())
            y1 = min(self._drag_start.y(), end.y())
            x2 = max(self._drag_start.x(), end.x())
            y2 = max(self._drag_start.y(), end.y())
            w = x2 - x1
            h = y2 - y1

            if (
                w >= self._min_drag_px
                and h >= self._min_drag_px
                and self._on_selection is not None
            ):
                # Turn yellow box red to show it's been captured
                if self._rubber_band is not None:
                    pen = QPen(QColor(255, 50, 50, 220))
                    pen.setWidth(2)
                    pen.setStyle(Qt.PenStyle.SolidLine)
                    self._rubber_band.setPen(pen)
                    self._rubber_band.setBrush(QBrush(QColor(255, 50, 50, 20)))
                self._on_selection(int(x1), int(y1), int(w), int(h))
            else:
                # Too small — remove the rubber band
                if self._rubber_band is not None:
                    self.scene().removeItem(self._rubber_band)
                    self._rubber_band = None

            self._drag_start = None
        event.accept()

    def clear_rubber_band(self) -> None:
        """Remove the rubber band selection rectangle if present."""
        if self._rubber_band is not None:
            self.scene().removeItem(self._rubber_band)
            self._rubber_band = None

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
        # Remove stale scan layer to prevent scene item leak
        if self._scan_layer is not None:
            self._clock.unregister(self._scan_layer.tick)
            self.scene().removeItem(self._scan_layer)
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
        # Remove stale donut cloud to prevent scene item leak
        if self._donut_cloud is not None:
            self._clock.unregister(self._donut_cloud.tick)
            self.scene().removeItem(self._donut_cloud)
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
    # Countdown / Abort / Flash / Click-through / Card glow pulse
    # ------------------------------------------------------------------

    def show_countdown(self, seconds: int = 3) -> CountdownWidget:
        """Create and start a cursor-following countdown widget.

        Args:
            seconds: Number of seconds to count down.

        Returns:
            The widget so caller can connect to countdown_finished signal.
        """
        if self._countdown is None:
            self._countdown = CountdownWidget(self._clock, seconds)
            self.scene().addItem(self._countdown)
        self._countdown.start()
        return self._countdown

    def hide_countdown(self) -> None:
        """Stop and hide the countdown widget."""
        if self._countdown is not None:
            self._countdown.stop()

    def show_abort_confirm(self, step_count: int) -> AbortPanel:
        """Show the abort confirmation panel.

        Args:
            step_count: Number of recorded steps that will be lost.

        Returns:
            The panel for signal connection.
        """
        if self._abort_panel is None:
            self._abort_panel = AbortPanel(self._clock)
            self._abort_panel.set_screen_size(self._screen_w, self._screen_h)
            self.scene().addItem(self._abort_panel)
        self._abort_panel.show_panel(step_count)
        return self._abort_panel

    def hide_abort_confirm(self) -> None:
        """Hide the abort confirmation panel."""
        if self._abort_panel is not None:
            self._abort_panel.hide_panel()

    def show_mini_dialog(self, dialog_cls: type) -> Any:
        """Create and display a mini-dialog (WaitDialog or PromptDialog).

        Args:
            dialog_cls: The dialog class to instantiate.

        Returns:
            The dialog instance for signal connection.
        """
        self.hide_mini_dialog("")
        dialog = dialog_cls(self._clock, self._screen_w, self._screen_h)
        self.scene().addItem(dialog)
        dialog.setPos(
            self._screen_w / 2 - dialog._width / 2,
            self._screen_h / 2 - dialog._height / 2,
        )
        self._mini_dialog = dialog
        return dialog

    def hide_mini_dialog(self, kind: str) -> None:
        """Remove the current mini-dialog from the scene.

        Args:
            kind: Dialog kind hint (unused -- only one at a time).
        """
        if self._mini_dialog is not None:
            self.scene().removeItem(self._mini_dialog)
            self._mini_dialog = None

    def flash_success(self, bbox_rect: QRectF) -> None:
        """Show a 500ms green flash on the given bbox rect, then auto-clear.

        Uses a QGraphicsRectItem with ACCENT_GREEN at alpha 120.

        Args:
            bbox_rect: The rectangle to flash green.
        """
        from recorder.overlay.hud_common import ACCENT_GREEN

        if self._flash_bbox_rect is not None:
            self.scene().removeItem(self._flash_bbox_rect)
        pen = QPen(Qt.PenStyle.NoPen)
        brush = QBrush(
            QColor(
                ACCENT_GREEN.red(),
                ACCENT_GREEN.green(),
                ACCENT_GREEN.blue(),
                120,
            )
        )
        self._flash_bbox_rect = self.scene().addRect(bbox_rect, pen, brush)
        self._flash_bbox_rect.setZValue(55)  # just above bbox layer at 50
        # Timer to remove flash after 500ms
        self._flash_timer = QTimer()
        self._flash_timer.setSingleShot(True)
        self._flash_timer.setInterval(500)
        self._flash_timer.timeout.connect(self._clear_flash)
        self._flash_timer.start()

    def _clear_flash(self) -> None:
        """Remove the success flash rect."""
        if self._flash_bbox_rect is not None:
            self.scene().removeItem(self._flash_bbox_rect)
            self._flash_bbox_rect = None

    def set_click_through(self, enabled: bool) -> None:
        """Toggle click-through mode independently of overlay state.

        Used during dry-run execution (overlay visible but non-interactive).

        Args:
            enabled: If True, make overlay click-through. If False, restore.
        """
        if sys.platform == "win32":
            from recorder.overlay.platform_win32 import set_click_through_win32

            try:
                set_click_through_win32(int(self.winId()), enabled)
            except RuntimeError:
                logger.debug("Window not realised for click-through toggle")
        else:
            from recorder.overlay.platform_linux import set_click_through_linux

            set_click_through_linux(self, enabled)
        # Update cursor: system default when click-through, crosshair when not
        if enabled:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def start_card_glow_pulse(self) -> None:
        """Start card glow pulsing as a loading indicator.

        Per CONTEXT.md locked decision: card glow pulses as loading
        indicator during DETECTING and VLM_ANALYZING phases.
        """
        self._card_glow_pulsing = True
        if self._tag_dialog is not None:
            self._tag_dialog.set_glow_pulsing(True)

    def stop_card_glow_pulse(self) -> None:
        """Stop card glow pulsing. Called when detection/VLM completes."""
        self._card_glow_pulsing = False
        if self._tag_dialog is not None:
            self._tag_dialog.set_glow_pulsing(False)

    # ------------------------------------------------------------------
    # Replay widgets
    # ------------------------------------------------------------------

    def show_replay_badge(self) -> None:
        """Create and show the replay status badge at top-center."""
        if self._status_badge is None:
            self._status_badge = StatusBadge(
                self._clock, self._screen_w, self._screen_h,
            )
            self.scene().addItem(self._status_badge)
        self._status_badge.set_visible_animated(True)

    def hide_replay_badge(self) -> None:
        """Hide the replay status badge."""
        if self._status_badge is not None:
            self._status_badge.set_visible_animated(False)

    def set_replay_status(self, text: str) -> None:
        """Update the status badge text."""
        if self._status_badge is not None:
            self._status_badge.set_text(text)

    def show_target_highlight(self, x: int, y: int, w: int, h: int) -> None:
        """Show a brief purple highlight around a located element.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            w: Target width.
            h: Target height.
        """
        if self._target_highlight is None:
            self._target_highlight = TargetHighlight(self._clock)
            self.scene().addItem(self._target_highlight)
        self._target_highlight.highlight(x, y, w, h)

    def hide_target_highlight(self) -> None:
        """Hide the target highlight immediately."""
        if self._target_highlight is not None:
            self._target_highlight.hide()

    def camera_flash(self) -> None:
        """Trigger the camera flash effect."""
        if self._camera_flash is None:
            self._camera_flash = CameraFlash(
                self._clock, self._screen_w, self._screen_h,
            )
            self.scene().addItem(self._camera_flash)
        self._camera_flash.flash()

    def hide_camera_flash(self) -> None:
        """Hide the camera flash immediately."""
        if self._camera_flash is not None:
            self._camera_flash.hide()

    # ------------------------------------------------------------------
    # Keyboard fallback
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: Any) -> None:
        """Route key events to focused proxy widgets, or accept as fallback."""
        # If a proxy widget (text field) has focus, let it handle the key
        focus_item = self.scene().focusItem() if self.scene() else None
        if focus_item is not None:
            super().keyPressEvent(event)
            return
        event.accept()
