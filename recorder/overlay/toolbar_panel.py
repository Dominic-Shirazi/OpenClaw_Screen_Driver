"""Floating draggable toolbar panel for the overlay HUD.

QGraphicsObject rendered as a horizontal pill with context-sensitive
button sets.  The toolbar switches between RECORDING, TAG_OPEN, and
DRY_RUN modes with fade transitions between button sets.
"""
from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QGraphicsObject,
    QGraphicsProxyWidget,
    QPushButton,
    QStyleOptionGraphicsItem,
    QWidget,
)

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.card_glow import paint_card_glow
from recorder.overlay.hud_common import (
    FONT_SIZE_LABEL,
    FONT_WEIGHT_REGULAR,
    FROST_BG,
    SPACING,
    TEXT_PRIMARY,
    TOOLBAR_CORNER_RADIUS,
    Z_TOOLBAR,
)

logger = logging.getLogger(__name__)


class ToolbarMode(Enum):
    """Context modes for the floating toolbar."""

    RECORDING = auto()   # [Pause] [Undo Last]
    TAG_OPEN = auto()    # [Confirm] [Dismiss] [Skip]
    DRY_RUN = auto()     # [Run Step] [Skip Step] [Finish]
    VALIDATING = auto()  # [Yes] [Edit Tags] [Re-capture] [Retry]
    BBOX_EDITING = auto()  # [Keep My Drag]
    UPDATE = auto()          # [OK] [Edit Step] [Fork Here] [Delete Step]
    UPDATE_NOT_FOUND = auto()  # [Skip] [Edit] [Delete] [Abort Update]


# Button definitions per mode: list of (label, signal_name)
_MODE_BUTTONS: dict[ToolbarMode, list[tuple[str, str]]] = {
    ToolbarMode.RECORDING: [
        ("Pause", "pause"),
        ("Undo Last", "undo_last"),
        ("Look Here", "look_here"),
        ("Add Wait", "add_wait"),
        ("Add Loop", "add_loop"),
        ("Add Prompt", "add_prompt"),
    ],
    ToolbarMode.TAG_OPEN: [
        ("Confirm", "confirm"),
        ("Redraw Box", "redraw"),
        ("Dismiss", "dismiss"),
        ("Skip", "skip"),
    ],
    ToolbarMode.DRY_RUN: [
        ("Run Step", "run_step"),
        ("Skip Step", "skip_step"),
        ("Finish", "finish"),
    ],
    ToolbarMode.VALIDATING: [
        ("Yes", "yes"),
        ("Edit Tags", "edit_tags"),
        ("Re-capture", "recapture"),
        ("Retry", "retry"),
    ],
    ToolbarMode.BBOX_EDITING: [
        ("Accept AI Box", "accept_ai_bbox"),
        ("Keep My Drag", "keep_drag"),
    ],
    ToolbarMode.UPDATE: [
        ("OK", "ok"),
        ("Edit Step", "edit"),
        ("Fork Here", "fork_here"),
        ("Delete Step", "delete_step"),
    ],
    ToolbarMode.UPDATE_NOT_FOUND: [
        ("Skip", "skip"),
        ("Edit", "edit"),
        ("Delete", "delete_step"),
        ("Abort Update", "abort"),
    ],
}

_BUTTON_STYLE_GREEN: str = (
    "QPushButton { "
    "background: rgba(50, 200, 50, 180); "
    "color: rgba(240, 240, 245, 230); "
    f"font-size: {FONT_SIZE_LABEL}px; "
    f"font-weight: {FONT_WEIGHT_REGULAR}; "
    "border: none; "
    "border-radius: 4px; "
    "padding: 4px 14px; "
    "} "
    "QPushButton:hover { "
    "background: rgba(50, 200, 50, 220); "
    "}"
)

_BUTTON_STYLE: str = (
    "QPushButton { "
    "background: transparent; "
    f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
    f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
    f"font-size: {FONT_SIZE_LABEL}px; "
    f"font-weight: {FONT_WEIGHT_REGULAR}; "
    "border: none; "
    "padding: 4px 14px; "
    "} "
    "QPushButton:hover { "
    "color: rgba(50, 200, 50, 230); "
    "}"
)


class ToolbarPanel(QGraphicsObject):
    """Floating draggable toolbar with context-sensitive buttons.

    Renders as a horizontal pill shape with card border glow.
    Buttons switch between recording, tag-open, and dry-run sets
    with fade transitions.

    Signals:
        button_clicked: Emitted with the button action name string.

    Args:
        clock: AnimationClock for tick-driven animations.
        screen_w: Screen width for positioning and drag clamping.
        screen_h: Screen height for drag clamping.
        parent: Optional parent QGraphicsObject.
    """

    button_clicked = pyqtSignal(str)

    def __init__(
        self,
        clock: AnimationClock,
        screen_w: int = 1920,
        screen_h: int = 1080,
        parent: QGraphicsObject | None = None,
    ) -> None:
        super().__init__(parent)

        self._button_proxies: dict[ToolbarMode, list[QGraphicsProxyWidget]] = {}
        self._width: float = self._compute_width_for_mode(ToolbarMode.RECORDING)
        self._height: float = 40.0
        self._corner_radius: float = TOOLBAR_CORNER_RADIUS

        self._opacity: float = 0.0
        self._target_opacity: float = 0.0
        self._glow_phase: float = 0.0

        self._mode: ToolbarMode = ToolbarMode.RECORDING
        self._screen_w: int = screen_w
        self._screen_h: int = screen_h

        self._clock = clock
        self._clock.register(self._tick)

        # Z-value and flags
        self.setZValue(Z_TOOLBAR)
        # NOTE: ItemIsMovable intentionally NOT set — it steals mouse events
        # from child proxy widgets (buttons). Drag is handled manually below.
        self.setFlag(
            QGraphicsObject.GraphicsItemFlag.ItemSendsGeometryChanges, True,
        )
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self._drag_offset: QPointF | None = None

        # Default position: top-right corner
        default_x = screen_w - self._width - SPACING.lg
        default_y = float(SPACING.lg)
        self.setPos(default_x, default_y)

        # Create button sets for all modes
        self._create_buttons()

        # Show only current mode's buttons
        self._update_button_visibility()

        logger.debug(
            "ToolbarPanel created (z=%d, pos=%.0f,%.0f)",
            Z_TOOLBAR,
            default_x,
            default_y,
        )

    # ------------------------------------------------------------------
    # Button creation
    # ------------------------------------------------------------------

    def _create_buttons(self) -> None:
        """Create all button proxy widgets for every mode."""
        for mode, button_defs in _MODE_BUTTONS.items():
            proxies: list[QGraphicsProxyWidget] = []
            for btn_idx, (label, action_name) in enumerate(button_defs):
                btn = QPushButton(label)
                # Use green style for first button in VALIDATING mode ("Yes")
                if mode == ToolbarMode.VALIDATING and btn_idx == 0:
                    btn.setStyleSheet(_BUTTON_STYLE_GREEN)
                else:
                    btn.setStyleSheet(_BUTTON_STYLE)
                def _on_btn_click(_checked: bool, name: str = action_name) -> None:
                    logger.info("Button clicked: %s", name)
                    self.button_clicked.emit(name)

                btn.clicked.connect(_on_btn_click)
                proxy = QGraphicsProxyWidget(self)
                proxy.setWidget(btn)
                proxy.setFlag(
                    QGraphicsProxyWidget.GraphicsItemFlag.ItemIsPanel, True,
                )
                proxy.setZValue(1)  # above parent's paint
                proxies.append(proxy)
            self._button_proxies[mode] = proxies

        self._width = self._compute_width_for_mode(self._mode)
        self._relayout_buttons()

    def _relayout_buttons(self) -> None:
        """Center the current mode's buttons horizontally within the pill."""
        self._width = self._compute_width_for_mode(self._mode)
        for mode, proxies in self._button_proxies.items():
            if not proxies:
                continue

            # Calculate total width of buttons + gaps
            btn_widths: list[float] = []
            for proxy in proxies:
                widget = proxy.widget()
                if widget is not None:
                    widget.adjustSize()
                    btn_widths.append(float(widget.sizeHint().width()))
                else:
                    btn_widths.append(60.0)

            total_w = sum(btn_widths) + SPACING.sm * (len(proxies) - 1)
            start_x = (self._width - total_w) / 2.0
            btn_y = (self._height - 28.0) / 2.0  # center vertically

            x = start_x
            for proxy, w in zip(proxies, btn_widths):
                proxy.setPos(x, btn_y)
                x += w + SPACING.sm

    def _update_button_visibility(self) -> None:
        """Show only the current mode's buttons, hide all others."""
        for mode, proxies in self._button_proxies.items():
            visible = mode == self._mode
            for proxy in proxies:
                proxy.setVisible(visible)

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _compute_width_for_mode(self, mode: ToolbarMode) -> float:
        """Compute pill width based on actual button sizes.

        Measures real sizeHint widths when proxies exist, otherwise
        falls back to a per-character estimate.

        Args:
            mode: The toolbar mode to measure.

        Returns:
            Dynamic width in pixels, minimum 250.0.
        """
        proxies = self._button_proxies.get(mode)
        if proxies:
            total = 0.0
            for proxy in proxies:
                widget = proxy.widget()
                if widget is not None:
                    widget.adjustSize()
                    total += float(widget.sizeHint().width())
                else:
                    total += 70.0
            total += SPACING.sm * max(0, len(proxies) - 1)
            return max(250.0, total + SPACING.lg * 2)

        # Fallback before buttons are created: estimate from label text
        btn_defs = _MODE_BUTTONS.get(mode, [])
        total_chars = sum(len(label) for label, _ in btn_defs)
        return max(250.0, total_chars * 9.0 + 28.0 * len(btn_defs) + SPACING.lg * 2)

    def set_mode(self, mode: ToolbarMode) -> None:
        """Switch toolbar to a new context mode with button swap.

        Args:
            mode: The toolbar mode to display.
        """
        if mode == self._mode:
            return

        self._mode = mode
        self._width = self._compute_width_for_mode(mode)
        self._update_button_visibility()
        self._relayout_buttons()

        logger.debug("ToolbarPanel mode -> %s", mode.name)

    # ------------------------------------------------------------------
    # Show / hide
    # ------------------------------------------------------------------

    def show_toolbar(self) -> None:
        """Fade the toolbar in."""
        self._target_opacity = 1.0

    def hide_toolbar(self) -> None:
        """Fade the toolbar out."""
        self._target_opacity = 0.0

    # ------------------------------------------------------------------
    # Avoidance rect
    # ------------------------------------------------------------------

    def get_avoidance_rect(self) -> QRectF:
        """Return the scene bounding rect for shimmer avoidance.

        Returns:
            QRectF in scene coordinates covering this toolbar.
        """
        pos = self.pos()
        return QRectF(pos.x(), pos.y(), self._width, self._height)

    # ------------------------------------------------------------------
    # Manual drag (only on pill background, not on buttons)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: Any) -> None:
        """Start drag only if clicking on pill background, not a button."""
        # Check if click is on a child proxy widget
        child = self.scene().itemAt(
            self.mapToScene(event.pos()),
            self.scene().views()[0].viewportTransform(),
        ) if self.scene() and self.scene().views() else None

        if child is not None and child is not self:
            # Click is on a button — let the proxy handle it
            event.ignore()
            return

        # Click is on pill background — start drag
        self._drag_offset = event.pos()
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        """Drag the toolbar if active."""
        if self._drag_offset is not None:
            new_pos = self.mapToScene(event.pos()) - self._drag_offset
            # Clamp to screen
            x = max(0.0, min(new_pos.x(), self._screen_w - self._width))
            y = max(0.0, min(new_pos.y(), self._screen_h - self._height))
            self.setPos(x, y)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: Any) -> None:
        """End drag."""
        if self._drag_offset is not None:
            self._drag_offset = None
            event.accept()
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        """Return bounding rect with padding for card glow overflow.

        Returns:
            QRectF with 20px padding on all sides.
        """
        return QRectF(-60, -60, self._width + 120, self._height + 120)

    def itemChange(
        self,
        change: QGraphicsObject.GraphicsItemChange,
        value: object,
    ) -> object:
        """Clamp position to screen bounds on drag.

        Args:
            change: The type of change.
            value: The new value.

        Returns:
            Clamped position or original value.
        """
        if change == QGraphicsObject.GraphicsItemChange.ItemPositionChange:
            from PyQt6.QtCore import QPointF

            if isinstance(value, QPointF):
                x = max(0.0, min(value.x(), self._screen_w - self._width))
                y = max(0.0, min(value.y(), self._screen_h - self._height))
                return QPointF(x, y)
        return super().itemChange(change, value)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint pill background and card border glow.

        Args:
            painter: Active QPainter.
            option: Style option (unused).
            widget: Target widget (unused).
        """
        if self._opacity < 0.01:
            return

        painter.save()
        painter.setOpacity(self._opacity)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(0, 0, self._width, self._height)

        # Card border glow — clip to OUTSIDE the pill so nothing bleeds through
        pill_path = QPainterPath()
        pill_path.addRoundedRect(rect, self._corner_radius, self._corner_radius)

        outer = QPainterPath()
        margin = 60.0
        outer.addRect(rect.adjusted(-margin, -margin, margin, margin))
        glow_clip = outer - pill_path

        painter.save()
        painter.setClipPath(glow_clip)
        paint_card_glow(
            painter,
            rect,
            brightness=1.0,
            phase=self._glow_phase,
            light_count=10,
            glow_radius=35.0,
        )
        painter.restore()
        painter.setOpacity(self._opacity)

        # Pill background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(FROST_BG)
        painter.drawPath(pill_path)

        painter.restore()

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def _tick(self, dt: float) -> None:
        """Advance animations by delta-time.

        Args:
            dt: Elapsed seconds since last tick.
        """
        # Continuous time for organic flicker
        self._glow_phase += dt * 0.8

        # Opacity interpolation
        diff = self._target_opacity - self._opacity
        if abs(diff) > 0.001:
            speed = 6.0  # ~150ms transition
            self._opacity += diff * min(1.0, dt * speed)
        else:
            self._opacity = self._target_opacity

        self.update()
