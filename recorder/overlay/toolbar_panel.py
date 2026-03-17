"""Floating draggable toolbar panel for the overlay HUD.

QGraphicsObject rendered as a horizontal pill with context-sensitive
button sets.  The toolbar switches between RECORDING, TAG_OPEN, and
DRY_RUN modes with fade transitions between button sets.
"""
from __future__ import annotations

import logging
from enum import Enum, auto

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
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


# Button definitions per mode: list of (label, signal_name)
_MODE_BUTTONS: dict[ToolbarMode, list[tuple[str, str]]] = {
    ToolbarMode.RECORDING: [("Pause", "pause"), ("Undo Last", "undo_last")],
    ToolbarMode.TAG_OPEN: [
        ("Confirm", "confirm"),
        ("Dismiss", "dismiss"),
        ("Skip", "skip"),
    ],
    ToolbarMode.DRY_RUN: [
        ("Run Step", "run_step"),
        ("Skip Step", "skip_step"),
        ("Finish", "finish"),
    ],
}

_BUTTON_STYLE: str = (
    "background: rgba(40, 40, 55, 200); "
    f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
    f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
    f"font-size: {FONT_SIZE_LABEL}px; "
    f"font-weight: {FONT_WEIGHT_REGULAR}; "
    "border-radius: 4px; "
    "padding: 4px 10px;"
)

_BUTTON_HOVER_STYLE: str = (
    "QPushButton { "
    "background: rgba(40, 40, 55, 200); "
    f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
    f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
    f"font-size: {FONT_SIZE_LABEL}px; "
    f"font-weight: {FONT_WEIGHT_REGULAR}; "
    "border-radius: 4px; "
    "padding: 4px 10px; "
    "} "
    "QPushButton:hover { "
    "background: rgba(60, 60, 75, 220); "
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

        self._width: float = 250.0
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
        self.setFlag(
            QGraphicsObject.GraphicsItemFlag.ItemIsMovable, True,
        )
        self.setFlag(
            QGraphicsObject.GraphicsItemFlag.ItemSendsGeometryChanges, True,
        )

        # Default position: top-right corner
        default_x = screen_w - self._width - SPACING.lg
        default_y = float(SPACING.lg)
        self.setPos(default_x, default_y)

        # Create button sets for all modes
        self._button_proxies: dict[ToolbarMode, list[QGraphicsProxyWidget]] = {}
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
            for label, action_name in button_defs:
                btn = QPushButton(label)
                btn.setStyleSheet(_BUTTON_HOVER_STYLE)
                btn.clicked.connect(
                    lambda _checked, name=action_name: self.button_clicked.emit(name),
                )
                proxy = QGraphicsProxyWidget(self)
                proxy.setWidget(btn)
                proxies.append(proxy)
            self._button_proxies[mode] = proxies

        self._relayout_buttons()

    def _relayout_buttons(self) -> None:
        """Center the current mode's buttons horizontally within the pill."""
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

    def set_mode(self, mode: ToolbarMode) -> None:
        """Switch toolbar to a new context mode with button swap.

        Args:
            mode: The toolbar mode to display.
        """
        if mode == self._mode:
            return

        self._mode = mode
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
    # Geometry
    # ------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        """Return bounding rect with padding for card glow overflow.

        Returns:
            QRectF with 20px padding on all sides.
        """
        return QRectF(-20, -20, self._width + 40, self._height + 40)

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

        # Pill background
        path = QPainterPath()
        path.addRoundedRect(rect, self._corner_radius, self._corner_radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(FROST_BG)
        painter.drawPath(path)

        # Card border glow (fewer lights for smaller widget)
        paint_card_glow(
            painter,
            rect,
            brightness=1.0,
            phase=self._glow_phase,
            light_count=12,
            glow_radius=20.0,
        )

        painter.restore()

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def _tick(self, dt: float) -> None:
        """Advance animations by delta-time.

        Args:
            dt: Elapsed seconds since last tick.
        """
        # Glow sweep (5s period)
        self._glow_phase = (self._glow_phase + dt / 5.0) % 1.0

        # Opacity interpolation
        diff = self._target_opacity - self._opacity
        if abs(diff) > 0.001:
            speed = 6.0  # ~150ms transition
            self._opacity += diff * min(1.0, dt * speed)
        else:
            self._opacity = self._target_opacity

        self.update()
