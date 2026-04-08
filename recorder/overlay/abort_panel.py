"""Abort confirmation panel for the recording overlay.

Centered frosted-glass panel with Discard and Keep Recording buttons.
Shown when the user attempts to abort an active recording session
with unsaved steps.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QGraphicsObject,
    QGraphicsProxyWidget,
    QLabel,
    QPushButton,
    QStyleOptionGraphicsItem,
    QWidget,
)

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.card_glow import paint_card_glow
from recorder.overlay.hud_common import (
    CORNER_RADIUS,
    DISMISS_BG,
    DISMISS_TEXT,
    FONT_FAMILY,
    FONT_SIZE_INPUT,
    FONT_SIZE_LABEL,
    FONT_WEIGHT_LIGHT,
    FONT_WEIGHT_REGULAR,
    FROST_BG,
    SPACING,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    Z_ABORT_PANEL,
)

logger = logging.getLogger(__name__)

_PANEL_WIDTH: float = 320.0
_DISCARD_BG = QColor(255, 50, 50, 180)
_DISCARD_TEXT = QColor(240, 240, 245, 230)

_BTN_STYLE_DISCARD: str = (
    "QPushButton { "
    f"background: rgba({_DISCARD_BG.red()}, {_DISCARD_BG.green()}, "
    f"{_DISCARD_BG.blue()}, {_DISCARD_BG.alpha()}); "
    f"color: rgba({_DISCARD_TEXT.red()}, {_DISCARD_TEXT.green()}, "
    f"{_DISCARD_TEXT.blue()}, {_DISCARD_TEXT.alpha()}); "
    f"font-size: {FONT_SIZE_LABEL}px; "
    f"font-weight: {FONT_WEIGHT_REGULAR}; "
    "border: none; border-radius: 4px; padding: 6px 14px; "
    "} "
    "QPushButton:hover { "
    f"background: rgba({_DISCARD_BG.red()}, {_DISCARD_BG.green()}, "
    f"{_DISCARD_BG.blue()}, 220); "
    "}"
)

_BTN_STYLE_KEEP: str = (
    "QPushButton { "
    f"background: rgba({DISMISS_BG.red()}, {DISMISS_BG.green()}, "
    f"{DISMISS_BG.blue()}, {DISMISS_BG.alpha()}); "
    f"color: rgba({DISMISS_TEXT.red()}, {DISMISS_TEXT.green()}, "
    f"{DISMISS_TEXT.blue()}, {DISMISS_TEXT.alpha()}); "
    f"font-size: {FONT_SIZE_LABEL}px; "
    f"font-weight: {FONT_WEIGHT_REGULAR}; "
    "border: none; border-radius: 4px; padding: 6px 14px; "
    "} "
    "QPushButton:hover { "
    f"background: rgba({DISMISS_BG.red()}, {DISMISS_BG.green()}, "
    f"{DISMISS_BG.blue()}, 200); "
    "}"
)


class AbortPanel(QGraphicsObject):
    """Centered abort confirmation panel with frosted-glass style.

    Signals:
        discard_clicked: Emitted when user clicks Discard.
        keep_clicked: Emitted when user clicks Keep Recording.

    Args:
        parent: Optional parent QGraphicsObject.
    """

    discard_clicked = pyqtSignal()
    keep_clicked = pyqtSignal()

    def __init__(
        self,
        clock: AnimationClock,
        parent: QGraphicsObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.setZValue(Z_ABORT_PANEL)
        self.setVisible(False)

        self._clock = clock
        self._width: float = _PANEL_WIDTH
        self._height: float = 140.0
        self._corner_radius: float = CORNER_RADIUS
        self._screen_w: int = 1920
        self._screen_h: int = 1080
        self._step_count: int = 0
        self._glow_phase: float = 0.0

        # Build UI
        self._heading_proxy: QGraphicsProxyWidget | None = None
        self._body_proxy: QGraphicsProxyWidget | None = None
        self._body_label: QLabel | None = None
        self._discard_proxy: QGraphicsProxyWidget | None = None
        self._keep_proxy: QGraphicsProxyWidget | None = None
        self._create_widgets()

        # Register tick for glow animation
        self._clock.register(self._tick)

        logger.debug("AbortPanel created (z=%d)", Z_ABORT_PANEL)

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _make_font(self, size: int, weight: int = FONT_WEIGHT_LIGHT) -> QFont:
        """Create a QFont with project typography settings.

        Args:
            size: Font size in pixels.
            weight: Font weight.

        Returns:
            Configured QFont.
        """
        font = QFont()
        if FONT_FAMILY:
            font.setFamily(FONT_FAMILY)
        font.setPixelSize(size)
        font.setWeight(QFont.Weight(weight))
        return font

    def _create_widgets(self) -> None:
        """Create heading, body text, and buttons as proxy widgets."""
        pad = float(SPACING.md)
        content_w = self._width - 2 * pad
        y = pad

        # Heading
        heading = QLabel("Abort Recording?")
        heading.setFont(self._make_font(FONT_SIZE_INPUT, FONT_WEIGHT_REGULAR))
        heading.setStyleSheet(
            f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
            f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
            "background: transparent;"
        )
        heading.setFixedWidth(int(content_w))
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heading_proxy = QGraphicsProxyWidget(self)
        self._heading_proxy.setWidget(heading)
        self._heading_proxy.setPos(pad, y)
        y += 24

        # Body
        body = QLabel("")
        body.setFont(self._make_font(FONT_SIZE_LABEL, FONT_WEIGHT_LIGHT))
        body.setStyleSheet(
            f"color: rgba({TEXT_SECONDARY.red()}, {TEXT_SECONDARY.green()}, "
            f"{TEXT_SECONDARY.blue()}, {TEXT_SECONDARY.alpha()}); "
            "background: transparent;"
        )
        body.setFixedWidth(int(content_w))
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._body_label = body
        self._body_proxy = QGraphicsProxyWidget(self)
        self._body_proxy.setWidget(body)
        self._body_proxy.setPos(pad, y)
        y += 36

        # Buttons
        btn_w = int((content_w - SPACING.sm) / 2)
        y += float(SPACING.sm)

        discard_btn = QPushButton("Discard")
        discard_btn.setFixedWidth(btn_w)
        discard_btn.setStyleSheet(_BTN_STYLE_DISCARD)
        discard_btn.clicked.connect(self.discard_clicked.emit)
        self._discard_proxy = QGraphicsProxyWidget(self)
        self._discard_proxy.setWidget(discard_btn)
        self._discard_proxy.setPos(pad, y)
        self._discard_proxy.setFlag(
            QGraphicsProxyWidget.GraphicsItemFlag.ItemIsPanel, True,
        )

        keep_btn = QPushButton("Keep Recording")
        keep_btn.setFixedWidth(btn_w)
        keep_btn.setStyleSheet(_BTN_STYLE_KEEP)
        keep_btn.clicked.connect(self.keep_clicked.emit)
        self._keep_proxy = QGraphicsProxyWidget(self)
        self._keep_proxy.setWidget(keep_btn)
        self._keep_proxy.setPos(pad + btn_w + SPACING.sm, y)
        self._keep_proxy.setFlag(
            QGraphicsProxyWidget.GraphicsItemFlag.ItemIsPanel, True,
        )

        y += 36 + pad
        self._height = y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_screen_size(self, w: int, h: int) -> None:
        """Set screen size for centering calculations.

        Args:
            w: Screen width.
            h: Screen height.
        """
        self._screen_w = w
        self._screen_h = h

    def show_panel(self, step_count: int) -> None:
        """Show the abort confirmation panel.

        Args:
            step_count: Number of recorded steps that will be lost.
        """
        self._step_count = step_count
        if self._body_label is not None:
            self._body_label.setText(
                f"All {step_count} recorded steps from this session will be lost."
            )

        # Center on screen
        x = (self._screen_w - self._width) / 2.0
        y = (self._screen_h - self._height) / 2.0
        self.setPos(x, y)
        self.setVisible(True)
        logger.debug("AbortPanel shown (steps=%d)", step_count)

    def hide_panel(self) -> None:
        """Hide the abort confirmation panel."""
        self.setVisible(False)
        logger.debug("AbortPanel hidden")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        """Return bounding rect with glow padding.

        Returns:
            QRectF with 40px padding for card glow.
        """
        pad = 40.0
        return QRectF(-pad, -pad, self._width + 2 * pad, self._height + 2 * pad)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint frosted-glass background and card glow.

        Args:
            painter: Active QPainter.
            option: Style option (unused).
            widget: Target widget (unused).
        """
        if not self.isVisible():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(0, 0, self._width, self._height)

        # Card glow
        card_path = QPainterPath()
        card_path.addRoundedRect(rect, self._corner_radius, self._corner_radius)

        outer = QPainterPath()
        margin = 40.0
        outer.addRect(rect.adjusted(-margin, -margin, margin, margin))
        glow_clip = outer - card_path

        painter.save()
        painter.setClipPath(glow_clip)
        paint_card_glow(
            painter,
            rect,
            brightness=1.0,
            phase=self._glow_phase,
            light_count=8,
            glow_radius=25.0,
        )
        painter.restore()

        # Frosted glass background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(FROST_BG)
        painter.drawPath(card_path)

        painter.restore()

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def _tick(self, dt: float) -> None:
        """Advance glow animation phase.

        Args:
            dt: Elapsed seconds since last tick.
        """
        if not self.isVisible():
            return
        self._glow_phase += dt * 0.8
        self.update()

    def cleanup(self) -> None:
        """Unregister tick callback from animation clock."""
        self._clock.unregister(self._tick)
        logger.debug("AbortPanel cleanup: tick unregistered")
