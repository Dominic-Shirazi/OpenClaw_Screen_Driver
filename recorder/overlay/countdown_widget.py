"""Cursor-following countdown spinner widget.

Renders a 52px diameter frosted-glass circle with a countdown digit
(3, 2, 1) that follows the mouse cursor.  The digit animates with a
shrink+fade between ticks for a smooth, cinematic feel.

Uses AnimationClock for smooth per-frame digit animation and a QTimer
for the 1-second integer countdown ticks.
"""
from __future__ import annotations

import logging
import math

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QGraphicsObject,
    QStyleOptionGraphicsItem,
    QWidget,
)

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.card_glow import paint_card_glow
from recorder.overlay.hud_common import (
    ACCENT_GREEN,
    BORDER_SUBTLE,
    FONT_FAMILY,
    FONT_SIZE_COUNTDOWN,
    FONT_WEIGHT_LIGHT,
    FROST_BG,
    Z_COUNTDOWN,
)

logger = logging.getLogger(__name__)

_DIAMETER: float = 52.0
_RADIUS: float = _DIAMETER / 2.0
_CURSOR_OFFSET: float = 20.0
_TICK_DURATION_MS: int = 1000
_FADE_DURATION: float = 0.8  # seconds for digit shrink+fade animation


class CountdownWidget(QGraphicsObject):
    """Cursor-following countdown spinner with frosted-glass style.

    Displays a circular badge near the cursor that counts down from
    ``seconds`` to 1, then emits ``countdown_finished``.

    Signals:
        countdown_finished: Emitted when the countdown reaches zero.

    Args:
        clock: AnimationClock for smooth dt-driven digit animation.
        seconds: Number of seconds to count down (default 3).
        parent: Optional parent QGraphicsObject.
    """

    countdown_finished = pyqtSignal()

    def __init__(
        self,
        clock: AnimationClock,
        seconds: int = 3,
        parent: QGraphicsObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # Updated: decorative layer, pass clicks through
        self.setZValue(Z_COUNTDOWN)
        self.setVisible(False)

        self._clock = clock
        self._total_seconds: int = seconds
        self._remaining: int = seconds
        self._digit_scale: float = 1.0
        self._digit_alpha: float = 230.0
        self._tick_elapsed: float = 0.0
        self._glow_phase: float = 0.0
        self._active: bool = False

        # 1-second timer for integer countdown ticks
        self._tick_timer = QTimer()
        self._tick_timer.setInterval(_TICK_DURATION_MS)
        self._tick_timer.timeout.connect(self._on_tick)

        logger.debug(
            "CountdownWidget created (z=%d, seconds=%d)", Z_COUNTDOWN, seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the countdown and make the widget visible."""
        self._remaining = self._total_seconds
        self._digit_scale = 1.0
        self._digit_alpha = 230.0
        self._tick_elapsed = 0.0
        self._active = True
        self.setVisible(True)
        self._clock.register(self._animate)
        self._tick_timer.start()
        logger.debug("CountdownWidget started (%d seconds)", self._total_seconds)

    def stop(self) -> None:
        """Stop the countdown and hide the widget."""
        self._active = False
        self.setVisible(False)
        self._tick_timer.stop()
        self._clock.unregister(self._animate)
        logger.debug("CountdownWidget stopped")

    def set_position(self, x: float, y: float) -> None:
        """Update position to follow cursor with offset.

        Args:
            x: Cursor x coordinate.
            y: Cursor y coordinate.
        """
        self.setPos(x + _CURSOR_OFFSET, y + _CURSOR_OFFSET)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        """Return bounding rect with glow padding.

        Returns:
            QRectF with 30px padding around the circle.
        """
        pad = 30.0
        return QRectF(-pad, -pad, _DIAMETER + 2 * pad, _DIAMETER + 2 * pad)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the frosted-glass circle with countdown digit.

        Args:
            painter: Active QPainter.
            option: Style option (unused).
            widget: Target widget (unused).
        """
        if not self._active:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(0, 0, _DIAMETER, _DIAMETER)

        # Card glow (thin underglow)
        circle_path = QPainterPath()
        circle_path.addEllipse(rect)

        outer = QPainterPath()
        margin = 30.0
        outer.addRect(rect.adjusted(-margin, -margin, margin, margin))
        glow_clip = outer - circle_path

        painter.save()
        painter.setClipPath(glow_clip)
        paint_card_glow(
            painter,
            rect,
            brightness=1.0,
            phase=self._glow_phase,
            light_count=6,
            glow_radius=20.0,
        )
        painter.restore()

        # Frosted glass circle
        painter.setPen(QColor(BORDER_SUBTLE))
        painter.setBrush(FROST_BG)
        painter.drawEllipse(rect)

        # Countdown digit with scale + alpha animation
        if self._remaining > 0:
            font = QFont()
            if FONT_FAMILY:
                font.setFamily(FONT_FAMILY)
            font.setPixelSize(FONT_SIZE_COUNTDOWN)
            font.setWeight(QFont.Weight(FONT_WEIGHT_LIGHT))
            painter.setFont(font)

            digit_color = QColor(ACCENT_GREEN)
            digit_color.setAlpha(int(self._digit_alpha))
            painter.setPen(digit_color)

            # Scale transform around center
            cx = _RADIUS
            cy = _RADIUS
            painter.translate(cx, cy)
            painter.scale(self._digit_scale, self._digit_scale)
            painter.translate(-cx, -cy)

            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self._remaining))

        painter.restore()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _animate(self, dt: float) -> None:
        """Smooth per-frame digit animation driven by AnimationClock.

        Args:
            dt: Delta seconds since last frame.
        """
        self._glow_phase += dt * 0.8
        self._tick_elapsed += dt

        # Digit shrink + fade: scale 1.0->0.7, alpha 230->0 over _FADE_DURATION
        progress = min(self._tick_elapsed / _FADE_DURATION, 1.0)
        self._digit_scale = 1.0 - 0.3 * progress
        self._digit_alpha = 230.0 * (1.0 - progress)

        self.update()

    def _on_tick(self) -> None:
        """Handle 1-second countdown tick."""
        self._remaining -= 1
        self._tick_elapsed = 0.0
        self._digit_scale = 1.0
        self._digit_alpha = 230.0

        if self._remaining <= 0:
            self.stop()
            self.countdown_finished.emit()
            logger.debug("CountdownWidget finished")
