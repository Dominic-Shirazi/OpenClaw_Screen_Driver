"""Quick purple glow around located element bbox.

Shows for ~300ms then fades out automatically. Used during replay
to show which element the system located before executing the action.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

from recorder.overlay.animation_clock import AnimationClock

logger = logging.getLogger(__name__)

_Z_TARGET_HIGHLIGHT: int = 290
_PURPLE: tuple[int, int, int] = (160, 80, 220)


class TargetHighlight(QGraphicsObject):
    """Quick purple glow around located element bbox.

    Shows for ~300ms then fades out automatically. Used during replay
    to show which element the system located before executing the action.

    Args:
        clock: AnimationClock for tick registration.
    """

    def __init__(self, clock: AnimationClock) -> None:
        super().__init__()
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # Updated: click-through during replay
        self._clock = clock
        self._rect: QRectF | None = None
        self._opacity: float = 0.0
        self._lifetime: float = 0.0
        self._duration: float = 0.3
        self.setZValue(_Z_TARGET_HIGHLIGHT)
        clock.register(self.tick)
        logger.debug("TargetHighlight created (z=%d)", _Z_TARGET_HIGHLIGHT)

    def boundingRect(self) -> QRectF:
        """Return bounding rect (expanded target rect or empty).

        Returns:
            QRectF covering the highlight area with padding.
        """
        if self._rect is not None:
            return self._rect.adjusted(-10, -10, 10, 10)
        return QRectF(0, 0, 0, 0)

    def highlight(self, x: int, y: int, w: int, h: int) -> None:
        """Set the target rect and start the fade timer.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            w: Target width.
            h: Target height.
        """
        self._rect = QRectF(x, y, w, h)
        self._opacity = 1.0
        self._lifetime = 0.0
        self.update()

    def hide(self) -> None:
        """Immediately hide the highlight."""
        self._rect = None
        self._opacity = 0.0
        self.update()

    def tick(self, dt: float) -> None:
        """Advance the fade-out animation.

        Args:
            dt: Delta seconds since last tick.
        """
        if self._rect is None or self._opacity <= 0:
            return

        self._lifetime += dt
        if self._lifetime >= self._duration:
            self._opacity = 0.0
            self._rect = None
        else:
            self._opacity = 1.0 - (self._lifetime / self._duration)
        self.update()

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the purple glow border around the target rect.

        Args:
            painter: Active QPainter.
            option: Style option (unused).
            widget: Target widget (unused).
        """
        if self._rect is None or self._opacity <= 0:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        r, g, b = _PURPLE

        # Outer soft glow
        glow_pen = QPen(QColor(r, g, b, int(60 * self._opacity)), 6.0)
        painter.setPen(glow_pen)
        painter.drawRoundedRect(self._rect, 4.0, 4.0)

        # Inner sharp border
        border_pen = QPen(QColor(r, g, b, int(180 * self._opacity)), 3.0)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self._rect, 4.0, 4.0)

        painter.restore()
