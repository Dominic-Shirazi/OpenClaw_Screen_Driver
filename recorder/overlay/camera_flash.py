"""15px border camera flash effect after screenshot capture.

Triggered after each screenshot during replay. Shows a bright white
15px border that fades out over ~200ms, simulating a camera flash.
The flash confirms to the user/agent that a screenshot was captured.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

from recorder.overlay.animation_clock import AnimationClock

logger = logging.getLogger(__name__)

_Z_CAMERA_FLASH: int = 280


class CameraFlash(QGraphicsObject):
    """15px border camera flash effect after screenshot capture.

    Triggered after each screenshot during replay. Shows a bright white
    15px border that fades out over ~200ms, simulating a camera flash.

    Args:
        clock: AnimationClock for tick registration.
        screen_w: Logical screen width in pixels.
        screen_h: Logical screen height in pixels.
    """

    def __init__(self, clock: AnimationClock, screen_w: int, screen_h: int) -> None:
        super().__init__()
        self._clock = clock
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._opacity: float = 0.0
        self._lifetime: float = 0.0
        self._active: bool = False
        self._duration: float = 0.2
        self._border_width: int = 15
        self.setZValue(_Z_CAMERA_FLASH)
        clock.register(self.tick)
        logger.debug("CameraFlash created (z=%d)", _Z_CAMERA_FLASH)

    def boundingRect(self) -> QRectF:
        """Return full screen rect as bounding box.

        Returns:
            QRectF covering the entire screen area.
        """
        return QRectF(0, 0, self._screen_w, self._screen_h)

    def flash(self) -> None:
        """Trigger the camera flash effect."""
        self._opacity = 1.0
        self._lifetime = 0.0
        self._active = True
        self.update()

    def hide(self) -> None:
        """Immediately hide the flash."""
        self._active = False
        self._opacity = 0.0
        self.update()

    def tick(self, dt: float) -> None:
        """Advance the flash fade-out animation.

        Args:
            dt: Delta seconds since last tick.
        """
        if not self._active:
            return

        self._lifetime += dt
        if self._lifetime >= self._duration:
            self._active = False
            self._opacity = 0.0
        else:
            self._opacity = 1.0 - (self._lifetime / self._duration)
        self.update()

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the white border flash.

        Args:
            painter: Active QPainter.
            option: Style option (unused).
            widget: Target widget (unused).
        """
        if not self._active or self._opacity <= 0:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        bw = self._border_width
        pen = QPen(QColor(255, 255, 255, int(220 * self._opacity)), bw)
        painter.setPen(pen)
        half = bw / 2.0
        painter.drawRect(QRectF(half, half, self._screen_w - bw, self._screen_h - bw))

        painter.restore()
