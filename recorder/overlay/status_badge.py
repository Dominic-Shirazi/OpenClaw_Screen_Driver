"""Frosted-glass pill badge showing replay step progress.

Positioned at top-center of screen. Shows text like "Step 3/8: Click Submit".
Used during routine replay to indicate which step is currently executing.
"""
from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

from recorder.overlay.animation_clock import AnimationClock

logger = logging.getLogger(__name__)

_BADGE_W: float = 400.0
_BADGE_H: float = 36.0
_BADGE_RADIUS: float = 18.0
_TOP_MARGIN: float = 20.0
_Z_STATUS_BADGE: int = 300

_BG_COLOR: QColor = QColor(20, 20, 30, 200)
_BORDER_COLOR: QColor = QColor(160, 80, 220, 120)
_TEXT_COLOR: QColor = QColor(240, 240, 245, 230)

_FONT_SIZE: int = 13


def _font_family() -> str:
    """Return platform-appropriate font family."""
    if sys.platform == "win32":
        return "Segoe UI"
    if sys.platform == "darwin":
        return "SF Pro"
    return "sans-serif"


class StatusBadge(QGraphicsObject):
    """Frosted-glass pill badge showing replay step progress.

    Positioned at top-center of screen. Shows text like "Step 3/8: Click Submit".

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
        self._text: str = ""
        self.setZValue(_Z_STATUS_BADGE)
        self.setPos(screen_w / 2 - _BADGE_W / 2, _TOP_MARGIN)
        logger.debug("StatusBadge created at top-center (z=%d)", _Z_STATUS_BADGE)

    def boundingRect(self) -> QRectF:
        """Return the pill bounding rect.

        Returns:
            QRectF covering the badge area.
        """
        return QRectF(0, 0, _BADGE_W, _BADGE_H)

    def set_text(self, text: str) -> None:
        """Update displayed text.

        Args:
            text: Step label text (e.g. "Step 3/8: Click Submit").
        """
        self._text = text
        self.update()

    def set_visible_animated(self, visible: bool) -> None:
        """Show or hide the badge.

        Per CONTEXT.md -- minimal UX, instant cleanup. No animation needed.

        Args:
            visible: Whether badge should be visible.
        """
        self.setVisible(visible)

    def tick(self, dt: float) -> None:
        """Animation tick (no-op for static widget).

        Args:
            dt: Delta seconds since last tick.
        """

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the frosted-glass pill with centered text.

        Args:
            painter: Active QPainter.
            option: Style option (unused).
            widget: Target widget (unused).
        """
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(0, 0, _BADGE_W, _BADGE_H)

        # Background fill
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_BG_COLOR)
        painter.drawRoundedRect(rect, _BADGE_RADIUS, _BADGE_RADIUS)

        # Purple border
        pen = QPen(_BORDER_COLOR, 1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, _BADGE_RADIUS, _BADGE_RADIUS)

        # Centered text
        if self._text:
            font = QFont()
            font.setFamily(_font_family())
            font.setPixelSize(_FONT_SIZE)
            painter.setFont(font)
            painter.setPen(_TEXT_COLOR)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._text)

        painter.restore()
