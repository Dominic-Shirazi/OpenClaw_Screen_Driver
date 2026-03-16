"""Colored border around screen edges indicating overlay state.

Draws four thin rectangles (top, bottom, left, right) forming a
continuous border around the screen perimeter.  The color is updated
by the view/controller when the overlay state changes.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsItemGroup, QGraphicsRectItem

logger = logging.getLogger(__name__)

BORDER_WIDTH: int = 4
"""Width of each border rectangle in logical pixels."""


class BorderLayer(QGraphicsItemGroup):
    """Screen-edge border indicating the current overlay state.

    Composed of four ``QGraphicsRectItem`` children (top, bottom,
    left, right) that together form a continuous coloured frame.

    Args:
        screen_w: Logical screen width in pixels.
        screen_h: Logical screen height in pixels.
    """

    def __init__(self, screen_w: int, screen_h: int) -> None:
        super().__init__()
        self.setZValue(10)

        bw = BORDER_WIDTH
        w = screen_w
        h = screen_h

        # Top edge
        self._top = QGraphicsRectItem(QRectF(0, 0, w, bw), self)
        # Bottom edge
        self._bottom = QGraphicsRectItem(QRectF(0, h - bw, w, bw), self)
        # Left edge
        self._left = QGraphicsRectItem(QRectF(0, 0, bw, h), self)
        # Right edge
        self._right = QGraphicsRectItem(QRectF(w - bw, 0, bw, h), self)

        self._rects = [self._top, self._bottom, self._left, self._right]

        # Remove default pen so only the brush colour is visible
        no_pen = QPen(Qt.PenStyle.NoPen)
        for rect in self._rects:
            rect.setPen(no_pen)

    def set_color(self, r: int, g: int, b: int, a: int) -> None:
        """Update all four border rectangles to the given RGBA colour.

        Args:
            r: Red channel (0-255).
            g: Green channel (0-255).
            b: Blue channel (0-255).
            a: Alpha channel (0-255).
        """
        brush = QBrush(QColor(r, g, b, a))
        for rect in self._rects:
            rect.setBrush(brush)
