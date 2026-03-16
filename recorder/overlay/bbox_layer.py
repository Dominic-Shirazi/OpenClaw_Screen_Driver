"""Bounding box rendering with corner resize handles.

Each ``BboxLayer`` represents a single detected UI element drawn on
the overlay scene.  Corner handles are included for future resize
support (actual drag logic is Phase 2+).
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

logger = logging.getLogger(__name__)

_HANDLE_SIZE: int = 8
"""Side length of corner handle squares in pixels."""


class BboxLayer(QGraphicsItemGroup):
    """Visual bounding box with corner handles and optional label.

    Args:
        x: Left edge in scene coordinates.
        y: Top edge in scene coordinates.
        w: Width in pixels.
        h: Height in pixels.
        color_rgba: Border colour as ``(r, g, b, a)`` tuple.
        label: Optional text label shown above the box.
        confidence: Detection confidence (0.0 -- 1.0), shown in label.
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color_rgba: tuple[int, int, int, int],
        label: str = "",
        confidence: float = 0.0,
    ) -> None:
        super().__init__()
        self.setZValue(50)

        self._x = x
        self._y = y
        self._w = w
        self._h = h

        r, g, b, a = color_rgba

        # Main border rectangle
        pen = QPen(QColor(r, g, b, min(a + 60, 255)))
        pen.setWidth(2)
        brush = QBrush(QColor(r, g, b, a // 4))
        self._rect = QGraphicsRectItem(QRectF(x, y, w, h), self)
        self._rect.setPen(pen)
        self._rect.setBrush(brush)

        # Optional label text above the box
        self._label: QGraphicsSimpleTextItem | None = None
        if label:
            text_str = f"{label} ({confidence:.0%})"
            self._label = QGraphicsSimpleTextItem(text_str, self)
            self._label.setPos(x, max(0, y - 16))
            self._label.setBrush(QBrush(QColor(r, g, b, 230)))
            font = QFont("Segoe UI", 9)
            font.setBold(True)
            self._label.setFont(font)

        # Corner handles (small white squares at each corner)
        self._handles: list[QGraphicsRectItem] = []
        handle_brush = QBrush(QColor(255, 255, 255, 200))
        handle_pen = QPen(QColor(r, g, b, 220))
        handle_pen.setWidth(1)

        for hx, hy in self._corner_positions(x, y, w, h):
            handle = QGraphicsRectItem(
                QRectF(hx, hy, _HANDLE_SIZE, _HANDLE_SIZE),
                self,
            )
            handle.setPen(handle_pen)
            handle.setBrush(handle_brush)
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True
            )
            self._handles.append(handle)

    @staticmethod
    def _corner_positions(
        x: int, y: int, w: int, h: int,
    ) -> list[tuple[float, float]]:
        """Return top-left positions for corner handle rects.

        Args:
            x: Box left edge.
            y: Box top edge.
            w: Box width.
            h: Box height.

        Returns:
            Four ``(hx, hy)`` tuples for TL, TR, BL, BR handles.
        """
        hs = _HANDLE_SIZE
        half = hs / 2
        return [
            (x - half, y - half),               # top-left
            (x + w - half, y - half),            # top-right
            (x - half, y + h - half),            # bottom-left
            (x + w - half, y + h - half),        # bottom-right
        ]

    def get_rect(self) -> tuple[int, int, int, int]:
        """Return the current bounding box as ``(x, y, w, h)``.

        Returns:
            Tuple of ``(x, y, width, height)`` accounting for any
            handle dragging.
        """
        return (self._x, self._y, self._w, self._h)

    def highlight(self, active: bool) -> None:
        """Highlight or dim this bounding box.

        Args:
            active: If True, render at full opacity.  If False, dim to
                40% opacity for de-emphasis.
        """
        self.setOpacity(1.0 if active else 0.4)

    def reset_highlight(self) -> None:
        """Restore the bounding box to default (full) opacity."""
        self.setOpacity(1.0)
