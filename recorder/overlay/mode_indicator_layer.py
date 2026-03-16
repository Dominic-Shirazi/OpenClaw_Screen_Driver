"""Text label showing the current overlay mode and hotkey hints.

Renders a semi-transparent dark background rectangle with a coloured
text label indicating READY / RECORDING / PAUSED and the available
keyboard shortcuts.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from PyQt6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

from recorder.overlay.state import STATE_COLORS, OverlayState

logger = logging.getLogger(__name__)

# Mode text templates
_MODE_TEXT: dict[OverlayState, str] = {
    OverlayState.READY: "[READY] F2 = record | Ctrl+Q = quit | ESC = abort",
    OverlayState.RECORDING: "[RECORDING] F2 = pause | Ctrl+Q = save | ESC = abort",
    OverlayState.PAUSED: "[PAUSED] F2 = resume | Ctrl+Q = save | ESC = abort",
}


class ModeIndicatorLayer(QGraphicsItemGroup):
    """Overlay HUD label showing current mode and available hotkeys.

    Contains a dark background rectangle and a coloured text item.
    Call :meth:`update_mode` when the overlay state changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setZValue(100)

        # Background rectangle
        self._bg = QGraphicsRectItem(self)
        self._bg.setPen(QPen(Qt.PenStyle.NoPen))
        self._bg.setBrush(QBrush(QColor(0, 0, 0, 180)))

        # Text label
        self._text = QGraphicsSimpleTextItem(self)
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        self._text.setFont(font)
        self._text.setPos(15, 10)

        # Set default state
        self.update_mode(OverlayState.READY)

    def update_mode(self, state: OverlayState) -> None:
        """Update the label text and colour for the given overlay state.

        Args:
            state: The current overlay state.
        """
        text = _MODE_TEXT.get(state, "")
        self._text.setText(text)

        # Use the state colour for the text
        r, g, b, _a = STATE_COLORS[state]
        self._text.setBrush(QBrush(QColor(r, g, b, 240)))

        # Resize background to fit text
        br = self._text.boundingRect()
        self._bg.setRect(QRectF(6, 6, br.width() + 18, br.height() + 8))
