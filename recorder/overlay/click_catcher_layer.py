"""Invisible full-screen rectangle for mouse hit-testing in record mode.

When the overlay is in RECORDING state the click catcher is added to
the scene so that mouse events are intercepted by the overlay instead
of passing through to the desktop.  In READY/PAUSED states it is
removed from the scene entirely.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsRectItem

logger = logging.getLogger(__name__)


class ClickCatcherLayer(QGraphicsRectItem):
    """Nearly-invisible full-screen rect for mouse event capture.

    Uses alpha=1 so it is effectively invisible but still receives
    mouse events from the Qt graphics scene.

    Args:
        screen_w: Logical screen width in pixels.
        screen_h: Logical screen height in pixels.
    """

    def __init__(self, screen_w: int, screen_h: int) -> None:
        super().__init__(QRectF(0, 0, screen_w, screen_h))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setBrush(QBrush(QColor(0, 0, 0, 1)))
        self.setZValue(-100)
