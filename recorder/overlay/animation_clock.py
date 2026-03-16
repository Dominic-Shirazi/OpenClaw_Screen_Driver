"""Shared animation clock delivering frame-independent delta-time.

Drives all overlay animations via a single 16ms QTimer with
QElapsedTimer-based delta-time correction.  Registered callbacks
receive the elapsed seconds since the last tick, capped at 0.1s
to prevent spiral-of-death after long stalls.
"""

from __future__ import annotations

import logging
from typing import Callable

from PyQt6.QtCore import QElapsedTimer, QObject, Qt, QTimer

logger = logging.getLogger(__name__)


class AnimationClock(QObject):
    """Central animation tick for all overlay animations.

    Uses a 16ms PreciseTimer (~60 fps target) and QElapsedTimer
    to deliver frame-independent delta-time to registered callbacks.

    Args:
        parent: Optional QObject parent for Qt ownership.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._elapsed = QElapsedTimer()
        self._callbacks: list[Callable[[float], None]] = []

    def start(self) -> None:
        """Start the animation clock and begin delivering ticks."""
        self._elapsed.start()
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        logger.debug("AnimationClock started (16ms interval)")

    def stop(self) -> None:
        """Stop the animation clock and cease tick delivery."""
        self._timer.stop()
        try:
            self._timer.timeout.disconnect(self._tick)
        except TypeError:
            pass  # Already disconnected
        logger.debug("AnimationClock stopped")

    def register(self, callback: Callable[[float], None]) -> None:
        """Register a callback to receive delta_seconds each frame.

        Args:
            callback: Function accepting a single float (delta seconds).
        """
        self._callbacks.append(callback)
        logger.debug("Registered animation callback: %s", callback)

    def unregister(self, callback: Callable[[float], None]) -> None:
        """Remove a previously registered callback.

        Args:
            callback: The callback to remove. No error if not found.
        """
        try:
            self._callbacks.remove(callback)
            logger.debug("Unregistered animation callback: %s", callback)
        except ValueError:
            pass  # Callback was not registered

    def _tick(self) -> None:
        """Internal tick handler -- measure delta-time, dispatch to callbacks."""
        dt = self._elapsed.restart() / 1000.0
        dt = min(dt, 0.1)  # Cap to prevent spiral of death
        for cb in list(self._callbacks):
            cb(dt)
