"""Timer-driven per-field character insertion engine.

Drives simultaneous typewriter fill across multiple QLineEdit fields,
registered with an AnimationClock for frame-independent timing.  Each
field fills at a randomly chosen speed between TYPEWRITER_MIN_CPS and
TYPEWRITER_MAX_CPS.
"""
from __future__ import annotations

import logging
import math
import random

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QLineEdit

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.hud_common import TYPEWRITER_MAX_CPS, TYPEWRITER_MIN_CPS

logger = logging.getLogger(__name__)


class _FieldState:
    """Internal state for one typewriter-driven field."""

    def __init__(self, widget: QLineEdit, target: str, cps: float) -> None:
        self.widget = widget
        self.target = target
        self.cps = cps
        self.pos: int = 0
        self.accum: float = 0.0
        self.done: bool = False


class TypewriterEngine(QObject):
    """Drives simultaneous typewriter fill across multiple QLineEdit fields.

    Registered with an AnimationClock to receive delta-time ticks.
    Each field fills at a randomly chosen speed between TYPEWRITER_MIN_CPS
    and TYPEWRITER_MAX_CPS.  Emits char_inserted(int) on each character
    insertion (field index) for glow synchronization.  Emits finished()
    when all fields are complete.

    Args:
        clock: The AnimationClock to register with for tick callbacks.
        parent: Optional QObject parent for Qt ownership.
    """

    char_inserted = pyqtSignal(int)
    """Emitted on each character insertion with the field index."""

    finished = pyqtSignal()
    """Emitted when all fields have completed typing."""

    def __init__(self, clock: AnimationClock, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._clock = clock
        self._fields: list[_FieldState] = []
        self._active: bool = False

    @property
    def is_active(self) -> bool:
        """Return True while any field is still typing.

        Returns:
            True if the engine is actively filling fields.
        """
        return self._active

    def start(self, fields: list[tuple[QLineEdit, str]]) -> None:
        """Begin typewriter fill.

        Args:
            fields: List of (widget, target_text) tuples.
                Each field gets a random speed between TYPEWRITER_MIN_CPS
                and TYPEWRITER_MAX_CPS.
        """
        self._fields = []
        for widget, target in fields:
            cps = random.uniform(TYPEWRITER_MIN_CPS, TYPEWRITER_MAX_CPS)
            state = _FieldState(widget, target, cps)
            # Fields with empty target are immediately done
            if len(target) == 0:
                state.done = True
            self._fields.append(state)

        # Check if all fields are already done (empty targets)
        if all(f.done for f in self._fields):
            self._active = False
            self.finished.emit()
            return

        self._active = True
        self._clock.register(self.tick)
        logger.debug(
            "TypewriterEngine started with %d fields", len(self._fields),
        )

    def tick(self, dt: float) -> None:
        """Advance each field by dt seconds.

        For each non-done field:
        1. Accumulate dt * cps into accum
        2. If accum >= 1.0, insert floor(accum) characters
        3. Call widget.setText(target[:new_pos])
        4. Emit char_inserted(field_index) for each new character
        5. If pos >= len(target), mark field as done

        When all fields done, call _finish().

        Args:
            dt: Elapsed seconds since last tick.
        """
        if not self._active:
            return

        any_active = False

        for idx, field in enumerate(self._fields):
            if field.done:
                continue

            field.accum += dt * field.cps
            chars_to_add = int(math.floor(field.accum))

            if chars_to_add < 1:
                any_active = True
                continue

            field.accum -= chars_to_add
            old_pos = field.pos
            new_pos = min(field.pos + chars_to_add, len(field.target))
            field.pos = new_pos

            if new_pos > old_pos:
                field.widget.setText(field.target[:new_pos])
                # Emit char_inserted for each new character
                for _ in range(new_pos - old_pos):
                    self.char_inserted.emit(idx)

            if field.pos >= len(field.target):
                field.done = True
            else:
                any_active = True

        if not any_active and all(f.done for f in self._fields):
            self._finish()

    def interrupt_field(self, field_index: int) -> None:
        """Stop typewriter for a specific field (user started editing).

        Args:
            field_index: Index of the field in the original start() list.
        """
        if 0 <= field_index < len(self._fields):
            self._fields[field_index].done = True
            logger.debug("TypewriterEngine field %d interrupted", field_index)

            # If all fields now done, finish
            if all(f.done for f in self._fields):
                self._finish()

    def stop(self) -> None:
        """Stop all typewriter activity immediately."""
        if self._active:
            self._clock.unregister(self.tick)
            for field in self._fields:
                field.done = True
            self._active = False
            logger.debug("TypewriterEngine stopped")

    def _finish(self) -> None:
        """Internal: all fields complete."""
        if self._active:
            self._clock.unregister(self.tick)
            self._active = False
            self.finished.emit()
            logger.debug("TypewriterEngine finished (all fields complete)")
