"""Recording session state manager.

Tracks the lifecycle of a recording session including elements captured,
timing, and undo history.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Literal

logger = logging.getLogger(__name__)


class RecordingSession:
    """Manages state for a single recording session.

    Tracks recorded elements, timing, undo history, and the current
    recording mode (workflow or diagram).

    Attributes:
        session_id: Unique identifier for this session.
        start_time: Monotonic timestamp when the session started.
        current_mode: Either "workflow" or "diagram".
        is_active: Whether the session is currently recording.
    """

    def __init__(
        self,
        mode: Literal["workflow", "diagram"] = "workflow",
    ) -> None:
        """Initializes a new recording session.

        Args:
            mode: The recording mode, either "workflow" or "diagram".
        """
        self.session_id: str = str(uuid.uuid4())
        self.start_time: float = time.monotonic()
        self.current_mode: Literal["workflow", "diagram"] = mode
        self.is_active: bool = True
        self._elements: list[dict] = []

        logger.info(
            "Recording session %s started in '%s' mode",
            self.session_id[:8],
            self.current_mode,
        )

    def add_element(self, elem: dict) -> None:
        """Adds a recorded element to the session.

        Args:
            elem: A dict representing a captured UI element or action.

        Raises:
            RuntimeError: If the session is no longer active.
        """
        if not self.is_active:
            raise RuntimeError(
                f"Session {self.session_id[:8]} is finalized; "
                "cannot add elements."
            )
        self._elements.append(elem)
        logger.debug(
            "Session %s: added element #%d",
            self.session_id[:8],
            len(self._elements),
        )

    def undo_last(self) -> dict | None:
        """Removes and returns the last recorded element.

        Returns:
            The removed element dict, or None if the list is empty.

        Raises:
            RuntimeError: If the session is no longer active.
        """
        if not self.is_active:
            raise RuntimeError(
                f"Session {self.session_id[:8]} is finalized; "
                "cannot undo."
            )
        if not self._elements:
            logger.debug(
                "Session %s: undo called on empty element list",
                self.session_id[:8],
            )
            return None

        removed = self._elements.pop()
        logger.debug(
            "Session %s: undid element, %d remaining",
            self.session_id[:8],
            len(self._elements),
        )
        return removed

    def finalize(self) -> list[dict]:
        """Finalizes the session and returns all recorded elements.

        Marks the session as inactive. No further elements can be added
        or undone after finalization.

        Returns:
            A copy of the list of recorded element dicts.
        """
        self.is_active = False
        elements = list(self._elements)
        logger.info(
            "Session %s finalized with %d elements (%.1fs elapsed)",
            self.session_id[:8],
            len(elements),
            self.elapsed_seconds(),
        )
        return elements

    def elapsed_seconds(self) -> float:
        """Returns the number of seconds since the session started.

        Returns:
            Elapsed time in seconds as a float.
        """
        return time.monotonic() - self.start_time

    @property
    def element_count(self) -> int:
        """Returns the current number of recorded elements."""
        return len(self._elements)

    def __repr__(self) -> str:
        return (
            f"RecordingSession(id={self.session_id[:8]}, "
            f"mode={self.current_mode!r}, "
            f"elements={len(self._elements)}, "
            f"active={self.is_active})"
        )
