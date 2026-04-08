"""Thread-safe run state tracking for OCSD API.

Provides RunManager for tracking the lifecycle of routine execution runs,
including start, step updates, waiting/paused states, completion, and failure.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """Status of a routine execution run."""

    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class RunAlreadyActiveError(Exception):
    """Raised when attempting to start a run while one is already active."""

    pass


class RunNotFoundError(Exception):
    """Raised when a run_id cannot be found."""

    pass


@dataclass
class ActiveRun:
    """State of an in-progress or completed routine run.

    Internal mutable object — never returned to callers directly.
    Use ``RunManager.get_run()`` which returns a frozen :class:`RunSnapshot`.

    Attributes:
        run_id: Unique identifier for this run.
        routine_id: Name/ID of the routine being executed.
        status: Current run status.
        current_step: Index of the step currently executing.
        total_steps: Total number of steps in the routine.
        error: Error message if the run failed.
        prompt_text: Text of the current prompt if status is WAITING.
        abort_event: Threading event to signal run abortion.
        thread: The thread executing this run.
    """

    run_id: str
    routine_id: str
    status: RunStatus = RunStatus.RUNNING
    current_step: int = 0
    total_steps: int = 0
    error: str | None = None
    prompt_text: str | None = None
    abort_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def snapshot(self) -> RunSnapshot:
        """Return a frozen, thread-safe copy of the current run state."""
        return RunSnapshot(
            run_id=self.run_id,
            routine_id=self.routine_id,
            status=self.status,
            current_step=self.current_step,
            total_steps=self.total_steps,
            error=self.error,
            prompt_text=self.prompt_text,
        )


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable, thread-safe snapshot of a run's state.

    Returned by ``RunManager.get_run()`` so callers can inspect run state
    without holding the manager lock and without risk of unsynchronised
    mutation.

    Attributes:
        run_id: Unique identifier for this run.
        routine_id: Name/ID of the routine being executed.
        status: Current run status.
        current_step: Index of the step currently executing.
        total_steps: Total number of steps in the routine.
        error: Error message if the run failed.
        prompt_text: Text of the current prompt if status is WAITING.
    """

    run_id: str
    routine_id: str
    status: RunStatus = RunStatus.RUNNING
    current_step: int = 0
    total_steps: int = 0
    error: str | None = None
    prompt_text: str | None = None


class RunManager:
    """Thread-safe manager for routine execution runs.

    Ensures only one run is active at a time and provides methods
    to track run state transitions.
    """

    _MAX_HISTORY: int = 100

    def __init__(self) -> None:
        """Initialize the RunManager with an empty state."""
        self._lock = threading.Lock()
        self._active: ActiveRun | None = None
        self._history: dict[str, ActiveRun] = {}

    def start_run(self, routine_id: str) -> str:
        """Start a new run for the given routine.

        Args:
            routine_id: Name/ID of the routine to run.

        Returns:
            The generated run_id.

        Raises:
            RunAlreadyActiveError: If a run is already in progress.
        """
        with self._lock:
            if self._active is not None and self._active.status in (
                RunStatus.RUNNING,
                RunStatus.WAITING,
                RunStatus.PAUSED,
            ):
                raise RunAlreadyActiveError(
                    f"Run {self._active.run_id} is already active "
                    f"(status={self._active.status.value})"
                )
            run_id = uuid.uuid4().hex[:12]
            run = ActiveRun(run_id=run_id, routine_id=routine_id)
            self._active = run
            logger.info("Started run %s for routine '%s'", run_id, routine_id)
            return run_id

    def get_run(self, run_id: str) -> RunSnapshot | None:
        """Get a frozen snapshot of a run by its ID.

        Args:
            run_id: The run identifier to look up.

        Returns:
            A frozen RunSnapshot if found, None otherwise.
        """
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                return self._active.snapshot()
            hist = self._history.get(run_id)
            return hist.snapshot() if hist is not None else None

    def update_step(self, run_id: str, step_index: int, total_steps: int) -> None:
        """Update the current step progress for an active run.

        Args:
            run_id: The run to update.
            step_index: Current step index.
            total_steps: Total number of steps.
        """
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                self._active.current_step = step_index
                self._active.total_steps = total_steps

    def mark_waiting(self, run_id: str, prompt_text: str) -> None:
        """Mark a run as waiting for user input.

        Args:
            run_id: The run to update.
            prompt_text: The prompt being displayed to the user.
        """
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                self._active.status = RunStatus.WAITING
                self._active.prompt_text = prompt_text

    def mark_paused(self, run_id: str) -> None:
        """Mark a run as paused.

        Args:
            run_id: The run to pause.
        """
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                self._active.status = RunStatus.PAUSED

    def mark_complete(self, run_id: str) -> None:
        """Mark a run as completed and archive it.

        Args:
            run_id: The run to complete.
        """
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                self._active.status = RunStatus.COMPLETED
                self._archive(self._active)

    def mark_failed(self, run_id: str, error: str) -> None:
        """Mark a run as failed and archive it.

        Args:
            run_id: The run that failed.
            error: Error message describing the failure.
        """
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                self._active.status = RunStatus.FAILED
                self._active.error = error
                self._archive(self._active)

    def _archive(self, run: ActiveRun) -> None:
        """Move a run from active to history, evicting oldest if at capacity.

        Args:
            run: The run to archive.
        """
        self._history[run.run_id] = run
        self._active = None
        # Evict oldest entries when history exceeds the cap
        if len(self._history) > self._MAX_HISTORY:
            excess = len(self._history) - self._MAX_HISTORY
            oldest_keys = list(self._history.keys())[:excess]
            for key in oldest_keys:
                del self._history[key]
            logger.debug("Evicted %d old run(s) from history", excess)
        logger.info(
            "Archived run %s (status=%s)", run.run_id, run.status.value
        )
