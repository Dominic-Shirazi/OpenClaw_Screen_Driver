"""Adapter connecting routine runner events to overlay visual updates.

The runner emits events on a background thread. This adapter receives
them and updates the overlay controller on the Qt main thread via a
pyqtSignal. If ``replay.show_overlay`` config is False, all operations
are no-ops.
"""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from core.config import get_config
from routine.runner import RunEvent

logger = logging.getLogger(__name__)


class ReplayOverlayAdapter(QObject):
    """Thread-safe bridge from RunEvent callbacks to OverlayController.

    Subclasses QObject to use pyqtSignal for crossing from the runner's
    background thread to the Qt main thread. The adapter instance is
    passed directly as the run_routine() callback.

    Usage:
        adapter = ReplayOverlayAdapter(controller)
        run_routine(routine_dir, callback=adapter)

    Args:
        controller: OverlayController instance, or None for headless mode.
        parent: Optional QObject parent.
    """

    # Signal: (event_value: int, data: object)
    # Using int + object because pyqtSignal doesn't support custom Enum types
    _event_signal = pyqtSignal(int, object)

    def __init__(
        self,
        controller: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._active: bool = True

        # Check config
        config = get_config()
        show_overlay = config.get("replay", {}).get("show_overlay", True)
        if not show_overlay:
            self._active = False

        # Headless mode: no controller
        if controller is None:
            self._active = False

        # Connect signal to main-thread slot
        self._event_signal.connect(self._on_event)

    def __call__(self, event: RunEvent, data: dict[str, Any]) -> None:
        """Receive event from runner background thread and forward via signal.

        Args:
            event: The run event that occurred.
            data: Event-specific data dictionary.
        """
        if not self._active:
            return
        # Signal emission is thread-safe in Qt
        self._event_signal.emit(event.value, data)

    def _on_event(self, event_value: int, data: dict[str, Any]) -> None:
        """Handle event on the main Qt thread. Safe to call controller.

        Args:
            event_value: Integer value of RunEvent enum member.
            data: Event-specific data dictionary.
        """
        try:
            event = RunEvent(event_value)
        except ValueError:
            return

        try:
            match event:
                case RunEvent.RUN_START:
                    self._controller.set_replay_mode(True)
                    self._controller.show()
                    name = data.get("routine_name", "")
                    total = data.get("total_steps", 0)
                    self._controller.set_replay_status(
                        f"Starting: {name} ({total} steps)"
                    )

                case RunEvent.STEP_START:
                    idx = data.get("step_index", 0) + 1
                    total = data.get("total_steps", 0)
                    label = data.get("label", "")
                    action = data.get("action", "")
                    self._controller.set_replay_status(
                        f"Step {idx}/{total}: {action.title()} {label}"
                    )

                case RunEvent.ELEMENT_LOCATED:
                    point = data.get("point", {"x": 0, "y": 0})
                    rect = data.get("rect")
                    if rect:
                        self._controller.show_target_highlight(
                            rect["x"], rect["y"], rect["w"], rect["h"],
                        )
                    else:
                        px = point.get("x", 0) if isinstance(point, dict) else point[0]
                        py = point.get("y", 0) if isinstance(point, dict) else point[1]
                        self._controller.show_target_highlight(
                            px - 20, py - 20, 40, 40,
                        )

                case RunEvent.SCREENSHOT_TAKEN:
                    self._controller.camera_flash()

                case RunEvent.STEP_RETRY:
                    stage = data.get("stage", "?")
                    self._controller.set_replay_status(
                        f"Retry: stage {stage}"
                    )

                case RunEvent.STEP_COMPLETE:
                    idx = data.get("step_index", 0) + 1
                    total = data.get("total_steps", 0)
                    self._controller.set_replay_status(
                        f"Step {idx}/{total}: OK"
                    )

                case RunEvent.STEP_FAILED:
                    idx = data.get("step_index", 0) + 1
                    label = data.get("label", "")
                    self._controller.set_replay_status(
                        f"FAILED at step {idx}: {label}"
                    )

                case RunEvent.RUN_PAUSED:
                    # Abort triggers pause UX per CONTEXT.md:
                    # - Shimmer turns green (READY state)
                    # - Status badge shows "Paused"
                    # - Overlay stays open for human-at-keyboard decision
                    self._controller.set_replay_mode(False)  # READY = green shimmer
                    self._controller.set_replay_status("Paused")
                    # Do NOT call close() or _cleanup_after_failure()
                    logger.info(
                        "Run paused via abort -- overlay showing pause state"
                    )

                case RunEvent.RUN_COMPLETE:
                    self._controller.set_replay_status("Complete")
                    self._controller.set_replay_mode(False)
                    self._controller.close()

                case RunEvent.RUN_FAILED:
                    reason = data.get("failure_reason", "unknown")
                    self._controller.set_replay_status(
                        f"Failed: {reason[:50]}"
                    )
                    # Brief delay then clear via QTimer (non-blocking)
                    from PyQt6.QtCore import QTimer

                    QTimer.singleShot(
                        2000, lambda: self._cleanup_after_failure(),
                    )
        except Exception as e:
            logger.debug("Replay overlay event %s failed: %s", event, e)

    def _cleanup_after_failure(self) -> None:
        """Clean up overlay after showing failure message briefly."""
        try:
            self._controller.set_replay_mode(False)
            self._controller.close()
        except Exception as e:
            logger.debug("Replay overlay cleanup failed: %s", e)
