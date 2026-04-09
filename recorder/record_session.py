"""RecordSession orchestrator for the recording pipeline.

Manages the full recording state machine from AWAITING_CLICK through
SUCCESS_FLASH, including dry-run execution, save/abort flow, and
loop-back to AWAITING_CLICK for multi-step recording.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from PyQt6.QtCore import QRectF, QTimer

from core.config import get_config
from recorder.overlay.pipeline_bridge import PipelineBridge
from recorder.overlay.record_phase import RecordPhase
from recorder.overlay.toolbar_panel import ToolbarMode

logger = logging.getLogger(__name__)


def _compute_region_hint(x: int, y: int, screen_w: int, screen_h: int) -> str:
    """Return a 3x3 grid region hint based on bbox position.

    Args:
        x: Horizontal position in pixels.
        y: Vertical position in pixels.
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.

    Returns:
        Region hint string like "top_left", "center", "bottom_right".
    """
    x_pct = x / screen_w if screen_w > 0 else 0.5
    y_pct = y / screen_h if screen_h > 0 else 0.5

    if y_pct < 0.33:
        v = "top"
    elif y_pct < 0.66:
        v = "center"
    else:
        v = "bottom"

    if x_pct < 0.33:
        h = "left"
    elif x_pct < 0.66:
        h = "center"
    else:
        h = "right"

    if v == "center" and h == "center":
        return "center"
    return f"{v}_{h}"


def _step_to_json(step: dict, index: int, screen_w: int, screen_h: int) -> dict:
    """Convert internal step dict to routine.json v1 step format.

    Delegates to :func:`routine.format.build_v1_step` for the actual
    step construction, ensuring consistency with the ocsd-routine-v1 schema.

    Args:
        step: Internal step dictionary with tag_data and bbox.
        index: Step index in the sequence.
        screen_w: Screen width for percentage calculations.
        screen_h: Screen height for percentage calculations.

    Returns:
        JSON-serializable step dictionary matching ocsd-routine-v1 schema.
    """
    from routine.format import build_v1_step

    node_id = step.get("node_id", str(uuid4()))
    # Strip non-serializable fields before passing to build_v1_step
    clean_step = {k: v for k, v in step.items() if k != "screenshot"}
    return build_v1_step(clean_step, index, node_id, screen_w, screen_h)


class RecordSession:
    """Orchestrates the recording pipeline from click through save/abort.

    RecordSession is the brain of the recording flow.  It receives events
    from the overlay controller (selection, toolbar buttons, tag dialog)
    and coordinates detection, VLM analysis, dry-run execution, step
    accumulation, and save/abort.

    Args:
        controller: OverlayController instance (already created).
        routine_name: Validated routine name from TUI prompt.
        start_from: Either ``"desktop"`` or ``"app"`` for initial context.
    """

    def __init__(
        self,
        controller: Any,
        routine_name: str,
        start_from: str = "desktop",
    ) -> None:
        self._controller = controller
        self._routine_name = routine_name
        self._start_from = start_from

        self._bridge = PipelineBridge()
        self._bridge.detection_ready.connect(self._on_detection_ready)
        self._bridge.vlm_ready.connect(self._on_vlm_ready)
        self._bridge.vlm_failed.connect(self._on_vlm_failed)
        self._bridge.execution_complete.connect(self._on_execution_complete)
        self._bridge.save_complete.connect(self._on_save_complete)
        self._bridge.save_failed.connect(self._on_save_failed)

        self._phase: RecordPhase = RecordPhase.AWAITING_CLICK
        self._steps: list[dict] = []
        self._current_step: dict | None = None
        self._current_bbox: tuple[int, int, int, int] | None = None
        self._original_drag_rect: tuple[int, int, int, int] | None = None
        self._screenshot: np.ndarray | None = None
        self._is_drag_capture: bool = False
        self._is_look_here: bool = False
        self._click_x: int = 0
        self._click_y: int = 0
        self._on_session_complete: Callable[[bool], None] | None = None
        self._save_in_progress: bool = False

        logger.info(
            "RecordSession created: routine=%s, start_from=%s",
            routine_name,
            start_from,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Return the number of accumulated steps."""
        return len(self._steps)

    @property
    def phase(self) -> RecordPhase:
        """Return the current recording phase."""
        return self._phase

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Wire controller callbacks, enter AWAITING_CLICK.

        Toolbar is deferred until the first F2 press (RECORDING transition)
        so it does not appear during the initial green passthrough.
        """
        self._controller._on_selection = self.on_selection
        self._controller._on_save = self.on_save_requested
        self._controller._on_abort = self.on_abort_requested
        self._controller._on_state_changed = self._on_overlay_state_changed
        self._toolbar_shown = False

        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)
        self._set_phase(RecordPhase.AWAITING_CLICK)

        logger.info("RecordSession started, awaiting first click")

    def _on_overlay_state_changed(self, state: Any) -> None:
        """Show toolbar on first transition to RECORDING."""
        from recorder.overlay.state import OverlayState

        if state == OverlayState.RECORDING and not self._toolbar_shown:
            self._controller.show_toolbar(on_action=self.on_toolbar_action)
            self._toolbar_shown = True

    def set_on_complete(self, callback: Callable[[bool], None]) -> None:
        """Register a session completion callback.

        Args:
            callback: Called with True on save, False on abort.
        """
        self._on_session_complete = callback

    def on_selection(self, x: int, y: int, w: int, h: int) -> None:
        """Handle overlay click/drag selection.

        Only accepts selections when phase is AWAITING_CLICK.
        Determines click vs drag and runs the capture pipeline.

        Args:
            x: Left edge of selection (or click X).
            y: Top edge of selection (or click Y).
            w: Width of selection (0 for clicks).
            h: Height of selection (0 for clicks).
        """
        if self._phase == RecordPhase.AWAITING_REGION_DRAG:
            # "Look Here" region drag -- skip detection, go straight to VLM
            if w <= 0 or h <= 0:
                logger.info("Look Here requires drag, not click -- ignoring")
                return
            self._is_drag_capture = True
            self._is_look_here = True
            self._current_step = {
                "click_x": x, "click_y": y,
                "is_drag": True, "is_look_here": True,
            }
            self._original_drag_rect = (x, y, w, h)
            self._current_bbox = (x, y, w, h)
            # Skip detection, go directly to VLM (Look Here doesn't need AI bbox)
            self._run_capture_pipeline_for_region(x, y, w, h)
            return

        if self._phase == RecordPhase.AWAITING_DRAG_TARGET:
            # Second click/drag for click_drag target
            self._handle_drag_target_selection(x, y, w, h)
            return

        if self._phase != RecordPhase.AWAITING_CLICK:
            logger.debug(
                "Ignoring selection in phase %s (gate: AWAITING_CLICK only)",
                self._phase.name,
            )
            return

        self._click_x = x
        self._click_y = y
        self._is_drag_capture = w > 0 and h > 0

        if self._is_drag_capture:
            self._original_drag_rect = (x, y, w, h)
        else:
            self._original_drag_rect = None

        self._current_step = {
            "click_x": x,
            "click_y": y,
            "is_drag": self._is_drag_capture,
        }

        logger.info(
            "Selection received: (%d, %d) %dx%d (drag=%s)",
            x, y, w, h, self._is_drag_capture,
        )

        self._run_capture_pipeline()

    def on_toolbar_action(self, action: str) -> None:
        """Route toolbar button clicks to appropriate handlers.

        Args:
            action: The action name string from toolbar button.
        """
        logger.debug("Toolbar action: %s (phase=%s)", action, self._phase.name)

        if action == "confirm":
            tag_data = self._controller.get_tag_data()
            if tag_data is not None:
                self.on_tag_confirmed(tag_data)
        elif action == "dismiss":
            tag_data = self._controller.get_tag_data()
            self.on_tag_dismissed(tag_data or {})
        elif action == "redraw":
            self._handle_redraw()
        elif action == "accept_ai_bbox":
            self._accept_bbox()
        elif action == "keep_drag":
            self._handle_keep_drag()
        elif action == "yes":
            self._handle_validate_yes()
        elif action == "edit_tags":
            self._handle_edit_tags()
        elif action == "recapture":
            self._handle_recapture()
        elif action == "retry":
            self._handle_retry()
        elif action == "pause":
            logger.info("Pause requested")
        elif action == "undo_last":
            self._handle_undo_last()
        elif action == "look_here":
            self._handle_look_here()
        elif action == "add_wait":
            self._handle_add_wait()
        elif action == "add_loop":
            self._handle_add_loop()
        elif action == "add_prompt":
            self._handle_add_prompt()
        else:
            logger.debug("Unhandled toolbar action: %s", action)

    def on_tag_confirmed(self, data: dict) -> None:
        """Handle tag dialog confirmation -- start dry-run countdown.

        Merges tag data with bbox data, dismisses tag dialog, and
        transitions to COUNTDOWN for dry-run execution.

        Args:
            data: Form data dict from the tag dialog.
        """
        if self._current_step is None:
            logger.warning("Tag confirmed but no current step")
            return

        # Merge step data with tag dialog data
        # Remap "action_type" from tag dialog form to "action" used by
        # the rest of the pipeline (dry-run, save, runner).  The dialog's
        # get_form_data() returns "action_type"; every consumer expects
        # "action".  Without this, the user's dropdown selection is silently
        # ignored and the VLM default ("click") is kept.
        if "action_type" in data and "action" not in data:
            data["action"] = data.pop("action_type")

        self._current_step = {
            **self._current_step,
            "tag_data": data,
            "bbox": self._current_bbox,
            # P1.6 fix: capture this step's screenshot so snippets are
            # cropped from the correct frame, not the final one
            "screenshot": self._screenshot,
        }

        self._controller.dismiss_tag_dialog()

        logger.info(
            "Tag confirmed: label=%s, type=%s, action=%s",
            data.get("label", "?"),
            data.get("element_type", "?"),
            data.get("action", "click"),
        )
        logger.info("Merged action: %s", data.get("action"))

        # click_drag needs a second bbox capture for the drag target
        action = data.get("action", "click")
        if action == "click_drag":
            self._set_phase(RecordPhase.AWAITING_DRAG_TARGET)
            self._controller.set_toolbar_mode(ToolbarMode.RECORDING)
            logger.info("click_drag: awaiting drag target click")
            return

        # Reset look_here flag after tag confirm
        self._is_look_here = False

        # Transition to COUNTDOWN
        self._set_phase(RecordPhase.COUNTDOWN)
        self._controller.set_toolbar_mode(ToolbarMode.DRY_RUN)

        widget = self._controller.show_countdown(3)
        if widget is not None:
            # Updated: disconnect before connect to prevent signal accumulation
            try:
                widget.countdown_finished.disconnect(self._on_countdown_finished)
            except TypeError:
                pass  # No existing connection -- that's fine
            widget.countdown_finished.connect(self._on_countdown_finished)

    def on_tag_dismissed(self, data: dict) -> None:
        """Handle tag dialog dismissal -- discard step, return to awaiting.

        Args:
            data: Form data dict (discarded).
        """
        self._current_step = None
        self._current_bbox = None
        self._controller.dismiss_tag_dialog()
        self._clear_scan()  # Updated: remove scan highlight — dismissed tag means step discarded — 2026-04-03
        self._set_phase(RecordPhase.AWAITING_CLICK)
        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)

        logger.info("Tag dismissed, returning to AWAITING_CLICK")

    def on_save_requested(self) -> None:
        """Handle Ctrl+Q -- save all accumulated steps.

        Guarded by ``_save_in_progress`` to prevent signal accumulation
        from triggering multiple concurrent save threads (Bug #12).
        """
        if self._save_in_progress:
            logger.debug("Save already in progress, ignoring duplicate request")
            return

        logger.info("Save requested with %d steps", len(self._steps))

        if len(self._steps) == 0:
            # No steps -- close silently
            self._controller.close()
            if self._on_session_complete is not None:
                self._on_session_complete(False)
            return

        self._save_in_progress = True

        # Save in background thread
        thread = threading.Thread(
            target=self._save_routine,
            daemon=True,
        )
        thread.start()

    def on_abort_requested(self) -> None:
        """Handle ESC -- show abort confirm or close silently."""
        if self._steps:
            logger.info(
                "Abort requested with %d unsaved steps", len(self._steps),
            )
            panel = self._controller.show_abort_confirm(len(self._steps))
            if panel is not None:
                # Disconnect before connect to prevent signal accumulation
                try:
                    panel.discard_clicked.disconnect(self._on_abort_confirmed)
                except TypeError:
                    pass  # No existing connection -- that's fine
                try:
                    panel.keep_clicked.disconnect(self._on_abort_cancelled)
                except TypeError:
                    pass  # No existing connection -- that's fine
                panel.discard_clicked.connect(self._on_abort_confirmed)
                panel.keep_clicked.connect(self._on_abort_cancelled)
        else:
            logger.info("Abort requested (no steps recorded)")
            self._controller.close()
            if self._on_session_complete is not None:
                self._on_session_complete(False)

    # ------------------------------------------------------------------
    # Private: toolbar quick-add handlers
    # ------------------------------------------------------------------

    def _handle_look_here(self) -> None:
        """Enter region-drag mode for observation actions (read, snip_and_search)."""
        if self._phase != RecordPhase.AWAITING_CLICK:
            logger.debug("Look Here only available in AWAITING_CLICK")
            return
        self._set_phase(RecordPhase.AWAITING_REGION_DRAG)
        if hasattr(self._controller, "set_cursor_crosshair"):
            self._controller.set_cursor_crosshair()
        logger.info("Look Here: waiting for region drag")

    def _handle_add_wait(self) -> None:
        """Open wait condition mini-dialog from toolbar."""
        if self._phase != RecordPhase.AWAITING_CLICK:
            return
        self._set_phase(RecordPhase.WAIT_CONFIGURING)

        dialog = self._controller.show_wait_dialog()
        if dialog is not None:
            # Disconnect before connect to prevent signal accumulation
            try:
                dialog.confirmed.disconnect(self._on_wait_configured)
            except TypeError:
                pass  # No existing connection -- that's fine
            try:
                dialog.dismissed.disconnect(self._on_wait_dismissed)
            except TypeError:
                pass  # No existing connection -- that's fine
            dialog.confirmed.connect(self._on_wait_configured)
            dialog.dismissed.connect(self._on_wait_dismissed)
        logger.info("Add Wait: showing configuration dialog")

    def _on_wait_configured(self, config: dict) -> None:
        """Handle wait dialog confirmation -- create wait step.

        Args:
            config: Wait configuration dict from WaitDialog.
        """
        self._controller.hide_wait_dialog()
        self._current_step = {
            "click_x": 0, "click_y": 0,
            "is_drag": False, "is_wait": True,
            "tag_data": {
                "action": "wait",
                "element_type": "unknown",
                "label": f"Wait: {config.get('condition_type', 'timer')}",
                "caption": f"Timeout: {config.get('timeout', 30)}s",
                "confidence": 1.0,
            },
            "wait_definition": config,
            "bbox": (0, 0, 0, 0),
            "dry_run_passed": True,
        }
        self._current_bbox = (0, 0, 0, 0)
        self._steps.append(self._current_step)
        logger.info("Wait step added: %s", config)
        self._current_step = None
        self._set_phase(RecordPhase.SUCCESS_FLASH)
        QTimer.singleShot(500, self._loop_back_to_awaiting)

    def _on_wait_dismissed(self) -> None:
        """Handle wait dialog cancellation."""
        self._controller.hide_wait_dialog()
        self._set_phase(RecordPhase.AWAITING_CLICK)
        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)

    def _handle_add_loop(self) -> None:
        """Open loop definition dialog from toolbar."""
        if self._phase != RecordPhase.AWAITING_CLICK:
            return
        if not self._steps:
            logger.info("Add Loop: no steps recorded yet, nothing to loop")
            return

        self._set_phase(RecordPhase.LOOP_DEFINING)

        dialog = self._controller.show_loop_dialog(self._steps)
        if dialog is not None:
            # Disconnect before connect to prevent signal accumulation
            try:
                dialog.confirmed.disconnect(self._on_loop_configured)
            except TypeError:
                pass  # No existing connection -- that's fine
            try:
                dialog.dismissed.disconnect(self._on_loop_dismissed)
            except TypeError:
                pass  # No existing connection -- that's fine
            dialog.confirmed.connect(self._on_loop_configured)
            dialog.dismissed.connect(self._on_loop_dismissed)
        logger.info(
            "Add Loop: showing definition dialog with %d steps",
            len(self._steps),
        )

    def _on_loop_configured(self, config: dict) -> None:
        """Handle loop dialog confirmation -- create loop step.

        Uses node_id references (not step indices) for body steps.
        This ensures stability across future editing (Phase 8).
        Indices are stored temporarily and resolved to node_ids at save time.

        Args:
            config: Loop config dict from LoopDialog.confirmed signal.
        """
        self._controller.hide_loop_dialog()

        start_idx, end_idx = config.get("body_step_range", (0, 0))

        # Store step indices temporarily; resolve to node_ids at save time
        body_step_indices = list(range(start_idx, end_idx + 1))

        exit_condition = config.get("exit_condition", {})

        self._current_step = {
            "click_x": 0, "click_y": 0,
            "is_drag": False, "is_loop": True,
            "tag_data": {
                "action": "loop",
                "element_type": "unknown",
                "label": f"Loop steps {start_idx + 1}-{end_idx + 1}",
                "caption": f"Until: {exit_condition.get('type', 'n_iterations')}",
                "confidence": 1.0,
            },
            "loop_definition": {
                "body_step_indices": body_step_indices,
                "exit_condition": exit_condition,
            },
            "bbox": (0, 0, 0, 0),
            "dry_run_passed": True,  # No loop-level dry-run per user decision
        }
        self._steps.append(self._current_step)
        logger.info(
            "Loop step added: steps %d-%d, exit=%s",
            start_idx + 1, end_idx + 1,
            exit_condition.get("type", "?"),
        )
        self._current_step = None
        self._set_phase(RecordPhase.SUCCESS_FLASH)
        QTimer.singleShot(500, self._loop_back_to_awaiting)

    def _on_loop_dismissed(self) -> None:
        """Handle loop dialog cancellation."""
        self._controller.hide_loop_dialog()
        self._set_phase(RecordPhase.AWAITING_CLICK)
        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)

    def _handle_add_prompt(self) -> None:
        """Open prompt question mini-dialog from toolbar."""
        if self._phase != RecordPhase.AWAITING_CLICK:
            return
        self._set_phase(RecordPhase.PROMPT_CONFIGURING)

        dialog = self._controller.show_prompt_dialog()
        if dialog is not None:
            # Disconnect before connect to prevent signal accumulation
            try:
                dialog.confirmed.disconnect(self._on_prompt_configured)
            except TypeError:
                pass  # No existing connection -- that's fine
            try:
                dialog.dismissed.disconnect(self._on_prompt_dismissed)
            except TypeError:
                pass  # No existing connection -- that's fine
            dialog.confirmed.connect(self._on_prompt_configured)
            dialog.dismissed.connect(self._on_prompt_dismissed)
        logger.info("Add Prompt: showing question dialog")

    def _on_prompt_configured(self, config: dict) -> None:
        """Handle prompt dialog confirmation -- create prompt_user step.

        Args:
            config: Config dict with question_text from PromptDialog.
        """
        self._controller.hide_prompt_dialog()
        question = config.get("question_text", "")
        self._current_step = {
            "click_x": 0, "click_y": 0,
            "is_drag": False,
            "tag_data": {
                "action": "prompt_user",
                "element_type": "unknown",
                "label": "Prompt User",
                "caption": question[:50],
                "confidence": 1.0,
                "question_text": question,
            },
            "bbox": (0, 0, 0, 0),
            "dry_run_passed": True,
        }
        self._steps.append(self._current_step)
        logger.info("Prompt user step added: %s", question[:50])
        self._current_step = None
        self._set_phase(RecordPhase.SUCCESS_FLASH)
        QTimer.singleShot(500, self._loop_back_to_awaiting)

    def _on_prompt_dismissed(self) -> None:
        """Handle prompt dialog cancellation."""
        self._controller.hide_prompt_dialog()
        self._set_phase(RecordPhase.AWAITING_CLICK)
        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)

    def _run_capture_pipeline_for_region(
        self, x: int, y: int, w: int, h: int,
    ) -> None:
        """Run capture for a pre-selected region (Look Here flow).

        Args:
            x: Left edge of the region.
            y: Top edge of the region.
            w: Width of the region.
            h: Height of the region.
        """
        self._set_phase(RecordPhase.CAPTURING)
        self._controller.hide_for_capture()
        cfg = get_config()
        delay_ms = cfg.get("overlay", {}).get("capture_delay_ms", 80)

        # P8.1 fix: use QTimer instead of time.sleep() to keep Qt event loop responsive
        def _after_compositor_wait() -> None:
            try:
                from core.capture import screenshot_full

                self._screenshot = screenshot_full()
            except Exception as e:
                logger.error("Screenshot failed: %s", e)
                self._screenshot = None
                self._controller.show_after_capture()
                self._set_phase(RecordPhase.AWAITING_CLICK)
                return
            self._controller.show_after_capture()
            self._set_phase(RecordPhase.VLM_ANALYZING)
            if hasattr(self._controller, "start_card_glow_pulse"):
                self._controller.start_card_glow_pulse()
            if self._screenshot is not None:
                self._start_vlm(self._screenshot, (x, y, w, h))

        QTimer.singleShot(delay_ms, _after_compositor_wait)

    def _handle_drag_target_selection(
        self, x: int, y: int, w: int, h: int,
    ) -> None:
        """Capture drag target bbox for click_drag step.

        Args:
            x: Left edge (or click X).
            y: Top edge (or click Y).
            w: Width of selection (0 for clicks).
            h: Height of selection (0 for clicks).
        """
        self._set_phase(RecordPhase.CAPTURING)
        self._controller.hide_for_capture()
        cfg = get_config()
        delay_ms = cfg.get("overlay", {}).get("capture_delay_ms", 80)

        # P8.1 fix: use QTimer instead of time.sleep() to keep Qt event loop responsive
        # P8.2 fix: store screenshot return value for drag-target visual reference
        def _after_compositor_wait() -> None:
            try:
                from core.capture import screenshot_full

                target_screenshot = screenshot_full()
            except Exception as e:
                logger.error("Target screenshot failed: %s", e)
                self._controller.show_after_capture()
                self._set_phase(RecordPhase.AWAITING_CLICK)
                return
            self._controller.show_after_capture()

            # Use click position or drag center as target bbox
            if w > 0 and h > 0:
                target_bbox = {"x": x, "y": y, "w": w, "h": h}
            else:
                target_bbox = {"x": x - 30, "y": y - 30, "w": 60, "h": 60}

            # Store drag target and its screenshot in current step
            if self._current_step is not None:
                node_id = self._current_step.get("node_id", str(uuid4()))
                self._current_step["drag_target"] = {
                    "bbox": target_bbox,
                    "screenshot": target_screenshot,
                    "anchors": {
                        "visual_match": f"snippets/{node_id}_target.png",
                    },
                }
                logger.info("Drag target captured: %s", target_bbox)

            # Now proceed to countdown/dry-run
            self._set_phase(RecordPhase.COUNTDOWN)
            self._controller.set_toolbar_mode(ToolbarMode.DRY_RUN)
            widget = self._controller.show_countdown(3)
            if widget is not None:
                # Updated: disconnect before connect to prevent signal accumulation
                try:
                    widget.countdown_finished.disconnect(self._on_countdown_finished)
                except TypeError:
                    pass  # No existing connection -- that's fine
                widget.countdown_finished.connect(self._on_countdown_finished)

        QTimer.singleShot(delay_ms, _after_compositor_wait)

    # ------------------------------------------------------------------
    # Private: pipeline methods
    # ------------------------------------------------------------------

    def _clear_scan(self) -> None:
        """Remove the scan highlight from the overlay view.

        Safely calls view.remove_scan() to tear down the ScanLayer
        graphics item and unregister its animation tick.
        """
        # Updated: new helper — centralises scan cleanup after tag/validate/recapture — 2026-04-03
        if (
            self._controller._view is not None
            and hasattr(self._controller._view, "remove_scan")
        ):
            self._controller._view.remove_scan()

    def _set_phase(self, phase: RecordPhase) -> None:
        """Update the current phase and log the transition.

        Args:
            phase: The new RecordPhase to transition to.
        """
        old = self._phase
        self._phase = phase
        logger.debug("Phase: %s -> %s", old.name, phase.name)

    def _run_capture_pipeline(self) -> None:
        """Execute capture sequence: hide overlay, screenshot, detect."""
        self._set_phase(RecordPhase.CAPTURING)
        self._controller.hide_for_capture()

        cfg = get_config()
        delay_ms = cfg.get("overlay", {}).get("capture_delay_ms", 80)

        # P8.1 fix: use QTimer instead of time.sleep() to keep Qt event loop responsive
        def _after_compositor_wait() -> None:
            try:
                from core.capture import screenshot_full

                self._screenshot = screenshot_full()
            except Exception as e:
                logger.error("Screenshot failed: %s", e)
                self._screenshot = None
                self._controller.show_after_capture()
                self._set_phase(RecordPhase.AWAITING_CLICK)
                return

            self._controller.show_after_capture()

            self._set_phase(RecordPhase.DETECTING)

            # Start scan animation immediately at selection rect (don't wait for AI)
            if self._is_drag_capture and self._original_drag_rect is not None:
                dx, dy, dw, dh = self._original_drag_rect
                self._controller.start_scan(dx, dy, dw, dh)
            else:
                # For clicks, scan a 60x60 region around the click point
                self._controller.start_scan(
                    self._click_x - 30, self._click_y - 30, 60, 60,
                )

            # Start card glow pulsing during detection
            if hasattr(self._controller, "start_card_glow_pulse"):
                self._controller.start_card_glow_pulse()

            # Start background detection thread
            self._start_detection(
                self._screenshot,
                self._click_x,
                self._click_y,
                self._is_drag_capture,
            )

        QTimer.singleShot(delay_ms, _after_compositor_wait)

    def _start_detection(
        self,
        screenshot: np.ndarray,
        click_x: int,
        click_y: int,
        is_drag: bool,
    ) -> None:
        """Run detection in a background thread.

        For clicks: tries click-local crop first, falls back to full screen.
        For drags: attempts IoU refinement against AI-detected bboxes.

        Args:
            screenshot: Full screen capture as BGR numpy array.
            click_x: Click X coordinate.
            click_y: Click Y coordinate.
            is_drag: Whether this was a drag selection.
        """
        thread = threading.Thread(
            target=self._detection_worker,
            args=(screenshot, click_x, click_y, is_drag),
            daemon=True,
        )
        thread.start()

    def _detection_worker(
        self,
        screenshot: np.ndarray,
        click_x: int,
        click_y: int,
        is_drag: bool,
    ) -> None:
        """Background worker for detection. Emits result via bridge.

        Args:
            screenshot: Full screen capture.
            click_x: Click X coordinate.
            click_y: Click Y coordinate.
            is_drag: Whether this was a drag selection.
        """
        result: dict[str, Any] = {
            "bbox": None,
            "type_guess": "unknown",
            "florence_caption": "",
            "candidates": [],
        }

        try:
            from core.detection import get_detector

            detector = get_detector()
        except (ImportError, OSError, ValueError) as e:
            logger.debug("Detection not available: %s", e)
            self._bridge.detection_ready.emit(result)
            return

        try:
            if is_drag and self._original_drag_rect is not None:
                # Drag capture: run IoU refinement
                dx, dy, dw, dh = self._original_drag_rect
                candidates = detector.detect(screenshot)
                result["candidates"] = candidates

                # Find best IoU match
                best_iou = 0.0
                best_rect = None
                best_candidate = None
                for c in candidates:
                    r = c.get("rect", {})
                    rx, ry = r.get("x", 0), r.get("y", 0)
                    rw, rh = r.get("w", 0), r.get("h", 0)
                    ix1 = max(dx, rx)
                    iy1 = max(dy, ry)
                    ix2 = min(dx + dw, rx + rw)
                    iy2 = min(dy + dh, ry + rh)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = dw * dh + rw * rh - inter
                        iou = inter / union if union > 0 else 0
                        if iou > best_iou:
                            best_iou = iou
                            best_rect = r
                            best_candidate = c

                if best_rect and best_iou > 0.3:
                    result["bbox"] = {
                        "x": best_rect["x"],
                        "y": best_rect["y"],
                        "w": best_rect["w"],
                        "h": best_rect["h"],
                    }
                    result["type_guess"] = best_candidate.get(
                        "type_guess", "unknown",
                    )
                    result["florence_caption"] = best_candidate.get(
                        "florence_caption", "",
                    )
                else:
                    # No IoU match: use drag rect as-is
                    result["bbox"] = {
                        "x": dx, "y": dy, "w": dw, "h": dh,
                    }
            else:
                # Click capture: try local crop first
                sh, sw = screenshot.shape[:2]
                radius = 120
                x1 = max(0, click_x - radius)
                y1 = max(0, click_y - radius)
                x2 = min(sw, click_x + radius)
                y2 = min(sh, click_y + radius)
                crop = screenshot[y1:y2, x1:x2]

                if crop.size > 0:
                    candidates = detector.detect(crop)
                    if candidates:
                        # Find closest to click in crop coords
                        cx_local = click_x - x1
                        cy_local = click_y - y1
                        best = None
                        best_dist = float("inf")
                        for c in candidates:
                            r = c["rect"]
                            mid_x = r["x"] + r["w"] / 2
                            mid_y = r["y"] + r["h"] / 2
                            dist = (
                                (mid_x - cx_local) ** 2
                                + (mid_y - cy_local) ** 2
                            ) ** 0.5
                            if dist < best_dist:
                                best = c
                                best_dist = dist

                        if best is not None:
                            r = best["rect"]
                            result["bbox"] = {
                                "x": r["x"] + x1,
                                "y": r["y"] + y1,
                                "w": r["w"],
                                "h": r["h"],
                            }
                            result["type_guess"] = best.get(
                                "type_guess", "unknown",
                            )
                            result["florence_caption"] = best.get(
                                "florence_caption", "",
                            )
                            result["candidates"] = [
                                {
                                    **c,
                                    "rect": {
                                        "x": c["rect"]["x"] + x1,
                                        "y": c["rect"]["y"] + y1,
                                        "w": c["rect"]["w"],
                                        "h": c["rect"]["h"],
                                    },
                                }
                                for c in candidates
                            ]

                # Fall back to full-screen if no local result
                if result["bbox"] is None:
                    candidates = detector.detect(screenshot)
                    result["candidates"] = candidates
                    if candidates:
                        # Find smallest bbox containing click point
                        best = None
                        best_area = float("inf")
                        for c in candidates:
                            r = c["rect"]
                            if (
                                r["x"] <= click_x <= r["x"] + r["w"]
                                and r["y"] <= click_y <= r["y"] + r["h"]
                            ):
                                area = r["w"] * r["h"]
                                if area < best_area:
                                    best = c
                                    best_area = area

                        if best is not None:
                            r = best["rect"]
                            result["bbox"] = {
                                "x": r["x"],
                                "y": r["y"],
                                "w": r["w"],
                                "h": r["h"],
                            }
                            result["type_guess"] = best.get(
                                "type_guess", "unknown",
                            )
                            result["florence_caption"] = best.get(
                                "florence_caption", "",
                            )
        except Exception as e:
            logger.warning("Detection worker error: %s", e)

        # Florence-2 caption enrichment — runs on the detected element crop
        bbox = result.get("bbox")
        if bbox and not result.get("florence_caption"):
            try:
                from core.florence import caption_crop

                bx, by = bbox["x"], bbox["y"]
                bw, bh = bbox["w"], bbox["h"]
                sh, sw = screenshot.shape[:2]
                cx1 = max(0, bx)
                cy1 = max(0, by)
                cx2 = min(sw, bx + bw)
                cy2 = min(sh, by + bh)
                element_crop = screenshot[cy1:cy2, cx1:cx2]
                if element_crop.size > 0:
                    florence_caption = caption_crop(element_crop)
                    if florence_caption:
                        result["florence_caption"] = florence_caption
                        logger.debug("Florence-2 caption: %r", florence_caption)
            except (ImportError, RuntimeError, Exception) as e:
                logger.debug("Florence-2 captioning unavailable: %s", e)

        logger.debug(
            "Detection result: bbox=%s, florence_caption=%r, type_guess=%r",
            result.get("bbox"),
            result.get("florence_caption"),
            result.get("type_guess"),
        )
        self._bridge.detection_ready.emit(result)

    def _on_detection_ready(self, result: dict) -> None:
        """Handle detection result on main thread.

        Stops card glow, shows scan animation or default bbox,
        and transitions to BBOX_EDITING or VLM_ANALYZING.

        Args:
            result: Detection result dict with bbox, type_guess, etc.
        """
        # Stop card glow pulsing
        if hasattr(self._controller, "stop_card_glow_pulse"):
            self._controller.stop_card_glow_pulse()

        bbox = result.get("bbox")

        if bbox is not None:
            bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            self._current_bbox = (bx, by, bw, bh)

            # Snap the already-running scan to the AI-fitted bbox
            self._controller.finish_scan(bx, by, bw, bh)

            self._set_phase(RecordPhase.BBOX_EDITING)

            if self._is_drag_capture:
                self._controller.set_toolbar_mode(ToolbarMode.BBOX_EDITING)
            else:
                # For click captures, auto-accept the bbox
                self._accept_bbox()
        else:
            # No bbox found: use drag rect or default 60x60 around click
            if self._is_drag_capture and self._original_drag_rect is not None:
                bx, by, bw, bh = self._original_drag_rect
            else:
                default_size = 60
                bx = self._click_x - default_size // 2
                by = self._click_y - default_size // 2
                bw, bh = default_size, default_size
            self._current_bbox = (bx, by, bw, bh)

            # Snap scan to the selection rect (ends the laser loop)
            self._controller.finish_scan(bx, by, bw, bh)

            self._set_phase(RecordPhase.BBOX_EDITING)
            self._accept_bbox()

    def _accept_bbox(self) -> None:
        """Accept the current bbox and start VLM analysis."""
        self._set_phase(RecordPhase.VLM_ANALYZING)

        # Start card glow pulsing during VLM analysis
        if hasattr(self._controller, "start_card_glow_pulse"):
            self._controller.start_card_glow_pulse()

        # Show tag dialog immediately in loading state (spinner)
        if self._current_bbox is not None:
            bx, by, bw, bh = self._current_bbox
            element_rect = QRectF(bx, by, bw, bh)
        else:
            element_rect = QRectF(
                self._click_x - 30, self._click_y - 30, 60, 60,
            )
        self._controller.show_tag_dialog(element_rect, vlm_data=None)

        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)

        if self._screenshot is not None and self._current_bbox is not None:
            self._start_vlm(self._screenshot, self._current_bbox)
        else:
            logger.warning("Cannot start VLM: missing screenshot or bbox")
            self._bridge.vlm_failed.emit("Missing screenshot or bbox")

    def _handle_keep_drag(self) -> None:
        """Reject AI bbox and keep user's original drag rect.

        Called when user clicks "Keep My Drag" during BBOX_EDITING.
        Reverts to the original drag rect and proceeds to VLM.
        """
        if self._original_drag_rect is None:
            logger.warning("Keep drag requested but no original drag rect")
            return

        self._current_bbox = self._original_drag_rect

        # Finish scan at the original drag rect coordinates
        dx, dy, dw, dh = self._original_drag_rect
        self._controller.finish_scan(dx, dy, dw, dh)

        logger.info(
            "Keeping original drag rect: (%d, %d) %dx%d", dx, dy, dw, dh,
        )

        self._accept_bbox()

    def _start_vlm(
        self,
        screenshot: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Run VLM analysis in a background thread.

        Crops the screenshot with 30% padding around the bbox and sends
        to the VLM.  Retries once on failure with shorter timeout.

        Args:
            screenshot: Full screen capture.
            bbox: Bounding box as (x, y, w, h).
        """
        thread = threading.Thread(
            target=self._vlm_worker,
            args=(screenshot, bbox),
            daemon=True,
        )
        thread.start()

    def _vlm_worker(
        self,
        screenshot: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Background worker for VLM analysis. Emits via bridge.

        Args:
            screenshot: Full screen capture.
            bbox: Bounding box as (x, y, w, h).
        """
        bx, by, bw, bh = bbox
        sh, sw = screenshot.shape[:2]

        # 30% padded crop
        pad_x = int(bw * 0.30)
        pad_y = int(bh * 0.30)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(sw, bx + bw + pad_x)
        y2 = min(sh, by + bh + pad_y)
        crop = screenshot[y1:y2, x1:x2]

        if crop.size == 0:
            self._bridge.vlm_failed.emit("Empty crop region")
            return

        try:
            from core.vision import analyze_crop_array

            result = analyze_crop_array(crop, "Identify this UI element")
            if result is not None:
                self._bridge.vlm_ready.emit(result)
                return
        except Exception as e:
            logger.warning("VLM first attempt failed: %s", e)

        # Retry once
        try:
            from core.vision import analyze_crop_array

            result = analyze_crop_array(crop, "Identify this UI element")
            if result is not None:
                self._bridge.vlm_ready.emit(result)
                return
        except Exception as e:
            logger.warning("VLM retry failed: %s", e)

        self._bridge.vlm_failed.emit("VLM analysis failed after retry")

    def _on_vlm_ready(self, result: dict) -> None:
        """Handle successful VLM analysis on main thread.

        Populates the already-visible tag dialog with VLM data.

        Args:
            result: VLM analysis result dict.
        """
        logger.info("VLM ready, populating tag dialog: %s", result.get("element_type", "?"))
        # Stop card glow pulsing
        if hasattr(self._controller, "stop_card_glow_pulse"):
            self._controller.stop_card_glow_pulse()

        self._set_phase(RecordPhase.TAG_DIALOG)

        vlm_data = {
            "element_type": result.get("element_type", "unknown"),
            "label": result.get("label_guess", ""),
            "caption": result.get("florence_caption", ""),
            "confidence": result.get("confidence", 0.0),
            "ocr_text": result.get("ocr_text"),
        }
        if self._is_look_here:
            vlm_data["action_type"] = "read"

        logger.debug(
            "VLM data for tag dialog: caption=%r, label=%r, type=%r",
            vlm_data.get("caption"),
            vlm_data.get("label"),
            vlm_data.get("element_type"),
        )
        self._controller.update_tag_dialog_data(vlm_data)
        self._controller.set_toolbar_mode(ToolbarMode.TAG_OPEN)

    def _on_vlm_failed(self, error: str) -> None:
        """Handle VLM failure on main thread.

        Populates the already-visible tag dialog with empty data so user
        can tag manually.

        Args:
            error: Error message string.
        """
        # Stop card glow pulsing
        if hasattr(self._controller, "stop_card_glow_pulse"):
            self._controller.stop_card_glow_pulse()

        self._set_phase(RecordPhase.TAG_DIALOG)

        logger.warning("VLM failed: %s — populating tag dialog for manual entry", error)

        partial_data = {
            "element_type": "unknown",
            "label": "",
            "caption": "VLM unavailable",
            "confidence": 0.0,
            "ocr_text": None,
        }
        if self._is_look_here:
            partial_data["action_type"] = "read"

        self._controller.update_tag_dialog_data(partial_data)
        self._controller.set_toolbar_mode(ToolbarMode.TAG_OPEN)

    # ------------------------------------------------------------------
    # Private: dry-run execution
    # ------------------------------------------------------------------

    def _on_countdown_finished(self) -> None:
        """Handle countdown completion -- execute dry-run action."""
        self._controller.hide_countdown()
        self._set_phase(RecordPhase.EXECUTING)
        # Switch to purple shimmer + click-through during dry-run
        if self._controller._view is not None:
            from recorder.overlay.state import OverlayState
            self._controller._view.apply_state(OverlayState.REPLAYING)
        self._controller.set_click_through(True)

        # Nuclear option: hide the overlay entirely during dry-run execution
        # so it CANNOT intercept clicks (e.g. second press of a double-click).
        # The 150ms pre-delay in _execute_dry_run gives time for the hide to
        # propagate through the compositor before any input is sent.
        logger.info("DRY-RUN: calling hide_for_capture()")
        self._controller.hide_for_capture()
        logger.info("DRY-RUN: hide_for_capture() returned")

        # Flush Qt event loop + Win32 compositor to ensure overlay is truly gone
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        import sys
        if sys.platform == "win32":
            try:
                from recorder.overlay.platform_win32 import dwm_flush
                dwm_flush()
            except Exception:
                import time as _time
                _time.sleep(0.1)  # fallback: 100ms for compositor
        else:
            import time as _time
            _time.sleep(0.1)  # non-Windows: sleep for compositor

        # Execute action in background thread
        thread = threading.Thread(
            target=self._execute_dry_run,
            daemon=True,
        )
        thread.start()

    def _execute_dry_run(self) -> None:
        """Background worker: execute the recorded action based on action type.

        Dispatches to the correct executor function based on tag_data["action"].
        Emits execution_complete signal via PipelineBridge (thread-safe
        Qt AutoConnection).
        """
        # Safety margin after compositor flush (hide is already processed)
        logger.info("DRY-RUN: pre-delay starting (0.05s)")
        time.sleep(0.05)
        logger.info("DRY-RUN: pre-delay done, executing action")

        try:
            if self._current_bbox is None or self._current_step is None:
                logger.warning("No bbox/step for dry-run execution")
                return

            bx, by, bw, bh = self._current_bbox
            center_x = bx + bw // 2
            center_y = by + bh // 2
            tag_data = self._current_step.get("tag_data", {})
            action = tag_data.get("action", "click")
            logger.info("DRY-RUN: pre-delay done, executing action '%s'", action)

            # --- CLIP validation: verify element can be re-found ---
            if action in ("click", "double_click", "right_click", "type", "click_drag"):
                import tempfile
                import shutil

                import cv2

                tmp_dir: Path | None = None
                try:
                    from core.locate import locate_element_from_step
                    from core.capture import screenshot_region
                    from core.embeddings import generate_embedding

                    tmp_dir = Path(tempfile.mkdtemp(prefix="ocsd_dryrun_"))
                    tmp_id = uuid4().hex[:12]

                    # Create temp snippet + embedding
                    (tmp_dir / "snippets").mkdir(exist_ok=True)
                    (tmp_dir / "embeddings").mkdir(exist_ok=True)

                    snippet = screenshot_region(bx, by, bw, bh)
                    snippet_path = f"snippets/{tmp_id}.png"
                    cv2.imwrite(str(tmp_dir / snippet_path), snippet)

                    emb = generate_embedding(snippet)
                    emb_path = f"embeddings/{tmp_id}.npy"
                    np.save(str(tmp_dir / emb_path), emb)

                    # Build minimal step dict for locate
                    screen_w = self._screenshot.shape[1] if self._screenshot is not None else 1920
                    screen_h = self._screenshot.shape[0] if self._screenshot is not None else 1080
                    cx, cy = bx + bw // 2, by + bh // 2

                    locate_step = {
                        "node_id": tmp_id,
                        "snippet_path": snippet_path,
                        "embedding_path": emb_path,
                        "label": tag_data.get("label", ""),
                        "element_type": tag_data.get("element_type", "unknown"),
                        "anchors": {
                            "visual_match": snippet_path,
                            "ocr_text": tag_data.get("ocr_text"),
                            "position_pct": {
                                "x_pct": round(cx / screen_w, 4),
                                "y_pct": round(cy / screen_h, 4),
                            },
                        },
                    }

                    locate_result = locate_element_from_step(
                        locate_step, tmp_dir, skip_vlm=True,
                    )

                    if locate_result and locate_result.point:
                        logger.info(
                            "DRY-RUN: CLIP validation found element at (%d, %d) "
                            "via %s (confidence=%.2f)",
                            locate_result.point.x, locate_result.point.y,
                            locate_result.method, locate_result.confidence,
                        )
                        center_x = locate_result.point.x
                        center_y = locate_result.point.y
                    else:
                        logger.warning(
                            "DRY-RUN: CLIP validation failed to re-locate element, "
                            "using bbox center",
                        )

                except Exception as e:
                    logger.warning(
                        "DRY-RUN: CLIP validation error: %s, using bbox center", e,
                    )
                finally:
                    if tmp_dir is not None and tmp_dir.exists():
                        shutil.rmtree(tmp_dir, ignore_errors=True)

            from core.executor import (
                click as exec_click,
                double_click as exec_double_click,
                right_click as exec_right_click,
                type_text as exec_type_text,
                scroll as exec_scroll,
                press_enter as exec_press_enter,
            )

            match action:
                case "click":
                    exec_click(center_x, center_y)
                case "double_click":
                    exec_double_click(center_x, center_y)
                case "right_click":
                    exec_right_click(center_x, center_y)
                case "type":
                    exec_click(center_x, center_y)
                    text = tag_data.get("text_to_type", "")
                    if text:
                        exec_type_text(text)
                    if tag_data.get("press_enter", False):
                        exec_press_enter()
                case "scroll":
                    direction_raw = tag_data.get("direction_amount", "down 3")
                    parts = direction_raw.strip().split()
                    direction = parts[0] if parts else "down"
                    amount = int(parts[1]) if len(parts) > 1 else 3
                    exec_scroll(center_x, center_y, direction, amount)
                case "click_drag":
                    from core.executor import drag as exec_drag

                    drag_target = self._current_step.get("drag_target", {})
                    target_bbox = drag_target.get("bbox", {})
                    tx = target_bbox.get("x", 0) + target_bbox.get("w", 0) // 2
                    ty = target_bbox.get("y", 0) + target_bbox.get("h", 0) // 2
                    exec_drag(center_x, center_y, tx, ty)
                case "read" | "snip_and_search":
                    logger.info(
                        "Dry-run %s: observation step, no action", action,
                    )
                case "select_all_extract":
                    from core.executor import select_all_extract

                    select_all_extract()
                case "wait" | "loop":
                    logger.info(
                        "Dry-run %s: skipped (no dry-run for flow actions)",
                        action,
                    )
                case "prompt_user":
                    from core.executor import prompt_user_blocking

                    prompt_user_blocking(
                        tag_data.get("question_text", ""),
                        dry_run=True,
                    )
                    logger.info("Dry-run prompt_user: would pause and wait for /respond")
                case _:
                    # P8.3 fix: skip execution for unknown actions instead of falling through to click
                    logger.warning(
                        "Unknown action type for dry-run: %s — skipping execution", action,
                    )

            logger.info(
                "Dry-run %s executed at (%d, %d)", action, center_x, center_y,
            )

            # Updated: allow OS to fully process all input events before restoring overlay
            logger.info("DRY-RUN: action complete, post-delay starting (0.3s)")
            time.sleep(0.3)
            logger.info("DRY-RUN: post-delay done, emitting execution_complete")
        except Exception as e:
            logger.error("Dry-run execution failed: %s", e)
        finally:
            self._bridge.execution_complete.emit()

    def _on_execution_complete(self) -> None:
        """Handle dry-run completion on main thread.

        Restores overlay visibility, click-through, switches back to red,
        transitions to VALIDATING.
        """
        # Re-show overlay before restoring interaction state.
        # Must happen before set_click_through(False) so the window is
        # visible when we re-enable hit-testing on it.
        logger.info("DRY-RUN: calling show_after_capture()")
        self._controller.show_after_capture()
        # Flush Qt event loop to ensure overlay is fully visible before restoring interaction
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        logger.info("DRY-RUN: show_after_capture() returned, restoring RECORDING state")
        self._controller.set_click_through(False)
        # Switch back to red shimmer for user interaction
        if self._controller._view is not None:
            from recorder.overlay.state import OverlayState
            self._controller._view.apply_state(OverlayState.RECORDING)
        self._set_phase(RecordPhase.VALIDATING)
        self._controller.set_toolbar_mode(ToolbarMode.VALIDATING)

    # ------------------------------------------------------------------
    # Private: validation toolbar action handlers
    # ------------------------------------------------------------------

    def _handle_validate_yes(self) -> None:
        """Handle 'Yes' from VALIDATING -- lock step, flash success, loop back."""
        if self._current_step is not None:
            self._current_step["dry_run_passed"] = True
            self._steps.append(self._current_step)
            logger.info(
                "Step %d validated: label=%s",
                len(self._steps),
                self._current_step.get("tag_data", {}).get("label", "?"),
            )

        self._set_phase(RecordPhase.SUCCESS_FLASH)

        # Flash success on the bbox
        if self._current_bbox is not None:
            bx, by, bw, bh = self._current_bbox
            self._controller.flash_success(QRectF(bx, by, bw, bh))

        # Clear current step data
        self._current_step = None
        self._current_bbox = None

        # After 500ms, loop back to AWAITING_CLICK (REC-10)
        QTimer.singleShot(
            500,
            self._loop_back_to_awaiting,
        )

    def _loop_back_to_awaiting(self) -> None:
        """Return to AWAITING_CLICK after success flash."""
        self._clear_scan()  # Updated: remove scan highlight on loop-back — prevents stale red rect — 2026-04-03
        self._set_phase(RecordPhase.AWAITING_CLICK)
        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)

    def _handle_edit_tags(self) -> None:
        """Handle 'Edit Tags' from VALIDATING -- reopen tag dialog."""
        logger.info("Edit tags requested")

        if self._current_step is not None and self._current_bbox is not None:
            bx, by, bw, bh = self._current_bbox
            element_rect = QRectF(bx, by, bw, bh)
            vlm_data = self._current_step.get("tag_data", {})
            self._controller.show_tag_dialog(
                element_rect, vlm_data=vlm_data, edit_mode=True,
            )
            self._set_phase(RecordPhase.TAG_DIALOG)
            self._controller.set_toolbar_mode(ToolbarMode.TAG_OPEN)

    def _handle_redraw(self) -> None:
        """Handle 'Redraw Box' -- clear bbox, return to drawing.

        Works from both TAG_DIALOG and BBOX_EDITING phases.  Allows
        the user to re-draw the bounding box when the current selection
        is wrong.  Similar to dismiss but explicitly signals intent to
        retry the bbox rather than abandon the step entirely.
        """
        prev_phase = self._phase
        self._current_step = None
        self._current_bbox = None
        if prev_phase == RecordPhase.TAG_DIALOG:
            self._controller.dismiss_tag_dialog()
        self._clear_scan()
        self._set_phase(RecordPhase.AWAITING_CLICK)
        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)
        logger.info("Redraw requested (from %s): returning to AWAITING_CLICK", prev_phase.name)

    def _handle_recapture(self) -> None:
        """Handle 'Recapture' -- discard current step, return to awaiting."""
        self._current_step = None
        self._current_bbox = None
        self._clear_scan()  # Updated: remove scan highlight on recapture — prevents stale red rect — 2026-04-03
        self._set_phase(RecordPhase.AWAITING_CLICK)
        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)
        logger.info("Recapture: returning to AWAITING_CLICK")

    def _handle_retry(self) -> None:
        """Handle 'Retry' from VALIDATING -- re-run countdown + dry-run."""
        logger.info("Retry requested -- restarting countdown")
        self._set_phase(RecordPhase.COUNTDOWN)
        self._controller.set_toolbar_mode(ToolbarMode.DRY_RUN)

        widget = self._controller.show_countdown(3)
        if widget is not None:
            # Updated: disconnect before connect to prevent signal accumulation
            try:
                widget.countdown_finished.disconnect(self._on_countdown_finished)
            except TypeError:
                pass  # No existing connection -- that's fine
            widget.countdown_finished.connect(self._on_countdown_finished)

    def _handle_undo_last(self) -> None:
        """Handle 'Undo Last' -- remove last recorded step."""
        if self._steps:
            removed = self._steps.pop()
            logger.info(
                "Undid step: %s",
                removed.get("tag_data", {}).get("label", "?"),
            )
        else:
            logger.info("Nothing to undo")

    # ------------------------------------------------------------------
    # Private: save flow
    # ------------------------------------------------------------------

    def _save_routine(self) -> None:
        """Background worker: save routine to disk.

        Creates ~/.ocsd/routines/{name}/ with routine.json, snippets/,
        and embeddings/ directories.
        """
        try:
            import pyautogui

            screen_w, screen_h = pyautogui.size()

            save_dir = Path.home() / ".ocsd" / "routines" / self._routine_name
            snippets_dir = save_dir / "snippets"
            embeddings_dir = save_dir / "embeddings"

            save_dir.mkdir(parents=True, exist_ok=True)
            snippets_dir.mkdir(exist_ok=True)
            embeddings_dir.mkdir(exist_ok=True)

            # Build OCSDGraph from steps
            from mapper.graph import OCSDGraph

            graph = OCSDGraph()
            prev_node_id: str | None = None
            node_ids: list[str] = []

            for i, step in enumerate(self._steps):
                tag = step.get("tag_data", {})
                bbox = step.get("bbox")
                bx, by, bw, bh = bbox if bbox else (0, 0, 0, 0)

                element_type = tag.get("element_type", "unknown")
                label = tag.get("label", "")

                x_pct = (bx + bw / 2) / screen_w if screen_w > 0 else 0.5
                y_pct = (by + bh / 2) / screen_h if screen_h > 0 else 0.5
                w_pct = bw / screen_w if screen_w > 0 else 0.0
                h_pct = bh / screen_h if screen_h > 0 else 0.0

                node_id = graph.add_node(
                    element_type=element_type,
                    label=label,
                    x_pct=x_pct,
                    y_pct=y_pct,
                    w_pct=w_pct,
                    h_pct=h_pct,
                    resolution=(screen_w, screen_h),
                )
                node_ids.append(node_id)
                # P1.2 fix: store graph node_id in step so _step_to_json
                # uses the same ID instead of generating a new uuid4()
                step["node_id"] = node_id

                if prev_node_id is not None:
                    action_type = "button"
                    if element_type == "textbox":
                        action_type = "textbox"
                    elif element_type in ("tab", "dropdown", "toggle"):
                        action_type = element_type
                    graph.add_edge(prev_node_id, node_id, action_type=action_type)

                prev_node_id = node_id

            # Save snippets and embeddings
            self._save_snippets_and_embeddings(
                save_dir, node_ids, screen_w, screen_h,
            )

            # Build and save routine using v1 Routine model
            from routine.format import Routine, resolve_loop_node_ids

            routine = Routine(
                name=self._routine_name,
                description=f"Recorded routine: {self._routine_name}",
                start_from=self._start_from,
                resolution=[screen_w, screen_h],
                steps=[
                    _step_to_json(s, i, screen_w, screen_h)
                    for i, s in enumerate(self._steps)
                ],
                graph=graph,
            )

            # Resolve loop body step indices to stable node_id references
            resolve_loop_node_ids(routine.steps, node_ids)

            # Auto-detect theme from screenshot luminance
            if self._screenshot is not None:
                try:
                    import cv2

                    gray = cv2.cvtColor(self._screenshot, cv2.COLOR_BGR2GRAY)
                    routine.theme = (
                        "dark" if float(np.mean(gray)) < 128 else "light"
                    )
                except Exception:
                    logger.debug("Could not auto-detect theme")

            # Auto-detect foreground program
            try:
                if __import__("sys").platform == "win32":
                    from core.capture import get_window_title

                    title = get_window_title()
                    if title:
                        routine.programs = [title]
            except Exception:
                logger.debug("Could not auto-detect foreground program")

            routine.save(save_dir)
            logger.info("Routine saved to %s", save_dir)
            self._bridge.save_complete.emit(str(save_dir))

        except Exception as e:
            logger.error("Failed to save routine: %s", e)
            self._bridge.save_failed.emit(str(e))

    def _save_snippets_and_embeddings(
        self,
        save_dir: Path,
        node_ids: list[str],
        screen_w: int,
        screen_h: int,
    ) -> None:
        """Save element crops and CLIP embeddings for each step.

        Args:
            save_dir: Root save directory for the routine.
            node_ids: Graph node IDs for each step.
            screen_w: Screen width.
            screen_h: Screen height.
        """
        cfg = get_config()
        crop_buffer = cfg.get("detection", {}).get("crop_buffer_pct", 0.30)

        for i, (step, node_id) in enumerate(zip(self._steps, node_ids)):
            bbox = step.get("bbox")
            if bbox is None:
                continue

            bx, by, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                continue

            # P1.6 fix: use each step's own screenshot (captured at record
            # time) instead of self._screenshot which only holds the last one
            step_screenshot = step.get("screenshot", self._screenshot)
            if step_screenshot is None:
                logger.debug("No screenshot for step %d, skipping snippet", i)
                continue

            # 30% padded crop
            buf_w = int(bw * crop_buffer)
            buf_h = int(bh * crop_buffer)
            x1 = max(0, bx - buf_w)
            y1 = max(0, by - buf_h)
            x2 = min(screen_w, bx + bw + buf_w)
            y2 = min(screen_h, by + bh + buf_h)

            sh, sw = step_screenshot.shape[:2]
            x2 = min(x2, sw)
            y2 = min(y2, sh)

            crop = step_screenshot[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Save snippet PNG
            snippet_path = save_dir / "snippets" / f"{node_id}.png"
            try:
                import cv2
                cv2.imwrite(str(snippet_path), crop)
            except Exception as e:
                logger.debug("Could not save snippet %d: %s", i, e)

            # Generate CLIP embedding
            embedding_path = save_dir / "embeddings" / f"{node_id}.npy"
            try:
                import cv2
                from core.embeddings import generate_embedding

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                embedding = generate_embedding(rgb_crop)
                np.save(str(embedding_path), embedding)
            except ImportError:
                logger.debug("CLIP not available, skipping embeddings")
                break
            except Exception as e:
                logger.debug("Could not generate embedding %d: %s", i, e)

    def _on_save_complete(self, path: str) -> None:
        """Handle successful save on main thread.

        Args:
            path: Directory path where routine was saved.
        """
        self._save_in_progress = False
        logger.info("Routine saved successfully to %s", path)
        self._controller.close()
        if self._on_session_complete is not None:
            self._on_session_complete(True)

    def _on_save_failed(self, error: str) -> None:
        """Handle save failure on main thread.

        Args:
            error: Error message string.
        """
        self._save_in_progress = False
        logger.error("Routine save failed: %s", error)
        # Don't lose session data -- user can retry

    # ------------------------------------------------------------------
    # Private: abort flow
    # ------------------------------------------------------------------

    def _on_abort_confirmed(self) -> None:
        """Handle abort confirmation -- discard and close."""
        self._steps.clear()
        self._controller.close()
        logger.info("Recording aborted, all steps discarded")
        if self._on_session_complete is not None:
            self._on_session_complete(False)

    def _on_abort_cancelled(self) -> None:
        """Handle abort cancellation -- return to recording."""
        self._controller.hide_abort_confirm()
        self._set_phase(RecordPhase.AWAITING_CLICK)
        logger.info("Abort cancelled, returning to AWAITING_CLICK")
