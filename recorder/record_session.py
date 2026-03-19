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
    return build_v1_step(step, index, node_id, screen_w, screen_h)


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
        self._click_x: int = 0
        self._click_y: int = 0
        self._on_session_complete: Callable[[bool], None] | None = None

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
        """Wire controller callbacks, show toolbar, enter AWAITING_CLICK."""
        self._controller._on_selection = self.on_selection
        self._controller._on_save = self.on_save_requested
        self._controller._on_abort = self.on_abort_requested

        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)
        self._controller.show_toolbar()
        self._set_phase(RecordPhase.AWAITING_CLICK)

        logger.info("RecordSession started, awaiting first click")

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
        self._current_step = {
            **self._current_step,
            "tag_data": data,
            "bbox": self._current_bbox,
        }

        self._controller.dismiss_tag_dialog()

        logger.info(
            "Tag confirmed: label=%s, type=%s -- starting countdown",
            data.get("label", "?"),
            data.get("element_type", "?"),
        )

        # Transition to COUNTDOWN
        self._set_phase(RecordPhase.COUNTDOWN)
        self._controller.set_toolbar_mode(ToolbarMode.DRY_RUN)

        widget = self._controller.show_countdown(3)
        if widget is not None:
            widget.countdown_finished.connect(self._on_countdown_finished)

    def on_tag_dismissed(self, data: dict) -> None:
        """Handle tag dialog dismissal -- discard step, return to awaiting.

        Args:
            data: Form data dict (discarded).
        """
        self._current_step = None
        self._current_bbox = None
        self._controller.dismiss_tag_dialog()
        self._set_phase(RecordPhase.AWAITING_CLICK)
        self._controller.set_toolbar_mode(ToolbarMode.RECORDING)

        logger.info("Tag dismissed, returning to AWAITING_CLICK")

    def on_save_requested(self) -> None:
        """Handle Ctrl+Q -- save all accumulated steps."""
        logger.info("Save requested with %d steps", len(self._steps))

        if len(self._steps) == 0:
            # No steps -- close silently
            self._controller.close()
            if self._on_session_complete is not None:
                self._on_session_complete(False)
            return

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
                panel.discard_clicked.connect(self._on_abort_confirmed)
                panel.keep_clicked.connect(self._on_abort_cancelled)
        else:
            logger.info("Abort requested (no steps recorded)")
            self._controller.close()
            if self._on_session_complete is not None:
                self._on_session_complete(False)

    # ------------------------------------------------------------------
    # Private: pipeline methods
    # ------------------------------------------------------------------

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
        time.sleep(delay_ms / 1000)

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

            # Show scan animation
            self._controller.start_scan(bx, by, bw, bh)

            self._set_phase(RecordPhase.BBOX_EDITING)

            if self._is_drag_capture:
                self._controller.set_toolbar_mode(ToolbarMode.BBOX_EDITING)
            else:
                # For click captures, auto-accept the bbox
                self._accept_bbox()
        else:
            # No bbox found: create default 60x60 around click
            default_size = 60
            bx = self._click_x - default_size // 2
            by = self._click_y - default_size // 2
            self._current_bbox = (bx, by, default_size, default_size)

            self._set_phase(RecordPhase.BBOX_EDITING)
            # Skip directly to VLM for no-bbox case
            self._accept_bbox()

    def _accept_bbox(self) -> None:
        """Accept the current bbox and start VLM analysis."""
        self._set_phase(RecordPhase.VLM_ANALYZING)

        # Start card glow pulsing during VLM analysis
        if hasattr(self._controller, "start_card_glow_pulse"):
            self._controller.start_card_glow_pulse()

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

            # First attempt with 15s timeout
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

        Opens tag dialog with VLM data and element rect.

        Args:
            result: VLM analysis result dict.
        """
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

        if self._current_bbox is not None:
            bx, by, bw, bh = self._current_bbox
            element_rect = QRectF(bx, by, bw, bh)
        else:
            element_rect = QRectF(
                self._click_x - 30, self._click_y - 30, 60, 60,
            )

        self._controller.show_tag_dialog(element_rect, vlm_data=vlm_data)
        self._controller.set_toolbar_mode(ToolbarMode.TAG_OPEN)

    def _on_vlm_failed(self, error: str) -> None:
        """Handle VLM failure on main thread.

        Opens tag dialog with partial data so user can still tag manually.

        Args:
            error: Error message string.
        """
        # Stop card glow pulsing
        if hasattr(self._controller, "stop_card_glow_pulse"):
            self._controller.stop_card_glow_pulse()

        self._set_phase(RecordPhase.TAG_DIALOG)

        logger.warning("VLM failed: %s", error)

        partial_data = {
            "element_type": "unknown",
            "label": "",
            "caption": "VLM unavailable",
            "confidence": 0.0,
            "ocr_text": None,
        }

        if self._current_bbox is not None:
            bx, by, bw, bh = self._current_bbox
            element_rect = QRectF(bx, by, bw, bh)
        else:
            element_rect = QRectF(
                self._click_x - 30, self._click_y - 30, 60, 60,
            )

        self._controller.show_tag_dialog(
            element_rect, vlm_data=partial_data,
        )
        self._controller.set_toolbar_mode(ToolbarMode.TAG_OPEN)

    # ------------------------------------------------------------------
    # Private: dry-run execution
    # ------------------------------------------------------------------

    def _on_countdown_finished(self) -> None:
        """Handle countdown completion -- execute dry-run action."""
        self._controller.hide_countdown()
        self._set_phase(RecordPhase.EXECUTING)
        self._controller.set_click_through(True)

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
        try:
            if self._current_bbox is None or self._current_step is None:
                logger.warning("No bbox/step for dry-run execution")
                return

            bx, by, bw, bh = self._current_bbox
            center_x = bx + bw // 2
            center_y = by + bh // 2
            tag_data = self._current_step.get("tag_data", {})
            action = tag_data.get("action", "click")

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
                case "read" | "snip_and_search":
                    logger.info(
                        "Dry-run %s: observation step, no action", action,
                    )
                case "select_all_extract":
                    logger.info(
                        "Dry-run select_all_extract: no action during dry-run",
                    )
                case "wait" | "loop" | "prompt_user":
                    logger.info(
                        "Dry-run %s: skipped (no dry-run for flow actions)",
                        action,
                    )
                case _:
                    logger.warning(
                        "Unknown action type for dry-run: %s", action,
                    )
                    exec_click(center_x, center_y)

            logger.info(
                "Dry-run %s executed at (%d, %d)", action, center_x, center_y,
            )
        except Exception as e:
            logger.error("Dry-run execution failed: %s", e)
        finally:
            self._bridge.execution_complete.emit()

    def _on_execution_complete(self) -> None:
        """Handle dry-run completion on main thread.

        Restores click-through and transitions to VALIDATING.
        """
        self._controller.set_click_through(False)
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

    def _handle_recapture(self) -> None:
        """Handle 'Recapture' -- discard current step, return to awaiting."""
        self._current_step = None
        self._current_bbox = None
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
            from routine.format import Routine

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
        if self._screenshot is None:
            logger.debug("No screenshot available for snippets")
            return

        cfg = get_config()
        crop_buffer = cfg.get("detection", {}).get("crop_buffer_pct", 0.30)

        for i, (step, node_id) in enumerate(zip(self._steps, node_ids)):
            bbox = step.get("bbox")
            if bbox is None:
                continue

            bx, by, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                continue

            # 30% padded crop
            buf_w = int(bw * crop_buffer)
            buf_h = int(bh * crop_buffer)
            x1 = max(0, bx - buf_w)
            y1 = max(0, by - buf_h)
            x2 = min(screen_w, bx + bw + buf_w)
            y2 = min(screen_h, by + bh + buf_h)

            sh, sw = self._screenshot.shape[:2]
            x2 = min(x2, sw)
            y2 = min(y2, sh)

            crop = self._screenshot[y1:y2, x1:x2]
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
        logger.info("Routine saved successfully to %s", path)
        self._controller.close()
        if self._on_session_complete is not None:
            self._on_session_complete(True)

    def _on_save_failed(self, error: str) -> None:
        """Handle save failure on main thread.

        Args:
            error: Error message string.
        """
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
