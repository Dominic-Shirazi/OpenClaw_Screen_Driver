"""Execution engine for running recorded skill paths.

Executes a sequence of nodes from the OCSD graph, locating each element
on screen via a 4-step fallback cascade, performing the action, and
validating the result. Emits events throughout for monitoring.

Fallback cascade for locate_element:
1. OCR search — fast, text-only
2. CLIP embedding search — visual, medium speed
3. VLM confirmation of candidates — slow, most reliable
4. FAIL → ElementNotFoundError
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Any, Callable

import numpy as np

from core.capture import pixel_diff, save_snippet, screenshot_full, screenshot_region
from core.config import get_config
from core.types import (
    ConfirmResult,
    ElementNotFoundError,
    LocateResult,
    LowConfidenceError,
    Point,
    Rect,
    ReplayLog,
    ReplayStep,
)
from executor.validator import validate_action
from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)


class RunnerEventType(Enum):
    """Events emitted during skill execution."""
    STEP_START = auto()
    ELEMENT_LOCATED = auto()
    LOW_CONFIDENCE = auto()
    ACTION_EXECUTED = auto()
    VALIDATION_PASSED = auto()
    VALIDATION_FAILED = auto()
    ELEMENT_NOT_FOUND = auto()
    STEP_COMPLETE = auto()
    PATH_COMPLETE = auto()
    PATH_FAILED = auto()


# Type alias for event callback
EventCallback = Callable[[RunnerEventType, dict[str, Any]], None]


def locate_element(
    node_data: dict[str, Any],
    *,
    dry_run: bool = False,
) -> LocateResult:
    """Locates an element on the current screen using the 4-step fallback cascade.

    1. OCR search if ocr_text is available
    2. CLIP embedding search within position radius
    3. VLM confirmation of best candidates
    4. FAIL with ElementNotFoundError

    Args:
        node_data: Full node dict from the graph (must include node_id,
                  ocr_text, embedding_id, element_type, label,
                  relative_position, confidence_threshold).
        dry_run: If True, skip actual screen capture and return a dummy result.

    Returns:
        LocateResult with the located point, method used, and confidence.

    Raises:
        ElementNotFoundError: If the element cannot be found after all fallbacks.
    """
    node_id = node_data.get("node_id", "unknown")
    ocr_text = node_data.get("ocr_text")
    confidence_threshold = node_data.get("confidence_threshold", 0.75)
    position = node_data.get("relative_position", {})
    x_pct = position.get("x_pct", 0.5)
    y_pct = position.get("y_pct", 0.5)

    if dry_run:
        # Return a plausible position for dry run
        return LocateResult(
            point=Point(int(x_pct * 1920), int(y_pct * 1080)),
            method="dry_run",
            confidence=1.0,
        )

    # STEP 1: OCR search
    if ocr_text:
        try:
            from core.ocr import find_text_on_screen

            match = find_text_on_screen(ocr_text)
            if match and match.confidence > 0.8:
                logger.debug(
                    "OCR located '%s' at (%d, %d) conf=%.2f",
                    ocr_text,
                    match.point.x,
                    match.point.y,
                    match.confidence,
                )
                return LocateResult(
                    point=match.point,
                    method="ocr",
                    confidence=match.confidence,
                    rect=match.rect,
                )
        except Exception as e:
            logger.debug("OCR search failed for '%s': %s", ocr_text, e)

    # STEP 2: CLIP embedding search within position radius
    try:
        from core.embeddings import generate_embedding, search_index

        current_screenshot = screenshot_full()
        # Generate embedding from the region around the expected position
        h, w = current_screenshot.shape[:2]
        crop_x = max(0, int(x_pct * w) - 100)
        crop_y = max(0, int(y_pct * h) - 100)
        crop_w = min(200, w - crop_x)
        crop_h = min(200, h - crop_y)
        region = current_screenshot[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        if region.size > 0:
            # Convert BGR to RGB for CLIP
            import cv2
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            query_emb = generate_embedding(region_rgb)

            candidates = search_index(
                query_emb,
                top_k=5,
                radius_pct=0.20,
                query_x_pct=x_pct,
                query_y_pct=y_pct,
            )

            if candidates:
                best = candidates[0]
                if best["score"] > confidence_threshold:
                    # Convert pct coordinates back to pixels
                    px = int(best["x_pct"] * w)
                    py = int(best["y_pct"] * h)
                    logger.debug(
                        "CLIP located element at (%d, %d) score=%.2f",
                        px,
                        py,
                        best["score"],
                    )
                    return LocateResult(
                        point=Point(px, py),
                        method="embedding",
                        confidence=best["score"],
                    )
    except Exception as e:
        logger.debug("CLIP search failed: %s", e)

    # STEP 3: VLM confirmation of candidates
    try:
        from core.vision import analyze_crop_array

        current_screenshot = screenshot_full()
        h, w = current_screenshot.shape[:2]

        # Define search regions around expected position
        search_radius_x = int(w * 0.15)
        search_radius_y = int(h * 0.15)
        center_x = int(x_pct * w)
        center_y = int(y_pct * h)

        # Crop a region around expected position
        rx = max(0, center_x - search_radius_x)
        ry = max(0, center_y - search_radius_y)
        rw = min(search_radius_x * 2, w - rx)
        rh = min(search_radius_y * 2, h - ry)

        if rw > 0 and rh > 0:
            region = current_screenshot[ry:ry + rh, rx:rx + rw]
            element_type = node_data.get("element_type", "unknown")
            label = node_data.get("label", "")

            result = analyze_crop_array(
                region,
                f"Find the {element_type} labeled '{label}' in this region",
            )

            if result.get("confidence", 0) > 0.75:
                # Return center of the search region as best estimate
                logger.debug(
                    "VLM confirmed element near (%d, %d) conf=%.2f",
                    center_x,
                    center_y,
                    result["confidence"],
                )
                return LocateResult(
                    point=Point(center_x, center_y),
                    method="vlm",
                    confidence=result["confidence"],
                )
    except Exception as e:
        logger.debug("VLM search failed: %s", e)

    # STEP 4: FAIL
    raise ElementNotFoundError(node_id)


def _execute_action(
    node_data: dict[str, Any],
    located: LocateResult,
    edge_data: dict[str, Any] | None,
    *,
    dry_run: bool = False,
) -> None:
    """Performs the action associated with a node.

    Args:
        node_data: Full node dict from the graph.
        located: Where the element was found on screen.
        edge_data: Edge data dict (for action_type and payload), or None.
        dry_run: If True, log without executing.
    """
    from core import executor

    element_type = node_data.get("element_type", "unknown")
    action_payload = edge_data.get("action_payload") if edge_data else None
    x, y = located.point.x, located.point.y

    if element_type == "textbox" and action_payload:
        # Type text into the field
        executor.click(x, y, dry_run=dry_run)
        executor.type_text(action_payload, dry_run=dry_run)
    elif element_type in ("button", "button_nav", "toggle", "tab"):
        executor.click(x, y, dry_run=dry_run)
    elif element_type == "dropdown":
        executor.click(x, y, dry_run=dry_run)
    elif element_type == "drag_source" and action_payload:
        # action_payload should be "target_x,target_y"
        try:
            tx, ty = action_payload.split(",")
            executor.drag(x, y, int(tx), int(ty), dry_run=dry_run)
        except (ValueError, TypeError):
            executor.click(x, y, dry_run=dry_run)
    elif element_type == "scrollbar":
        direction = action_payload or "down"
        executor.scroll(x, y, direction, 3, dry_run=dry_run)
    elif element_type == "read_here":
        # No action needed — this is a data read point
        pass
    elif action_payload == "tab":
        executor.hotkey("tab", dry_run=dry_run)
    else:
        # Default: click
        executor.click(x, y, dry_run=dry_run)


def run_path(
    path: list[str],
    graph: OCSDGraph,
    *,
    dry_run: bool = False,
    event_callback: EventCallback | None = None,
    skip_vlm_validation: bool = False,
) -> ReplayLog:
    """Executes a sequence of nodes from the graph.

    For each node in the path:
    1. Take a before screenshot
    2. Locate the element via fallback cascade
    3. Optionally validate with VLM before acting
    4. Execute the action
    5. Take an after screenshot
    6. Validate the result
    7. Record execution stats
    8. Append to replay log

    Args:
        path: Ordered list of node_ids to execute.
        graph: The OCSDGraph containing node/edge data.
        dry_run: If True, log actions without executing them.
        event_callback: Optional callback for execution events.
        skip_vlm_validation: If True, skip VLM for faster execution.

    Returns:
        ReplayLog with step-by-step results.
    """
    replay_log = ReplayLog(skill_id=graph.skill_id)
    start_time = time.time()

    def emit(event_type: RunnerEventType, data: dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.error("Event callback error: %s", e)

    logger.info(
        "Starting path execution: %d steps, dry_run=%s",
        len(path),
        dry_run,
    )

    for step_idx, node_id in enumerate(path):
        emit(RunnerEventType.STEP_START, {"node_id": node_id, "step": step_idx})

        try:
            node_data = graph.get_node(node_id)
        except KeyError:
            logger.error("Node not found in graph: %s", node_id)
            step = ReplayStep(
                node_id=node_id,
                located_at=None,
                locate_method="failed",
                vlm_confidence=0.0,
                pixel_diff_pct=0.0,
                success=False,
                error=f"Node not found: {node_id}",
            )
            replay_log.append_step(step)
            emit(RunnerEventType.ELEMENT_NOT_FOUND, {"node_id": node_id})
            continue

        # Determine the edge data (for action type/payload)
        edge_data: dict[str, Any] | None = None
        if step_idx + 1 < len(path):
            next_node_id = path[step_idx + 1]
            try:
                edge_data = graph.get_edge(node_id, next_node_id)
            except KeyError:
                pass

        # Step 1: Before screenshot
        before_img = None
        if not dry_run:
            try:
                before_img = screenshot_full()
            except Exception as e:
                logger.warning("Before screenshot failed: %s", e)

        # Step 2: Locate element
        try:
            located = locate_element(node_data, dry_run=dry_run)
            emit(
                RunnerEventType.ELEMENT_LOCATED,
                {
                    "node_id": node_id,
                    "point": (located.point.x, located.point.y),
                    "method": located.method,
                    "confidence": located.confidence,
                },
            )
        except ElementNotFoundError:
            step = ReplayStep(
                node_id=node_id,
                located_at=None,
                locate_method="failed",
                vlm_confidence=0.0,
                pixel_diff_pct=0.0,
                success=False,
                error="Element not found after all fallbacks",
            )
            replay_log.append_step(step)
            emit(RunnerEventType.ELEMENT_NOT_FOUND, {"node_id": node_id})

            # Record failure in graph
            if step_idx > 0:
                prev_id = path[step_idx - 1]
                graph.record_execution(prev_id, node_id, success=False)

            logger.warning("Element not found: %s (%s)", node_id[:8], node_data.get("label"))
            continue

        # Check confidence threshold
        confidence_threshold = node_data.get("confidence_threshold", 0.75)
        if located.confidence < confidence_threshold:
            emit(
                RunnerEventType.LOW_CONFIDENCE,
                {
                    "node_id": node_id,
                    "confidence": located.confidence,
                    "threshold": confidence_threshold,
                },
            )
            logger.warning(
                "Low confidence for %s: %.2f < %.2f",
                node_id[:8],
                located.confidence,
                confidence_threshold,
            )

        # Step 3: Execute the action
        try:
            _execute_action(node_data, located, edge_data, dry_run=dry_run)
            emit(
                RunnerEventType.ACTION_EXECUTED,
                {"node_id": node_id, "action": node_data.get("element_type")},
            )
        except Exception as e:
            step = ReplayStep(
                node_id=node_id,
                located_at=located.point,
                locate_method=located.method,
                vlm_confidence=located.confidence,
                pixel_diff_pct=0.0,
                success=False,
                error=f"Action failed: {e}",
            )
            replay_log.append_step(step)
            logger.error("Action failed for %s: %s", node_id[:8], e)
            continue

        # Step 4: After screenshot + validation
        diff_pct = 0.0
        validation_success = True
        vlm_confidence = located.confidence

        if not dry_run and before_img is not None:
            try:
                # Brief pause for UI to update
                time.sleep(0.2)
                after_img = screenshot_full()
                diff_pct = pixel_diff(before_img, after_img)

                # Validate the action result
                element_type = node_data.get("element_type", "unknown")
                label = node_data.get("label", "")
                intended_action = f"{element_type}: {label}"

                validation = validate_action(
                    before_img,
                    after_img,
                    intended_action,
                    skip_vlm=skip_vlm_validation,
                )
                validation_success = validation.success
                vlm_confidence = validation.confidence

                if validation_success:
                    emit(
                        RunnerEventType.VALIDATION_PASSED,
                        {"node_id": node_id, "diff_pct": diff_pct},
                    )
                else:
                    emit(
                        RunnerEventType.VALIDATION_FAILED,
                        {
                            "node_id": node_id,
                            "notes": validation.notes,
                            "diff_pct": diff_pct,
                        },
                    )
            except Exception as e:
                logger.warning("Post-action validation failed: %s", e)

        # Step 5: Record results
        step = ReplayStep(
            node_id=node_id,
            located_at=located.point,
            locate_method=located.method,
            vlm_confidence=vlm_confidence,
            pixel_diff_pct=diff_pct,
            success=validation_success,
        )
        replay_log.append_step(step)

        # Update graph execution stats
        if step_idx > 0:
            prev_id = path[step_idx - 1]
            graph.record_execution(prev_id, node_id, success=validation_success)

        emit(
            RunnerEventType.STEP_COMPLETE,
            {
                "node_id": node_id,
                "step": step_idx,
                "success": validation_success,
            },
        )

        logger.info(
            "Step %d/%d: %s (%s) via %s — %s",
            step_idx + 1,
            len(path),
            node_data.get("label", "?"),
            node_data.get("element_type", "?"),
            located.method,
            "OK" if validation_success else "FAIL",
        )

    # Finalize
    elapsed_ms = int((time.time() - start_time) * 1000)
    replay_log.duration_ms = elapsed_ms

    if replay_log.overall_success:
        emit(RunnerEventType.PATH_COMPLETE, {"duration_ms": elapsed_ms})
        logger.info("Path complete: %d steps in %dms", len(path), elapsed_ms)
    else:
        emit(RunnerEventType.PATH_FAILED, {"duration_ms": elapsed_ms})
        logger.warning(
            "Path failed: %d/%d steps succeeded in %dms",
            sum(1 for s in replay_log.steps if s.success),
            len(replay_log.steps),
            elapsed_ms,
        )

    return replay_log
