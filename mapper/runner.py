"""Replay orchestrator for OCSD skill execution.

Takes a recorded UI automation graph (OCSDGraph), finds the optimal
path from start to goal, and replays each step using the locate
cascade, action executor, and post-action validator.

Locate cascade (Stage 1):
1. OCR — find element by visible text
2. YOLO-E + position — STUB (Stage 2)
3. YOLO-E + VLM — STUB (Stage 2)
4. VLM full scan — STUB (Stage 2)
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from core.capture import screenshot_full
from core.config import get_config
from core.executor import click, type_text, scroll
from core.ocr import find_text_on_screen
from core.types import (
    ConfirmResult,
    ElementNotFoundError,
    LocateResult,
    ReplayLog,
    ReplayStep,
)
from mapper.graph import OCSDGraph
from mapper.pathfinder import get_execution_plan, get_fingerprint_checkpoints
from mapper.validator import validate_action

logger = logging.getLogger(__name__)


def locate_element(graph: OCSDGraph, node_id: str) -> LocateResult:
    """Locates an element on screen using the cascading strategy.

    Cascade order:
    1. OCR text match
    2. YOLO-E + position (stub)
    3. YOLO-E + VLM (stub)
    4. VLM full scan (stub)

    Args:
        graph: The OCSDGraph containing the node data.
        node_id: The ID of the node to locate.

    Returns:
        LocateResult with the screen location and match info.

    Raises:
        ElementNotFoundError: If all cascade stages fail.
        KeyError: If node_id does not exist in the graph.
    """
    node_data = graph.get_node(node_id)

    # Stage 1: OCR
    ocr_text = node_data.get("ocr_text")
    if ocr_text:
        logger.debug("Locate [%s] via OCR: %r", node_id[:8], ocr_text)
        result = find_text_on_screen(ocr_text)
        if result is not None:
            logger.info(
                "Located [%s] via OCR at (%d, %d) conf=%.2f",
                node_id[:8],
                result.point.x,
                result.point.y,
                result.confidence,
            )
            return result

    # Stage 2: YOLO-E + position (stub)
    logger.debug("YOLO-E + position locate not yet implemented, skipping")

    # Stage 3: YOLO-E + VLM (stub)
    logger.debug("YOLO-E + VLM locate not yet implemented, skipping")

    # Stage 4: VLM full scan (stub)
    logger.debug("VLM full scan locate not yet implemented, skipping")

    # Stage 5 (fallback): Use stored position coordinates
    pos = node_data.get("relative_position", {})
    x_pct = pos.get("x_pct")
    y_pct = pos.get("y_pct")
    if x_pct is not None and y_pct is not None:
        import pyautogui
        from core.types import Point

        sw, sh = pyautogui.size()
        px = int(x_pct * sw)
        py = int(y_pct * sh)
        w_pct = pos.get("w_pct", 0.0)
        h_pct = pos.get("h_pct", 0.0)
        if w_pct > 0 and h_pct > 0:
            logger.warning(
                "Located [%s] via bbox center fallback at (%d, %d) "
                "bbox=%dx%d — no visual confirmation",
                node_id[:8], px, py, int(w_pct * sw), int(h_pct * sh),
            )
        else:
            logger.warning(
                "Located [%s] via position fallback at (%d, %d) — "
                "no visual confirmation",
                node_id[:8], px, py,
            )
        return LocateResult(
            point=Point(px, py),
            confidence=0.3,  # low confidence — blind replay
            method="direct",
            snippet=None,
        )

    raise ElementNotFoundError(node_id, f"All locate stages failed for node {node_id[:8]}")


def _action_type_for_node(graph: OCSDGraph, node_id: str) -> str:
    """Determines the action type for a node based on its element type.

    Args:
        graph: The graph containing the node.
        node_id: The node to get the action type for.

    Returns:
        Action type string (e.g., "textbox", "button", "button_nav").
    """
    node_data = graph.get_node(node_id)
    etype = node_data.get("element_type", "unknown")
    mapping = {
        "textbox": "textbox",
        "button": "button",
        "button_nav": "button_nav",
        "icon": "button",
        "link": "button_nav",
        "tab": "tab",
        "dropdown": "dropdown",
        "toggle": "toggle",
        "scrollbar": "scrollbar",
    }
    return mapping.get(etype, "button")


def execute_node(
    graph: OCSDGraph,
    node_id: str,
    next_node_id: str | None = None,
    dry_run: bool = False,
) -> ReplayStep:
    """Executes an action on a single node.

    Locates the element on screen, takes before/after screenshots,
    executes the action, validates the result, and records stats.

    Args:
        graph: The OCSDGraph containing the node data.
        node_id: The node to interact with.
        next_node_id: The next node in the path (for edge stats tracking).
        dry_run: If True, logs actions without executing them.

    Returns:
        ReplayStep with execution details and success status.
    """
    action_type = _action_type_for_node(graph, node_id)

    # Get action payload from incoming edge if it exists
    action_payload = ""
    if next_node_id:
        try:
            edge_data = graph.get_edge(node_id, next_node_id)
            action_payload = edge_data.get("action_payload", "")
        except (KeyError, Exception):
            pass

    # Locate the element
    try:
        location = locate_element(graph, node_id)
    except ElementNotFoundError:
        logger.error("Cannot locate element for node %s", node_id[:8])
        if next_node_id:
            graph.record_execution(node_id, next_node_id, False)
        return ReplayStep(
            node_id=node_id,
            located_at=None,
            locate_method="failed",
            vlm_confidence=0.0,
            pixel_diff_pct=0.0,
            success=False,
            error=f"Element not found: {node_id[:8]}",
        )

    point = location.point

    # Before screenshot
    before_img = screenshot_full()

    # Execute the action
    logger.info(
        "Executing %s on [%s] at (%d, %d)",
        action_type,
        node_id[:8],
        point.x,
        point.y,
    )

    if action_type in ("button", "button_nav", "toggle", "dropdown", "tab", "click"):
        click(point.x, point.y, dry_run=dry_run)
    elif action_type == "textbox":
        click(point.x, point.y, dry_run=dry_run)
        if action_payload:
            type_text(action_payload, dry_run=dry_run)
    elif action_type == "scrollbar":
        scroll(point.x, point.y, direction="down", amount=3, dry_run=dry_run)
    elif action_type == "drag_source":
        logger.warning("Drag actions not yet implemented, skipping")
    else:
        logger.warning("Unknown action type %r, defaulting to click", action_type)
        click(point.x, point.y, dry_run=dry_run)

    # Post-action delay (human-like jitter)
    time.sleep(random.uniform(0.3, 0.5))

    # After screenshot
    after_img = screenshot_full()

    # Validate — skip pixel-diff for same-page actions (textbox focus,
    # intermediate clicks) where minimal visual change is expected.
    # Navigation actions (button_nav) always validate.
    # Only pixel-diff validate actions that should cause major screen changes.
    # Textbox clicks, same-page buttons, etc. produce minimal diff.
    # Skip validation for non-navigation actions AND for the last node
    # (terminal action has nothing to validate against).
    is_last_node = next_node_id is None
    skip_validation = action_type not in ("button_nav",) or is_last_node
    if skip_validation:
        logger.debug("Skipping pixel-diff validation for %s action", action_type)
        validation = ConfirmResult(
            success=True,
            confidence=0.6,
            notes=f"Validation skipped for {action_type} (locate succeeded)",
        )
    else:
        intended = f"{action_type} on {graph.get_node(node_id).get('label', node_id[:8])}"
        validation = validate_action(before_img, after_img, intended)

    # Record stats on the graph edge
    if next_node_id:
        graph.record_execution(node_id, next_node_id, validation.success)

    return ReplayStep(
        node_id=node_id,
        located_at=point,
        locate_method=location.method,
        vlm_confidence=validation.confidence,
        pixel_diff_pct=0.0,  # TODO: expose raw diff from validator
        success=validation.success,
        error=None if validation.success else validation.notes,
    )


def run_skill(
    graph: OCSDGraph,
    start_id: str,
    goal_id: str,
    dry_run: bool = False,
) -> ReplayLog:
    """Executes a full skill from start node to goal node.

    Finds the optimal path, iterates through each edge, executing
    actions and validating results. Records full timing and step log.

    Args:
        graph: The OCSDGraph containing the recorded skill.
        start_id: The starting node ID.
        goal_id: The goal/destination node ID.
        dry_run: If True, simulates without actual input.

    Returns:
        ReplayLog with complete execution history and timing.

    Raises:
        PathNotFoundError: If no path exists between start and goal.
    """
    config = get_config()
    abort_on_failure = config.get("execution", {}).get("abort_on_failure", True)

    start_time = time.monotonic()
    replay_log = ReplayLog(skill_id=graph.skill_id)

    # Build execution plan
    plan = get_execution_plan(graph, start_id, goal_id)
    path = plan["path"]
    fingerprints = plan["fingerprint_checks"]
    branches = plan["branch_points"]
    reliability = plan["estimated_reliability"]

    logger.info(
        "Skill replay: %d nodes to click, reliability=%.2f, "
        "%d fingerprints, %d branches",
        len(path),
        reliability,
        len(fingerprints),
        len(branches),
    )

    # Execute every node in the path (click each element in sequence)
    for i, node_id in enumerate(path):
        next_id = path[i + 1] if i < len(path) - 1 else None
        node_label = graph.get_node(node_id).get("label", node_id[:8])
        logger.info(
            "Step %d/%d: %s",
            i + 1,
            len(path),
            node_label,
        )

        step = execute_node(graph, node_id, next_node_id=next_id, dry_run=dry_run)
        replay_log.append_step(step)

        if not step.success:
            logger.error(
                "Step %d failed: %s",
                i + 1,
                step.error or "unknown error",
            )
            if abort_on_failure:
                logger.info("Aborting skill replay due to failure")
                break

    # Finalize timing
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    replay_log.duration_ms = elapsed_ms

    logger.info(
        "Skill replay complete: success=%s, %d steps, %dms",
        replay_log.overall_success,
        len(replay_log.steps),
        elapsed_ms,
    )

    return replay_log
