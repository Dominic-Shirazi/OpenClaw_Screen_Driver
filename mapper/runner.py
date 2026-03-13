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

    raise ElementNotFoundError(node_id, f"All locate stages failed for node {node_id[:8]}")


def execute_action(
    graph: OCSDGraph,
    source_id: str,
    target_id: str,
    dry_run: bool = False,
) -> ReplayStep:
    """Executes a single action edge from the graph.

    Locates the source element, takes before/after screenshots,
    executes the action, validates the result, and records stats.

    Args:
        graph: The OCSDGraph containing the execution paths.
        source_id: The node where the action is performed.
        target_id: The node representing the expected result state.
        dry_run: If True, logs actions without executing them.

    Returns:
        ReplayStep with execution details and success status.
    """
    edge_data = graph.get_edge(source_id, target_id)
    action_type = edge_data.get("action_type", "button")
    action_payload = edge_data.get("action_payload", "")

    # Locate the source element
    try:
        location = locate_element(graph, source_id)
    except ElementNotFoundError:
        logger.error("Cannot locate element for node %s", source_id[:8])
        graph.record_execution(source_id, target_id, False)
        return ReplayStep(
            node_id=source_id,
            located_at=None,
            locate_method="failed",
            vlm_confidence=0.0,
            pixel_diff_pct=0.0,
            success=False,
            error=f"Element not found: {source_id[:8]}",
        )

    point = location.point

    # Before screenshot
    before_img = screenshot_full()

    # Execute the action
    logger.info(
        "Executing %s on [%s] at (%d, %d)",
        action_type,
        source_id[:8],
        point.x,
        point.y,
    )

    if action_type in ("button", "button_nav", "toggle", "dropdown", "tab"):
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

    # Validate
    intended = f"{action_type} on {graph.get_node(source_id).get('label', source_id[:8])}"
    validation = validate_action(before_img, after_img, intended)

    # Record stats on the graph
    graph.record_execution(source_id, target_id, validation.success)

    return ReplayStep(
        node_id=source_id,
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
        "Skill replay: %d steps, reliability=%.2f, "
        "%d fingerprints, %d branches",
        len(path) - 1,
        reliability,
        len(fingerprints),
        len(branches),
    )

    # Execute each edge in the path
    for i in range(len(path) - 1):
        src = path[i]
        tgt = path[i + 1]

        src_label = graph.get_node(src).get("label", src[:8])
        tgt_label = graph.get_node(tgt).get("label", tgt[:8])
        logger.info(
            "Step %d/%d: %s -> %s",
            i + 1,
            len(path) - 1,
            src_label,
            tgt_label,
        )

        step = execute_action(graph, src, tgt, dry_run=dry_run)
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
