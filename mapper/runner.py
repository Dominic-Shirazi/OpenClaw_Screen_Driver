"""Replay orchestrator for OCSD skill execution.

Takes a recorded UI automation graph (OCSDGraph), finds the optimal
path from start to goal, and replays each step using the locate
cascade, action executor, and post-action validator.

Locate cascade (ordered by speed, cheapest first):
1. OmniParser detect+match — saved snippet → detect boxes → CLIP match (~50ms)
2. CLIP embedding — compare SAVED embedding against OmniParser candidates
3. OCR text match — SCOPED to region around expected position (~200ms)
4. VLM full scan — screenshot → LiteLLM → match by label (expensive)
5. Position fallback — blind click at recorded coordinates (no confirmation)
"""

from __future__ import annotations

import logging
import random
import time
from enum import Enum, auto
from typing import Any, Callable

import pyautogui

from core.capture import screenshot_full
from core.config import get_config
from core.executor import click, type_text, scroll
from core.ocr import find_text_on_screen
from core.types import (
    ConfirmResult,
    ElementNotFoundError,
    LocateResult,
    Point,
    ReplayLog,
    ReplayStep,
)
from mapper.graph import OCSDGraph
from mapper.pathfinder import get_execution_plan, get_fingerprint_checkpoints
from mapper.validator import validate_action

logger = logging.getLogger(__name__)


class RunnerEventType(Enum):
    """Events emitted during skill execution."""
    STEP_START = auto()
    STEP_PREVIEW = auto()
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


def _resolve_position_hint(
    node_data: dict,
) -> tuple[int | None, int | None, int, int]:
    """Extracts expected pixel position from node's relative_position.

    Returns:
        (hint_x, hint_y, screen_w, screen_h) — hint values are None
        if the node has no stored position.
    """
    sw, sh = pyautogui.size()
    pos = node_data.get("relative_position", {})
    x_pct = pos.get("x_pct")
    y_pct = pos.get("y_pct")
    if x_pct is not None and y_pct is not None:
        return int(x_pct * sw), int(y_pct * sh), sw, sh
    return None, None, sw, sh


def locate_element(
    graph: OCSDGraph,
    node_id: str,
    skill_id: str = "",
) -> LocateResult:
    """Locates an element on screen using the cascading strategy.

    Cascade order (fastest / cheapest first):
    1. **OmniParser detect+match** — give it the saved snippet from recording
       and approximate position → detect all UI boxes then CLIP-match, ~50 ms.
    2. **CLIP embedding** — retrieve the SAVED embedding (from recording),
       compare against crops of the current screen near the expected
       position.  Confirms OmniParser hits or finds the element independently.
    3. **OCR text match** — search ONLY within a region around the expected
       position (not full screen!) to avoid false positives from ads etc.
    4. **VLM full-screen scan** — expensive LiteLLM call, most reliable.
    5. **Position fallback** — blind click at recorded coordinates.

    Args:
        graph: The OCSDGraph containing the node data.
        node_id: The ID of the node to locate.
        skill_id: Skill name (for loading snippet PNGs).

    Returns:
        LocateResult with the screen location and match info.

    Raises:
        ElementNotFoundError: If all cascade stages fail.
        KeyError: If node_id does not exist in the graph.
    """
    node_data = graph.get_node(node_id)
    hint_x, hint_y, sw, sh = _resolve_position_hint(node_data)

    # ------------------------------------------------------------------
    # Stage 1: OmniParser detect + CLIP match
    # "here's the snippet from recording → detect all boxes → CLIP-match"
    # ------------------------------------------------------------------
    try:
        from core.capture import load_snippet
        from core.detection import get_detector

        snippet = load_snippet(skill_id, node_id[:12])
        if snippet is not None:
            screen = screenshot_full()
            cfg = get_config()
            detector = get_detector()
            result = detector.detect_and_match(
                screen, snippet, hint_x or 0, hint_y or 0,
                match_threshold=cfg.get("detection", {}).get("match_threshold", 0.7),
                search_radius=cfg.get("detection", {}).get("search_radius", 400),
            )
            if result is not None:
                logger.info(
                    "Located [%s] via OmniParser at (%d, %d) conf=%.2f",
                    node_id[:8], result.point.x, result.point.y, result.confidence,
                )
                return result
            logger.debug("OmniParser found no match for [%s]", node_id[:8])
        else:
            logger.debug("No snippet on disk for [%s], skipping Stage 1", node_id[:8])
    except ImportError:
        logger.debug("Detection module not available, skipping Stage 1")
    except Exception as e:
        logger.debug("OmniParser locate error: %s", e)

    # ------------------------------------------------------------------
    # Stage 2: CLIP embedding — CORRECT direction
    # Retrieve the SAVED embedding from FAISS (recorded at capture time),
    # then screenshot a region of the current screen around the expected
    # position, embed that crop, and compare via cosine similarity.
    # ------------------------------------------------------------------
    try:
        from core.capture import screenshot_region
        from core.embeddings import generate_embedding, get_embedding_by_id

        saved_emb = get_embedding_by_id(node_id)
        if saved_emb is not None and hint_x is not None and hint_y is not None:
            import cv2
            import numpy as np

            # Crop a region around the expected position
            search_r = 300
            rx = max(0, hint_x - search_r)
            ry = max(0, hint_y - search_r)
            rw = min(search_r * 2, sw - rx)
            rh = min(search_r * 2, sh - ry)

            crop = screenshot_region(rx, ry, rw, rh)
            if crop.size > 0:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                current_emb = generate_embedding(rgb_crop)

                # Cosine similarity (both are L2-normalized → dot product)
                score = float(np.dot(saved_emb, current_emb.T).item())

                if score > 0.75:
                    # High similarity — element is still roughly where it was
                    logger.info(
                        "Located [%s] via CLIP at (%d, %d) score=%.3f",
                        node_id[:8], hint_x, hint_y, score,
                    )
                    return LocateResult(
                        point=Point(hint_x, hint_y),
                        confidence=min(score, 0.85),
                        method="clip",
                    )
                else:
                    logger.debug(
                        "CLIP score %.3f too low for [%s]", score, node_id[:8],
                    )
    except ImportError:
        logger.debug("CLIP/FAISS not available, skipping Stage 2")
    except Exception as e:
        logger.debug("CLIP search error: %s", e)

    # ------------------------------------------------------------------
    # Stage 3: OCR text match — SCOPED to region around expected position
    # Avoids false positives from ads / unrelated UI text.
    # ------------------------------------------------------------------
    ocr_text = node_data.get("ocr_text")
    if ocr_text:
        logger.debug("Locate [%s] via OCR: %r", node_id[:8], ocr_text)
        result = find_text_on_screen(
            ocr_text,
            hint_x=hint_x,
            hint_y=hint_y,
            search_radius=400,
        )
        if result is not None:
            logger.info(
                "Located [%s] via OCR at (%d, %d) conf=%.2f",
                node_id[:8],
                result.point.x,
                result.point.y,
                result.confidence,
            )
            return result

    # ------------------------------------------------------------------
    # Stage 4: VLM full-screen analysis (expensive, high reliability)
    # ------------------------------------------------------------------
    try:
        from core.vision import first_pass_map_array

        full_img = screenshot_full()
        candidates = first_pass_map_array(full_img)

        target_label = node_data.get("label", "").lower()

        if target_label and candidates:
            for c in candidates:
                c_label = c.get("label_guess", "").lower()
                if target_label in c_label or c_label in target_label:
                    rect = c.get("rect", {})
                    cx = rect.get("x", 0) + rect.get("w", 0) // 2
                    cy = rect.get("y", 0) + rect.get("h", 0) // 2
                    conf = c.get("confidence", 0.5)
                    logger.info(
                        "Located [%s] via VLM at (%d, %d) conf=%.2f",
                        node_id[:8], cx, cy, conf,
                    )
                    return LocateResult(
                        point=Point(cx, cy),
                        confidence=conf,
                        method="vlm",
                    )
    except ImportError:
        logger.debug("VLM module not available, skipping Stage 4")
    except Exception as e:
        logger.debug("VLM scan error: %s", e)

    # ------------------------------------------------------------------
    # Stage 5: Position fallback — blind click at recorded coordinates
    # ------------------------------------------------------------------
    if hint_x is not None and hint_y is not None:
        pos = node_data.get("relative_position", {})
        w_pct = pos.get("w_pct", 0.0)
        h_pct = pos.get("h_pct", 0.0)
        if w_pct > 0 and h_pct > 0:
            logger.warning(
                "Located [%s] via bbox center fallback at (%d, %d) "
                "bbox=%dx%d — no visual confirmation",
                node_id[:8], hint_x, hint_y, int(w_pct * sw), int(h_pct * sh),
            )
        else:
            logger.warning(
                "Located [%s] via position fallback at (%d, %d) — "
                "no visual confirmation",
                node_id[:8], hint_x, hint_y,
            )
        return LocateResult(
            point=Point(hint_x, hint_y),
            confidence=0.3,
            method="direct",
        )

    raise ElementNotFoundError(
        node_id, f"All locate stages failed for node {node_id[:8]}",
    )


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
    skill_id: str = "",
) -> ReplayStep:
    """Executes an action on a single node.

    Locates the element on screen, takes before/after screenshots,
    executes the action, validates the result, and records stats.

    Args:
        graph: The OCSDGraph containing the node data.
        node_id: The node to interact with.
        next_node_id: The next node in the path (for edge stats tracking).
        dry_run: If True, logs actions without executing them.
        skill_id: Skill name for loading snippet PNGs during locate.

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
        location = locate_element(graph, node_id, skill_id=skill_id)
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
        # Resolve text from input_spec (variable/literal), fall back to edge payload
        text_to_type = _resolve_input_text(node_data)
        if not text_to_type and action_payload:
            text_to_type = action_payload
        if text_to_type:
            type_text(text_to_type, dry_run=dry_run)
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


def _resolve_input_text(
    node_data: dict[str, Any],
    execution_params: dict[str, str] | None = None,
) -> str:
    """Resolves the text to type for a textbox node.

    Resolution chain:
    1. input_spec.type == "literal" → use the stored string directly
    2. input_spec.type == "variable" → look up in execution_params dict
    3. Fallback to env var OCSD_{VARIABLE_NAME}
    4. If nothing found, log warning and return empty string

    Args:
        node_data: The node's data dict from the graph.
        execution_params: Optional dict of variable→value mappings
            passed by the agent at runtime.

    Returns:
        The resolved text string to type, or empty string if unresolved.
    """
    spec = node_data.get("input_spec")
    if not spec:
        return ""

    spec_type = spec.get("type", "")
    spec_value = spec.get("value", "")

    if spec_type == "literal":
        return spec_value

    if spec_type == "variable":
        # Extract variable name from {{var_name}} format
        var_name = spec_value.strip("{}")
        if not var_name:
            return ""

        # Check execution_params first
        if execution_params and var_name in execution_params:
            return execution_params[var_name]

        # Fallback to environment variable
        import os
        env_key = f"OCSD_{var_name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val

        logger.warning(
            "Variable '%s' not provided in params or env (%s) — skipping",
            var_name, env_key,
        )
        return ""

    logger.warning("Unknown input_spec type: %s", spec_type)
    return ""


def run_skill(
    graph: OCSDGraph,
    start_id: str,
    goal_id: str,
    dry_run: bool = False,
    execution_params: dict[str, str] | None = None,
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

        step = execute_node(
            graph, node_id, next_node_id=next_id,
            dry_run=dry_run, skill_id=graph.skill_id,
        )
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


def run_path(
    path: list[str],
    graph: OCSDGraph,
    *,
    dry_run: bool = False,
    event_callback: EventCallback | None = None,
    skip_vlm_validation: bool = False,
) -> ReplayLog:
    """Executes a sequence of nodes with event callbacks.

    Lighter-weight alternative to run_skill() that takes a pre-computed
    path and emits RunnerEventType events for monitoring.

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
    start_time = time.monotonic()

    def emit(event_type: RunnerEventType, data: dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.error("Event callback error: %s", e)

    for step_idx, node_id in enumerate(path):
        emit(RunnerEventType.STEP_START, {"node_id": node_id, "step": step_idx})

        try:
            node_data = graph.get_node(node_id)
        except KeyError:
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

        # Dry-run shortcut: produce a plausible result without screen interaction
        if dry_run:
            pos = node_data.get("relative_position", {})
            x_pct = pos.get("x_pct", 0.5)
            y_pct = pos.get("y_pct", 0.5)
            located_point = Point(int(x_pct * 1920), int(y_pct * 1080))

            emit(
                RunnerEventType.ELEMENT_LOCATED,
                {
                    "node_id": node_id,
                    "point": (located_point.x, located_point.y),
                    "method": "dry_run",
                    "confidence": 1.0,
                },
            )

            step = ReplayStep(
                node_id=node_id,
                located_at=located_point,
                locate_method="dry_run",
                vlm_confidence=1.0,
                pixel_diff_pct=0.0,
                success=True,
            )
            replay_log.append_step(step)

            emit(
                RunnerEventType.STEP_COMPLETE,
                {"node_id": node_id, "step": step_idx, "success": True},
            )
            continue

        # Live execution: delegate to execute_node
        next_id = path[step_idx + 1] if step_idx + 1 < len(path) else None
        step = execute_node(
            graph, node_id, next_node_id=next_id,
            dry_run=False, skill_id=graph.skill_id,
        )
        replay_log.append_step(step)

        if step.success:
            emit(
                RunnerEventType.ELEMENT_LOCATED,
                {
                    "node_id": node_id,
                    "point": (step.located_at.x, step.located_at.y) if step.located_at else (0, 0),
                    "method": step.locate_method,
                    "confidence": step.vlm_confidence,
                },
            )
            emit(RunnerEventType.STEP_COMPLETE, {"node_id": node_id, "step": step_idx, "success": True})
        else:
            emit(RunnerEventType.ELEMENT_NOT_FOUND, {"node_id": node_id})

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    replay_log.duration_ms = elapsed_ms

    if replay_log.overall_success:
        emit(RunnerEventType.PATH_COMPLETE, {"duration_ms": elapsed_ms})
    else:
        emit(RunnerEventType.PATH_FAILED, {"duration_ms": elapsed_ms})

    return replay_log
