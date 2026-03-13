"""Full replay orchestrator with pre-flight checks and recovery.

Wraps the runner's core replay logic with two Wave 5 capabilities:

1. **Pre-flight fingerprint check** — before replay begins, captures
   the screen and asks the VLM "does this look like the expected
   starting state?"  Aborts early if the app is on the wrong page.

2. **Recovery LLM** — when a step fails (element not found, low
   confidence), captures the screen and asks the VLM to diagnose
   what went wrong and suggest a recovery action (e.g. scroll,
   wait, navigate back).

The orchestrator delegates actual node execution to mapper.runner.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.config import get_config
from core.types import ReplayLog, ReplayStep
from mapper.graph import OCSDGraph
from mapper.pathfinder import get_execution_plan
from mapper.runner import (
    EventCallback,
    RunnerEventType,
    execute_node,
    locate_element,
)

logger = logging.getLogger(__name__)


def preflight_check(
    graph: OCSDGraph,
    start_id: str,
    *,
    skip_vlm: bool = False,
) -> dict[str, Any]:
    """Validates that the screen matches the expected starting state.

    Captures the current screen and compares it against the start node's
    expected position and visual context using VLM analysis.

    Args:
        graph: The skill graph.
        start_id: The node where replay will begin.
        skip_vlm: If True, skip VLM validation and return optimistic result.

    Returns:
        Dict with:
        - "passed": bool — whether the preflight check passed
        - "confidence": float — 0.0-1.0
        - "notes": str — human-readable explanation
        - "screen_state": str — VLM description of current screen (if available)
    """
    if skip_vlm:
        return {
            "passed": True,
            "confidence": 0.5,
            "notes": "VLM skipped, assuming correct starting state",
            "screen_state": "",
        }

    try:
        node_data = graph.get_node(start_id)
    except KeyError:
        return {
            "passed": False,
            "confidence": 0.0,
            "notes": f"Start node {start_id[:8]} not found in graph",
            "screen_state": "",
        }

    label = node_data.get("label", "")
    element_type = node_data.get("element_type", "unknown")

    # Try to locate the start element — if we can find it, the screen
    # is likely in the right state
    try:
        result = locate_element(graph, start_id, skill_id=graph.skill_id)
        if result.confidence >= 0.5:
            logger.info(
                "Preflight: start element [%s] located at (%d, %d) conf=%.2f",
                start_id[:8], result.point.x, result.point.y, result.confidence,
            )
            return {
                "passed": True,
                "confidence": result.confidence,
                "notes": (
                    f"Start element '{label}' ({element_type}) found "
                    f"via {result.method} (confidence: {result.confidence:.2f})"
                ),
                "screen_state": f"Element '{label}' visible on screen",
            }
        else:
            return {
                "passed": False,
                "confidence": result.confidence,
                "notes": (
                    f"Start element '{label}' found but low confidence "
                    f"({result.confidence:.2f}) — screen may not be in expected state"
                ),
                "screen_state": "",
            }
    except Exception as e:
        logger.warning("Preflight: could not locate start element: %s", e)

    # Fallback: try VLM destination validation
    try:
        from core.capture import screenshot_full
        from mapper.validator import validate_destination

        screenshot = screenshot_full()
        expected_desc = (
            f"Screen should show a '{element_type}' element labeled '{label}'"
        )
        result = validate_destination(screenshot, expected_desc)

        return {
            "passed": result.success,
            "confidence": result.confidence,
            "notes": result.notes,
            "screen_state": result.notes,
        }
    except Exception as e:
        logger.warning("Preflight VLM validation failed: %s", e)
        return {
            "passed": False,
            "confidence": 0.0,
            "notes": f"Preflight check failed: {e}",
            "screen_state": "",
        }


def diagnose_failure(
    graph: OCSDGraph,
    failed_node_id: str,
    step_error: str | None = None,
) -> dict[str, Any]:
    """Uses VLM to diagnose why a step failed and suggest recovery.

    Captures the current screen and asks the VLM to analyze what
    happened and what the user/orchestrator could do next.

    Args:
        graph: The skill graph.
        failed_node_id: The node that failed to execute.
        step_error: The error message from the failed step.

    Returns:
        Dict with:
        - "diagnosis": str — what the VLM thinks went wrong
        - "suggestion": str — suggested recovery action
        - "action": str — one of "retry", "scroll_down", "scroll_up",
                          "wait", "navigate_back", "abort"
        - "confidence": float
    """
    try:
        node_data = graph.get_node(failed_node_id)
    except KeyError:
        return {
            "diagnosis": f"Node {failed_node_id[:8]} not found",
            "suggestion": "Check graph integrity",
            "action": "abort",
            "confidence": 0.0,
        }

    label = node_data.get("label", "unknown")
    element_type = node_data.get("element_type", "unknown")

    try:
        import tempfile
        from pathlib import Path

        import cv2

        from core.capture import screenshot_full
        from core.vision import analyze_crop

        screenshot = screenshot_full()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = Path(tmp.name)
            cv2.imwrite(str(img_path), screenshot)

        try:
            prompt = (
                f"I was trying to find a '{element_type}' element labeled "
                f"'{label}' on this screen but could not. "
                f"Error: {step_error or 'element not found'}. "
                f"What do you see on the screen? Is the element visible? "
                f"If not, what should I do — scroll down, wait for loading, "
                f"go back, or something else?"
            )
            result = analyze_crop(str(img_path), prompt)

            # Parse the VLM response to extract an action
            vlm_label = str(result.get("label_guess", "")).lower()
            vlm_confidence = float(result.get("confidence", 0.3))

            action = _parse_recovery_action(vlm_label)

            return {
                "diagnosis": vlm_label,
                "suggestion": f"VLM suggests: {vlm_label}",
                "action": action,
                "confidence": vlm_confidence,
            }
        finally:
            if img_path.exists():
                img_path.unlink()

    except ImportError:
        logger.debug("VLM not available for failure diagnosis")
    except Exception as e:
        logger.warning("Failure diagnosis error: %s", e)

    # No VLM available — suggest generic retry
    return {
        "diagnosis": f"Could not find '{label}' ({element_type})",
        "suggestion": "Try scrolling or waiting for the page to load",
        "action": "retry",
        "confidence": 0.2,
    }


def _parse_recovery_action(vlm_text: str) -> str:
    """Parses VLM response text into a recovery action keyword.

    Args:
        vlm_text: Raw text from VLM analysis.

    Returns:
        One of: "retry", "scroll_down", "scroll_up", "wait",
                "navigate_back", "abort".
    """
    text = vlm_text.lower()

    if "scroll down" in text or "below" in text:
        return "scroll_down"
    if "scroll up" in text or "above" in text:
        return "scroll_up"
    if "wait" in text or "loading" in text or "spinner" in text:
        return "wait"
    if "back" in text or "previous" in text or "navigate back" in text:
        return "navigate_back"
    if "not found" in text or "wrong page" in text or "different" in text:
        return "abort"
    return "retry"


def _execute_recovery(action: str) -> bool:
    """Executes a recovery action suggested by the VLM.

    Args:
        action: Recovery action keyword.

    Returns:
        True if recovery was attempted, False if action is "abort".
    """
    import time

    if action == "scroll_down":
        try:
            from core.executor import scroll
            import pyautogui
            sw, sh = pyautogui.size()
            scroll(sw // 2, sh // 2, direction="down", amount=5)
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.warning("Recovery scroll_down failed: %s", e)
            return False

    elif action == "scroll_up":
        try:
            from core.executor import scroll
            import pyautogui
            sw, sh = pyautogui.size()
            scroll(sw // 2, sh // 2, direction="up", amount=5)
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.warning("Recovery scroll_up failed: %s", e)
            return False

    elif action == "wait":
        time.sleep(2.0)
        return True

    elif action == "navigate_back":
        try:
            import pyautogui
            pyautogui.hotkey("alt", "left")
            time.sleep(1.0)
            return True
        except Exception as e:
            logger.warning("Recovery navigate_back failed: %s", e)
            return False

    elif action == "abort":
        return False

    else:  # "retry"
        time.sleep(0.5)
        return True


def orchestrate_skill(
    graph: OCSDGraph,
    start_id: str,
    goal_id: str,
    *,
    dry_run: bool = False,
    skip_preflight: bool = False,
    skip_vlm: bool = False,
    max_retries: int = 2,
    event_callback: EventCallback | None = None,
) -> ReplayLog:
    """Full replay orchestration with preflight and recovery.

    This is the top-level entry point for executing a skill with
    all Wave 5 features enabled:

    1. Pre-flight fingerprint check
    2. Path execution with per-step failure recovery
    3. VLM-based diagnosis on failure

    Args:
        graph: The OCSDGraph containing the recorded skill.
        start_id: The starting node ID.
        goal_id: The goal/destination node ID.
        dry_run: If True, simulates without actual input.
        skip_preflight: If True, skip the preflight check.
        skip_vlm: If True, skip all VLM calls.
        max_retries: Max recovery attempts per failed step.
        event_callback: Optional callback for execution events.

    Returns:
        ReplayLog with complete execution history.
    """
    config = get_config()
    abort_on_failure = config.get("execution", {}).get("abort_on_failure", True)

    def emit(event_type: RunnerEventType, data: dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.error("Event callback error: %s", e)

    # ------------------------------------------------------------------
    # 1. Pre-flight check
    # ------------------------------------------------------------------
    if not skip_preflight and not dry_run:
        logger.info("Running pre-flight check...")
        preflight = preflight_check(graph, start_id, skip_vlm=skip_vlm)
        if not preflight["passed"]:
            logger.error(
                "Pre-flight FAILED: %s (confidence: %.2f)",
                preflight["notes"],
                preflight["confidence"],
            )
            replay_log = ReplayLog(skill_id=graph.skill_id)
            replay_log.append_step(ReplayStep(
                node_id=start_id,
                located_at=None,
                locate_method="preflight",
                vlm_confidence=preflight["confidence"],
                pixel_diff_pct=0.0,
                success=False,
                error=f"Pre-flight failed: {preflight['notes']}",
            ))
            emit(RunnerEventType.PATH_FAILED, {"reason": "preflight_failed"})
            return replay_log
        logger.info("Pre-flight PASSED: %s", preflight["notes"])

    # ------------------------------------------------------------------
    # 2. Build execution plan
    # ------------------------------------------------------------------
    start_time = time.monotonic()
    replay_log = ReplayLog(skill_id=graph.skill_id)

    plan = get_execution_plan(graph, start_id, goal_id)
    path = plan["path"]

    logger.info(
        "Orchestrator: %d nodes, reliability=%.2f",
        len(path), plan["estimated_reliability"],
    )

    # ------------------------------------------------------------------
    # 3. Execute with recovery
    # ------------------------------------------------------------------
    for i, node_id in enumerate(path):
        next_id = path[i + 1] if i < len(path) - 1 else None
        node_label = graph.get_node(node_id).get("label", node_id[:8])

        emit(RunnerEventType.STEP_START, {
            "node_id": node_id, "step": i, "label": node_label,
        })

        logger.info("Step %d/%d: %s", i + 1, len(path), node_label)

        # Try executing the step, with retries on failure
        step: ReplayStep | None = None
        for attempt in range(1 + max_retries):
            step = execute_node(
                graph, node_id, next_node_id=next_id,
                dry_run=dry_run, skill_id=graph.skill_id,
            )

            if step.success:
                emit(RunnerEventType.STEP_COMPLETE, {
                    "node_id": node_id, "step": i, "success": True,
                    "attempt": attempt + 1,
                })
                break

            # Step failed — attempt recovery if retries remain
            if attempt < max_retries and not dry_run and not skip_vlm:
                logger.warning(
                    "Step %d failed (attempt %d/%d), diagnosing...",
                    i + 1, attempt + 1, max_retries + 1,
                )
                diagnosis = diagnose_failure(
                    graph, node_id, step_error=step.error,
                )
                logger.info(
                    "Diagnosis: %s -> action: %s",
                    diagnosis["diagnosis"][:80],
                    diagnosis["action"],
                )

                recovered = _execute_recovery(diagnosis["action"])
                if not recovered:
                    logger.info("Recovery action '%s' indicates abort", diagnosis["action"])
                    break
            else:
                break

        assert step is not None
        replay_log.append_step(step)

        if not step.success:
            logger.error("Step %d failed: %s", i + 1, step.error or "unknown")
            emit(RunnerEventType.VALIDATION_FAILED, {
                "node_id": node_id, "step": i, "error": step.error,
            })
            if abort_on_failure:
                logger.info("Aborting orchestration due to step failure")
                break

    # ------------------------------------------------------------------
    # 4. Finalize
    # ------------------------------------------------------------------
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    replay_log.duration_ms = elapsed_ms

    if replay_log.overall_success:
        emit(RunnerEventType.PATH_COMPLETE, {"duration_ms": elapsed_ms})
        logger.info("Orchestration complete: SUCCESS (%dms)", elapsed_ms)
    else:
        emit(RunnerEventType.PATH_FAILED, {"duration_ms": elapsed_ms})
        logger.info("Orchestration complete: FAILED (%dms)", elapsed_ms)

    return replay_log
