"""Step-sequential routine replay engine.

Loads a recorded routine, validates pre-flight conditions, iterates
through steps using the locate cascade, executes actions with human-like
timing, validates results, and handles failures with a 5-stage recovery
cascade. Emits events for overlay integration and creates run logs with
annotated screenshots for post-mortem debugging.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pyautogui

from core.capture import screenshot_full, screenshot_region
from core.config import get_config
from core.executor import (
    click,
    double_click,
    drag,
    hotkey,
    press_enter,
    prompt_user_blocking,
    right_click,
    scroll,
    select_all_extract,
    type_text,
)
from core.locate import locate_element_from_step
from core.types import ElementNotFoundError, LocateResult, Point
from routine.format import Routine
from routine.run_log import (
    annotate_screenshot,
    create_run_dir,
    prune_old_runs,
    save_annotated_screenshot,
    save_run_result,
    setup_run_logger,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class RunEvent(Enum):
    """Events emitted during routine execution."""

    RUN_START = auto()
    PREFLIGHT_OK = auto()
    PREFLIGHT_FAILED = auto()
    STEP_START = auto()
    ELEMENT_LOCATED = auto()
    SCREENSHOT_TAKEN = auto()
    ACTION_EXECUTED = auto()
    VALIDATION_PASSED = auto()
    VALIDATION_FAILED = auto()
    STEP_RETRY = auto()
    STEP_FAILED = auto()
    STEP_COMPLETE = auto()
    LOOP_ITERATION = auto()
    RUN_COMPLETE = auto()
    RUN_FAILED = auto()
    RUN_PAUSED = auto()


class RunCallback(Protocol):
    """Protocol for run event callbacks."""

    def __call__(self, event: RunEvent, data: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a routine execution run."""

    success: bool
    routine_name: str
    run_id: str
    run_dir: Path
    steps_completed: int
    total_steps: int
    duration_ms: int
    failure_step: int | None = None
    failure_reason: str | None = None
    step_results: list[dict[str, Any]] = field(default_factory=list)


class PreflightError(Exception):
    """Raised when pre-flight checks fail."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Pre-flight failed: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit(
    callback: RunCallback | None,
    event: RunEvent,
    data: dict[str, Any],
) -> None:
    """Emit an event via callback if provided."""
    if callback is not None:
        try:
            callback(event, data)
        except Exception:
            logger.debug("Callback error for %s", event.name, exc_info=True)


def preflight_check(routine: Routine, routine_dir: Path) -> list[str]:
    """Validate that all required assets exist before running.

    Args:
        routine: Loaded routine object.
        routine_dir: Root directory of the routine.

    Returns:
        List of error strings. Empty list means all checks passed.
    """
    errors: list[str] = []

    for i, step in enumerate(routine.steps):
        action = step.get("action", "click")
        # Loop steps don't have their own snippet/embedding
        if action in ("loop", "wait", "prompt_user"):
            continue

        snippet_rel = step.get("snippet_path")
        if snippet_rel:
            snippet_path = routine_dir / snippet_rel
            if not snippet_path.exists():
                errors.append(
                    f"Step {i}: snippet file missing: {snippet_rel}"
                )

        embedding_rel = step.get("embedding_path")
        if embedding_rel:
            emb_path = routine_dir / embedding_rel
            if not emb_path.exists():
                errors.append(
                    f"Step {i}: embedding file missing: {embedding_rel}"
                )

    # VLM connectivity check (warning only, not an error)
    try:
        from core.vision import analyze_crop_array  # noqa: F401
        logger.debug("VLM module available for replay")
    except ImportError:
        logger.warning("VLM module not available -- replay will skip VLM stages")

    return errors


def _compute_search_region(
    pos_pct: dict[str, Any],
    screen_w: int,
    screen_h: int,
) -> tuple[int, int, int, int]:
    """Compute a 35% search region centered on the recorded position.

    Args:
        pos_pct: Dict with ``x_pct`` and ``y_pct`` keys.
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.

    Returns:
        Tuple of (rx, ry, rw, rh) clamped to screen bounds.
    """
    cx = int(pos_pct.get("x_pct", 0.5) * screen_w)
    cy = int(pos_pct.get("y_pct", 0.5) * screen_h)
    rw = int(screen_w * 0.35)
    rh = int(screen_h * 0.35)
    rx = max(0, cx - rw // 2)
    ry = max(0, cy - rh // 2)
    # Clamp right/bottom edges
    rw = min(rw, screen_w - rx)
    rh = min(rh, screen_h - ry)
    return rx, ry, rw, rh


def _build_ai_fallback_prompt(
    routine: Routine,
    step_index: int,
    step: dict[str, Any],
    screenshot_b64: str,
) -> str:
    """Build a prompt for LiteLLM AI fallback element location.

    Args:
        routine: The routine being executed.
        step_index: Current step index.
        step: Current step dictionary.
        screenshot_b64: Base64-encoded screenshot PNG.

    Returns:
        Prompt string for the AI model.
    """
    step_summary = []
    for i, s in enumerate(routine.steps):
        marker = ">>>" if i == step_index else "   "
        label = s.get("label", "?")
        action = s.get("action", "click")
        step_summary.append(f"{marker} Step {i}: [{action}] {label}")

    return (
        f"You are helping locate a UI element on screen.\n\n"
        f"Routine: {routine.name}\n"
        f"Steps:\n" + "\n".join(step_summary) + "\n\n"
        f"Current step {step_index}:\n"
        f"  Action: {step.get('action', 'click')}\n"
        f"  Label: {step.get('label', '?')}\n"
        f"  Element type: {step.get('element_type', 'unknown')}\n"
        f"  Expected region: {step.get('region_hint', 'unknown')}\n\n"
        f"Find this element in the screenshot and return JSON: "
        f'{{"x": <int>, "y": <int>, "confidence": <0.0-1.0>}}'
    )


def _failure_cascade(
    step: dict[str, Any],
    routine: Routine,
    routine_dir: Path,
    run_dir: Path,
    callback: RunCallback | None,
    step_index: int = 0,
) -> LocateResult | None:
    """5-stage failure recovery cascade.

    Stages:
    1. Retry at position (2 attempts)
    2. 25% region scan
    3. Full-screen cascade
    4. LiteLLM AI fallback
    5. Abort (return None)

    Args:
        step: Current step dictionary.
        routine: The routine being executed.
        routine_dir: Root directory of the routine.
        run_dir: Current run directory for saving debug screenshots.
        callback: Optional event callback.
        step_index: Current step index for logging.

    Returns:
        LocateResult if found, None if all stages fail.
    """
    label = step.get("label", "?")

    # Stage 1: Retry at position (2 attempts)
    _emit(callback, RunEvent.STEP_RETRY, {
        "stage": 1, "description": "retry_at_position", "step_index": step_index,
    })
    for attempt in range(2):
        try:
            result = locate_element_from_step(step, routine_dir)
            if result is not None:
                logger.info(
                    "Cascade Stage 1: found '%s' on retry %d", label, attempt + 1
                )
                return result
        except ElementNotFoundError:
            time.sleep(1.0)

    # Stage 2: 25% region scan
    _emit(callback, RunEvent.STEP_RETRY, {
        "stage": 2, "description": "region_scan", "step_index": step_index,
    })
    try:
        anchors = step.get("anchors", {})
        pos_pct = anchors.get("position_pct", {})
        sw, sh = pyautogui.size()
        rx, ry, rw, rh = _compute_search_region(pos_pct, sw, sh)
        region_img = screenshot_region(rx, ry, rw, rh)
        if callback:
            callback(RunEvent.SCREENSHOT_TAKEN, {
                "purpose": "region_scan", "step_index": step_index,
            })
        # Try locate on region -- skip position fallback to avoid blind clicks
        result = locate_element_from_step(
            step, routine_dir, skip_position_fallback=True,
        )
        if result is not None:
            logger.info("Cascade Stage 2: found '%s' via region scan", label)
            return result
    except (ElementNotFoundError, Exception) as exc:
        logger.debug("Cascade Stage 2 failed: %s", exc)

    # Stage 3: Full-screen cascade (all stages enabled)
    _emit(callback, RunEvent.STEP_RETRY, {
        "stage": 3, "description": "full_screen", "step_index": step_index,
    })
    try:
        result = locate_element_from_step(step, routine_dir)
        if result is not None:
            logger.info("Cascade Stage 3: found '%s' via full-screen", label)
            return result
    except ElementNotFoundError:
        logger.debug("Cascade Stage 3: full-screen locate failed for '%s'", label)

    # Stage 4: LiteLLM AI fallback
    _emit(callback, RunEvent.STEP_RETRY, {
        "stage": 4, "description": "litellm_fallback", "step_index": step_index,
    })
    try:
        import cv2
        import litellm

        full_img = screenshot_full()
        if callback:
            callback(RunEvent.SCREENSHOT_TAKEN, {
                "purpose": "ai_fallback", "step_index": step_index,
            })
        _, buf = cv2.imencode(".png", full_img)
        screenshot_b64 = base64.b64encode(buf).decode("utf-8")

        prompt = _build_ai_fallback_prompt(routine, step_index, step, screenshot_b64)
        config = get_config()
        litellm_cfg = config.get("recovery", {})
        model = litellm_cfg.get("model", "claude-opus")

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_b64}",
                            },
                        },
                    ],
                }
            ],
            timeout=15,
        )

        content = response.choices[0].message.content
        # Parse JSON from response
        import re
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            parsed = json.loads(json_match.group())
            ai_x = int(parsed.get("x", 0))
            ai_y = int(parsed.get("y", 0))
            ai_conf = float(parsed.get("confidence", 0.0))

            if ai_conf >= 0.5:
                logger.info(
                    "Cascade Stage 4: LiteLLM found '%s' at (%d, %d) conf=%.2f",
                    label, ai_x, ai_y, ai_conf,
                )
                # Save annotated screenshot
                bboxes = [{"x": ai_x - 20, "y": ai_y - 20, "w": 40, "h": 40,
                           "color": (0, 165, 255), "label": f"AI: {label}"}]
                annotated = annotate_screenshot(full_img, bboxes, f"Stage 4: {label}")
                save_annotated_screenshot(run_dir, annotated, f"step{step_index}_ai_found")
                return LocateResult(
                    point=Point(ai_x, ai_y),
                    confidence=ai_conf,
                    method="litellm",
                )
            logger.debug("Cascade Stage 4: AI confidence too low (%.2f)", ai_conf)
    except ImportError:
        logger.debug("LiteLLM not available, skipping Stage 4")
    except Exception as exc:
        logger.debug("Cascade Stage 4 failed: %s", exc)

    # Stage 5: Abort -- save failure screenshot
    logger.warning("Cascade ABORT: could not find '%s' after all stages", label)
    try:
        fail_img = screenshot_full()
        if callback:
            callback(RunEvent.SCREENSHOT_TAKEN, {
                "purpose": "failure_log", "step_index": step_index,
            })
        annotated = annotate_screenshot(fail_img, [], f"FAILED: {label}")
        save_annotated_screenshot(run_dir, annotated, f"step{step_index}_failed")
    except Exception:
        logger.debug("Could not save failure screenshot", exc_info=True)

    return None


def _dispatch_action(
    step: dict[str, Any],
    locate_result: LocateResult,
    dry_run: bool = False,
    prompt_timeout_s: float | None = None,
) -> str:
    """Execute the appropriate action for a step.

    Args:
        step: Step dictionary with action type and parameters.
        locate_result: Location result for element targeting.
        dry_run: If True, log without executing.

    Returns:
        String describing what was executed.

    Raises:
        ValueError: If action is 'loop' (handled separately).
    """
    action = step.get("action", "click")
    point = locate_result.point

    if action == "click":
        click(point.x, point.y, dry_run=dry_run)
        return "clicked"

    elif action == "double_click":
        double_click(point.x, point.y, dry_run=dry_run)
        return "double_clicked"

    elif action == "right_click":
        right_click(point.x, point.y, dry_run=dry_run)
        return "right_clicked"

    elif action == "click_drag":
        drag_target = step.get("drag_target", {})
        # Resolve target from bbox_pct
        sw, sh = pyautogui.size()
        dt_pct = drag_target.get("bbox_pct", {})
        tx = int(dt_pct.get("x_pct", 0.5) * sw + dt_pct.get("w_pct", 0) * sw / 2)
        ty = int(dt_pct.get("y_pct", 0.5) * sh + dt_pct.get("h_pct", 0) * sh / 2)
        drag(point.x, point.y, tx, ty, dry_run=dry_run)
        return "dragged"

    elif action == "type":
        click(point.x, point.y, dry_run=dry_run)
        text = step.get("text_to_type", "")
        type_text(text, dry_run=dry_run)
        if step.get("press_enter"):
            press_enter(dry_run=dry_run)
        return "typed"

    elif action == "scroll":
        scroll_def = step.get("scroll", {})
        direction = scroll_def.get("direction", "down")
        amount = scroll_def.get("amount", 3)
        scroll(point.x, point.y, direction, amount, dry_run=dry_run)
        return "scrolled"

    elif action in ("read", "snip_and_search"):
        try:
            from core.vision import analyze_crop_array

            full_img = screenshot_full()
            bbox = step.get("bbox", {})
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("w", full_img.shape[1])
            h = bbox.get("h", full_img.shape[0])
            crop = full_img[y : y + h, x : x + w]
            vlm_prompt = step.get("vlm_prompt", "Describe what you see")
            result = analyze_crop_array(crop, vlm_prompt)
            return str(result) if result else "read_empty"
        except ImportError:
            logger.warning("VLM not available for read/snip_and_search")
            return "read_skipped"

    elif action == "select_all_extract":
        click(point.x, point.y, dry_run=dry_run)
        text = select_all_extract(dry_run=dry_run)
        return text if text else "extract_empty"

    elif action == "prompt_user":
        question = step.get("question_text", "")
        response = prompt_user_blocking(question, dry_run=dry_run, timeout=prompt_timeout_s)
        return response if response else "prompt_empty"

    elif action == "wait":
        from core.conditions import ConditionChecker

        wait_def = step.get("wait", {})
        cond_type = wait_def.get("condition_type", "fixed_timer")
        params = wait_def.get("params", wait_def)
        timeout = wait_def.get("timeout", 30.0)
        poll_interval = wait_def.get("poll_interval", 2.0)
        checker = ConditionChecker(
            cond_type, params,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        cond_result = checker.poll_until()
        return f"wait_{cond_type}: met={cond_result.met}, elapsed={cond_result.elapsed:.1f}s"

    elif action == "loop":
        raise ValueError(
            "Loop steps must be handled by _handle_loop_step, not _dispatch_action"
        )

    else:
        logger.warning("Unknown action '%s', defaulting to click", action)
        click(point.x, point.y, dry_run=dry_run)
        return f"clicked (unknown action: {action})"


def _handle_loop_step(
    step: dict[str, Any],
    routine: Routine,
    routine_dir: Path,
    run_dir: Path,
    step_index: int,
    callback: RunCallback | None,
    dry_run: bool,
    step_results: list[dict[str, Any]],
    abort_event: threading.Event | None = None,
    prompt_timeout_s: float | None = None,
) -> None:
    """Execute a loop step by replaying body steps and checking exit condition.

    Args:
        step: Loop step dictionary.
        routine: The full routine.
        routine_dir: Root directory of the routine.
        run_dir: Current run directory.
        step_index: Index of the loop step.
        callback: Optional event callback.
        dry_run: If True, log without executing.
        step_results: Accumulator for step results.
    """
    loop_def = step.get("loop", {})
    body_node_ids = loop_def.get("body_step_node_ids", [])
    max_iterations = loop_def.get("max_iterations", 100)

    # Find body steps by matching node_id
    body_steps = []
    for node_id in body_node_ids:
        for s in routine.steps:
            if s.get("node_id") == node_id:
                body_steps.append(s)
                break

    # Build exit condition checker
    exit_cond = loop_def.get("exit_condition", {})
    cond_type = exit_cond.get("condition_type", "n_iterations")
    cond_params = exit_cond.get("params", exit_cond)
    timeout = exit_cond.get("timeout", 0)

    from core.conditions import ConditionChecker

    # For n_iterations, handle iteration counting directly in the loop
    # (creating a new ConditionChecker per iteration would reset the counter)
    n_iter_target: int | None = None
    if cond_type == "n_iterations":
        n_iter_target = cond_params.get("count", 1)

    iteration = 0
    while iteration < max_iterations:
        # Check abort flag between loop iterations
        if abort_event is not None and abort_event.is_set():
            logger.info("Abort requested during loop at iteration %d", iteration)
            break

        iteration += 1
        _emit(callback, RunEvent.LOOP_ITERATION, {
            "step_index": step_index,
            "iteration": iteration,
            "max_iterations": max_iterations,
        })

        # Execute body steps
        for body_step in body_steps:
            body_action = body_step.get("action", "click")
            if body_action in ("wait", "prompt_user"):
                lr = LocateResult(point=Point(0, 0), confidence=1.0, method="none")
            else:
                try:
                    lr = locate_element_from_step(body_step, routine_dir)
                except ElementNotFoundError:
                    lr = _failure_cascade(
                        body_step, routine, routine_dir, run_dir, callback, step_index,
                    )
                    if lr is None:
                        logger.warning(
                            "Loop body step '%s' failed to locate, skipping",
                            body_step.get("label", "?"),
                        )
                        continue

            _dispatch_action(body_step, lr, dry_run=dry_run, prompt_timeout_s=prompt_timeout_s)

        # Check exit condition
        if n_iter_target is not None:
            # Simple iteration count -- no ConditionChecker needed
            if iteration >= n_iter_target:
                logger.info("Loop n_iterations met after %d iterations", iteration)
                break
        else:
            # Use ConditionChecker for non-count conditions (single poll)
            checker = ConditionChecker(
                cond_type, cond_params,
                poll_interval=0.1,
                timeout=timeout if timeout > 0 else 0.5,
                max_iterations=1,
            )
            cond_result = checker.poll_until()
            if cond_result.met:
                logger.info("Loop exit condition met after %d iterations", iteration)
                break

    step_results.append({
        "step_index": step_index,
        "action": "loop",
        "iterations": iteration,
        "label": step.get("label", "loop"),
    })


def run_routine(
    routine_dir: Path,
    *,
    callback: RunCallback | None = None,
    dry_run: bool = False,
    abort_event: threading.Event | None = None,
    prompt_timeout_s: float | None = None,
) -> RunResult:
    """Execute a routine from its directory.

    Main entry point for the replay engine. Loads the routine, runs
    pre-flight checks, iterates through steps with locate/execute/validate,
    handles failures with the 5-stage cascade, and creates run logs.

    Args:
        routine_dir: Path to the routine directory containing routine.json.
        callback: Optional event callback for overlay integration.
        dry_run: If True, log actions without executing them.
        abort_event: Optional threading.Event; when set, pauses run between steps.

    Returns:
        RunResult with execution outcome and metadata.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()

    # Load routine
    routine = Routine.load(routine_dir)
    total_steps = len(routine.steps)

    # Pre-flight checks
    errors = preflight_check(routine, routine_dir)
    if errors:
        _emit(callback, RunEvent.PREFLIGHT_FAILED, {"errors": errors})
        logger.error("Pre-flight failed: %s", errors)
        # Still create a run dir for the failure record
        run_dir = create_run_dir(routine_dir)
        duration_ms = int((time.monotonic() - start_time) * 1000)
        result = RunResult(
            success=False,
            routine_name=routine.name,
            run_id=run_dir.name,
            run_dir=run_dir,
            steps_completed=0,
            total_steps=total_steps,
            duration_ms=duration_ms,
            failure_step=None,
            failure_reason=f"Pre-flight failed: {'; '.join(errors)}",
        )
        save_run_result(run_dir, {
            "run_id": run_dir.name,
            "routine_name": routine.name,
            "status": "failed",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "steps_completed": 0,
            "total_steps": total_steps,
            "failure_step": None,
            "failure_reason": result.failure_reason,
        })
        return result

    _emit(callback, RunEvent.PREFLIGHT_OK, {"routine_name": routine.name})

    # Security scan
    try:
        from hub.scanner import scan_routine
        scan_result = scan_routine(routine.to_dict())
        if not scan_result.is_safe:
            scan_errors = [
                f"Security scan failed (risk={scan_result.risk_score:.2f}): {w}"
                for w in scan_result.warnings
            ]
            _emit(callback, RunEvent.PREFLIGHT_FAILED, {"errors": scan_errors})
            logger.error("Security scan blocked routine: %s", scan_errors)
            run_dir = create_run_dir(routine_dir)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            result = RunResult(
                success=False,
                routine_name=routine.name,
                run_id=run_dir.name,
                run_dir=run_dir,
                steps_completed=0,
                total_steps=total_steps,
                duration_ms=duration_ms,
                failure_step=None,
                failure_reason=scan_errors[0] if scan_errors else "Security scan failed",
            )
            save_run_result(run_dir, {
                "run_id": run_dir.name,
                "routine_name": routine.name,
                "status": "failed",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration_ms,
                "steps_completed": 0,
                "total_steps": total_steps,
                "failure_step": None,
                "failure_reason": result.failure_reason,
            })
            prune_old_runs(routine_dir)
            return result
    except ImportError:
        logger.debug("hub.scanner not available, skipping security scan")

    # Create run directory and logger
    run_dir = create_run_dir(routine_dir)
    handler = setup_run_logger(run_dir)
    logging.getLogger().addHandler(handler)

    # Desktop setup
    config = get_config()
    replay_cfg = config.get("replay", {})
    if routine.start_from == "desktop" and replay_cfg.get("auto_minimize", True):
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["powershell", "-c",
                     "(New-Object -ComObject Shell.Application).MinimizeAll()"],
                    timeout=5, capture_output=True,
                )
                time.sleep(1.0)
            except Exception:
                logger.debug("Could not minimize windows", exc_info=True)

    _emit(callback, RunEvent.RUN_START, {
        "routine_name": routine.name,
        "total_steps": total_steps,
        "run_dir": str(run_dir),
    })

    # Step loop
    step_results: list[dict[str, Any]] = []
    steps_completed = 0
    failure_step: int | None = None
    failure_reason: str | None = None

    for i, step in enumerate(routine.steps):
        # Check abort flag between steps
        if abort_event is not None and abort_event.is_set():
            logger.info("Abort requested, pausing run at step %d", i)
            _emit(callback, RunEvent.RUN_PAUSED, {
                "routine_name": routine.name,
                "paused_step": i,
                "reason": "Aborted by user/agent",
            })
            failure_step = i
            failure_reason = "Paused by user/agent"
            break

        action = step.get("action", "click")
        label = step.get("label", f"step_{i}")

        # Handle loop steps inline
        if action == "loop":
            _handle_loop_step(
                step, routine, routine_dir, run_dir, i,
                callback, dry_run, step_results,
                abort_event=abort_event,
                prompt_timeout_s=prompt_timeout_s,
            )
            steps_completed += 1
            continue

        _emit(callback, RunEvent.STEP_START, {
            "step_index": i,
            "total_steps": total_steps,
            "label": label,
            "action": action,
        })

        # Locate element (skip for actions that don't need location)
        locate_result: LocateResult | None = None
        if action in ("wait", "prompt_user"):
            locate_result = LocateResult(
                point=Point(0, 0), confidence=1.0, method="none",
            )
        else:
            try:
                locate_result = locate_element_from_step(step, routine_dir)
                _emit(callback, RunEvent.ELEMENT_LOCATED, {
                    "step_index": i,
                    "point": {"x": locate_result.point.x, "y": locate_result.point.y},
                    "method": locate_result.method,
                    "confidence": locate_result.confidence,
                })
            except ElementNotFoundError:
                # Enter failure cascade
                locate_result = _failure_cascade(
                    step, routine, routine_dir, run_dir, callback, i,
                )
                if locate_result is None:
                    # ABORT -- never blind-click
                    failure_step = i
                    failure_reason = (
                        f"Could not find element '{label}' after full cascade"
                    )
                    _emit(callback, RunEvent.STEP_FAILED, {
                        "step_index": i,
                        "label": label,
                        "reason": failure_reason,
                    })
                    break

        # Screenshot before action
        before_img: np.ndarray | None = None
        if action not in ("wait", "prompt_user"):
            before_img = screenshot_full()
            if callback:
                callback(RunEvent.SCREENSHOT_TAKEN, {
                    "purpose": "before_action", "step_index": i,
                })

        # Execute action
        action_result = _dispatch_action(step, locate_result, dry_run=dry_run, prompt_timeout_s=prompt_timeout_s)
        _emit(callback, RunEvent.ACTION_EXECUTED, {
            "step_index": i,
            "action": action,
            "result": action_result,
        })

        # Post-action validation
        skip_validation = action in (
            "wait", "prompt_user", "read", "snip_and_search", "select_all_extract",
        )
        if not skip_validation and before_img is not None:
            after_img = screenshot_full()
            if callback:
                callback(RunEvent.SCREENSHOT_TAKEN, {
                    "purpose": "after_action", "step_index": i,
                })

            try:
                from mapper.validator import validate_action

                validation = validate_action(
                    before_img, after_img,
                    f"{action} on '{label}'",
                )
                if not validation.success:
                    _emit(callback, RunEvent.VALIDATION_FAILED, {
                        "step_index": i,
                        "confidence": validation.confidence,
                        "notes": validation.notes,
                    })
                    logger.warning(
                        "Step %d validation failed: %s", i, validation.notes,
                    )
                else:
                    _emit(callback, RunEvent.VALIDATION_PASSED, {
                        "step_index": i,
                        "confidence": validation.confidence,
                    })
            except Exception as exc:
                logger.debug("Validation error (non-fatal): %s", exc)

        _emit(callback, RunEvent.STEP_COMPLETE, {
            "step_index": i,
            "label": label,
            "action": action,
            "result": action_result,
        })
        steps_completed += 1
        step_results.append({
            "step_index": i,
            "action": action,
            "label": label,
            "result": action_result,
            "located_method": locate_result.method if locate_result else None,
        })

    # Build final result
    duration_ms = int((time.monotonic() - start_time) * 1000)
    success = failure_reason is None
    completed_at = datetime.now(timezone.utc).isoformat()

    run_result = RunResult(
        success=success,
        routine_name=routine.name,
        run_id=run_dir.name,
        run_dir=run_dir,
        steps_completed=steps_completed,
        total_steps=total_steps,
        duration_ms=duration_ms,
        failure_step=failure_step,
        failure_reason=failure_reason,
        step_results=step_results,
    )

    # Save run result
    save_run_result(run_dir, {
        "run_id": run_dir.name,
        "routine_name": routine.name,
        "status": "success" if success else "failed",
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "steps_completed": steps_completed,
        "total_steps": total_steps,
        "failure_step": failure_step,
        "failure_reason": failure_reason,
    })

    # Emit final event
    if success:
        _emit(callback, RunEvent.RUN_COMPLETE, {
            "routine_name": routine.name,
            "steps_completed": steps_completed,
            "duration_ms": duration_ms,
        })
    else:
        _emit(callback, RunEvent.RUN_FAILED, {
            "routine_name": routine.name,
            "failure_step": failure_step,
            "failure_reason": failure_reason,
        })

    # Cleanup
    logging.getLogger().removeHandler(handler)
    handler.close()

    # Prune old runs
    try:
        prune_old_runs(routine_dir)
    except Exception:
        logger.debug("Prune error (non-fatal)", exc_info=True)

    return run_result
