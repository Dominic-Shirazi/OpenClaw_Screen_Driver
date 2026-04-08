"""Shared condition-checking engine for wait and loop actions.

Polls a condition on the calling thread (expected to be a background thread).
Delivers results via callback. Used by both wait steps (bodyless) and loop
steps (with body actions between polls).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConditionResult:
    """Result of a condition check cycle."""

    met: bool
    timed_out: bool
    iterations: int
    elapsed: float
    detail: str = ""


class ConditionChecker:
    """Polls a condition until met or timeout/max-iterations reached.

    Args:
        condition_type: One of "fixed_timer", "element_appears",
            "screen_change", "vlm_check", "text_matches", "n_iterations".
        params: Type-specific parameters dict.
        poll_interval: Seconds between polls (default 2.0).
        timeout: Maximum seconds before giving up. Must be positive for all
            types except "n_iterations".
        max_iterations: Maximum poll cycles (auto-set for n_iterations).
    """

    VALID_TYPES = frozenset({
        "fixed_timer",
        "element_appears",
        "screen_change",
        "vlm_check",
        "text_matches",
        "n_iterations",
    })

    def __init__(
        self,
        condition_type: str,
        params: dict[str, Any],
        poll_interval: float = 2.0,
        timeout: float = 30.0,
        max_iterations: int | None = None,
    ) -> None:
        if condition_type not in self.VALID_TYPES:
            raise ValueError(f"Unknown condition type: {condition_type}")
        if timeout <= 0 and condition_type not in ("n_iterations", "fixed_timer"):
            raise ValueError("timeout must be positive")
        self.condition_type = condition_type
        self.params = params
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.max_iterations = max_iterations
        self._baseline: np.ndarray | None = None
        self._cancelled = False
        self._iteration_count = 0

        # For n_iterations, auto-set max_iterations from params
        if condition_type == "n_iterations":
            self.max_iterations = params.get("count", 1)

    def cancel(self) -> None:
        """Request cancellation of the polling loop."""
        self._cancelled = True

    def poll_until(self) -> ConditionResult:
        """Run polling loop on the calling thread. Blocks until done.

        Returns:
            ConditionResult with met, timed_out, iterations, elapsed.
        """
        start = time.monotonic()
        iterations = 0

        # Take baseline screenshot for screen_change
        if self.condition_type == "screen_change":
            self._baseline = self._take_screenshot()

        while not self._cancelled:
            iterations += 1
            elapsed = time.monotonic() - start

            # Check timeout (skip for n_iterations with timeout <= 0)
            if self.timeout > 0 and elapsed >= self.timeout:
                return ConditionResult(
                    met=False,
                    timed_out=True,
                    iterations=iterations,
                    elapsed=elapsed,
                    detail="Timeout reached",
                )

            # Check condition
            met = self._check()
            if met:
                return ConditionResult(
                    met=True,
                    timed_out=False,
                    iterations=iterations,
                    elapsed=time.monotonic() - start,
                )

            # Check max iterations (after check, so n_iterations can resolve on Nth)
            if self.max_iterations and iterations >= self.max_iterations:
                return ConditionResult(
                    met=False,
                    timed_out=True,
                    iterations=iterations,
                    elapsed=time.monotonic() - start,
                    detail=f"Max iterations ({self.max_iterations}) reached",
                )

            # Sleep before next poll (except for fixed_timer which handles its own)
            if self.condition_type != "fixed_timer":
                time.sleep(self.poll_interval)

        return ConditionResult(
            met=False,
            timed_out=False,
            iterations=iterations,
            elapsed=time.monotonic() - start,
            detail="Cancelled",
        )

    def _check(self) -> bool:
        """Dispatch to the appropriate check function."""
        match self.condition_type:
            case "fixed_timer":
                return self._check_fixed_timer()
            case "screen_change":
                return self._check_screen_change()
            case "element_appears":
                return self._check_element_appears()
            case "vlm_check":
                return self._check_vlm()
            case "text_matches":
                return self._check_text_matches()
            case "n_iterations":
                return self._check_n_iterations()
            case _:
                return False

    def _check_fixed_timer(self) -> bool:
        """Sleep for the specified duration, then return True."""
        seconds = self.params.get("seconds", self.timeout)
        time.sleep(seconds)
        return True

    def _check_screen_change(self) -> bool:
        """Compare current screenshot against baseline using pixel diff."""
        import cv2

        current = self._take_screenshot()
        if self._baseline is None or current is None:
            return False
        threshold = self.params.get("threshold", 0.05)
        gray_base = cv2.cvtColor(self._baseline, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_base, gray_curr)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed = cv2.countNonZero(thresh)
        total = thresh.shape[0] * thresh.shape[1]
        return (changed / total) > threshold if total > 0 else False

    def _check_element_appears(self) -> bool:
        """Locate element using cascade with adaptive VLM escalation."""
        step = self.params.get("step")
        routine_dir = self.params.get("routine_dir")
        if not step or not routine_dir:
            logger.warning("element_appears missing step or routine_dir params")
            return False

        from pathlib import Path

        from core.locate import locate_element_from_step

        # Updated: increment _iteration_count here — was only incremented by
        # _check_n_iterations, so VLM escalation never fired — 2026-04-03
        self._iteration_count += 1
        # Adaptive: first 3 polls skip VLM (stages 1-3 only), then include VLM
        use_vlm = self._iteration_count >= 3
        try:
            result = locate_element_from_step(
                step,
                Path(routine_dir),
                skip_vlm=not use_vlm,
                skip_position_fallback=True,  # Never use position for conditions
            )
            return result is not None
        except Exception:
            return False

    def _check_vlm(self) -> bool:
        """Ask VLM a yes/no question about the current screen."""
        prompt = self.params.get("prompt", "")
        if not prompt:
            return False
        try:
            screenshot = self._take_screenshot()
            if screenshot is None:
                return False
            from core.vision import analyze_crop_array

            result = analyze_crop_array(screenshot, prompt)
            if result and isinstance(result, dict):
                answer = str(result.get("answer", "")).lower()
                return answer in ("yes", "true", "1")
            return False
        except Exception as e:
            logger.warning("VLM check failed: %s", e)
            return False

    def _check_text_matches(self) -> bool:
        """Fuzzy OCR text matching with Levenshtein tolerance."""
        target = self.params.get("target_text", "")
        if not target:
            return False

        from core.ocr import find_text_on_screen

        # Try exact match first via scoped OCR
        hint = self.params.get("position_hint", {})
        result = find_text_on_screen(
            target,
            hint_x=hint.get("x"),
            hint_y=hint.get("y"),
            search_radius=400,
        )
        if result is not None:
            return True

        # Fuzzy fallback: OCR full region and check Levenshtein distance
        try:
            import cv2
            import pytesseract
            from rapidfuzz.distance import Levenshtein

            screenshot = self._take_screenshot()
            if screenshot is None:
                return False

            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            texts = [t for t in ocr_data.get("text", []) if t.strip()]

            # Threshold: at least 2 edits, plus 1 per 5 chars, capped at 5
            threshold = min(5, max(2, len(target) // 5))
            for text in texts:
                dist = Levenshtein.distance(target.lower(), text.lower())
                if dist <= threshold:
                    logger.info("text_matches: '%s' ~= '%s' (dist=%d)", target, text, dist)
                    return True
        except ImportError:
            logger.warning("rapidfuzz not installed, fuzzy text matching unavailable")
        except Exception as e:
            logger.warning("text_matches fuzzy check failed: %s", e)

        return False

    def _check_n_iterations(self) -> bool:
        """Track iteration count and return True on the Nth call."""
        count = self.params.get("count", 1)
        self._iteration_count += 1
        return self._iteration_count >= count

    def _take_screenshot(self) -> np.ndarray | None:
        """Capture a full-screen screenshot, returning None on failure."""
        try:
            from core.capture import screenshot_full

            return screenshot_full()
        except Exception as e:
            logger.warning("Screenshot failed in condition check: %s", e)
            return None
