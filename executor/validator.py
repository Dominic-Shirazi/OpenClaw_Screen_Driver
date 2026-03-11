"""Post-action validation for the OCSD execution engine.

Implements a decision tree to determine whether an action succeeded:
1. Pixel diff detection — did the screen change at all?
2. VLM confirmation — does the change match the intended action?
3. Confidence threshold enforcement — is the VLM confident enough?

Used by runner.py after every action to decide whether to continue
or flag a problem.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.capture import pixel_diff
from core.config import get_config
from core.types import ConfirmResult

logger = logging.getLogger(__name__)


def _save_temp_image(img: np.ndarray) -> str:
    """Saves a numpy array to a temporary PNG file.

    Args:
        img: BGR image as numpy array.

    Returns:
        Path to the temporary file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    tmp.close()
    return tmp.name


def validate_action(
    before_img: np.ndarray,
    after_img: np.ndarray,
    intended_action: str,
    *,
    confidence_threshold: float | None = None,
    diff_threshold: float | None = None,
    skip_vlm: bool = False,
) -> ConfirmResult:
    """Validates whether an action produced the expected result.

    Decision tree:
    1. Compute pixel diff between before and after screenshots
    2. If diff is below threshold → no visible change → likely failed
    3. If diff exceeds threshold → something changed → ask VLM to confirm
    4. If VLM confidence is below threshold → uncertain result
    5. Return final verdict

    Args:
        before_img: Screenshot taken before the action (BGR numpy array).
        after_img: Screenshot taken after the action (BGR numpy array).
        intended_action: Human-readable description of what should have happened.
        confidence_threshold: Minimum VLM confidence to consider success.
                            Defaults to config execution.default_confidence.
        diff_threshold: Pixel diff threshold to trigger VLM confirmation.
                       Defaults to config execution.pixel_diff_threshold.
        skip_vlm: If True, skip VLM confirmation and decide based on
                 pixel diff alone. Useful for fast validation or when
                 Ollama is unavailable.

    Returns:
        ConfirmResult with success, confidence, and notes.
    """
    config = get_config()
    exec_config = config.get("execution", {})

    if confidence_threshold is None:
        confidence_threshold = exec_config.get("default_confidence", 0.75)
    if diff_threshold is None:
        diff_threshold = exec_config.get("pixel_diff_threshold", 0.08)

    # Step 1: Pixel diff
    diff_pct = pixel_diff(before_img, after_img)
    logger.debug(
        "Pixel diff: %.4f (threshold: %.4f)", diff_pct, diff_threshold
    )

    # Step 2: No-change detection
    if diff_pct < diff_threshold:
        return ConfirmResult(
            success=False,
            confidence=1.0 - diff_pct,  # High confidence it didn't change
            notes=(
                f"No visible change detected (diff={diff_pct:.4f}, "
                f"threshold={diff_threshold:.4f}). "
                f"Action '{intended_action}' may not have had any effect."
            ),
        )

    # Step 3: If skip_vlm, decide based on pixel diff alone
    if skip_vlm:
        return ConfirmResult(
            success=True,
            confidence=min(diff_pct * 2, 0.8),  # Moderate confidence from diff alone
            notes=(
                f"Screen changed (diff={diff_pct:.4f}). "
                f"VLM confirmation skipped. "
                f"Assuming action '{intended_action}' succeeded based on pixel diff."
            ),
        )

    # Step 4: VLM confirmation
    vlm_result = _vlm_confirm(before_img, after_img, intended_action)

    # Step 5: Confidence threshold enforcement
    if vlm_result.confidence < confidence_threshold:
        return ConfirmResult(
            success=False,
            confidence=vlm_result.confidence,
            notes=(
                f"VLM confidence too low: {vlm_result.confidence:.2f} < "
                f"{confidence_threshold:.2f}. {vlm_result.notes}"
            ),
        )

    return vlm_result


def _vlm_confirm(
    before_img: np.ndarray,
    after_img: np.ndarray,
    intended_action: str,
) -> ConfirmResult:
    """Calls the VLM to confirm whether an action succeeded.

    Handles VLM errors gracefully — returns a low-confidence result
    rather than crashing.

    Args:
        before_img: Before screenshot (BGR numpy array).
        after_img: After screenshot (BGR numpy array).
        intended_action: What the action was supposed to do.

    Returns:
        ConfirmResult from VLM, or a fallback result on error.
    """
    before_path = _save_temp_image(before_img)
    after_path = _save_temp_image(after_img)

    try:
        from core.vision import confirm_action

        result_dict = confirm_action(before_path, after_path, intended_action)
        return ConfirmResult(
            success=result_dict.get("success", False),
            confidence=result_dict.get("confidence", 0.0),
            notes=result_dict.get("notes", ""),
        )
    except Exception as e:
        logger.warning("VLM confirmation failed: %s", e)
        return ConfirmResult(
            success=False,
            confidence=0.0,
            notes=f"VLM confirmation failed: {e}",
        )
    finally:
        # Clean up temp files
        Path(before_path).unlink(missing_ok=True)
        Path(after_path).unlink(missing_ok=True)


def validate_element_located(
    screenshot: np.ndarray,
    element_rect: tuple[int, int, int, int],
    expected_type: str,
    expected_label: str,
    *,
    confidence_threshold: float | None = None,
    skip_vlm: bool = False,
) -> ConfirmResult:
    """Validates that a located element is actually what we expected.

    Pre-action validation. Crops the element region from the screenshot
    and asks the VLM if it matches the expected type and label.

    Args:
        screenshot: Full-screen screenshot (BGR numpy array).
        element_rect: (x, y, w, h) bounding box of the located element.
        expected_type: Expected ElementType value string.
        expected_label: Expected human label.
        confidence_threshold: Minimum confidence. Defaults to config value.
        skip_vlm: If True, skip VLM and return optimistic result.

    Returns:
        ConfirmResult indicating whether the element matches expectations.
    """
    config = get_config()
    if confidence_threshold is None:
        confidence_threshold = config.get("execution", {}).get(
            "default_confidence", 0.75
        )

    x, y, w, h = element_rect

    # Bounds checking
    img_h, img_w = screenshot.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return ConfirmResult(
            success=False,
            confidence=0.0,
            notes="Invalid element rect: zero or negative dimensions",
        )

    if skip_vlm:
        return ConfirmResult(
            success=True,
            confidence=0.6,
            notes="VLM validation skipped; assuming element is correct.",
        )

    # Crop the element region
    crop = screenshot[y : y + h, x : x + w]

    # Save crop to temp file for VLM
    crop_path = _save_temp_image(crop)
    try:
        from core.vision import analyze_crop

        result = analyze_crop(
            crop_path,
            f"Is this a {expected_type} labeled '{expected_label}'?",
        )

        vlm_type = result.get("element_type", "unknown")
        vlm_confidence = result.get("confidence", 0.0)

        # Check if the VLM agrees with expected type
        type_match = vlm_type == expected_type
        if not type_match:
            return ConfirmResult(
                success=False,
                confidence=vlm_confidence,
                notes=(
                    f"Type mismatch: expected '{expected_type}', "
                    f"VLM detected '{vlm_type}' "
                    f"(confidence: {vlm_confidence:.2f})"
                ),
            )

        if vlm_confidence < confidence_threshold:
            return ConfirmResult(
                success=False,
                confidence=vlm_confidence,
                notes=(
                    f"Low confidence: {vlm_confidence:.2f} < "
                    f"{confidence_threshold:.2f} for {expected_type}"
                ),
            )

        return ConfirmResult(
            success=True,
            confidence=vlm_confidence,
            notes=(
                f"Element confirmed as {vlm_type} "
                f"(confidence: {vlm_confidence:.2f}). "
                f"Label guess: {result.get('label_guess', 'N/A')}"
            ),
        )
    except Exception as e:
        logger.warning("Element validation failed: %s", e)
        return ConfirmResult(
            success=False,
            confidence=0.0,
            notes=f"Validation error: {e}",
        )
    finally:
        Path(crop_path).unlink(missing_ok=True)
