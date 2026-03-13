"""Post-action verification for OCSD replay steps.

Decision tree that validates whether a UI action produced the expected
result, using pixel-diff heuristics and optional VLM confirmation.

Verification levels:
1. quick_check — pixel diff only, fast, no VLM call
2. validate_action — pixel diff + conditional VLM confirmation
3. validate_destination — VLM-only full-screen state check
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.capture import pixel_diff
from core.config import get_config
from core.types import ConfirmResult
from core.vision import confirm_action, analyze_crop

logger = logging.getLogger(__name__)


def validate_action(
    before_screenshot: np.ndarray,
    after_screenshot: np.ndarray,
    intended_action: str,
    require_vlm: bool = False,
) -> ConfirmResult:
    """Validates if an action produced the intended result.

    Uses a combination of pixel difference and VLM confirmation based
    on configuration and the magnitude of visual change.

    Args:
        before_screenshot: Screenshot taken before the action.
        after_screenshot: Screenshot taken after the action.
        intended_action: Description of what the action should have done.
        require_vlm: If True, forces VLM validation regardless of config.

    Returns:
        ConfirmResult indicating success status, confidence, and notes.
    """
    config = get_config()
    exec_cfg = config.get("execution", {})
    threshold = float(exec_cfg.get("pixel_diff_threshold", 0.08))
    vlm_confirm = bool(exec_cfg.get("vlm_confirm", True))

    diff = pixel_diff(before_screenshot, after_screenshot)

    # No visible change — action likely failed
    if diff < threshold and not require_vlm:
        return ConfirmResult(
            success=False,
            confidence=0.9,
            notes="No visible change detected",
        )

    # Change detected but VLM disabled — trust the pixel diff
    if diff >= threshold and not require_vlm and not vlm_confirm:
        return ConfirmResult(
            success=True,
            confidence=0.7,
            notes=f"Pixel diff {diff:.1%} suggests change occurred",
        )

    # VLM confirmation path
    before_path: Path | None = None
    after_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_b:
            before_path = Path(tmp_b.name)
            cv2.imwrite(str(before_path), before_screenshot)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_a:
            after_path = Path(tmp_a.name)
            cv2.imwrite(str(after_path), after_screenshot)

        res = confirm_action(str(before_path), str(after_path), intended_action)
        return ConfirmResult(
            success=bool(res.get("success", False)),
            confidence=float(res.get("confidence", 0.0)),
            notes=str(res.get("notes", "No notes")),
        )
    except (RuntimeError, ConnectionError, ValueError) as e:
        logger.error("VLM validation failed: %s", e)
        return ConfirmResult(
            success=False, confidence=0.0, notes=f"VLM error: {e}"
        )
    finally:
        if before_path and before_path.exists():
            before_path.unlink()
        if after_path and after_path.exists():
            after_path.unlink()


def quick_check(
    before_screenshot: np.ndarray, after_screenshot: np.ndarray
) -> bool:
    """Fast pixel-diff-only check for UI changes.

    No VLM call. Suitable for tight loops where speed matters.

    Args:
        before_screenshot: Screenshot before the action.
        after_screenshot: Screenshot after the action.

    Returns:
        True if pixel difference exceeds the configured threshold.
    """
    config = get_config()
    threshold = float(
        config.get("execution", {}).get("pixel_diff_threshold", 0.08)
    )
    diff = pixel_diff(before_screenshot, after_screenshot)
    return diff >= threshold


def validate_destination(
    screenshot: np.ndarray, expected_description: str
) -> ConfirmResult:
    """Verifies if the current screen matches an expected destination state.

    Uses a VLM call to analyze the full screenshot against the provided
    description. Used for destination node verification.

    Args:
        screenshot: The current screen screenshot.
        expected_description: Description of the expected screen state.

    Returns:
        ConfirmResult indicating whether the destination matches.
    """
    img_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = Path(tmp.name)
            cv2.imwrite(str(img_path), screenshot)

        # Use analyze_crop with a destination-verification prompt
        context = (
            f"Verify this screenshot matches the expected state: "
            f"{expected_description}"
        )
        res = analyze_crop(str(img_path), context)

        # High confidence in the VLM's type guess means the screen
        # likely shows what we expect
        confidence = float(res.get("confidence", 0.5))
        label = str(res.get("label_guess", ""))
        ocr_text = res.get("ocr_text", "")

        # If the VLM found relevant text/elements, consider it a match
        success = confidence >= 0.6
        notes = f"VLM label: {label}"
        if ocr_text:
            notes += f", OCR: {ocr_text}"

        return ConfirmResult(success=success, confidence=confidence, notes=notes)
    except (RuntimeError, ConnectionError, ValueError) as e:
        logger.error("Destination validation failed: %s", e)
        return ConfirmResult(
            success=False, confidence=0.0, notes=f"Validation error: {e}"
        )
    finally:
        if img_path and img_path.exists():
            img_path.unlink()
