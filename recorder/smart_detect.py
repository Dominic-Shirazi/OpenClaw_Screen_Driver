"""Smart element detection for recording sessions.

During --record mode, automatically detects UI elements on screen using
YOLOE (fast object detection) and/or VLM (vision language model) to
pre-populate candidate bounding boxes on the overlay.

Detection pipeline:
1. Capture screenshot
2. Run YOLOE object detection (fast, ~20ms) for bounding boxes
3. Optionally run VLM (slow, 1-3s) for element classification/labeling
4. Return candidates compatible with OverlayController.set_candidates()

All AI calls are guarded with try/except so recording works even without
GPU libs installed — candidates will just be empty.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from core.config import get_config

logger = logging.getLogger(__name__)


def detect_ui_elements(
    screenshot: np.ndarray,
    *,
    use_vlm: bool = True,
    use_yoloe: bool = True,
) -> list[dict[str, Any]]:
    """Detects UI elements on a screenshot using available AI backends.

    Tries YOLOE first (fast), then VLM (comprehensive). Results are
    merged and deduplicated.

    Args:
        screenshot: BGR full-screen image as numpy array.
        use_vlm: Whether to use VLM for classification. Slower but gives
                 element types and labels.
        use_yoloe: Whether to use YOLOE for fast bbox detection.

    Returns:
        List of candidate dicts with keys:
        - rect: {x, y, w, h} in pixels
        - type_guess: element type string
        - label_guess: human-readable label
        - confidence: 0.0-1.0
        - ocr_text: visible text (if detected)
    """
    candidates: list[dict[str, Any]] = []

    # Strategy: Use VLM for full detection (it returns typed + labeled candidates)
    # YOLOE is best for targeted re-finding, not general detection.
    # For recording, VLM's first_pass_map is the right tool.
    if use_vlm:
        vlm_candidates = _detect_via_vlm(screenshot)
        if vlm_candidates:
            candidates.extend(vlm_candidates)
            return candidates  # VLM gives the best results

    # Fallback: try OCR-based detection for text elements
    ocr_candidates = _detect_via_ocr(screenshot)
    if ocr_candidates:
        candidates.extend(ocr_candidates)

    return candidates


def detect_ui_elements_async(
    screenshot: np.ndarray,
    callback: Any,
    *,
    use_vlm: bool = True,
) -> threading.Thread:
    """Runs detection in a background thread and calls callback with results.

    Args:
        screenshot: BGR screenshot to analyze.
        callback: Callable that receives list[dict] of candidates.
                  Called on the background thread — use QTimer.singleShot
                  to marshal to Qt main thread if needed.
        use_vlm: Whether to use VLM.

    Returns:
        The started Thread object (for joining if needed).
    """
    def _worker() -> None:
        try:
            results = detect_ui_elements(screenshot, use_vlm=use_vlm)
            callback(results)
        except Exception as e:
            logger.error("Async detection failed: %s", e)
            callback([])

    thread = threading.Thread(target=_worker, daemon=True, name="smart-detect")
    thread.start()
    return thread


def _detect_via_vlm(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses VLM first_pass_map to detect and classify all UI elements.

    Args:
        screenshot: BGR full-screen image.

    Returns:
        List of candidate dicts, or empty list on failure.
    """
    try:
        from core.vision import first_pass_map_array

        candidates = first_pass_map_array(screenshot)
        logger.info("VLM detected %d UI elements", len(candidates))
        return candidates
    except ImportError:
        logger.debug("VLM module not available for smart detection")
        return []
    except Exception as e:
        logger.warning("VLM detection failed: %s", e)
        return []


def _detect_via_ocr(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses OCR to find text regions as potential UI elements.

    This is a lightweight fallback when VLM is unavailable. It finds
    text on screen and creates candidate boxes for each text region.

    Args:
        screenshot: BGR full-screen image.

    Returns:
        List of candidate dicts, or empty list on failure.
    """
    try:
        import cv2
        import pytesseract

        # Convert to grayscale for OCR
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Get bounding boxes for all detected text
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT
        )

        candidates: list[dict[str, Any]] = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            # Skip empty or low-confidence text
            if not text or conf < 40:
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            # Skip tiny boxes (noise)
            if w < 10 or h < 8:
                continue

            candidates.append({
                "rect": {"x": x, "y": y, "w": w, "h": h},
                "type_guess": "unknown",
                "label_guess": text[:30],
                "confidence": conf / 100.0,
                "ocr_text": text,
            })

        logger.info("OCR detected %d text regions", len(candidates))
        return candidates
    except ImportError:
        logger.debug("pytesseract not available for OCR detection")
        return []
    except Exception as e:
        logger.warning("OCR detection failed: %s", e)
        return []
