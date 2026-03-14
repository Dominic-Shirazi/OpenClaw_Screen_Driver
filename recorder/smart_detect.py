"""Smart element detection for recording sessions.

During --record mode, automatically detects UI elements on screen using
YOLOE text-prompt mode (fast, local object detection). VLM is NOT used
for detection — only for labeling individual crops on user click.

Detection pipeline (Wave 1):
1. Capture screenshot
2. Run YOLOE text-prompt detection → bounding boxes + element types
3. If YOLOE returns 0 results, fall back to OCR
4. Return candidates for OverlayController.set_candidates()

All AI calls are guarded with try/except so recording works even without
GPU libs installed — candidates will just be empty.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def detect_ui_elements(
    screenshot: np.ndarray,
    *,
    use_vlm: bool = True,
    use_yoloe: bool = True,
) -> list[dict[str, Any]]:
    """Detects UI elements on a screenshot using YOLOE text-prompt mode.

    Pipeline: YOLOE text-prompt (primary) → OCR (fallback).
    VLM is NOT used for detection. The use_vlm parameter is kept for
    API compatibility but has no effect on the detection pipeline.

    Args:
        screenshot: BGR full-screen image as numpy array.
        use_vlm: Kept for API compatibility. Has no effect (VLM removed
                 from detection pipeline in Wave 1).
        use_yoloe: Whether to use YOLOE for detection. If False,
                   falls through directly to OCR.

    Returns:
        List of candidate dicts with keys:
        - rect: {x, y, w, h} in pixels
        - type_guess: element type string
        - label_guess: human-readable label
        - confidence: 0.0-1.0
        - ocr_text: visible text (if detected via OCR)
    """
    candidates: list[dict[str, Any]] = []

    # Primary: YOLOE text-prompt detection (fast, local)
    if use_yoloe:
        yoloe_candidates = _detect_via_yoloe(screenshot)
        if yoloe_candidates:
            return yoloe_candidates

    # Fallback: OCR-based detection for text elements
    ocr_candidates = _detect_via_ocr(screenshot)
    if ocr_candidates:
        candidates.extend(ocr_candidates)

    return candidates


def detect_ui_elements_async(
    screenshot: np.ndarray,
    callback: Any,
    *,
    use_vlm: bool = True,
    use_yoloe: bool = True,
) -> threading.Thread:
    """Runs detection in a background thread and calls callback with results.

    Args:
        screenshot: BGR screenshot to analyze.
        callback: Callable that receives list[dict] of candidates.
                  Called on the background thread — use QTimer.singleShot
                  to marshal to Qt main thread if needed.
        use_vlm: Kept for API compatibility. Has no effect.
        use_yoloe: Whether to use YOLOE for detection.

    Returns:
        The started Thread object (for joining if needed).
    """
    def _worker() -> None:
        try:
            results = detect_ui_elements(
                screenshot, use_vlm=use_vlm, use_yoloe=use_yoloe,
            )
            callback(results)
        except Exception as e:
            logger.error("Async detection failed: %s", e)
            callback([])

    thread = threading.Thread(target=_worker, daemon=True, name="smart-detect")
    thread.start()
    return thread


def _detect_via_yoloe(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses YOLOE text-prompt mode to detect all UI elements.

    Calls detect_all_elements() from core.yoloe which uses pre-cached
    CLIP text embeddings for fast inference.

    Args:
        screenshot: BGR full-screen image.

    Returns:
        List of candidate dicts, or empty list on failure.
    """
    try:
        from core.yoloe import detect_all_elements

        candidates = detect_all_elements(screenshot)
        logger.info("YOLOE detected %d UI elements", len(candidates))
        return candidates
    except ImportError:
        logger.debug("YOLOE module not available for smart detection")
        return []
    except Exception as e:
        logger.warning("YOLOE detection failed: %s", e)
        return []


def _detect_via_ocr(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses OCR to find text regions as potential UI elements.

    This is a lightweight fallback when YOLOE is unavailable or returns
    zero results. It finds text on screen and creates candidate boxes
    for each text region.

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
