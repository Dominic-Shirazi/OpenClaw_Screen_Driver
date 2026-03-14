"""Smart element detection for recording sessions.

During --record mode, automatically detects UI elements on screen using
OmniParser (YOLOv8-based) for primary detection and Florence-2 for caption
enrichment. Falls back to OCR when detection returns no results.

Detection pipeline:
1. Capture screenshot
2. Run OmniParser detection → bounding boxes + element types
3. Enrich candidates with Florence-2 captions (label_guess update)
4. If OmniParser returns 0 results, fall back to OCR
5. Return candidates for OverlayController.set_candidates()

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
) -> list[dict[str, Any]]:
    """Detects UI elements on a screenshot using OmniParser + Florence-2.

    Pipeline: OmniParser (primary) → Florence-2 enrichment → OCR (fallback).

    Args:
        screenshot: BGR full-screen image as numpy array.

    Returns:
        List of candidate dicts with keys:
        - rect: {x, y, w, h} in pixels
        - type_guess: element type string
        - label_guess: human-readable label
        - confidence: 0.0-1.0
        - ocr_text: visible text (if detected via OCR)
    """
    # Primary: OmniParser detection
    candidates = _detect_via_omniparser(screenshot)
    if candidates:
        candidates = _enrich_with_florence(candidates, screenshot)
        return candidates

    # Fallback: OCR-based detection for text elements
    ocr_candidates = _detect_via_ocr(screenshot)
    if ocr_candidates:
        return ocr_candidates

    return []


def detect_ui_elements_async(
    screenshot: np.ndarray,
    callback: Any,
) -> threading.Thread:
    """Runs detection in a background thread and calls callback with results.

    Args:
        screenshot: BGR screenshot to analyze.
        callback: Callable that receives list[dict] of candidates.
                  Called on the background thread — use QTimer.singleShot
                  to marshal to Qt main thread if needed.

    Returns:
        The started Thread object (for joining if needed).
    """
    def _worker() -> None:
        try:
            results = detect_ui_elements(screenshot)
            callback(results)
        except Exception as e:
            logger.error("Async detection failed: %s", e)
            callback([])

    thread = threading.Thread(target=_worker, daemon=True, name="smart-detect")
    thread.start()
    return thread


def _detect_via_omniparser(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses OmniParser YOLOv8 to detect all UI elements.

    Calls get_detector().detect() from core.detection which uses the
    OmniParser model for fast, accurate UI element detection.

    Args:
        screenshot: BGR full-screen image.

    Returns:
        List of candidate dicts, or empty list on failure.
    """
    try:
        from core.detection import get_detector

        detector = get_detector()
        candidates = detector.detect(screenshot)
        logger.info("OmniParser detected %d UI elements", len(candidates))
        return candidates
    except ImportError:
        logger.debug("Detection module not available")
        return []
    except Exception as e:
        logger.warning("OmniParser detection failed: %s", e)
        return []


def _enrich_with_florence(
    candidates: list[dict[str, Any]],
    screenshot: np.ndarray,
) -> list[dict[str, Any]]:
    """Add Florence-2 captions to each candidate's label_guess.

    Crops each candidate bbox from the screenshot, runs Florence-2
    caption_batch, and updates label_guess fields.

    Args:
        candidates: List of detection candidate dicts with 'rect' keys.
        screenshot: BGR full-screen image used for cropping.

    Returns:
        The same candidates list, mutated in-place with updated label_guess
        and florence_caption fields where captions were produced.
    """
    try:
        import cv2
        from core.florence import caption_batch

        crops = []
        for c in candidates:
            r = c["rect"]
            x1 = max(0, r["x"])
            y1 = max(0, r["y"])
            x2 = r["x"] + r["w"]
            y2 = r["y"] + r["h"]
            crop = screenshot[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            else:
                crops.append(np.zeros((1, 1, 3), dtype=np.uint8))

        captions = caption_batch(crops)
        for c, cap in zip(candidates, captions):
            if cap:
                c["label_guess"] = cap
                c["florence_caption"] = cap

    except ImportError:
        logger.debug("Florence-2 not available, skipping captioning")
    except Exception as e:
        logger.warning("Florence-2 captioning failed: %s", e)

    return candidates


def _detect_via_ocr(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses OCR to find text regions as potential UI elements.

    This is a lightweight fallback when OmniParser is unavailable or returns
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
