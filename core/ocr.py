"""Pytesseract OCR wrapper for text detection and screen search."""
from __future__ import annotations

import logging
import os
import shutil
import sys

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError

from core.types import LocateResult, Point, Rect
import core.capture as capture

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "--oem 3 --psm 6"

# Well-known install paths per platform
_TESSERACT_HINTS: dict[str, list[str]] = {
    "win32": [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ],
    "darwin": [
        "/usr/local/bin/tesseract",      # Homebrew Intel
        "/opt/homebrew/bin/tesseract",    # Homebrew Apple Silicon
    ],
    "linux": [
        "/usr/bin/tesseract",
    ],
}


def _resolve_tesseract_cmd() -> str | None:
    """Find the Tesseract binary, checking env var, PATH, then well-known locations."""
    # 1. Explicit env-var override (from .env or system)
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        return env_cmd

    # 2. Already on PATH
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path

    # 3. Platform-specific well-known locations
    for hint in _TESSERACT_HINTS.get(sys.platform, []):
        if os.path.isfile(hint):
            return hint

    return None


def _ensure_tesseract_installed() -> None:
    """Check if Tesseract is installed and available.

    Tries env var → PATH → well-known locations before giving up.
    """
    resolved = _resolve_tesseract_cmd()
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved
        logger.debug("Tesseract binary: %s", resolved)

    try:
        pytesseract.get_tesseract_version()
    except TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR binary not found. "
            "Install Tesseract and ensure it's on PATH, or set TESSERACT_CMD "
            "in your .env file.\n"
            "Windows: winget install UB-Mannheim.TesseractOCR\n"
            "Mac:     brew install tesseract\n"
            "Linux:   sudo apt install tesseract-ocr"
        )
    except Exception as e:
        raise RuntimeError(f"Error accessing Tesseract: {e}")

def _prepare_image(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB as expected by pytesseract."""
    # OpenCV images are typically BGR, pytesseract expects RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img

def ocr_image(img: np.ndarray) -> str:
    """
    Extract raw text from an image.
    
    Args:
        img: The image as a numpy array (BGR format).
        
    Returns:
        The extracted text as a string.
    """
    _ensure_tesseract_installed()
    rgb_img = _prepare_image(img)
    text = pytesseract.image_to_string(rgb_img, config=DEFAULT_CONFIG)
    return text.strip()

def ocr_with_boxes(img: np.ndarray) -> list[dict]:
    """
    Extract text and bounding boxes from an image.
    
    Args:
        img: The image as a numpy array (BGR format).
        
    Returns:
        A list of dictionaries containing {text, rect: Rect, confidence: float} 
        for each detected text block.
    """
    _ensure_tesseract_installed()
    rgb_img = _prepare_image(img)
    
    # Use image_to_data to get boxes, confidences, and text
    data = pytesseract.image_to_data(rgb_img, config=DEFAULT_CONFIG, output_type=pytesseract.Output.DICT)
    
    results = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        # Filter out empty text and low confidence matches
        if text:
            try:
                conf = float(data['conf'][i])
            except ValueError:
                conf = 0.0
                
            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])
            
            # Normalize confidence to 0.0-1.0 range since pytesseract returns 0-100 (or sometimes -1 for empty)
            normalized_conf = max(0.0, min(1.0, conf / 100.0))
            
            if normalized_conf > 0:
                results.append({
                    "text": text,
                    "rect": Rect(x=x, y=y, w=w, h=h),
                    "confidence": normalized_conf
                })
            
    return results

def find_text_in_region(
    text: str,
    region_x: int,
    region_y: int,
    region_w: int,
    region_h: int,
) -> LocateResult | None:
    """Search for *text* within a specific screen region only.

    This avoids false positives from ads / unrelated UI that happen to
    contain the same words (e.g. an ad saying "Log in today!" when we
    want the actual Log-in button).

    Coordinates in the returned :class:`LocateResult` are in full-screen
    space (offset back from the crop).

    Args:
        text: Substring to search for (case-insensitive).
        region_x: Left edge of the search region (pixels).
        region_y: Top edge of the search region (pixels).
        region_w: Width of the search region (pixels).
        region_h: Height of the search region (pixels).

    Returns:
        LocateResult if found inside the region, otherwise None.
    """
    _ensure_tesseract_installed()

    img = capture.screenshot_region(region_x, region_y, region_w, region_h)
    blocks = ocr_with_boxes(img)

    best_match = None
    target_text = text.lower()

    for block in blocks:
        block_text = block["text"].lower()
        if target_text in block_text:
            if best_match is None or block["confidence"] > best_match["confidence"]:
                best_match = block

    if best_match:
        # Offset rect back to full-screen coordinates
        r = best_match["rect"]
        abs_rect = Rect(
            x=r.x + region_x,
            y=r.y + region_y,
            w=r.w,
            h=r.h,
        )
        return LocateResult(
            point=abs_rect.center,
            method="ocr",
            confidence=best_match["confidence"],
            rect=abs_rect,
        )

    return None


def find_text_on_screen(
    text: str,
    *,
    hint_x: int | None = None,
    hint_y: int | None = None,
    search_radius: int = 400,
) -> LocateResult | None:
    """Capture the screen and find the given text on it.

    If *hint_x*/*hint_y* are provided the search is scoped to a region
    of *search_radius* pixels around that point.  Otherwise the full
    screen is scanned (slower, higher false-positive risk).

    Args:
        text: The text substring to search for.
        hint_x: Optional expected X position (pixels).
        hint_y: Optional expected Y position (pixels).
        search_radius: Pixel radius around the hint.

    Returns:
        LocateResult if found, otherwise None.
    """
    _ensure_tesseract_installed()

    # --- Scoped search when we have a position hint ---
    if hint_x is not None and hint_y is not None:
        import pyautogui
        sw, sh = pyautogui.size()
        rx = max(0, hint_x - search_radius)
        ry = max(0, hint_y - search_radius)
        rw = min(search_radius * 2, sw - rx)
        rh = min(search_radius * 2, sh - ry)
        result = find_text_in_region(text, rx, ry, rw, rh)
        if result is not None:
            return result
        # Fall through to full-screen only if scoped search missed

    # --- Full-screen fallback ---
    img = capture.screenshot_full()
    blocks = ocr_with_boxes(img)

    best_match = None
    target_text = text.lower()

    for block in blocks:
        block_text = block['text'].lower()
        if target_text in block_text:
            if best_match is None or block['confidence'] > best_match['confidence']:
                best_match = block

    if best_match:
        rect = best_match['rect']
        return LocateResult(
            point=rect.center,
            method="ocr",
            confidence=best_match['confidence'],
            rect=rect
        )

    return None