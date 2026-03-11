"""Pytesseract OCR wrapper for text detection and screen search."""

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError

from core.types import LocateResult, Point, Rect
import core.capture as capture

DEFAULT_CONFIG = "--oem 3 --psm 6"


def _ensure_tesseract_installed():
    """Check if Tesseract is installed and available."""
    try:
        pytesseract.get_tesseract_version()
    except TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR binary not found. "
            "Please install Tesseract OCR and ensure it's in your system PATH.\n"
            "Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Mac: brew install tesseract\n"
            "Linux: sudo apt install tesseract-ocr"
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

def find_text_on_screen(text: str) -> LocateResult | None:
    """
    Capture the screen and find the given text on it.
    
    Args:
        text: The text substring to search for.
        
    Returns:
        LocateResult if found, otherwise None.
    """
    _ensure_tesseract_installed()
    
    # Capture full screen
    img = capture.screenshot_full()
    
    # Get all text blocks
    blocks = ocr_with_boxes(img)
    
    # Search for substring match
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