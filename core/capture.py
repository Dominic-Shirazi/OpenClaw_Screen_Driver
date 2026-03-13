from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

import cv2
import mss
import numpy as np

from core.config import get_config
from core.types import Point, Rect

logger = logging.getLogger(__name__)


def screenshot_full() -> np.ndarray:
    """Takes a full screen screenshot of the primary monitor."""
    with mss.mss() as sct:
        # monitor 1 is usually the primary monitor in mss
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        
        # Convert to numpy array
        img = np.array(sct_img)
        
        # Convert from BGRA (mss default) to BGR (OpenCV convention)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def screenshot_region(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Takes a screenshot of a specific bounding box region."""
    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": w, "height": h}
        sct_img = sct.grab(monitor)
        
        # Convert to numpy array
        img = np.array(sct_img)
        
        # Convert from BGRA to BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def pixel_diff(img_a: np.ndarray, img_b: np.ndarray, threshold: float = 0.05) -> float:
    """Calculates the percentage of pixels that have changed between two images.
    
    Args:
        img_a: First image (BGR).
        img_b: Second image (BGR).
        threshold: Intensity difference threshold (0.0-1.0) to consider a pixel changed.
        
    Returns:
        Float between 0.0 and 1.0 representing the percentage of changed pixels.
    """
    if img_a.shape != img_b.shape:
        # Resize b to match a if they differ (though ideally they shouldn't)
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
    # Absolute difference between the two images
    diff = cv2.absdiff(img_a, img_b)
    
    # Calculate difference as a float 0.0-1.0 per channel
    normalized_diff = diff.astype(np.float32) / 255.0
    
    # Average across the 3 channels (BGR) to get a per-pixel difference 0.0-1.0
    pixel_diffs = np.mean(normalized_diff, axis=2)
    
    # Count how many pixels exceed the threshold
    changed_pixels = np.sum(pixel_diffs > threshold)
    total_pixels = img_a.shape[0] * img_a.shape[1]
    
    return float(changed_pixels) / float(total_pixels)


def save_snippet(img: np.ndarray, skill_id: str, element_id: str) -> str:
    """Saves an image crop to the snippets directory.
    
    Path: assets/snippets/{skill_id}/{element_id}.png
    """
    config = get_config()
    snippets_dir = config.get("paths", {}).get("snippets_dir", "./assets/snippets")
    
    # Ensure skill directory exists
    save_dir = Path(snippets_dir) / skill_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / f"{element_id}.png"
    
    # Write the image
    cv2.imwrite(str(file_path), img)
    
    return str(file_path)


def get_window_title() -> str:
    """Retrieves the active window title.

    Uses Win32 API on Windows, falls back to empty string on other platforms.
    """
    if sys.platform != "win32":
        logger.debug("get_window_title() is only supported on Windows")
        return ""

    hwnd = ctypes.windll.user32.GetForegroundWindow()
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)

    # Create a buffer of the correct length
    buf = ctypes.create_unicode_buffer(length + 1)

    # Fill the buffer with the window text
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)

    return buf.value


def get_active_url() -> str | None:
    """Best-effort attempt to get the active browser URL via UIA.

    NOTE: Windows-only (requires pywinauto). Returns None on other platforms.
    This is heavily dependent on the browser and OS language,
    and is notoriously unreliable. Use only as a hint.
    """
    if sys.platform != "win32":
        return None

    try:
        from pywinauto import Application

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return None

        # Connect to the active window
        app = Application(backend="uia").connect(handle=hwnd, timeout=0.5)
        window = app.window(handle=hwnd)

        # Try to find an element that looks like an address bar
        # This regex attempts to catch common address bar UIA names
        address_bar = window.child_window(
            control_type="Edit",
            title_re=".*[Aa]ddress.*|.*[Ss]earch.*|.*URL.*",
        )

        if address_bar.exists():
            return address_bar.get_value()

    except Exception:
        # Ignore all errors, this is best-effort only
        pass

    return None