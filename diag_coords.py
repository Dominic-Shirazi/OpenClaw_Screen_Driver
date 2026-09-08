"""Diagnostic: compare OmniParser YOLO coordinates against screenshot.

Run: python diag_coords.py

Captures a screenshot, runs OmniParser detection, and saves an annotated
image (diag_coords_output.png) with the detected bounding boxes drawn
directly on the screenshot. If boxes align with UI elements in the image,
coordinates are correct. If they're shifted, you can measure the offset.
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("diag_coords")


def main() -> None:
    import cv2
    import numpy as np

    # Must set DPI awareness BEFORE capturing
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError):
                ctypes.windll.user32.SetProcessDPIAware()

    from core.config import load_config
    load_config()

    from core.capture import screenshot_full

    logger.info("Capturing screenshot...")
    screenshot = screenshot_full()
    h, w = screenshot.shape[:2]
    logger.info("Screenshot: %dx%d", w, h)

    # Save raw screenshot for reference
    cv2.imwrite("diag_coords_raw.png", screenshot)
    logger.info("Saved raw screenshot to diag_coords_raw.png")

    # Run OmniParser detection
    logger.info("Running OmniParser detection...")
    from core.detection import get_detector
    detector = get_detector()
    candidates = detector.detect(screenshot)
    logger.info("Detected %d candidates", len(candidates))

    # Draw bounding boxes on screenshot copy
    annotated = screenshot.copy()
    colors = {
        "button": (0, 200, 0),
        "textbox": (255, 150, 0),
        "icon": (0, 180, 230),
        "toggle": (0, 165, 255),
        "unknown": (128, 128, 128),
    }

    for i, c in enumerate(candidates):
        r = c["rect"]
        x, y, bw, bh = r["x"], r["y"], r["w"], r["h"]
        tg = c.get("type_guess", "unknown")
        conf = c.get("confidence", 0)
        color = colors.get(tg, (128, 128, 128))

        # Draw rect
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        # Draw label
        label = f"#{i} {tg} ({conf:.0%})"
        cv2.putText(
            annotated, label, (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

        # Draw center dot
        cx, cy = x + bw // 2, y + bh // 2
        cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

        logger.info(
            "  #%02d %-10s rect=(%4d,%4d %4dx%4d) center=(%4d,%4d) conf=%.2f",
            i, tg, x, y, bw, bh, cx, cy, conf,
        )

    cv2.imwrite("diag_coords_output.png", annotated)
    logger.info("Saved annotated screenshot to diag_coords_output.png")
    logger.info(
        "Open diag_coords_output.png and check if boxes align with UI elements."
    )
    logger.info(
        "If boxes are shifted right, measure the offset in pixels."
    )


if __name__ == "__main__":
    main()
