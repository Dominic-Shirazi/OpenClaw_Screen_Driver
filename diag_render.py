"""Diagnostic: test overlay rendering with hardcoded fake candidates.

Run: python diag_render.py

This bypasses detection entirely and renders fake boxes on the overlay.
If boxes appear → rendering works, bug is in detection data.
If boxes DON'T appear → rendering itself is broken.
"""
from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("diag_render")


def main() -> None:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    from core.config import load_config
    load_config()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    from recorder.overlay import OverlayController, OverlayMode

    # Fake candidates at known positions
    fake_candidates = [
        {
            "rect": {"x": 100, "y": 100, "w": 200, "h": 50},
            "type_guess": "button",
            "label_guess": "FAKE BUTTON 1",
            "confidence": 0.95,
        },
        {
            "rect": {"x": 400, "y": 300, "w": 300, "h": 40},
            "type_guess": "textbox",
            "label_guess": "FAKE TEXTBOX",
            "confidence": 0.88,
        },
        {
            "rect": {"x": 800, "y": 200, "w": 150, "h": 60},
            "type_guess": "icon",
            "label_guess": "FAKE ICON",
            "confidence": 0.72,
        },
        {
            "rect": {"x": 500, "y": 500, "w": 250, "h": 80},
            "type_guess": "unknown",
            "label_guess": "FAKE UNKNOWN",
            "confidence": 0.60,
        },
    ]

    def on_close() -> None:
        logger.info("Overlay closed")
        app.quit()

    overlay = OverlayController(
        on_element_clicked=lambda x, y, w, h, c: logger.info("Clicked: %d,%d %dx%d", x, y, w, h),
        on_close=on_close,
    )
    overlay.show(start_mode=OverlayMode.RECORD)

    # Render fake candidates after a brief delay
    def _render_fakes() -> None:
        logger.info("Rendering %d fake candidates...", len(fake_candidates))
        overlay.set_candidates(fake_candidates)

        # Log scene item count
        view = overlay._view
        if view:
            scene = view.scene()
            items = scene.items()
            logger.info("Scene has %d items after rendering", len(items))
            for item in items:
                rect = item.boundingRect()
                logger.info(
                    "  Item: %s z=%.0f rect=(%.0f,%.0f %.0fx%.0f) visible=%s opacity=%.2f",
                    type(item).__name__,
                    item.zValue(),
                    rect.x(), rect.y(), rect.width(), rect.height(),
                    item.isVisible(),
                    item.opacity(),
                )

    QTimer.singleShot(500, _render_fakes)

    logger.info("Overlay shown. Look for 4 colored boxes on screen.")
    logger.info("Press Ctrl+Q or ESC to close.")
    app.exec()


if __name__ == "__main__":
    main()
