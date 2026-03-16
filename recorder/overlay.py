"""Overlay controller for recording sessions.

Manages mode switching, candidate rendering, click capture,
review flow, and integration with the recording session pipeline.

The actual PyQt6 window lives in overlay_view.py; graphics primitives
in overlay_items.py; hotkey listeners in hotkeys.py.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum, auto
from typing import Any, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from recorder.hotkeys import _create_hotkey_listener

logger = logging.getLogger(__name__)

# Win32 constants (needed for _apply_win32_flags)
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020


class OverlayMode(Enum):
    """Overlay interaction modes."""
    PASSTHROUGH = auto()  # Clicks go through to underlying app
    RECORD = auto()       # Overlay captures clicks for tagging


class OverlayController:
    """Controls the transparent fullscreen overlay.

    Manages mode switching, candidate rendering, click capture,
    review flow, and integration with the recording session pipeline.
    """

    # Minimum drag distance (pixels) to count as a bounding box vs a click
    _MIN_DRAG_PX = 8

    def __init__(
        self,
        on_element_clicked: Callable[[int, int, int, int, dict[str, Any] | None], None] | None = None,
        on_mode_changed: Callable[[OverlayMode], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Initializes the overlay controller.

        Args:
            on_element_clicked: Callback when user selects in RECORD mode.
                               Args: (x, y, w, h, matched_candidate_dict_or_None).
                               For point clicks, w=0, h=0.
                               For bounding boxes, (x, y) is the top-left corner.
            on_mode_changed: Callback when overlay mode changes.
            on_close: Callback when overlay is closed via Ctrl+Q / ESC.
        """
        self._on_element_clicked = on_element_clicked
        self._on_mode_changed = on_mode_changed
        self._on_close = on_close

        self._mode = OverlayMode.PASSTHROUGH
        self._candidates: list[dict[str, Any]] = []
        self._view: Any = None  # _OverlayView — imported lazily to avoid circular
        self._is_active = False
        self._hotkey_listener: Any = None

    @property
    def mode(self) -> OverlayMode:
        """Returns the current overlay mode."""
        return self._mode

    @property
    def is_active(self) -> bool:
        """Returns whether the overlay is currently shown."""
        return self._is_active

    def show(self, *, start_mode: OverlayMode = OverlayMode.PASSTHROUGH) -> None:
        """Shows the overlay in the specified mode."""
        if self._view is not None and self._is_active:
            logger.debug("Overlay already active")
            return

        from recorder.overlay_view import _OverlayView

        self._view = _OverlayView(controller=self)
        if QApplication.primaryScreen():
            self._view.setGeometry(QApplication.primaryScreen().geometry())
        self._view.show()
        self._view.raise_()
        self._is_active = True
        self._set_mode(start_mode)

        self._hotkey_listener = _create_hotkey_listener(
            on_toggle=self.toggle_mode,
            on_close=self.close,
        )
        self._hotkey_listener.start()

        logger.info("Overlay shown in %s mode", start_mode.name)

    def close(self) -> None:
        """Closes and cleans up the overlay and hotkeys."""
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
            self._hotkey_listener = None

        if self._view is not None:
            self._view.close()
            self._view = None
        self._is_active = False
        self._candidates = []

        if self._on_close:
            try:
                self._on_close()
            except Exception as e:
                logger.error("on_close callback error: %s", e)

        logger.info("Overlay closed")

    def toggle_mode(self) -> None:
        """Toggles between PASSTHROUGH and RECORD modes."""
        if self._mode == OverlayMode.PASSTHROUGH:
            self._set_mode(OverlayMode.RECORD)
        else:
            self._set_mode(OverlayMode.PASSTHROUGH)

    def set_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Updates the list of candidate elements to render.

        Args:
            candidates: List of dicts with rect, type_guess, label_guess, confidence.
        """
        self._candidates = candidates
        if self._view is not None:
            self._view.render_candidates(candidates)
        for c in candidates:
            r = c.get("rect", {})
            logger.info(
                "  candidate: %s '%s' at (%d,%d) %dx%d conf=%.2f",
                c.get("type_guess", "?"), c.get("label_guess", ""),
                r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0),
                c.get("confidence", 0),
            )
        logger.info("Rendered %d candidates on overlay", len(candidates))

    def clear_candidates(self) -> None:
        """Removes all candidate renderings."""
        self._candidates = []
        if self._view is not None:
            self._view.clear_scene()

    def start_review(
        self,
        on_review_element: Callable[[int, dict[str, Any]], bool],
    ) -> None:
        """Starts one-by-one review of detected candidates.

        Iterates through each candidate, highlighting it on the overlay,
        and calling the callback which should open a TagDialog.

        Args:
            on_review_element: Called for each candidate with (index, candidate).
                              Should return True if accepted, False if skipped.
                              The candidate dict's rect may have been updated
                              by handle dragging.
        """
        if not self._candidates or self._view is None:
            return

        for i, candidate in enumerate(self._candidates):
            # Highlight current candidate, dim others
            self._view.highlight_candidate(i)
            QApplication.processEvents()

            # Update candidate rect from the (possibly resized) box
            rect_tuple = self._view.get_candidate_rect(i)
            if rect_tuple:
                x, y, w, h = rect_tuple
                candidate["rect"] = {"x": x, "y": y, "w": w, "h": h}

            on_review_element(i, candidate)

        # Restore all highlights after review
        self._view.reset_highlights()
        logger.info("Review complete — %d candidates reviewed", len(self._candidates))

    def highlight_candidate(self, index: int) -> None:
        """Highlights a single candidate on the overlay."""
        if self._view is not None:
            self._view.highlight_candidate(index)

    def get_candidate_rect(self, index: int) -> tuple[int, int, int, int] | None:
        """Returns the (possibly resized) rect for a candidate."""
        if self._view is not None:
            return self._view.get_candidate_rect(index)
        return None

    def _set_mode(self, mode: OverlayMode) -> None:
        """Sets the overlay mode and updates Win32 flags accordingly."""
        self._mode = mode

        if self._view is not None:
            if sys.platform == "win32":
                self._apply_win32_flags(mode)
            else:
                if mode == OverlayMode.PASSTHROUGH:
                    self._view.setAttribute(
                        Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
                    )
                else:
                    self._view.setAttribute(
                        Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
                    )

            if mode == OverlayMode.RECORD:
                self._view.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._view.setCursor(Qt.CursorShape.ArrowCursor)

            self._view.refresh_overlay()

        if self._on_mode_changed:
            try:
                self._on_mode_changed(mode)
            except Exception as e:
                logger.error("on_mode_changed callback error: %s", e)

        logger.info("Overlay mode: %s", mode.name)

    def _apply_win32_flags(self, mode: OverlayMode) -> None:
        """Applies Win32 extended window style flags for click-through."""
        if self._view is None:
            return

        import ctypes
        user32 = ctypes.windll.user32

        hwnd = int(self._view.winId())
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)

        if mode == OverlayMode.PASSTHROUGH:
            new_style = style | WS_EX_TRANSPARENT
        else:
            new_style = style & ~WS_EX_TRANSPARENT

        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    def _handle_selection(self, x: int, y: int, w: int, h: int) -> bool:
        """Handles a completed selection (point click or bounding box drag).

        Args:
            x: Top-left X (or click X for point clicks).
            y: Top-left Y (or click Y for point clicks).
            w: Width of bounding box (0 for point click).
            h: Height of bounding box (0 for point click).

        Returns:
            True if the element was accepted (recorded), False if skipped.
        """
        if w > 0 and h > 0:
            cx, cy = x + w // 2, y + h // 2
            matched = self._find_candidate_at(cx, cy)
            logger.info("Bbox selection: (%d, %d) %dx%d", x, y, w, h)
        else:
            matched = self._find_candidate_at(x, y)
            logger.info("Point click: (%d, %d)", x, y)

        accepted = False
        if self._on_element_clicked:
            try:
                result = self._on_element_clicked(x, y, w, h, matched)
                accepted = bool(result)
            except Exception as e:
                logger.error("on_element_clicked callback error: %s", e)

        return accepted

    def _find_candidate_at(self, x: int, y: int) -> dict[str, Any] | None:
        """Finds the candidate element closest to a screen coordinate."""
        best: dict[str, Any] | None = None
        best_area = float("inf")

        for candidate in self._candidates:
            rect = candidate.get("rect", {})
            rx = rect.get("x", 0)
            ry = rect.get("y", 0)
            rw = rect.get("w", 0)
            rh = rect.get("h", 0)

            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                area = rw * rh
                if area < best_area:
                    best = candidate
                    best_area = area

        return best
