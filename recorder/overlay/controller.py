"""Overlay controller: state machine, hotkey wiring, and lifecycle.

The ``OverlayController`` is the single public entry point for the
overlay subsystem.  It owns the state machine, creates/manages the
``OverlayView`` on demand, and wires global hotkeys for mode toggling
and save/abort.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

from PyQt6.QtCore import QRectF

from recorder.overlay.bbox_layer import BboxLayer
from recorder.overlay.state import OverlayState, transition
from recorder.overlay.toolbar_panel import ToolbarMode

logger = logging.getLogger(__name__)


class OverlayController:
    """Public API for the transparent fullscreen overlay.

    Manages the overlay lifecycle (show/close), state transitions via
    hotkeys, and delegates visual updates to the underlying
    ``OverlayView``.  The view is created lazily on first ``show()``
    to avoid circular imports and premature Qt widget creation.

    Args:
        on_selection: Called with ``(x, y, w, h)`` when the user
            completes a drag-to-draw bounding box.
        on_state_changed: Called with the new ``OverlayState`` after
            every state transition.
        on_save: Called when the user presses Ctrl+Q while recording.
        on_abort: Called when the user presses ESC or Ctrl+Q while
            in READY/PAUSED state.
    """

    def __init__(
        self,
        *,
        on_selection: Callable[[int, int, int, int], None] | None = None,
        on_state_changed: Callable[[OverlayState], None] | None = None,
        on_save: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        self._on_selection = on_selection
        self._on_state_changed = on_state_changed
        self._on_save = on_save
        self._on_abort = on_abort

        self._state: OverlayState = OverlayState.READY
        self._view: Any = None  # OverlayView, lazy-imported
        self._hotkey_listener: Any = None
        self._is_active: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> OverlayState:
        """The current overlay state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the overlay is currently visible and active."""
        return self._is_active

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Create the overlay view, display it, and start hotkey listening.

        The ``OverlayView`` is imported lazily to prevent circular
        dependencies (the view imports state but not the controller).
        """
        # Lazy import to avoid circular dependency
        from recorder.overlay.view import OverlayView

        self._view = OverlayView(on_selection=self._on_selection)

        # Size to primary screen
        from PyQt6.QtWidgets import QApplication

        primary = QApplication.primaryScreen()
        if primary is not None:
            self._view.setGeometry(primary.geometry())

        self._view.show()
        self._view.raise_()

        # Apply initial state
        self._view.apply_state(self._state)

        # Start hotkey listener
        from recorder.hotkeys import _create_hotkey_listener

        self._hotkey_listener = _create_hotkey_listener(
            on_toggle=self._handle_toggle,
            on_close=self._handle_close,
        )
        self._hotkey_listener.start()

        self._is_active = True
        logger.info("Overlay shown in %s state", self._state.name)

    def close(self) -> None:
        """Tear down the overlay: stop hotkeys, close view, reset state."""
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
            self._hotkey_listener = None

        if self._view is not None:
            self._view.close()
            self._view = None

        self._is_active = False
        self._state = OverlayState.READY
        logger.info("Overlay closed")

    def hide_for_capture(self) -> None:
        """Hide the overlay window before a screenshot capture."""
        if self._view is not None:
            self._view.hide_for_capture()

    def show_after_capture(self) -> None:
        """Restore the overlay window after a screenshot capture."""
        if self._view is not None:
            self._view.show_after_capture()

    def set_bboxes(self, bboxes: list[dict[str, Any]]) -> None:
        """Render bounding boxes on the overlay.

        Args:
            bboxes: List of dicts with keys ``x``, ``y``, ``w``, ``h``,
                ``color``, and optionally ``label`` and ``confidence``.
        """
        if self._view is not None:
            self._view.render_bboxes(bboxes)

    def clear_bboxes(self) -> None:
        """Remove all bounding boxes from the overlay."""
        if self._view is not None:
            self._view.clear_bboxes()

    # ------------------------------------------------------------------
    # HUD panel API
    # ------------------------------------------------------------------

    def show_tag_dialog(
        self,
        element_rect: QRectF,
        vlm_data: dict | None = None,
        edit_mode: bool = False,
    ) -> None:
        """Show the tag dialog panel near the captured element.

        Args:
            element_rect: Bounding rect of captured element.
            vlm_data: Optional VLM analysis data.
            edit_mode: If True, pre-fill without typewriter.
        """
        if self._view is not None:
            self._view.show_tag_dialog(element_rect, vlm_data, edit_mode)

    def dismiss_tag_dialog(self) -> None:
        """Dismiss the tag dialog."""
        if self._view is not None:
            self._view.dismiss_tag_dialog()

    def get_tag_data(self) -> dict | None:
        """Return current tag dialog form data.

        Returns:
            Dict of form field values, or None if no dialog is showing.
        """
        if self._view is not None:
            return self._view.get_tag_data()
        return None

    def show_toolbar(self) -> None:
        """Show the floating toolbar."""
        if self._view is not None:
            self._view.show_toolbar()

    def hide_toolbar(self) -> None:
        """Hide the floating toolbar."""
        if self._view is not None:
            self._view.hide_toolbar()

    def set_toolbar_mode(self, mode: ToolbarMode) -> None:
        """Switch toolbar context mode.

        Args:
            mode: The toolbar mode to display.
        """
        if self._view is not None:
            self._view.set_toolbar_mode(mode)

    # ------------------------------------------------------------------
    # Scan / Donut cloud API
    # ------------------------------------------------------------------

    def start_scan(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        bbox: BboxLayer | None = None,
    ) -> None:
        """Start a scan animation at the given coordinates.

        Args:
            x: Left edge of rough snip boundary.
            y: Top edge of rough snip boundary.
            w: Width of rough snip boundary.
            h: Height of rough snip boundary.
            bbox: Optional BboxLayer to morph when AI result arrives.
        """
        if self._view is not None:
            self._view.start_scan(x, y, w, h, bbox=bbox)

    def finish_scan(
        self,
        fitted_x: int,
        fitted_y: int,
        fitted_w: int,
        fitted_h: int,
    ) -> None:
        """Deliver AI-fitted bbox to the scan layer and trigger morph.

        Args:
            fitted_x: Left edge of AI-fitted bbox.
            fitted_y: Top edge of AI-fitted bbox.
            fitted_w: Width of AI-fitted bbox.
            fitted_h: Height of AI-fitted bbox.
        """
        if self._view is not None:
            self._view.finish_scan(fitted_x, fitted_y, fitted_w, fitted_h)

    def show_donut_cloud(
        self,
        center_x: float,
        center_y: float,
        radius: float = 60.0,
    ) -> None:
        """Show a donut cloud probability visualizer.

        Args:
            center_x: Cloud center X coordinate.
            center_y: Cloud center Y coordinate.
            radius: Cloud radius.
        """
        if self._view is not None:
            self._view.show_donut_cloud(center_x, center_y, radius)

    def accept_donut_cloud(self) -> None:
        """Transition the donut cloud color to green (accepted)."""
        if self._view is not None:
            self._view.accept_donut_cloud()

    # ------------------------------------------------------------------
    # Recording pipeline visual API
    # ------------------------------------------------------------------

    def set_click_through(self, enabled: bool) -> None:
        """Toggle click-through mode without changing overlay state.

        Used during dry-run execution.

        Args:
            enabled: If True, make overlay click-through.
        """
        if self._view is not None:
            self._view.set_click_through(enabled)

    def show_countdown(self, seconds: int = 3) -> Any:
        """Show cursor-following countdown widget.

        Args:
            seconds: Number of seconds to count down.

        Returns:
            The CountdownWidget for signal connection, or None.
        """
        if self._view is not None:
            return self._view.show_countdown(seconds)
        return None

    def hide_countdown(self) -> None:
        """Hide and stop the countdown widget."""
        if self._view is not None:
            self._view.hide_countdown()

    def show_abort_confirm(self, step_count: int) -> Any:
        """Show abort confirmation panel.

        Args:
            step_count: Number of steps that will be lost.

        Returns:
            The AbortPanel for signal connection, or None.
        """
        if self._view is not None:
            return self._view.show_abort_confirm(step_count)
        return None

    def hide_abort_confirm(self) -> None:
        """Hide the abort confirmation panel."""
        if self._view is not None:
            self._view.hide_abort_confirm()

    def flash_success(self, bbox_rect: QRectF) -> None:
        """Show a brief green flash on the given bbox rect.

        Args:
            bbox_rect: The rectangle to flash green.
        """
        if self._view is not None:
            self._view.flash_success(bbox_rect)

    def start_card_glow_pulse(self) -> None:
        """Start card glow pulsing as a loading indicator.

        Call during DETECTING and VLM_ANALYZING phases per CONTEXT.md.
        """
        if self._view is not None:
            self._view.start_card_glow_pulse()

    def stop_card_glow_pulse(self) -> None:
        """Stop card glow pulsing. Call when detection/VLM completes."""
        if self._view is not None:
            self._view.stop_card_glow_pulse()

    # ------------------------------------------------------------------
    # Private: hotkey handlers
    # ------------------------------------------------------------------

    def _handle_toggle(self) -> None:
        """Handle F2 / Ctrl+R: transition overlay state."""
        new_state = transition(self._state, "f2")
        if new_state is None:
            logger.debug(
                "No transition for trigger 'f2' in state %s",
                self._state.name,
            )
            return

        old_name = self._state.name
        self._state = new_state

        if self._view is not None:
            self._view.apply_state(new_state)

        self._fire_callback(self._on_state_changed, new_state)
        logger.info("State: %s -> %s", old_name, new_state.name)

    def _handle_close(self) -> None:
        """Handle Ctrl+Q / ESC: delegate to on_save or on_abort callback.

        The callback (RecordSession) decides whether to close the overlay.
        Only auto-close on abort (non-recording). Save callback handles
        closing after save completes.
        """
        if self._state == OverlayState.RECORDING:
            logger.info("Close requested while RECORDING -> save")
            self._fire_callback(self._on_save)
        else:
            logger.info("Close requested while %s -> abort", self._state.name)
            self._fire_callback(self._on_abort)
            self.close()

    def _fire_callback(
        self,
        cb: Callable[..., Any] | None,
        *args: Any,
    ) -> None:
        """Invoke a callback safely, logging any errors.

        Args:
            cb: The callback to invoke, or None.
            *args: Positional arguments forwarded to the callback.
        """
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as exc:
            logger.error(
                "Callback %s raised: %s",
                getattr(cb, "__name__", cb),
                exc,
                exc_info=True,
            )
