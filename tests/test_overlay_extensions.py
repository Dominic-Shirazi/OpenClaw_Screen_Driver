"""Tests for overlay controller and bbox layer extensions (Plan 04-02)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.bbox_layer import BboxLayer
from recorder.overlay.controller import OverlayController
from recorder.overlay.toolbar_panel import ToolbarMode


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


# ------------------------------------------------------------------
# Controller extension tests (safe without view)
# ------------------------------------------------------------------


class TestControllerExtensions:
    """Tests for new controller public API methods."""

    def test_controller_set_click_through(
        self, qapp: QApplication,
    ) -> None:
        """Controller.set_click_through does not raise without view."""
        ctrl = OverlayController()
        ctrl.set_click_through(True)
        ctrl.set_click_through(False)

    def test_controller_show_countdown(
        self, qapp: QApplication,
    ) -> None:
        """Controller.show_countdown returns None without view."""
        ctrl = OverlayController()
        result = ctrl.show_countdown(3)
        assert result is None

    def test_controller_hide_countdown(
        self, qapp: QApplication,
    ) -> None:
        """Controller.hide_countdown does not raise without view."""
        ctrl = OverlayController()
        ctrl.hide_countdown()

    def test_controller_show_abort_confirm(
        self, qapp: QApplication,
    ) -> None:
        """Controller.show_abort_confirm returns None without view."""
        ctrl = OverlayController()
        result = ctrl.show_abort_confirm(3)
        assert result is None

    def test_controller_hide_abort_confirm(
        self, qapp: QApplication,
    ) -> None:
        """Controller.hide_abort_confirm does not raise without view."""
        ctrl = OverlayController()
        ctrl.hide_abort_confirm()

    def test_controller_flash_success(
        self, qapp: QApplication,
    ) -> None:
        """Controller.flash_success does not raise without view."""
        ctrl = OverlayController()
        ctrl.flash_success(QRectF(0, 0, 100, 50))

    def test_controller_start_stop_card_glow_pulse(
        self, qapp: QApplication,
    ) -> None:
        """Controller card glow pulse methods do not raise without view."""
        ctrl = OverlayController()
        ctrl.start_card_glow_pulse()
        ctrl.stop_card_glow_pulse()

    def test_controller_has_all_new_methods(
        self, qapp: QApplication,
    ) -> None:
        """Controller has all 8 new public methods."""
        ctrl = OverlayController()
        methods = [
            "set_click_through",
            "show_countdown",
            "hide_countdown",
            "show_abort_confirm",
            "hide_abort_confirm",
            "flash_success",
            "start_card_glow_pulse",
            "stop_card_glow_pulse",
        ]
        for m in methods:
            assert hasattr(ctrl, m), f"missing method: {m}"

    def test_controller_handle_close_no_close_on_recording(
        self, qapp: QApplication,
    ) -> None:
        """_handle_close does NOT call close() when state is RECORDING.

        Verifies that save callback fires but overlay stays open.
        """
        from recorder.overlay.state import OverlayState

        save_called = []
        ctrl = OverlayController(on_save=lambda: save_called.append(True))
        ctrl._state = OverlayState.RECORDING
        ctrl._handle_close()
        assert save_called, "on_save should have been called"
        # Controller should NOT have closed (view is None so
        # close() would set _state=READY)
        assert ctrl._state == OverlayState.RECORDING


# ------------------------------------------------------------------
# BboxLayer editing tests
# ------------------------------------------------------------------


class TestBboxEditing:
    """Tests for interactive bbox editing."""

    def test_bbox_enable_editing(
        self, qapp: QApplication,
    ) -> None:
        """BboxLayer.enable_editing enables movable on handles."""
        from PyQt6.QtWidgets import QGraphicsItem

        b = BboxLayer(10, 20, 200, 150, (255, 0, 0, 100))
        b.enable_editing(True)
        for handle in b._handles:
            assert handle.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable

    def test_bbox_editing_get_rect(
        self, qapp: QApplication,
    ) -> None:
        """BboxLayer.get_edited_rect returns (x, y, w, h) tuple."""
        b = BboxLayer(10, 20, 200, 150, (255, 0, 0, 100))
        rect = b.get_edited_rect()
        assert isinstance(rect, tuple)
        assert len(rect) == 4
        assert rect == (10, 20, 200, 150)

    def test_bbox_accept_reject_edit(
        self, qapp: QApplication,
    ) -> None:
        """accept_edit finalizes, reject_edit reverts."""
        b = BboxLayer(10, 20, 200, 150, (255, 0, 0, 100))
        original = b.get_rect()

        b.enable_editing(True)
        b.accept_edit()
        assert b.get_rect() == (10, 20, 200, 150)
        assert not b._editing

        b.enable_editing(True)
        b.reject_edit((50, 60, 100, 80))
        assert b.get_rect() == (50, 60, 100, 80)
        assert not b._editing

    def test_bbox_eight_handles(
        self, qapp: QApplication,
    ) -> None:
        """BboxLayer has 8 handles (4 corners + 4 edge midpoints)."""
        b = BboxLayer(0, 0, 100, 100, (255, 0, 0, 100))
        assert len(b._handles) == 8
        positions = b._handle_positions(0, 0, 100, 100)
        assert len(positions) == 8
