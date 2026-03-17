"""Integration tests for HUD panels in overlay scene."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.tag_dialog_panel import TagDialogPanel
from recorder.overlay.toolbar_panel import ToolbarMode, ToolbarPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


class TestBothPanelsInScene:
    """Tests for both HUD panels coexisting."""

    def test_both_panels_create(self, qapp: QApplication) -> None:
        """Both panels can be instantiated without error."""
        clock = AnimationClock()
        tag = TagDialogPanel(clock)
        toolbar = ToolbarPanel(clock, 1920, 1080)
        assert tag is not None
        assert toolbar is not None

    def test_avoidance_rects_from_both(self, qapp: QApplication) -> None:
        """Both panels provide valid avoidance rects."""
        clock = AnimationClock()
        tag = TagDialogPanel(clock)
        toolbar = ToolbarPanel(clock, 1920, 1080)
        rects = []
        tag_rect = tag.get_avoidance_rect()
        toolbar_rect = toolbar.get_avoidance_rect()
        rects.append(tag_rect)
        rects.append(toolbar_rect)
        assert len(rects) == 2
        assert all(isinstance(r, QRectF) for r in rects)


class TestHideForCapture:
    """Tests for capture visibility management."""

    def test_both_panels_can_be_hidden(self, qapp: QApplication) -> None:
        """Both panels support setVisible(False) for capture hiding."""
        clock = AnimationClock()
        tag = TagDialogPanel(clock)
        toolbar = ToolbarPanel(clock, 1920, 1080)
        tag.setVisible(False)
        toolbar.setVisible(False)
        assert not tag.isVisible()
        assert not toolbar.isVisible()

    def test_both_panels_can_be_restored(self, qapp: QApplication) -> None:
        """Both panels restore visibility after capture."""
        clock = AnimationClock()
        tag = TagDialogPanel(clock)
        toolbar = ToolbarPanel(clock, 1920, 1080)
        tag.setVisible(False)
        toolbar.setVisible(False)
        tag.setVisible(True)
        toolbar.setVisible(True)
        assert tag.isVisible()
        assert toolbar.isVisible()


class TestControllerAPI:
    """Tests for controller HUD method existence and safety."""

    def test_controller_has_hud_methods(self, qapp: QApplication) -> None:
        """OverlayController has all required HUD API methods."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        assert hasattr(ctrl, "show_tag_dialog")
        assert hasattr(ctrl, "dismiss_tag_dialog")
        assert hasattr(ctrl, "get_tag_data")
        assert hasattr(ctrl, "show_toolbar")
        assert hasattr(ctrl, "hide_toolbar")
        assert hasattr(ctrl, "set_toolbar_mode")

    def test_controller_methods_safe_without_view(
        self, qapp: QApplication,
    ) -> None:
        """Controller HUD methods do not crash when view is None."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        # These should all be no-ops when _view is None
        ctrl.show_tag_dialog(QRectF(0, 0, 50, 30))
        ctrl.dismiss_tag_dialog()
        ctrl.get_tag_data()
        ctrl.show_toolbar()
        ctrl.hide_toolbar()
        ctrl.set_toolbar_mode(ToolbarMode.RECORDING)
