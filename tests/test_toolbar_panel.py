"""Tests for ToolbarPanel."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication, QGraphicsObject

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.toolbar_panel import ToolbarMode, ToolbarPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture()
def toolbar(qapp: QApplication) -> ToolbarPanel:
    """Create a ToolbarPanel instance for testing."""
    clock = AnimationClock()
    t = ToolbarPanel(clock, screen_w=1920, screen_h=1080)
    return t


class TestInstantiation:
    """Tests for toolbar construction defaults."""

    def test_z_value(self, toolbar: ToolbarPanel) -> None:
        """Toolbar z-value is 110."""
        assert toolbar.zValue() == 110

    def test_dimensions(self, toolbar: ToolbarPanel) -> None:
        """Toolbar has expected width and height (dynamic for RECORDING mode)."""
        # RECORDING mode has 6 buttons: 70*6 + 40 = 460
        assert toolbar._width == 460.0
        assert toolbar._height == 40.0

    def test_movable_flag(self, toolbar: ToolbarPanel) -> None:
        """Toolbar has ItemIsMovable flag set."""
        assert toolbar.flags() & QGraphicsObject.GraphicsItemFlag.ItemIsMovable

    def test_default_mode(self, toolbar: ToolbarPanel) -> None:
        """Toolbar starts in RECORDING mode."""
        assert toolbar._mode == ToolbarMode.RECORDING

    def test_button_signal(self, toolbar: ToolbarPanel) -> None:
        """button_clicked signal is connectable."""
        handler = MagicMock()
        toolbar.button_clicked.connect(handler)
        # Should not raise


class TestModeSwitch:
    """Tests for toolbar mode switching."""

    def test_switch_to_tag_open(self, toolbar: ToolbarPanel) -> None:
        """set_mode(TAG_OPEN) updates internal mode."""
        toolbar.set_mode(ToolbarMode.TAG_OPEN)
        assert toolbar._mode == ToolbarMode.TAG_OPEN

    def test_switch_to_dry_run(self, toolbar: ToolbarPanel) -> None:
        """set_mode(DRY_RUN) updates internal mode."""
        toolbar.set_mode(ToolbarMode.DRY_RUN)
        assert toolbar._mode == ToolbarMode.DRY_RUN

    def test_switch_back_to_recording(self, toolbar: ToolbarPanel) -> None:
        """Mode can switch back to RECORDING."""
        toolbar.set_mode(ToolbarMode.TAG_OPEN)
        toolbar.set_mode(ToolbarMode.RECORDING)
        assert toolbar._mode == ToolbarMode.RECORDING

    def test_tag_open_buttons_visible(self, toolbar: ToolbarPanel) -> None:
        """TAG_OPEN mode shows its buttons, hides others."""
        toolbar.set_mode(ToolbarMode.TAG_OPEN)
        # TAG_OPEN buttons should be visible
        for proxy in toolbar._button_proxies[ToolbarMode.TAG_OPEN]:
            assert proxy.isVisible()
        # RECORDING buttons should be hidden
        for proxy in toolbar._button_proxies[ToolbarMode.RECORDING]:
            assert not proxy.isVisible()


class TestAvoidanceRect:
    """Tests for avoidance rect API."""

    def test_returns_qrectf(self, toolbar: ToolbarPanel) -> None:
        """get_avoidance_rect returns a QRectF."""
        rect = toolbar.get_avoidance_rect()
        assert isinstance(rect, QRectF)
        assert rect.width() >= 250  # dynamic width, at least 250


class TestHideForCapture:
    """Tests for capture visibility control."""

    def test_hide_makes_invisible(self, toolbar: ToolbarPanel) -> None:
        """setVisible(False) hides toolbar."""
        toolbar.show_toolbar()
        toolbar.setVisible(False)
        assert not toolbar.isVisible()

    def test_show_after_hide(self, toolbar: ToolbarPanel) -> None:
        """setVisible(True) restores toolbar."""
        toolbar.setVisible(False)
        toolbar.setVisible(True)
        assert toolbar.isVisible()
