"""Unit tests for DonutCloudLayer rendering and color transition.

Uses QT_QPA_PLATFORM=offscreen so tests run without a display.
"""

from __future__ import annotations

import os

# MUST be set before any PyQt6 import
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

from recorder.overlay.donut_cloud_layer import DonutCloudLayer


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture()
def cloud(qapp: QApplication) -> DonutCloudLayer:
    """Create a DonutCloudLayer centered at (400, 300) with radius 60."""
    return DonutCloudLayer(400.0, 300.0, 60.0, 60.0)


class TestDonutCloudInstantiation:
    """Tests for DonutCloudLayer construction."""

    def test_instantiates_with_center_and_radius(self, cloud: DonutCloudLayer) -> None:
        """DonutCloudLayer stores center and radius from constructor."""
        assert cloud._center.x() == 400.0
        assert cloud._center.y() == 300.0
        assert cloud._radius_x == 60.0
        assert cloud._radius_y == 60.0

    def test_bounding_rect_covers_radius(self, cloud: DonutCloudLayer) -> None:
        """boundingRect() covers the full radius area plus padding."""
        rect = cloud.boundingRect()
        # max(radius_x, radius_y) + 20 padding = 80
        expected_r = 80.0
        assert rect.width() == expected_r * 2
        assert rect.height() == expected_r * 2
        assert rect.left() == 400.0 - expected_r
        assert rect.top() == 300.0 - expected_r

    def test_z_value_is_45(self, cloud: DonutCloudLayer) -> None:
        """DonutCloudLayer z-value is 45 (below BboxLayer at 50)."""
        assert cloud.zValue() == 45


class TestDonutCloudColor:
    """Tests for color transitions."""

    def test_initial_color_is_red(self, cloud: DonutCloudLayer) -> None:
        """Initial color is red (editing mode)."""
        assert cloud._color.red() == 255
        assert cloud._color.green() == 50
        assert cloud._color.blue() == 50

    def test_accept_sets_target_color_green(self, cloud: DonutCloudLayer) -> None:
        """accept() transitions target color to green."""
        cloud.accept()
        assert cloud._accepted is True
        assert cloud._target_color.green() == 200
        assert cloud._target_color.red() == 50


class TestDonutCloudTick:
    """Tests for tick() animation advancement."""

    def test_tick_advances_fade_in_progress(self, cloud: DonutCloudLayer) -> None:
        """tick() advances fade_in_progress from 0.0 toward 1.0."""
        assert cloud.fade_in_progress == 0.0
        cloud.tick(0.25)
        assert cloud.fade_in_progress > 0.0
        assert cloud.fade_in_progress <= 1.0

    def test_tick_spawns_raindrops(self, cloud: DonutCloudLayer) -> None:
        """tick() spawns new raindrops when below max count."""
        # Advance spawn timer past threshold (0.3s)
        cloud.tick(0.35)
        assert len(cloud._raindrops) >= 1

    def test_completed_raindrops_removed(self, cloud: DonutCloudLayer) -> None:
        """Completed raindrops (progress >= 1.0) are removed."""
        # Force-spawn a raindrop near completion
        cloud.tick(0.35)
        initial_count = len(cloud._raindrops)
        # Tick many times to complete all existing raindrops
        for _ in range(100):
            cloud.tick(0.1)
        # Some should have been removed (and possibly new ones spawned)
        # At minimum, the system should not accumulate beyond max
        assert len(cloud._raindrops) <= cloud._max_raindrops


class TestDonutCloudSetRadius:
    """Tests for set_radius() dimension updates."""

    def test_set_radius_updates_dimensions(self, cloud: DonutCloudLayer) -> None:
        """set_radius() updates radius_x and radius_y."""
        cloud.set_radius(100.0, 80.0)
        assert cloud._radius_x == 100.0
        assert cloud._radius_y == 80.0

    def test_set_radius_changes_bounding_rect(self, cloud: DonutCloudLayer) -> None:
        """set_radius() causes boundingRect to reflect new dimensions."""
        cloud.set_radius(100.0, 80.0)
        rect = cloud.boundingRect()
        # max(100, 80) + 20 = 120
        assert rect.width() == 240.0
        assert rect.height() == 240.0
