"""Unit tests for BboxLayer morph_to() animation capability.

Tests cover original API regression, morph target storage, morph
progress interpolation, and the is_morphing flag.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure offscreen rendering for headless CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

# Create QApplication once for all tests in this module
_app: QApplication | None = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from recorder.overlay.bbox_layer import BboxLayer


class TestBboxLayerOriginalAPI:
    """Verify original API still works after morph extension."""

    def test_construction_unchanged(self) -> None:
        """BboxLayer constructs with original (x, y, w, h, color) API."""
        layer = BboxLayer(10, 20, 100, 200, (255, 50, 50, 200))
        assert layer is not None

    def test_get_rect_returns_initial(self) -> None:
        """get_rect() returns initial coordinates."""
        layer = BboxLayer(10, 20, 100, 200, (255, 50, 50, 200))
        assert layer.get_rect() == (10, 20, 100, 200)

    def test_highlight_still_works(self) -> None:
        """highlight() method still sets opacity."""
        layer = BboxLayer(10, 20, 100, 200, (255, 50, 50, 200))
        layer.highlight(False)
        assert layer.opacity() == pytest.approx(0.4)
        layer.highlight(True)
        assert layer.opacity() == pytest.approx(1.0)

    def test_reset_highlight_still_works(self) -> None:
        """reset_highlight() restores full opacity."""
        layer = BboxLayer(10, 20, 100, 200, (255, 50, 50, 200))
        layer.highlight(False)
        layer.reset_highlight()
        assert layer.opacity() == pytest.approx(1.0)


class TestBboxLayerMorphTo:
    """Test morph_to() and related methods."""

    def test_morph_to_stores_target(self) -> None:
        """morph_to() stores the target coordinates."""
        layer = BboxLayer(10, 20, 100, 200, (255, 50, 50, 200))
        layer.morph_to(15, 25, 90, 180)
        assert layer._morph_end == (15, 25, 90, 180)

    def test_is_morphing_true_during_morph(self) -> None:
        """is_morphing returns True after morph_to() starts."""
        layer = BboxLayer(10, 20, 100, 200, (255, 50, 50, 200))
        assert layer.is_morphing is False
        layer.morph_to(15, 25, 90, 180)
        assert layer.is_morphing is True

    def test_apply_morph_progress_midpoint(self) -> None:
        """_apply_morph_progress(0.5) interpolates to midpoint."""
        layer = BboxLayer(0, 0, 100, 100, (255, 50, 50, 200))
        layer.morph_to(10, 20, 80, 60)
        # Manually apply midpoint
        layer._apply_morph_progress(0.5)
        x, y, w, h = layer.get_rect()
        assert x == pytest.approx(5, abs=1)
        assert y == pytest.approx(10, abs=1)
        assert w == pytest.approx(90, abs=1)
        assert h == pytest.approx(80, abs=1)

    def test_apply_morph_progress_complete(self) -> None:
        """_apply_morph_progress(1.0) produces exact target coordinates."""
        layer = BboxLayer(0, 0, 100, 100, (255, 50, 50, 200))
        layer.morph_to(10, 20, 80, 60)
        layer._apply_morph_progress(1.0)
        assert layer.get_rect() == (10, 20, 80, 60)

    def test_get_rect_updates_during_morph(self) -> None:
        """get_rect() returns interpolated coordinates during morph."""
        layer = BboxLayer(0, 0, 200, 200, (255, 50, 50, 200))
        layer.morph_to(20, 20, 160, 160)
        layer._apply_morph_progress(0.25)
        x, y, w, h = layer.get_rect()
        assert x == pytest.approx(5, abs=1)
        assert y == pytest.approx(5, abs=1)
        assert w == pytest.approx(190, abs=1)
        assert h == pytest.approx(190, abs=1)

    def test_morph_duration_default(self) -> None:
        """Default morph duration is 500ms."""
        layer = BboxLayer(0, 0, 100, 100, (255, 50, 50, 200))
        layer.morph_to(10, 20, 80, 60)
        assert layer._morph_anim is not None
        assert layer._morph_anim.duration() == 500
