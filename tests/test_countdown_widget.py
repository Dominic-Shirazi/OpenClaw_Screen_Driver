"""Tests for CountdownWidget and AbortPanel."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from recorder.overlay.abort_panel import AbortPanel
from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.countdown_widget import CountdownWidget


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


# -----------------------------------------------------------------------
# CountdownWidget
# -----------------------------------------------------------------------


class TestCountdownWidgetInstantiation:
    """Tests for CountdownWidget construction."""

    def test_default_seconds(self, qapp: QApplication) -> None:
        """Default countdown is 3 seconds."""
        clock = AnimationClock()
        w = CountdownWidget(clock)
        assert w._total_seconds == 3

    def test_z_value(self, qapp: QApplication) -> None:
        """Z-value is 300."""
        clock = AnimationClock()
        w = CountdownWidget(clock)
        assert w.zValue() == 300

    def test_starts_invisible(self, qapp: QApplication) -> None:
        """Widget is hidden on creation."""
        clock = AnimationClock()
        w = CountdownWidget(clock)
        assert not w.isVisible()


class TestCountdownWidgetPosition:
    """Tests for set_position offset."""

    def test_offset_applied(self, qapp: QApplication) -> None:
        """set_position applies +20px offset in both axes."""
        clock = AnimationClock()
        w = CountdownWidget(clock)
        w.set_position(100.0, 200.0)
        pos = w.pos()
        assert pos.x() == 120.0
        assert pos.y() == 220.0


class TestCountdownWidgetVisibility:
    """Tests for start/stop visibility."""

    def test_start_makes_visible(self, qapp: QApplication) -> None:
        """start() makes widget visible."""
        clock = AnimationClock()
        w = CountdownWidget(clock)
        w.start()
        assert w.isVisible()
        w.stop()  # cleanup

    def test_stop_makes_invisible(self, qapp: QApplication) -> None:
        """stop() makes widget invisible."""
        clock = AnimationClock()
        w = CountdownWidget(clock)
        w.start()
        w.stop()
        assert not w.isVisible()


class TestCountdownWidgetBoundingRect:
    """Tests for bounding rect."""

    def test_non_empty(self, qapp: QApplication) -> None:
        """boundingRect() returns non-empty rect."""
        clock = AnimationClock()
        w = CountdownWidget(clock)
        rect = w.boundingRect()
        assert rect.width() > 0
        assert rect.height() > 0


class TestCountdownWidgetSignal:
    """Tests for countdown_finished signal."""

    def test_signal_exists(self, qapp: QApplication) -> None:
        """countdown_finished signal is connectable."""
        clock = AnimationClock()
        w = CountdownWidget(clock)
        handler = MagicMock()
        w.countdown_finished.connect(handler)
        # Should not raise


# -----------------------------------------------------------------------
# AbortPanel
# -----------------------------------------------------------------------


class TestAbortPanelInstantiation:
    """Tests for AbortPanel construction."""

    def test_z_value(self, qapp: QApplication) -> None:
        """Z-value is 200."""
        p = AbortPanel()
        assert p.zValue() == 200

    def test_starts_invisible(self, qapp: QApplication) -> None:
        """Panel is hidden on creation."""
        p = AbortPanel()
        assert not p.isVisible()


class TestAbortPanelShowHide:
    """Tests for show/hide behavior."""

    def test_show_makes_visible(self, qapp: QApplication) -> None:
        """show_panel makes panel visible."""
        p = AbortPanel()
        p.show_panel(5)
        assert p.isVisible()

    def test_hide_makes_invisible(self, qapp: QApplication) -> None:
        """hide_panel makes panel invisible."""
        p = AbortPanel()
        p.show_panel(3)
        p.hide_panel()
        assert not p.isVisible()

    def test_body_text_contains_count(self, qapp: QApplication) -> None:
        """Body text contains the step count."""
        p = AbortPanel()
        p.show_panel(7)
        assert p._body_label is not None
        assert "7" in p._body_label.text()


class TestAbortPanelSignals:
    """Tests for abort panel signals."""

    def test_discard_signal_exists(self, qapp: QApplication) -> None:
        """discard_clicked signal is connectable."""
        p = AbortPanel()
        handler = MagicMock()
        p.discard_clicked.connect(handler)

    def test_keep_signal_exists(self, qapp: QApplication) -> None:
        """keep_clicked signal is connectable."""
        p = AbortPanel()
        handler = MagicMock()
        p.keep_clicked.connect(handler)
