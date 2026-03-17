"""Tests for TypewriterEngine."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication, QLineEdit

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.typewriter_engine import TypewriterEngine


@pytest.fixture(scope="module")
def qapp():
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    yield app


class TestSimultaneousFill:
    """Test that multiple fields fill at the same time."""

    def test_two_fields_advance_independently(self, qapp: QApplication) -> None:
        """Both fields should have text after enough ticks."""
        clock = AnimationClock()
        engine = TypewriterEngine(clock)
        w1 = QLineEdit()
        w2 = QLineEdit()
        engine.start([(w1, "Hello"), (w2, "World Test")])
        # Simulate 2 seconds of ticks at 60fps
        for _ in range(120):
            engine.tick(1 / 60)
        assert len(w1.text()) > 0
        assert len(w2.text()) > 0
        assert engine.is_active is False  # both should be done after 2s at 20+ cps

    def test_fields_complete_at_different_times(self, qapp: QApplication) -> None:
        """Shorter text should finish before longer text."""
        clock = AnimationClock()
        engine = TypewriterEngine(clock)
        w1 = QLineEdit()
        w2 = QLineEdit()
        # Patch random to give predictable speeds
        with patch("recorder.overlay.typewriter_engine.random") as mock_rng:
            mock_rng.uniform.side_effect = [30.0, 30.0]  # same speed
            engine.start([(w1, "Hi"), (w2, "Hello World")])
        # After short time, short field done, long field not
        for _ in range(10):
            engine.tick(1 / 60)
        # "Hi" at 30 cps = ~0.067s = ~4 ticks. After 10 ticks should be done
        assert w1.text() == "Hi"

    def test_finished_signal_emitted(self, qapp: QApplication) -> None:
        """finished signal fires when all fields complete."""
        clock = AnimationClock()
        engine = TypewriterEngine(clock)
        w1 = QLineEdit()
        handler = MagicMock()
        engine.finished.connect(handler)
        engine.start([(w1, "AB")])
        for _ in range(120):
            engine.tick(1 / 60)
        handler.assert_called_once()

    def test_char_inserted_signal(self, qapp: QApplication) -> None:
        """char_inserted signal fires for each character."""
        clock = AnimationClock()
        engine = TypewriterEngine(clock)
        w1 = QLineEdit()
        signals: list[int] = []
        engine.char_inserted.connect(lambda idx: signals.append(idx))
        engine.start([(w1, "ABC")])
        for _ in range(120):
            engine.tick(1 / 60)
        assert len(signals) == 3  # one per character
        assert all(s == 0 for s in signals)  # all from field index 0


class TestInterrupt:
    """Test user interruption of typewriter."""

    def test_interrupt_stops_field(self, qapp: QApplication) -> None:
        """Interrupting a field freezes its text."""
        clock = AnimationClock()
        engine = TypewriterEngine(clock)
        w1 = QLineEdit()
        engine.start([(w1, "Hello World")])
        engine.tick(0.05)  # type ~1 char
        engine.interrupt_field(0)
        text_at_interrupt = w1.text()
        engine.tick(1.0)  # big tick
        assert w1.text() == text_at_interrupt  # no change after interrupt


class TestEdgeCase:
    """Edge cases."""

    def test_empty_target_text(self, qapp: QApplication) -> None:
        """Empty target should immediately be done."""
        clock = AnimationClock()
        engine = TypewriterEngine(clock)
        w1 = QLineEdit()
        handler = MagicMock()
        engine.finished.connect(handler)
        engine.start([(w1, "")])
        engine.tick(0.016)
        assert engine.is_active is False

    def test_stop_unregisters(self, qapp: QApplication) -> None:
        """stop() should unregister from clock."""
        clock = AnimationClock()
        engine = TypewriterEngine(clock)
        w1 = QLineEdit()
        engine.start([(w1, "Test")])
        engine.stop()
        assert engine.is_active is False
