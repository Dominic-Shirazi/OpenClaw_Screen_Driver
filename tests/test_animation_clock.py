"""Unit tests for AnimationClock -- shared animation tick infrastructure.

Tests cover instantiation, callback registration/unregistration,
start/stop lifecycle, delta-time delivery, and delta-time cap.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists before any test
_app: QApplication | None = None


@pytest.fixture(autouse=True)
def qapp() -> QApplication:
    """Provide a QApplication instance for all tests."""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


def _make_clock():  # noqa: ANN202
    """Import and instantiate AnimationClock."""
    from recorder.overlay.animation_clock import AnimationClock

    return AnimationClock()


class TestAnimationClockInstantiation:
    """Test 1: AnimationClock can be instantiated without errors."""

    def test_instantiation(self) -> None:
        clock = _make_clock()
        assert clock is not None


class TestCallbackRegistration:
    """Test 2: register() adds a callback, unregister() removes it."""

    def test_register_adds_callback(self) -> None:
        clock = _make_clock()
        cb = MagicMock()
        clock.register(cb)
        assert cb in clock._callbacks

    def test_unregister_removes_callback(self) -> None:
        clock = _make_clock()
        cb = MagicMock()
        clock.register(cb)
        clock.unregister(cb)
        assert cb not in clock._callbacks

    def test_unregister_missing_callback_no_error(self) -> None:
        clock = _make_clock()
        cb = MagicMock()
        # Should not raise
        clock.unregister(cb)


class TestStartStopLifecycle:
    """Test 3 & 4: start/stop control callback invocation."""

    def test_start_delivers_delta_to_callback(self, qapp: QApplication) -> None:
        """After start(), registered callbacks receive a float delta > 0."""
        clock = _make_clock()
        received: list[float] = []

        def on_tick(dt: float) -> None:
            received.append(dt)
            if len(received) >= 1:
                clock.stop()
                qapp.quit()

        clock.register(on_tick)
        clock.start()
        # Safety timeout to prevent hang
        QTimer.singleShot(200, qapp.quit)
        qapp.exec()
        clock.stop()

        assert len(received) >= 1, "Callback should have been called at least once"
        assert isinstance(received[0], float)
        assert received[0] > 0

    def test_stop_prevents_further_callbacks(self, qapp: QApplication) -> None:
        """After stop(), no more callbacks are delivered."""
        clock = _make_clock()
        call_count = 0

        def on_tick(dt: float) -> None:
            nonlocal call_count
            call_count += 1

        clock.register(on_tick)
        clock.start()
        clock.stop()

        # Let event loop run briefly -- no callbacks should fire
        QTimer.singleShot(50, qapp.quit)
        qapp.exec()

        assert call_count == 0


class TestDeltaTimeCap:
    """Test 5: Delta-time is capped at 0.1 seconds."""

    def test_delta_capped_at_100ms(self, qapp: QApplication) -> None:
        """If elapsed time is large, delta is capped at 0.1s."""
        from recorder.overlay.animation_clock import AnimationClock

        clock = AnimationClock()
        received: list[float] = []

        def on_tick(dt: float) -> None:
            received.append(dt)
            clock.stop()
            qapp.quit()

        clock.register(on_tick)

        # Patch QElapsedTimer.restart to return 200ms
        with patch.object(
            clock._elapsed, "restart", return_value=200
        ):
            clock.start()
            # Force a tick manually
            clock._tick()

        clock.stop()

        assert len(received) >= 1
        assert received[0] <= 0.1


class TestMultipleCallbacks:
    """Test 6: Multiple callbacks all receive the same delta per tick."""

    def test_same_delta_for_all_callbacks(self) -> None:
        from recorder.overlay.animation_clock import AnimationClock

        clock = AnimationClock()
        deltas_a: list[float] = []
        deltas_b: list[float] = []

        clock.register(lambda dt: deltas_a.append(dt))
        clock.register(lambda dt: deltas_b.append(dt))

        # Patch elapsed to return a known value
        with patch.object(clock._elapsed, "restart", return_value=16):
            clock._tick()

        assert len(deltas_a) == 1
        assert len(deltas_b) == 1
        assert deltas_a[0] == deltas_b[0]
        assert abs(deltas_a[0] - 0.016) < 0.001
