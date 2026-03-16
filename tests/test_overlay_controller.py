"""Integration tests for OverlayController state machine and lifecycle.

Mocks the view and hotkey listener to test controller logic in
isolation without requiring a display or Qt event loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from recorder.overlay.state import OverlayState


@pytest.fixture()
def mock_view_cls() -> MagicMock:
    """Return a mock OverlayView class whose instances have the expected API."""
    view_instance = MagicMock()
    view_instance.apply_state = MagicMock()
    view_instance.hide_for_capture = MagicMock()
    view_instance.show_after_capture = MagicMock()
    view_instance.render_bboxes = MagicMock()
    view_instance.clear_bboxes = MagicMock()
    view_instance.close = MagicMock()
    view_instance.show = MagicMock()
    view_instance.raise_ = MagicMock()
    view_instance.setGeometry = MagicMock()
    cls_mock = MagicMock(return_value=view_instance)
    return cls_mock


@pytest.fixture()
def mock_hotkey_factory() -> MagicMock:
    """Return a mock _create_hotkey_listener that yields a start/stop listener."""
    listener = MagicMock()
    listener.start = MagicMock()
    listener.stop = MagicMock()
    factory = MagicMock(return_value=listener)
    return factory


@pytest.fixture()
def controller(
    mock_view_cls: MagicMock,
    mock_hotkey_factory: MagicMock,
) -> MagicMock:
    """Create an OverlayController with mocked view and hotkey listener."""
    with (
        patch("recorder.overlay.controller.OverlayView", mock_view_cls),
        patch("recorder.overlay.controller._create_hotkey_listener", mock_hotkey_factory),
        patch("recorder.overlay.controller.QApplication") as mock_qapp,
    ):
        mock_screen = MagicMock()
        mock_screen.geometry.return_value = MagicMock()
        mock_qapp.primaryScreen.return_value = mock_screen

        # Import here so patches apply to the lazy imports in show()
        # Actually we need to patch at module level -- the controller
        # lazy-imports inside show(). We'll patch at usage site.
        pass

    # Create controller without patches (they apply inside show())
    from recorder.overlay.controller import OverlayController

    return OverlayController()


class TestControllerInitialState:
    """Tests for controller default state before show() is called."""

    def test_initial_state_is_ready(self) -> None:
        """Controller starts in READY state."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        assert ctrl.state == OverlayState.READY

    def test_not_active_initially(self) -> None:
        """Controller is not active until show() is called."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        assert ctrl.is_active is False


class TestControllerStateTransitions:
    """Tests for F2 toggle state machine via _handle_toggle()."""

    def test_toggle_state_ready_to_recording(self) -> None:
        """After one toggle, state goes from READY to RECORDING."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        ctrl._handle_toggle()
        assert ctrl.state == OverlayState.RECORDING

    def test_toggle_state_recording_to_ready(self) -> None:
        """After two toggles, state returns to READY."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        ctrl._handle_toggle()
        ctrl._handle_toggle()
        assert ctrl.state == OverlayState.READY

    def test_state_changed_callback_fired(self) -> None:
        """on_state_changed callback is invoked with the new state."""
        from recorder.overlay.controller import OverlayController

        callback = MagicMock()
        ctrl = OverlayController(on_state_changed=callback)
        ctrl._handle_toggle()
        callback.assert_called_once_with(OverlayState.RECORDING)


class TestControllerCloseCallbacks:
    """Tests for close() and _handle_close() callback firing."""

    def test_close_fires_abort_when_ready(self) -> None:
        """Closing while READY fires the on_abort callback."""
        from recorder.overlay.controller import OverlayController

        abort_cb = MagicMock()
        ctrl = OverlayController(on_abort=abort_cb)
        ctrl._handle_close()
        abort_cb.assert_called_once()

    def test_close_fires_save_when_recording(self) -> None:
        """Closing while RECORDING fires the on_save callback."""
        from recorder.overlay.controller import OverlayController

        save_cb = MagicMock()
        ctrl = OverlayController(on_save=save_cb)
        ctrl._handle_toggle()  # READY -> RECORDING
        ctrl._handle_close()
        save_cb.assert_called_once()

    def test_close_resets_state(self) -> None:
        """After close(), state returns to READY and is_active is False."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        ctrl._handle_toggle()  # READY -> RECORDING
        ctrl._handle_close()
        assert ctrl.state == OverlayState.READY
        assert ctrl.is_active is False


class TestControllerDelegation:
    """Tests for delegation methods that forward to the view."""

    def test_hide_for_capture_delegates(self) -> None:
        """hide_for_capture() calls view.hide_for_capture()."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        mock_view = MagicMock()
        ctrl._view = mock_view

        ctrl.hide_for_capture()
        mock_view.hide_for_capture.assert_called_once()

    def test_show_after_capture_delegates(self) -> None:
        """show_after_capture() calls view.show_after_capture()."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        mock_view = MagicMock()
        ctrl._view = mock_view

        ctrl.show_after_capture()
        mock_view.show_after_capture.assert_called_once()

    def test_set_bboxes_delegates(self) -> None:
        """set_bboxes() calls view.render_bboxes() with the data."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        mock_view = MagicMock()
        ctrl._view = mock_view

        boxes = [{"x": 10, "y": 20, "w": 100, "h": 50}]
        ctrl.set_bboxes(boxes)
        mock_view.render_bboxes.assert_called_once_with(boxes)

    def test_clear_bboxes_delegates(self) -> None:
        """clear_bboxes() calls view.clear_bboxes()."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        mock_view = MagicMock()
        ctrl._view = mock_view

        ctrl.clear_bboxes()
        mock_view.clear_bboxes.assert_called_once()

    def test_hide_for_capture_noop_without_view(self) -> None:
        """hide_for_capture() does nothing when view is None."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        # Should not raise
        ctrl.hide_for_capture()

    def test_set_bboxes_noop_without_view(self) -> None:
        """set_bboxes() does nothing when view is None."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        # Should not raise
        ctrl.set_bboxes([{"x": 0, "y": 0, "w": 10, "h": 10}])
