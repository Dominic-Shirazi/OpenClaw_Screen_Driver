"""Comprehensive unit tests for RecordSession orchestrator and platform_utils."""
from __future__ import annotations

import os
import sys
import threading
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication

from recorder.overlay.record_phase import RecordPhase
from recorder.overlay.toolbar_panel import ToolbarMode
from recorder.platform_utils import minimize_all_windows


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


def _make_mock_controller() -> MagicMock:
    """Create a fully mocked OverlayController."""
    ctrl = MagicMock()
    ctrl.hide_for_capture = MagicMock()
    ctrl.show_after_capture = MagicMock()
    ctrl.show_tag_dialog = MagicMock()
    ctrl.dismiss_tag_dialog = MagicMock()
    ctrl.get_tag_data = MagicMock(return_value={"label": "test", "element_type": "button"})
    ctrl.set_toolbar_mode = MagicMock()
    ctrl.show_toolbar = MagicMock()
    ctrl.start_scan = MagicMock()
    ctrl.finish_scan = MagicMock()
    ctrl.start_card_glow_pulse = MagicMock()
    ctrl.stop_card_glow_pulse = MagicMock()
    return ctrl


def _make_session(ctrl: MagicMock | None = None) -> "RecordSession":
    """Create a RecordSession with mocked controller and config."""
    from recorder.record_session import RecordSession

    if ctrl is None:
        ctrl = _make_mock_controller()

    with patch("recorder.record_session.get_config", return_value={"overlay": {"capture_delay_ms": 0}}):
        session = RecordSession(ctrl, routine_name="test_routine", start_from="desktop")
    return session


import numpy as np

_DUMMY_SCREENSHOT = np.zeros((100, 100, 3), dtype=np.uint8)


class TestSessionInit:
    """REC-01: RecordSession initialization tests."""

    def test_session_init(self, qapp: QApplication) -> None:
        """RecordSession stores routine_name and start_from."""
        session = _make_session()
        assert session._routine_name == "test_routine"
        assert session._start_from == "desktop"
        assert session.step_count == 0

    def test_session_init_custom_start(self, qapp: QApplication) -> None:
        """RecordSession accepts custom start_from value."""
        ctrl = _make_mock_controller()
        with patch("recorder.record_session.get_config", return_value={}):
            from recorder.record_session import RecordSession
            session = RecordSession(ctrl, routine_name="r1", start_from="app")
        assert session._start_from == "app"


class TestSessionStart:
    """REC-03: Start phase tests."""

    def test_session_start_sets_awaiting_click(self, qapp: QApplication) -> None:
        """After start(), phase is AWAITING_CLICK."""
        session = _make_session()
        session.start()
        assert session.phase == RecordPhase.AWAITING_CLICK

    def test_session_start_shows_toolbar(self, qapp: QApplication) -> None:
        """start() calls set_toolbar_mode(RECORDING) and show_toolbar."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session.start()
        ctrl.set_toolbar_mode.assert_called_with(ToolbarMode.RECORDING)
        ctrl.show_toolbar.assert_called_once()


class TestClickGating:
    """REC-04: Phase gating on selection."""

    def test_click_gates_on_phase(self, qapp: QApplication) -> None:
        """on_selection is ignored when phase is not AWAITING_CLICK."""
        session = _make_session()
        session._phase = RecordPhase.DETECTING

        with patch("recorder.record_session.get_config", return_value={"overlay": {"capture_delay_ms": 0}}):
            session.on_selection(100, 200, 0, 0)

        # Should not have changed phase
        assert session.phase == RecordPhase.DETECTING
        assert session._current_step is None


class TestCapturePipeline:
    """REC-04: Capture pipeline tests."""

    def test_click_triggers_capture_pipeline(self, qapp: QApplication) -> None:
        """on_selection calls hide_for_capture and transitions to CAPTURING."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session.start()

        with patch("recorder.record_session.get_config", return_value={"overlay": {"capture_delay_ms": 0}}), \
             patch("core.capture.screenshot_full", return_value=_DUMMY_SCREENSHOT), \
             patch.object(session, "_start_detection"):
            session.on_selection(100, 200, 0, 0)

        ctrl.hide_for_capture.assert_called_once()
        ctrl.show_after_capture.assert_called_once()

    def test_card_glow_starts_on_detecting(self, qapp: QApplication) -> None:
        """_run_capture_pipeline calls start_card_glow_pulse when entering DETECTING."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session.start()

        with patch("recorder.record_session.get_config", return_value={"overlay": {"capture_delay_ms": 0}}), \
             patch("core.capture.screenshot_full", return_value=_DUMMY_SCREENSHOT), \
             patch.object(session, "_start_detection"):
            session.on_selection(50, 50, 0, 0)

        ctrl.start_card_glow_pulse.assert_called_once()


class TestDetection:
    """REC-04: Detection cascade tests."""

    def test_detection_cascade_local_first(self, qapp: QApplication) -> None:
        """_start_detection crops around click point first."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.DETECTING

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [{
            "rect": {"x": 10, "y": 10, "w": 40, "h": 20},
            "type_guess": "button",
            "florence_caption": "OK",
            "confidence": 0.9,
        }]

        # Run detection worker directly (not in thread)
        with patch("core.detection.get_detector", return_value=mock_detector):
            session._detection_worker(_DUMMY_SCREENSHOT, 50, 50, False)

        # Should have called detect at least once (local crop)
        mock_detector.detect.assert_called()

    def test_detection_result_shows_scan(self, qapp: QApplication) -> None:
        """_on_detection_ready with bbox calls start_scan."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.DETECTING
        session._screenshot = _DUMMY_SCREENSHOT

        # Mock _accept_bbox to prevent VLM thread from starting
        with patch.object(session, "_accept_bbox"):
            session._on_detection_ready({
                "bbox": {"x": 10, "y": 20, "w": 50, "h": 30},
                "type_guess": "button",
                "florence_caption": "",
                "candidates": [],
            })

        ctrl.start_scan.assert_called_once_with(10, 20, 50, 30)

    def test_card_glow_stops_on_detection_ready(self, qapp: QApplication) -> None:
        """_on_detection_ready calls stop_card_glow_pulse."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.DETECTING
        session._screenshot = _DUMMY_SCREENSHOT

        with patch.object(session, "_accept_bbox"):
            session._on_detection_ready({
                "bbox": {"x": 10, "y": 20, "w": 50, "h": 30},
                "type_guess": "button",
                "florence_caption": "",
                "candidates": [],
            })

        ctrl.stop_card_glow_pulse.assert_called_once()

    def test_detection_starts_daemon_thread(self, qapp: QApplication) -> None:
        """_start_detection creates a daemon Thread."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)

        threads_before = threading.active_count()

        with patch.object(session, "_detection_worker"):
            with patch("threading.Thread") as mock_thread_cls:
                mock_thread = MagicMock()
                mock_thread_cls.return_value = mock_thread
                session._start_detection(_DUMMY_SCREENSHOT, 50, 50, False)

                mock_thread_cls.assert_called_once()
                call_kwargs = mock_thread_cls.call_args
                assert call_kwargs.kwargs.get("daemon") is True
                mock_thread.start.assert_called_once()


class TestDragCapture:
    """REC-05: Drag capture and bbox editing tests."""

    def test_drag_capture_refines_bbox(self, qapp: QApplication) -> None:
        """on_selection with w>0, h>0 attempts IoU refinement."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session.start()

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [{
            "rect": {"x": 95, "y": 195, "w": 60, "h": 40},
            "type_guess": "button",
            "florence_caption": "Submit",
            "confidence": 0.8,
        }]

        with patch("recorder.record_session.get_config", return_value={"overlay": {"capture_delay_ms": 0}}), \
             patch("core.capture.screenshot_full", return_value=_DUMMY_SCREENSHOT), \
             patch("core.detection.get_detector", return_value=mock_detector):
            # Run selection with drag dimensions and let detection worker run inline
            session.on_selection(100, 200, 50, 30)
            # Wait briefly for daemon thread
            import time
            time.sleep(0.2)

        assert session._is_drag_capture is True
        assert session._original_drag_rect == (100, 200, 50, 30)

    def test_drag_capture_shows_bbox_editing_toolbar(self, qapp: QApplication) -> None:
        """_on_detection_ready for drag capture sets toolbar to BBOX_EDITING."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.DETECTING
        session._is_drag_capture = True
        session._screenshot = _DUMMY_SCREENSHOT
        session._original_drag_rect = (100, 200, 50, 30)

        session._on_detection_ready({
            "bbox": {"x": 95, "y": 195, "w": 60, "h": 40},
            "type_guess": "button",
            "florence_caption": "",
            "candidates": [],
        })

        ctrl.set_toolbar_mode.assert_called_with(ToolbarMode.BBOX_EDITING)

    def test_keep_drag_rejects_ai_bbox(self, qapp: QApplication) -> None:
        """on_toolbar_action("keep_drag") keeps original drag rect and proceeds to VLM."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.BBOX_EDITING
        session._is_drag_capture = True
        session._original_drag_rect = (100, 200, 50, 30)
        session._current_bbox = (95, 195, 60, 40)  # AI bbox
        session._screenshot = _DUMMY_SCREENSHOT

        with patch.object(session, "_start_vlm"):
            session.on_toolbar_action("keep_drag")

        assert session._current_bbox == (100, 200, 50, 30)
        ctrl.finish_scan.assert_called_once_with(100, 200, 50, 30)
        assert session.phase == RecordPhase.VLM_ANALYZING


class TestVLM:
    """REC-06, REC-07: VLM analysis tests."""

    def test_card_glow_starts_on_vlm_analyzing(self, qapp: QApplication) -> None:
        """_accept_bbox calls start_card_glow_pulse when entering VLM_ANALYZING."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.BBOX_EDITING
        session._screenshot = _DUMMY_SCREENSHOT
        session._current_bbox = (10, 20, 50, 30)

        # Reset mock to check only accept_bbox calls
        ctrl.start_card_glow_pulse.reset_mock()

        with patch.object(session, "_start_vlm"):
            session._accept_bbox()

        ctrl.start_card_glow_pulse.assert_called_once()
        assert session.phase == RecordPhase.VLM_ANALYZING

    def test_vlm_success_opens_tag_dialog(self, qapp: QApplication) -> None:
        """_on_vlm_ready opens tag dialog with vlm_data."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.VLM_ANALYZING
        session._current_bbox = (10, 20, 50, 30)

        session._on_vlm_ready({
            "element_type": "button",
            "label_guess": "Submit",
            "confidence": 0.9,
            "ocr_text": "Submit",
        })

        ctrl.show_tag_dialog.assert_called_once()
        call_args = ctrl.show_tag_dialog.call_args
        assert isinstance(call_args[0][0], QRectF)
        vlm_data = call_args[1].get("vlm_data") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["vlm_data"]
        assert vlm_data["element_type"] == "button"
        assert vlm_data["label"] == "Submit"
        assert session.phase == RecordPhase.TAG_DIALOG

    def test_card_glow_stops_on_vlm_ready(self, qapp: QApplication) -> None:
        """_on_vlm_ready calls stop_card_glow_pulse."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.VLM_ANALYZING
        session._current_bbox = (10, 20, 50, 30)

        session._on_vlm_ready({
            "element_type": "button",
            "label_guess": "OK",
            "confidence": 0.8,
            "ocr_text": None,
        })

        ctrl.stop_card_glow_pulse.assert_called_once()

    def test_vlm_failure_opens_tag_dialog_partial(self, qapp: QApplication) -> None:
        """_on_vlm_failed opens tag dialog without full vlm_data."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.VLM_ANALYZING
        session._current_bbox = (10, 20, 50, 30)

        session._on_vlm_failed("VLM timeout")

        ctrl.show_tag_dialog.assert_called_once()
        call_args = ctrl.show_tag_dialog.call_args
        vlm_data = call_args[1].get("vlm_data") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["vlm_data"]
        assert vlm_data["element_type"] == "unknown"
        assert vlm_data["caption"] == "VLM unavailable"
        assert session.phase == RecordPhase.TAG_DIALOG

    def test_card_glow_stops_on_vlm_failed(self, qapp: QApplication) -> None:
        """_on_vlm_failed calls stop_card_glow_pulse."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.VLM_ANALYZING
        session._current_bbox = (10, 20, 50, 30)

        session._on_vlm_failed("Error")

        ctrl.stop_card_glow_pulse.assert_called_once()


class TestTagDialog:
    """REC-10: Tag dialog confirm/dismiss tests."""

    def test_tag_confirmed_transitions_to_countdown(self, qapp: QApplication) -> None:
        """on_tag_confirmed merges step data and transitions to COUNTDOWN."""
        ctrl = _make_mock_controller()
        ctrl.show_countdown.return_value = MagicMock()
        session = _make_session(ctrl)
        session._phase = RecordPhase.TAG_DIALOG
        session._current_step = {"click_x": 100, "click_y": 200, "is_drag": False}
        session._current_bbox = (90, 190, 60, 40)

        session.on_tag_confirmed({"label": "Submit", "element_type": "button"})

        # Step is NOT added to _steps yet (happens after dry-run validation)
        assert session.step_count == 0
        # But current_step has merged tag data
        assert session._current_step["tag_data"]["label"] == "Submit"
        assert session._current_step["bbox"] == (90, 190, 60, 40)
        assert session.phase == RecordPhase.COUNTDOWN
        ctrl.show_countdown.assert_called_once_with(3)
        ctrl.set_toolbar_mode.assert_called_with(ToolbarMode.DRY_RUN)

    def test_tag_dismissed_returns_to_awaiting(self, qapp: QApplication) -> None:
        """on_tag_dismissed returns to AWAITING_CLICK without adding step."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.TAG_DIALOG
        session._current_step = {"click_x": 100, "click_y": 200, "is_drag": False}

        session.on_tag_dismissed({})

        assert session.step_count == 0
        assert session.phase == RecordPhase.AWAITING_CLICK
        assert session._current_step is None


class TestPlatformUtils:
    """Platform utility tests."""

    def test_minimize_windows_platform_guard(self) -> None:
        """minimize_all_windows returns False on non-win32."""
        with patch("recorder.platform_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = minimize_all_windows()
        assert result is False


class TestDryRunStages:
    """REC-08, REC-09: Dry-run stage transition tests."""

    def test_dry_run_stages(self, qapp: QApplication) -> None:
        """on_tag_confirmed transitions to COUNTDOWN; countdown wires signal."""
        ctrl = _make_mock_controller()
        mock_countdown_widget = MagicMock()
        ctrl.show_countdown.return_value = mock_countdown_widget
        session = _make_session(ctrl)
        session._phase = RecordPhase.TAG_DIALOG
        session._current_step = {"click_x": 100, "click_y": 200, "is_drag": False}
        session._current_bbox = (90, 190, 60, 40)

        # Track phase transitions
        phases_seen: list[RecordPhase] = []
        original_set_phase = session._set_phase

        def tracking_set_phase(phase: RecordPhase) -> None:
            phases_seen.append(phase)
            original_set_phase(phase)

        session._set_phase = tracking_set_phase

        session.on_tag_confirmed({"label": "OK", "element_type": "button"})

        # Should have transitioned to COUNTDOWN
        assert RecordPhase.COUNTDOWN in phases_seen
        assert session.phase == RecordPhase.COUNTDOWN
        # Countdown widget wired
        mock_countdown_widget.countdown_finished.connect.assert_called_once()
        # Step NOT yet added (added after dry-run "yes")
        assert session.step_count == 0


class TestToolbarRouting:
    """Toolbar action routing tests."""

    def test_toolbar_routes_confirm(self, qapp: QApplication) -> None:
        """on_toolbar_action('confirm') calls on_tag_confirmed -> COUNTDOWN."""
        ctrl = _make_mock_controller()
        ctrl.show_countdown.return_value = MagicMock()
        session = _make_session(ctrl)
        session._phase = RecordPhase.TAG_DIALOG
        session._current_step = {"click_x": 50, "click_y": 50, "is_drag": False}
        session._current_bbox = (40, 40, 60, 40)

        session.on_toolbar_action("confirm")

        # Step goes to countdown, not immediately stored
        assert session.phase == RecordPhase.COUNTDOWN
        ctrl.show_countdown.assert_called_once_with(3)

    def test_toolbar_routes_dismiss(self, qapp: QApplication) -> None:
        """on_toolbar_action('dismiss') calls on_tag_dismissed."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.TAG_DIALOG
        session._current_step = {"click_x": 50, "click_y": 50, "is_drag": False}

        session.on_toolbar_action("dismiss")

        assert session.step_count == 0
        assert session.phase == RecordPhase.AWAITING_CLICK

    def test_toolbar_routes_keep_drag(self, qapp: QApplication) -> None:
        """on_toolbar_action('keep_drag') calls _handle_keep_drag."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.BBOX_EDITING
        session._is_drag_capture = True
        session._original_drag_rect = (100, 200, 50, 30)
        session._current_bbox = (95, 195, 60, 40)
        session._screenshot = _DUMMY_SCREENSHOT

        with patch.object(session, "_start_vlm"):
            session.on_toolbar_action("keep_drag")

        assert session._current_bbox == (100, 200, 50, 30)
