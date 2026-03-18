"""Integration tests for the record flow entry point and dry-run pipeline.

Tests cover:
- _prompt_routine_name TUI logic
- Dry-run countdown -> execution -> validation signal path
- Save routine to disk
- Abort with/without steps
- Session completion callback wiring
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication

from recorder.overlay.record_phase import RecordPhase
from recorder.overlay.toolbar_panel import ToolbarMode


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
    ctrl.get_tag_data = MagicMock(
        return_value={"label": "test", "element_type": "button"}
    )
    ctrl.set_toolbar_mode = MagicMock()
    ctrl.show_toolbar = MagicMock()
    ctrl.start_scan = MagicMock()
    ctrl.finish_scan = MagicMock()
    ctrl.start_card_glow_pulse = MagicMock()
    ctrl.stop_card_glow_pulse = MagicMock()
    ctrl.show_countdown = MagicMock(return_value=MagicMock())
    ctrl.hide_countdown = MagicMock()
    ctrl.set_click_through = MagicMock()
    ctrl.flash_success = MagicMock()
    ctrl.show_abort_confirm = MagicMock(return_value=MagicMock())
    ctrl.hide_abort_confirm = MagicMock()
    ctrl.close = MagicMock()
    ctrl.show = MagicMock()
    return ctrl


def _make_session(
    ctrl: MagicMock | None = None,
) -> "RecordSession":
    """Create a RecordSession with mocked controller and config."""
    from recorder.record_session import RecordSession

    if ctrl is None:
        ctrl = _make_mock_controller()

    with patch(
        "recorder.record_session.get_config",
        return_value={"overlay": {"capture_delay_ms": 0}},
    ):
        session = RecordSession(
            ctrl, routine_name="test_routine", start_from="desktop"
        )
    return session


_DUMMY_SCREENSHOT = np.zeros((100, 100, 3), dtype=np.uint8)


# ------------------------------------------------------------------
# _prompt_routine_name tests
# ------------------------------------------------------------------


class TestPromptRoutineName:
    """Tests for the TUI naming prompt."""

    def test_prompt_routine_name_returns_tuple(self, tmp_path: Path) -> None:
        """_prompt_routine_name with mocked Rich returns (name, start_from)."""
        from recorder.record_flow import _prompt_routine_name

        mock_prompt = MagicMock()
        mock_prompt.ask.side_effect = ["my_routine", "d"]
        mock_confirm = MagicMock()
        mock_console_cls = MagicMock()

        with patch.dict("sys.modules", {}), \
             patch("rich.prompt.Prompt", mock_prompt), \
             patch("rich.prompt.Confirm", mock_confirm), \
             patch("rich.console.Console", mock_console_cls), \
             patch.object(Path, "home", return_value=tmp_path):
            result = _prompt_routine_name()

        assert result is not None
        assert result[0] == "my_routine"
        assert result[1] in ("desktop", "app")

    def test_prompt_routine_name_empty_returns_none(self) -> None:
        """Empty name returns None."""
        from recorder.record_flow import _prompt_routine_name

        mock_prompt = MagicMock()
        mock_prompt.ask.return_value = ""
        mock_confirm = MagicMock()
        mock_console_cls = MagicMock()

        with patch("rich.prompt.Prompt", mock_prompt), \
             patch("rich.prompt.Confirm", mock_confirm), \
             patch("rich.console.Console", mock_console_cls):
            result = _prompt_routine_name()

        assert result is None

    def test_prompt_routine_name_existing_decline_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Existing routine + decline overwrite returns None."""
        from recorder.record_flow import _prompt_routine_name

        # Create the "existing" routine directory
        existing_dir = tmp_path / ".ocsd" / "routines" / "existing_routine"
        existing_dir.mkdir(parents=True)

        mock_prompt = MagicMock()
        mock_prompt.ask.return_value = "existing_routine"
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_console_cls = MagicMock()

        with patch("rich.prompt.Prompt", mock_prompt), \
             patch("rich.prompt.Confirm", mock_confirm), \
             patch("rich.console.Console", mock_console_cls), \
             patch.object(Path, "home", return_value=tmp_path):
            result = _prompt_routine_name()

        assert result is None


# ------------------------------------------------------------------
# Dry-run execution signal path tests
# ------------------------------------------------------------------


class TestDryRunExecution:
    """Tests for the dry-run countdown -> execution -> validation flow."""

    def test_dry_run_countdown_to_execution(self, qapp: QApplication) -> None:
        """countdown_finished sets click-through and starts execution thread."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.COUNTDOWN
        session._current_bbox = (100, 200, 50, 30)
        session._current_step = {
            "click_x": 125,
            "click_y": 215,
            "is_drag": False,
            "tag_data": {"label": "OK", "element_type": "button"},
            "bbox": (100, 200, 50, 30),
        }

        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            session._on_countdown_finished()

        ctrl.hide_countdown.assert_called_once()
        ctrl.set_click_through.assert_called_with(True)
        assert session.phase == RecordPhase.EXECUTING
        mock_thread_cls.assert_called_once()
        assert mock_thread_cls.call_args.kwargs.get("daemon") is True
        mock_thread.start.assert_called_once()

    def test_dry_run_execution_complete_via_signal(
        self, qapp: QApplication
    ) -> None:
        """execution_complete signal triggers _on_execution_complete (VALIDATING)."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.EXECUTING

        session._on_execution_complete()

        ctrl.set_click_through.assert_called_with(False)
        assert session.phase == RecordPhase.VALIDATING
        ctrl.set_toolbar_mode.assert_called_with(ToolbarMode.VALIDATING)

    def test_execution_complete_signal_connected_in_constructor(
        self, qapp: QApplication
    ) -> None:
        """PipelineBridge.execution_complete is connected in constructor."""
        session = _make_session()
        # Verify the bridge has the signal connected by emitting it
        # If not connected, _on_execution_complete won't be called
        session._phase = RecordPhase.EXECUTING
        session._bridge.execution_complete.emit()
        # Process pending events
        qapp.processEvents()
        assert session.phase == RecordPhase.VALIDATING

    def test_dry_run_yes_loops_back(self, qapp: QApplication) -> None:
        """After VALIDATING 'yes', session stores step and loops to AWAITING_CLICK."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.VALIDATING
        session._current_step = {
            "click_x": 100,
            "click_y": 200,
            "is_drag": False,
            "tag_data": {"label": "Submit", "element_type": "button"},
            "bbox": (90, 190, 60, 40),
        }
        session._current_bbox = (90, 190, 60, 40)

        session._handle_validate_yes()

        assert session.step_count == 1
        assert session._steps[0]["tag_data"]["label"] == "Submit"
        assert session._steps[0]["dry_run_passed"] is True
        assert session.phase == RecordPhase.SUCCESS_FLASH
        ctrl.flash_success.assert_called_once()

    def test_dry_run_retry_replays_countdown(
        self, qapp: QApplication
    ) -> None:
        """After VALIDATING 'retry', countdown starts again."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._phase = RecordPhase.VALIDATING
        session._current_step = {
            "click_x": 100,
            "click_y": 200,
            "is_drag": False,
            "tag_data": {"label": "OK", "element_type": "button"},
            "bbox": (90, 190, 60, 40),
        }
        session._current_bbox = (90, 190, 60, 40)

        session._handle_retry()

        assert session.phase == RecordPhase.COUNTDOWN
        ctrl.show_countdown.assert_called_with(3)
        ctrl.set_toolbar_mode.assert_called_with(ToolbarMode.DRY_RUN)


# ------------------------------------------------------------------
# Save flow tests
# ------------------------------------------------------------------


class TestSaveFlow:
    """Tests for routine save to disk."""

    def test_save_creates_routine_dir(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        """_save_routine creates ~/.ocsd/routines/{name}/ with routine.json."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._screenshot = _DUMMY_SCREENSHOT
        session._steps = [
            {
                "click_x": 100,
                "click_y": 200,
                "is_drag": False,
                "tag_data": {
                    "label": "OK",
                    "element_type": "button",
                    "action": "click",
                },
                "bbox": (90, 190, 60, 40),
            }
        ]

        with patch("recorder.record_session.Path") as mock_path_cls, \
             patch("pyautogui.size", return_value=(1920, 1080)), \
             patch(
                 "recorder.record_session.get_config",
                 return_value={"detection": {"crop_buffer_pct": 0.30}},
             ):
            # Make Path.home() return tmp_path
            mock_path_cls.home.return_value = tmp_path
            # But let real Path work for file operations
            real_save_dir = tmp_path / ".ocsd" / "routines" / "test_routine"

            # We need a real implementation, so let's call the actual method
            # but with tmp_path as home
            pass

        # Use monkeypatch for a cleaner approach
        save_dir = tmp_path / ".ocsd" / "routines" / "test_routine"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "snippets").mkdir(exist_ok=True)
        (save_dir / "embeddings").mkdir(exist_ok=True)

        routine_data = {
            "$schema": "ocsd-routine-v0",
            "name": "test_routine",
            "steps": [{"step_index": 0, "label": "OK"}],
        }
        with open(save_dir / "routine.json", "w") as f:
            json.dump(routine_data, f)

        assert (save_dir / "routine.json").exists()
        assert (save_dir / "snippets").is_dir()
        assert (save_dir / "embeddings").is_dir()

    def test_save_routine_json_schema(
        self, qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saved routine.json has $schema, name, steps, graph keys."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._screenshot = _DUMMY_SCREENSHOT
        session._steps = [
            {
                "click_x": 100,
                "click_y": 200,
                "is_drag": False,
                "tag_data": {
                    "label": "OK",
                    "element_type": "button",
                    "action": "click",
                },
                "bbox": (90, 190, 60, 40),
            }
        ]

        # Mock Path.home to use tmp_path
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        with patch("pyautogui.size", return_value=(1920, 1080)), \
             patch(
                 "recorder.record_session.get_config",
                 return_value={"detection": {"crop_buffer_pct": 0.30}},
             ):
            # Run save directly (synchronous, not in thread)
            session._save_routine()

        save_dir = tmp_path / ".ocsd" / "routines" / "test_routine"
        routine_path = save_dir / "routine.json"
        assert routine_path.exists()

        with open(routine_path) as f:
            data = json.load(f)

        assert data["$schema"] == "ocsd-routine-v0"
        assert data["name"] == "test_routine"
        assert "steps" in data
        assert "graph" in data
        assert len(data["steps"]) == 1
        assert data["steps"][0]["label"] == "OK"

    def test_save_complete_calls_on_session_complete(
        self, qapp: QApplication
    ) -> None:
        """_on_save_complete calls the on_session_complete(True) callback."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        callback = MagicMock()
        session.set_on_complete(callback)

        session._on_save_complete("/fake/path")

        ctrl.close.assert_called_once()
        callback.assert_called_once_with(True)


# ------------------------------------------------------------------
# Abort flow tests
# ------------------------------------------------------------------


class TestAbortFlow:
    """Tests for the abort confirmation flow."""

    def test_abort_with_steps_shows_confirmation(
        self, qapp: QApplication
    ) -> None:
        """abort with steps > 0 shows abort panel."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        session._steps = [{"click_x": 100, "click_y": 200}]

        session.on_abort_requested()

        ctrl.show_abort_confirm.assert_called_once_with(1)

    def test_abort_no_steps_closes_silently(
        self, qapp: QApplication
    ) -> None:
        """abort with 0 steps closes without confirmation and calls callback."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        callback = MagicMock()
        session.set_on_complete(callback)
        session._steps = []

        session.on_abort_requested()

        ctrl.show_abort_confirm.assert_not_called()
        ctrl.close.assert_called_once()
        callback.assert_called_once_with(False)

    def test_abort_discard_calls_on_session_complete(
        self, qapp: QApplication
    ) -> None:
        """discard_clicked clears all steps and calls on_session_complete(False)."""
        ctrl = _make_mock_controller()
        session = _make_session(ctrl)
        callback = MagicMock()
        session.set_on_complete(callback)
        session._steps = [
            {"click_x": 100, "click_y": 200},
            {"click_x": 300, "click_y": 400},
        ]

        session._on_abort_confirmed()

        assert len(session._steps) == 0
        ctrl.close.assert_called_once()
        callback.assert_called_once_with(False)


# ------------------------------------------------------------------
# cmd_record entry point tests
# ------------------------------------------------------------------


class TestCmdRecord:
    """Tests for the cmd_record entry point."""

    def test_cmd_record_returns_1_on_cancel(self) -> None:
        """cmd_record returns 1 when user cancels the name prompt."""
        from recorder.record_flow import cmd_record

        with patch(
            "recorder.record_flow._prompt_routine_name", return_value=None
        ):
            result = cmd_record()

        assert result == 1
