"""Tests for CLI commands using Typer CliRunner.

Covers: CLI-01 through CLI-09 -- all subcommands, --speed wiring,
--json output, --param parsing, hub stub, loading screen skips for
read-only commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from cli.app import app
from cli.output import parse_params, resolve_routine_path

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_run_result(tmp_path: Path, name: str = "Name") -> MagicMock:
    """Create a mock RunResult."""
    return MagicMock(
        success=True, routine_name=name, run_id="r1",
        steps_completed=1, total_steps=1, duration_ms=100,
        failure_step=None, failure_reason=None, run_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# TUI / no-args
# ---------------------------------------------------------------------------


def test_no_args_invokes_tui() -> None:
    """ocsd with no args calls run_tui."""
    mock_run_tui = MagicMock()
    tui_module = MagicMock(run_tui=mock_run_tui)
    with patch.dict("sys.modules", {"cli.tui": tui_module}):
        result = runner.invoke(app, [])
    assert result.exit_code == 0
    mock_run_tui.assert_called_once()


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def test_record_command() -> None:
    """ocsd record 'Test' calls cmd_record with 'Test'."""
    mock_cmd = MagicMock(return_value=0)
    mock_loading = MagicMock()
    with patch.dict("sys.modules", {
        "cli.tui": MagicMock(show_loading_screen=mock_loading),
        "recorder.record_flow": MagicMock(cmd_record=mock_cmd),
    }):
        result = runner.invoke(app, ["record", "Test"])
    assert result.exit_code == 0
    mock_cmd.assert_called_once_with("Test")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def test_run_command_by_name(tmp_path: Path) -> None:
    """ocsd run 'MyRoutine' resolves path and calls run_routine."""
    routine_dir = tmp_path / "MyRoutine"
    routine_dir.mkdir()

    mock_result = _mock_run_result(routine_dir, "MyRoutine")
    mock_run = MagicMock(return_value=mock_result)
    mock_loading = MagicMock()

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("cli.app.collect_variables", return_value={}):
            with patch.dict("sys.modules", {
                "cli.tui": MagicMock(show_loading_screen=mock_loading),
                "routine.runner": MagicMock(run_routine=mock_run),
            }):
                with patch("core.config.get_config", return_value={"execution": {"human_delay": 1.0}}):
                    result = runner.invoke(app, ["run", "MyRoutine"])

    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_run_command_with_speed(tmp_path: Path) -> None:
    """ocsd run 'Name' --speed 2.0 sets human_delay to 0.5 during run."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()

    cfg = {"execution": {"human_delay": 1.0}}
    captured_delay: list[float] = []

    def capture_config_during_run(**kwargs):
        captured_delay.append(cfg["execution"]["human_delay"])
        return _mock_run_result(routine_dir)

    mock_loading = MagicMock()

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("cli.app.collect_variables", return_value={}):
            with patch("core.config.get_config", return_value=cfg):
                with patch.dict("sys.modules", {
                    "cli.tui": MagicMock(show_loading_screen=mock_loading),
                    "routine.runner": MagicMock(run_routine=capture_config_during_run),
                }):
                    result = runner.invoke(app, ["run", "Name", "--speed", "2.0"])

    assert result.exit_code == 0
    # During run, human_delay should have been 0.5 (1.0 / 2.0)
    assert captured_delay == [0.5]
    # After run, it should be restored to original
    assert cfg["execution"]["human_delay"] == 1.0


def test_run_command_with_json(tmp_path: Path) -> None:
    """ocsd run 'Name' --json outputs JSON."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()

    mock_result = _mock_run_result(routine_dir)
    mock_run = MagicMock(return_value=mock_result)
    mock_loading = MagicMock()

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("cli.app.collect_variables", return_value={}):
            with patch("core.config.get_config", return_value={"execution": {"human_delay": 1.0}}):
                with patch.dict("sys.modules", {
                    "cli.tui": MagicMock(show_loading_screen=mock_loading),
                    "routine.runner": MagicMock(run_routine=mock_run),
                }):
                    result = runner.invoke(app, ["run", "Name", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["success"] is True
    assert data["routine_name"] == "Name"


def test_run_command_with_param(tmp_path: Path) -> None:
    """ocsd run 'Name' --param search_term=news parses params correctly."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()

    mock_result = _mock_run_result(routine_dir)
    mock_run = MagicMock(return_value=mock_result)
    mock_loading = MagicMock()

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("cli.app.collect_variables", return_value={"search_term": "news"}):
            with patch("cli.app.prepare_run", return_value=routine_dir):
                with patch("core.config.get_config", return_value={"execution": {"human_delay": 1.0}}):
                    with patch.dict("sys.modules", {
                        "cli.tui": MagicMock(show_loading_screen=mock_loading),
                        "routine.runner": MagicMock(run_routine=mock_run),
                    }):
                        result = runner.invoke(app, ["run", "Name", "--param", "search_term=news"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_command() -> None:
    """ocsd list calls list_routines and outputs Rich table."""
    from routine.discovery import RoutineInfo

    routines = [
        RoutineInfo(name="routine1", path=Path("/fake/routine1"), schema_version="v1"),
    ]
    with patch("routine.discovery.list_routines", return_value=routines):
        result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "routine1" in result.output


def test_list_command_json() -> None:
    """ocsd list --json outputs JSON array."""
    from routine.discovery import RoutineInfo

    routines = [
        RoutineInfo(name="routine1", path=Path("/fake/routine1"), schema_version="v1"),
        RoutineInfo(name="routine2", path=Path("/fake/routine2"), schema_version="v0"),
    ]
    with patch("routine.discovery.list_routines", return_value=routines):
        result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 2
    assert data[0]["name"] == "routine1"
    assert data[1]["name"] == "routine2"


def test_list_command_no_loading_screen() -> None:
    """ocsd list does NOT call show_loading_screen."""
    from routine.discovery import RoutineInfo

    mock_loading = MagicMock()
    routines = [RoutineInfo(name="r1", path=Path("/r1"), schema_version="v1")]

    with patch("routine.discovery.list_routines", return_value=routines):
        with patch.dict("sys.modules", {
            "cli.tui": MagicMock(show_loading_screen=mock_loading),
        }):
            result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    mock_loading.assert_not_called()


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------


def test_inspect_command(tmp_path: Path) -> None:
    """ocsd inspect 'Name' calls inspect_routine."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()

    mock_inspect = MagicMock()
    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("routine.management.inspect_routine", mock_inspect):
            result = runner.invoke(app, ["inspect", "Name"])

    assert result.exit_code == 0
    mock_inspect.assert_called_once_with(routine_dir, as_json=False)


def test_inspect_command_no_loading_screen(tmp_path: Path) -> None:
    """ocsd inspect 'Name' does NOT call show_loading_screen."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()

    mock_loading = MagicMock()
    mock_inspect = MagicMock()

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("routine.management.inspect_routine", mock_inspect):
            with patch.dict("sys.modules", {
                "cli.tui": MagicMock(show_loading_screen=mock_loading),
            }):
                result = runner.invoke(app, ["inspect", "Name"])

    assert result.exit_code == 0
    mock_loading.assert_not_called()


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def test_update_command(tmp_path: Path) -> None:
    """ocsd update 'Name' resolves and dispatches."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()

    mock_routine = MagicMock(name="Name", steps=[{}, {}], version="1.0.0")
    mock_session = MagicMock()
    mock_loading = MagicMock()

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch.dict("sys.modules", {
            "cli.tui": MagicMock(show_loading_screen=mock_loading),
            "routine.format": MagicMock(Routine=MagicMock(load=MagicMock(return_value=mock_routine))),
            "routine.update_session": MagicMock(UpdateSession=mock_session),
        }):
            result = runner.invoke(app, ["update", "Name"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


def test_fork_command(tmp_path: Path) -> None:
    """ocsd fork 'Name' 'NewName' calls fork_routine."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()
    new_dir = tmp_path / "NewName"

    mock_fork = MagicMock(return_value=new_dir)
    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("routine.management.fork_routine", mock_fork):
            result = runner.invoke(app, ["fork", "Name", "NewName"])

    assert result.exit_code == 0
    mock_fork.assert_called_once_with(source_dir=routine_dir, new_name="NewName")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_command(tmp_path: Path) -> None:
    """ocsd delete 'Name' --yes calls delete_routine with skip_confirm=True."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()

    mock_delete = MagicMock(return_value=True)
    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("routine.management.delete_routine", mock_delete):
            result = runner.invoke(app, ["delete", "Name", "--yes"])

    assert result.exit_code == 0
    mock_delete.assert_called_once_with(routine_dir, skip_confirm=True)


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------


def test_hub_search_stub() -> None:
    """ocsd hub search 'query' prints V2 stub message."""
    result = runner.invoke(app, ["hub", "search", "my query"])
    assert result.exit_code == 0
    output_lower = result.output.lower()
    assert "v2" in output_lower or "coming soon" in output_lower


# ---------------------------------------------------------------------------
# Output helpers (direct unit tests)
# ---------------------------------------------------------------------------


def test_parse_params_valid() -> None:
    """parse_params(['a=b', 'c=d']) returns {'a': 'b', 'c': 'd'}."""
    result = parse_params(["a=b", "c=d"])
    assert result == {"a": "b", "c": "d"}


def test_parse_params_invalid() -> None:
    """parse_params(['invalid']) raises BadParameter."""
    with pytest.raises(typer.BadParameter):
        parse_params(["invalid"])


def test_resolve_routine_path_by_name(tmp_path: Path) -> None:
    """resolve_routine_path looks up name in ~/.ocsd/routines/."""
    routine_dir = tmp_path / "TestRoutine"
    routine_dir.mkdir()
    (routine_dir / "routine.json").write_text("{}")

    with patch("cli.output.get_routine_dir", return_value=tmp_path):
        result = resolve_routine_path("TestRoutine")

    assert result == routine_dir


def test_resolve_routine_path_not_found() -> None:
    """resolve_routine_path raises FileNotFoundError for missing routine."""
    with patch("cli.output.get_routine_dir", return_value=Path("/nonexistent/dir")):
        with pytest.raises(FileNotFoundError):
            resolve_routine_path("NoSuchRoutine")


# ---------------------------------------------------------------------------
# Variable collection tests
# ---------------------------------------------------------------------------


def _make_fixture_routine(
    routine_dir: Path,
    steps: list[dict],
) -> None:
    """Helper: create a minimal routine.json in routine_dir."""
    from routine.format import Routine

    routine = Routine(name="test_routine", steps=steps)
    routine.save(routine_dir)


def test_collect_variables_scans_steps(tmp_path: Path) -> None:
    """collect_variables prompts only for missing variables."""
    from cli.output import collect_variables

    routine_dir = tmp_path / "VarRoutine"
    _make_fixture_routine(routine_dir, [
        {
            "step_index": 0,
            "node_id": "n1",
            "action": "type",
            "label": "search",
            "input_spec": {"type": "variable", "value": "{search_term}", "hint": "Search query"},
        },
        {
            "step_index": 1,
            "node_id": "n2",
            "action": "type",
            "label": "user",
            "input_spec": {"type": "variable", "value": "{username}", "hint": "Your username"},
        },
        {
            "step_index": 2,
            "node_id": "n3",
            "action": "click",
            "label": "submit",
        },
    ])

    with patch("cli.output.Prompt.ask", return_value="john") as mock_ask:
        result = collect_variables(routine_dir, {"search_term": "test"})

    assert result == {"search_term": "test", "username": "john"}
    # Should only prompt for username, not search_term
    mock_ask.assert_called_once()
    assert "username" in mock_ask.call_args[0][0].lower() or "Your username" in mock_ask.call_args[0][0]


def test_collect_variables_all_provided(tmp_path: Path) -> None:
    """collect_variables does NOT prompt when all params provided."""
    from cli.output import collect_variables

    routine_dir = tmp_path / "VarRoutine2"
    _make_fixture_routine(routine_dir, [
        {
            "step_index": 0,
            "node_id": "n1",
            "action": "type",
            "label": "search",
            "input_spec": {"type": "variable", "value": "{search_term}"},
        },
        {
            "step_index": 1,
            "node_id": "n2",
            "action": "type",
            "label": "user",
            "input_spec": {"type": "variable", "value": "{username}"},
        },
    ])

    with patch("cli.output.Prompt.ask") as mock_ask:
        result = collect_variables(
            routine_dir,
            {"search_term": "test", "username": "john"},
        )

    mock_ask.assert_not_called()
    assert result == {"search_term": "test", "username": "john"}


def test_prepare_run_injects_values(tmp_path: Path) -> None:
    """prepare_run creates temp dir with text_to_type populated."""
    import shutil

    from cli.output import prepare_run
    from routine.format import Routine

    routine_dir = tmp_path / "InjectRoutine"
    _make_fixture_routine(routine_dir, [
        {
            "step_index": 0,
            "node_id": "n1",
            "action": "type",
            "label": "search box",
            "input_spec": {"type": "variable", "value": "{search_term}"},
            "text_to_type": "",
        },
    ])

    temp_dir = prepare_run(routine_dir, {"search_term": "hello"})
    try:
        loaded = Routine.load(temp_dir)
        assert loaded.steps[0]["text_to_type"] == "hello"
    finally:
        shutil.rmtree(temp_dir.parent, ignore_errors=True)


def test_run_command_with_variables(tmp_path: Path) -> None:
    """CLI run command with --param collects and injects correctly."""
    routine_dir = tmp_path / "Name"
    routine_dir.mkdir()

    mock_result = _mock_run_result(routine_dir)
    mock_run = MagicMock(return_value=mock_result)
    mock_loading = MagicMock()
    mock_collect = MagicMock(return_value={"search_term": "news"})
    mock_prepare = MagicMock(return_value=routine_dir)

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("cli.app.collect_variables", mock_collect):
            with patch("cli.app.prepare_run", mock_prepare):
                with patch("core.config.get_config", return_value={"execution": {"human_delay": 1.0}}):
                    with patch.dict("sys.modules", {
                        "cli.tui": MagicMock(show_loading_screen=mock_loading),
                        "routine.runner": MagicMock(run_routine=mock_run),
                    }):
                        result = runner.invoke(app, ["run", "Name", "--param", "search_term=news"])

    assert result.exit_code == 0
    mock_collect.assert_called_once()
    mock_prepare.assert_called_once()
    mock_run.assert_called_once()


def test_run_command_wires_overlay_callback(tmp_path: Path) -> None:
    """Verify run_command passes ReplayOverlayAdapter as callback to run_routine."""
    routine_dir = tmp_path / "OverlayRoutine"
    routine_dir.mkdir()

    mock_result = _mock_run_result(routine_dir, "OverlayRoutine")
    captured_kwargs: dict = {}

    def _capture_run(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_result

    mock_loading = MagicMock()

    # Mock QApplication to avoid needing a real display
    mock_qapp_instance = MagicMock()
    mock_qapp_instance.exec = MagicMock(return_value=0)
    mock_qapp_instance.quit = MagicMock()

    mock_qapp_cls = MagicMock()
    mock_qapp_cls.instance = MagicMock(return_value=mock_qapp_instance)

    mock_controller = MagicMock()
    mock_adapter = MagicMock()

    # We need to intercept the thread to run synchronously
    def _fake_thread_start(self_thread):
        self_thread._target()

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("cli.app.collect_variables", return_value={}):
            with patch("core.config.get_config", return_value={"execution": {"human_delay": 1.0}}):
                with patch.dict("sys.modules", {
                    "cli.tui": MagicMock(show_loading_screen=mock_loading),
                    "routine.runner": MagicMock(run_routine=_capture_run),
                    "PyQt6.QtWidgets": MagicMock(QApplication=mock_qapp_cls),
                    "recorder.overlay.controller": MagicMock(OverlayController=MagicMock(return_value=mock_controller)),
                    "routine.replay_overlay": MagicMock(ReplayOverlayAdapter=MagicMock(return_value=mock_adapter)),
                }):
                    with patch("threading.Thread") as mock_thread_cls:
                        mock_thread_obj = MagicMock()
                        mock_thread_cls.return_value = mock_thread_obj
                        # When start() is called, run the target synchronously
                        def _run_target():
                            target = mock_thread_cls.call_args[1].get("target") or mock_thread_cls.call_args[0][0]
                            target()
                        mock_thread_obj.start = _run_target
                        result = runner.invoke(app, ["run", "OverlayRoutine"])

    assert result.exit_code == 0
    assert captured_kwargs.get("callback") is not None, "callback= must be passed to run_routine"


def test_run_command_still_outputs_result_with_overlay(tmp_path: Path) -> None:
    """Verify run_command still shows result panel after overlay run."""
    routine_dir = tmp_path / "OutputRoutine"
    routine_dir.mkdir()

    mock_result = _mock_run_result(routine_dir, "OutputRoutine")
    mock_loading = MagicMock()

    mock_qapp_instance = MagicMock()
    mock_qapp_instance.exec = MagicMock(return_value=0)
    mock_qapp_instance.quit = MagicMock()
    mock_qapp_cls = MagicMock()
    mock_qapp_cls.instance = MagicMock(return_value=mock_qapp_instance)

    with patch("cli.app.resolve_routine_path", return_value=routine_dir):
        with patch("cli.app.collect_variables", return_value={}):
            with patch("core.config.get_config", return_value={"execution": {"human_delay": 1.0}}):
                with patch.dict("sys.modules", {
                    "cli.tui": MagicMock(show_loading_screen=mock_loading),
                    "routine.runner": MagicMock(run_routine=MagicMock(return_value=mock_result)),
                    "PyQt6.QtWidgets": MagicMock(QApplication=mock_qapp_cls),
                    "recorder.overlay.controller": MagicMock(OverlayController=MagicMock()),
                    "routine.replay_overlay": MagicMock(ReplayOverlayAdapter=MagicMock()),
                }):
                    with patch("threading.Thread") as mock_thread_cls:
                        mock_thread_obj = MagicMock()
                        mock_thread_cls.return_value = mock_thread_obj
                        def _run_target():
                            target = mock_thread_cls.call_args[1].get("target") or mock_thread_cls.call_args[0][0]
                            target()
                        mock_thread_obj.start = _run_target
                        result = runner.invoke(app, ["run", "OutputRoutine"])

    assert result.exit_code == 0
    assert "Run Result" in result.output


def test_main_redirect() -> None:
    """main.main is callable and redirects to cli.app.main."""
    from main import main as main_func

    assert callable(main_func)

    with patch("cli.app.main") as mock_cli_main:
        main_func()

    mock_cli_main.assert_called_once()
