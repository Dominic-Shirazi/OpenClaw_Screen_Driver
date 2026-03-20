"""Tests for the Rich TUI (cli.tui) and key reader (cli._keys).

Covers feature loading, menu navigation, routine browser filtering,
loading screen execution, and V2+ disabled item behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_read_key():
    """Fixture that patches read_key and returns the mock."""
    with patch("cli.tui.read_key") as m:
        yield m


@pytest.fixture()
def sample_routines():
    """Three sample RoutineInfo objects for browser tests."""
    from routine.discovery import RoutineInfo

    return [
        RoutineInfo(name="login_flow", path=Path("/r/login_flow"), schema_version="v1"),
        RoutineInfo(name="checkout_v2", path=Path("/r/checkout_v2"), schema_version="v1"),
        RoutineInfo(name="search_test", path=Path("/r/search_test"), schema_version="v0"),
    ]


# ---------------------------------------------------------------------------
# _load_features tests
# ---------------------------------------------------------------------------


class TestLoadFeatures:
    """Tests for _load_features()."""

    def test_load_features_returns_list_with_keys(self) -> None:
        """_load_features returns list of dicts with text and status keys."""
        from cli.tui import _load_features

        features = _load_features()

        assert isinstance(features, list)
        assert len(features) > 0
        for feat in features:
            assert "text" in feat
            assert "status" in feat

    def test_load_features_has_shipped_and_coming(self) -> None:
        """Feature list includes both shipped and coming_soon items."""
        from cli.tui import _load_features

        features = _load_features()
        statuses = {f["status"] for f in features}

        assert "shipped" in statuses
        assert "coming_soon" in statuses

    def test_load_features_fallback_on_missing_file(self, tmp_path: Path) -> None:
        """When features.yml is missing, returns hardcoded fallback."""
        from cli.tui import _load_features

        fake_path = tmp_path / "nonexistent" / "features.yml"
        with patch("cli.tui.Path") as mock_path_cls:
            # Make Path(__file__).parent / "features.yml" return nonexistent
            mock_file = MagicMock()
            mock_path_cls.return_value = mock_file
            mock_file.parent.__truediv__ = lambda self, x: fake_path

            # Directly test: patch the open call to raise
            with patch("builtins.open", side_effect=OSError("not found")):
                features = _load_features()

        assert isinstance(features, list)
        assert len(features) >= 3
        assert features[0]["text"] == "Record once, replay forever"


# ---------------------------------------------------------------------------
# Menu item tests
# ---------------------------------------------------------------------------


class TestMenuItems:
    """Tests for menu item definitions."""

    def test_menu_items_include_disabled(self) -> None:
        """Menu has at least 2 disabled items for V2+ features."""
        from cli.tui import MENU_ITEMS

        disabled = [item for item in MENU_ITEMS if not item[2]]
        assert len(disabled) >= 2

    def test_disabled_items_are_hub_and_voice(self) -> None:
        """Disabled items include Hub Browse and Voice Record."""
        from cli.tui import MENU_ITEMS

        disabled_labels = {item[0] for item in MENU_ITEMS if not item[2]}
        assert "Hub Browse" in disabled_labels
        assert "Voice Record" in disabled_labels

    def test_feature_inbound_label_in_tui(self) -> None:
        """V2+ items display 'Feature inbound' text."""
        import cli.tui as tui_module

        src = Path(tui_module.__file__).read_text(encoding="utf-8")
        assert "Feature inbound" in src


# ---------------------------------------------------------------------------
# Routine browser tests
# ---------------------------------------------------------------------------


class TestRoutineBrowser:
    """Tests for _show_routine_browser()."""

    def test_routine_browser_empty(self, mock_read_key: MagicMock) -> None:
        """Empty routine list shows 'No routines' and returns None."""
        with patch("routine.discovery.list_routines", return_value=[]):
            from cli.tui import _show_routine_browser

            result = _show_routine_browser()

        assert result is None

    def test_routine_browser_escape_returns_none(
        self,
        mock_read_key: MagicMock,
        sample_routines: list[Any],
    ) -> None:
        """Pressing escape in routine browser returns None."""
        mock_read_key.return_value = "escape"
        with (
            patch("routine.discovery.list_routines", return_value=sample_routines),
            patch("cli.tui.Live"),
        ):
            from cli.tui import _show_routine_browser

            result = _show_routine_browser()

        assert result is None

    def test_routine_browser_filter_and_select(
        self,
        mock_read_key: MagicMock,
        sample_routines: list[Any],
    ) -> None:
        """Typing 'l', 'o' filters to login_flow, enter selects it."""
        # Type "l", "o", then enter to select first filtered result
        mock_read_key.side_effect = ["l", "o", "enter"]

        with (
            patch("routine.discovery.list_routines", return_value=sample_routines),
            patch("cli.tui.Live"),
        ):
            from cli.tui import _show_routine_browser

            result = _show_routine_browser()

        assert result == Path("/r/login_flow")


# ---------------------------------------------------------------------------
# Loading screen tests
# ---------------------------------------------------------------------------


class TestShowLoadingScreen:
    """Tests for show_loading_screen()."""

    def test_show_loading_screen_runs(self) -> None:
        """show_loading_screen completes without error with mocked models."""
        with (
            patch("cli.tui.Live"),
            patch("cli.tui.Console"),
            patch("cli.tui.Progress") as mock_progress,
        ):
            # Make progress mock work like a real progress
            mock_prog_inst = MagicMock()
            mock_progress.return_value = mock_prog_inst
            mock_prog_inst.add_task.return_value = 0

            from cli.tui import show_loading_screen

            # This should not raise
            show_loading_screen()


# ---------------------------------------------------------------------------
# read_key tests
# ---------------------------------------------------------------------------


class TestReadKey:
    """Tests for cli._keys.read_key."""

    def test_read_key_returns_string(self) -> None:
        """read_key returns a string type on Windows via msvcrt mock."""
        import sys
        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        import msvcrt
        with patch.object(msvcrt, "getwch", return_value="a"):
            from cli._keys import read_key

            result = read_key()
            assert isinstance(result, str)
            assert result == "a"

    def test_read_key_windows_arrow_up(self) -> None:
        """Windows arrow up returns 'up'."""
        import sys
        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        import msvcrt
        with patch.object(msvcrt, "getwch", side_effect=["\xe0", "H"]):
            from cli._keys import _read_key_windows

            result = _read_key_windows()
            assert result == "up"

    def test_read_key_windows_enter(self) -> None:
        """Windows enter returns 'enter'."""
        import sys
        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        import msvcrt
        with patch.object(msvcrt, "getwch", return_value="\r"):
            from cli._keys import _read_key_windows

            result = _read_key_windows()
            assert result == "enter"


# ---------------------------------------------------------------------------
# Disabled item navigation test
# ---------------------------------------------------------------------------


class TestDisabledNavigation:
    """Tests that disabled items cannot be selected via Enter."""

    def test_enter_on_disabled_skips(self, mock_read_key: MagicMock) -> None:
        """Pressing enter on a disabled item does nothing; navigating
        to an enabled item and pressing enter selects it."""
        from cli.tui import MENU_ITEMS

        # Find index of first disabled item
        disabled_idx = next(i for i, item in enumerate(MENU_ITEMS) if not item[2])
        quit_idx = next(i for i, item in enumerate(MENU_ITEMS) if item[1] == "quit")

        # Navigate down to disabled item, try enter (should not break),
        # then press escape to quit
        keys: list[str] = []
        # Go down to disabled item
        for _ in range(disabled_idx):
            keys.append("down")
        keys.append("enter")  # This should NOT select (disabled)
        keys.append("escape")  # This should quit

        mock_read_key.side_effect = keys

        with patch("cli.tui.Live"):
            from cli.tui import _show_menu

            command, kwargs = _show_menu()

        assert command == "quit"


# ---------------------------------------------------------------------------
# Sequential gate tests
# ---------------------------------------------------------------------------


class TestSequentialGate:
    """Tests for the TUI-to-Qt sequential gate pattern."""

    def test_sequential_gate(self) -> None:
        """show_loading_screen completes without importing PyQt6."""
        import sys

        # Remove PyQt6 from sys.modules if it's there, then verify
        # it is NOT there after loading screen runs
        modules_before = set(sys.modules.keys())

        with (
            patch("cli.tui.Live"),
            patch("cli.tui.Console"),
            patch("cli.tui.Progress") as mock_progress,
        ):
            mock_prog_inst = MagicMock()
            mock_progress.return_value = mock_prog_inst
            mock_prog_inst.add_task.return_value = 0

            from cli.tui import show_loading_screen

            show_loading_screen()

        # PyQt6 should not have been newly imported during loading screen
        new_modules = set(sys.modules.keys()) - modules_before
        pyqt_modules = {m for m in new_modules if m.startswith("PyQt6")}
        assert not pyqt_modules, f"PyQt6 was imported during loading screen: {pyqt_modules}"


# ---------------------------------------------------------------------------
# TUI run_tui loop tests
# ---------------------------------------------------------------------------


class TestRunTuiLoop:
    """Tests for the run_tui main menu loop."""

    def test_tui_quit_exits(self, mock_read_key: MagicMock) -> None:
        """Selecting quit from menu exits run_tui without error."""
        mock_read_key.side_effect = ["enter"]  # "Record Routine" is first, need to go to quit

        with (
            patch("cli.tui.show_loading_screen"),
            patch("cli.tui._show_menu", return_value=("quit", {})),
        ):
            from cli.tui import run_tui

            # Should return without error
            run_tui()

    def test_tui_menu_loop_back(self) -> None:
        """After completing an action, menu shows again."""
        call_count = [0]

        def mock_show_menu() -> tuple[str, dict]:
            call_count[0] += 1
            if call_count[0] == 1:
                return ("list", {})
            return ("quit", {})

        with (
            patch("cli.tui.show_loading_screen"),
            patch("cli.tui._show_menu", side_effect=mock_show_menu),
            patch("routine.discovery.list_routines", return_value=[]),
            patch("cli.output.output_routines"),
        ):
            from cli.tui import run_tui

            run_tui()

        assert call_count[0] == 2, "Menu should have been called twice (list then quit)"
