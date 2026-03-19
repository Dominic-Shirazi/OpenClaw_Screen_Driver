"""Tests for the shared condition-checking engine and select_all_extract.

Covers: ConditionChecker (fixed_timer, screen_change, n_iterations,
element_appears, timeout safety) and select_all_extract clipboard + VLM fallback.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ConditionChecker tests
# ---------------------------------------------------------------------------

class TestFixedTimer:
    """fixed_timer condition sleeps for the specified duration then resolves."""

    def test_fixed_timer(self) -> None:
        from core.conditions import ConditionChecker

        checker = ConditionChecker(
            condition_type="fixed_timer",
            params={"seconds": 0.1},
            timeout=5.0,
        )
        t0 = time.monotonic()
        result = checker.poll_until()
        elapsed = time.monotonic() - t0

        assert result.met is True
        assert result.timed_out is False
        assert elapsed >= 0.08  # allow small timing slack
        assert elapsed < 2.0   # should not take anywhere near timeout


class TestScreenChange:
    """screen_change condition detects pixel differences between screenshots."""

    def test_screen_change(self) -> None:
        """Successive different screenshots -> met=True."""
        from core.conditions import ConditionChecker

        # First call returns baseline (black), second returns different (white)
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        call_count = 0

        def mock_screenshot() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            # First call is baseline capture, second is the poll check
            return black if call_count <= 1 else white

        with patch("core.conditions.ConditionChecker._take_screenshot", side_effect=mock_screenshot):
            checker = ConditionChecker(
                condition_type="screen_change",
                params={"threshold": 0.05},
                timeout=5.0,
                poll_interval=0.01,
            )
            result = checker.poll_until()

        assert result.met is True
        assert result.timed_out is False

    def test_screen_change_timeout(self) -> None:
        """Identical screenshots every time -> timed_out=True."""
        from core.conditions import ConditionChecker

        static = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("core.conditions.ConditionChecker._take_screenshot", return_value=static):
            checker = ConditionChecker(
                condition_type="screen_change",
                params={"threshold": 0.05},
                timeout=0.15,
                poll_interval=0.02,
            )
            result = checker.poll_until()

        assert result.met is False
        assert result.timed_out is True


class TestNIterations:
    """n_iterations condition resolves after exactly N polls."""

    def test_n_iterations(self) -> None:
        from core.conditions import ConditionChecker

        checker = ConditionChecker(
            condition_type="n_iterations",
            params={"count": 3},
            timeout=5.0,
            poll_interval=0.01,
        )
        result = checker.poll_until()

        assert result.met is True
        assert result.iterations == 3


class TestElementAppearsTimeout:
    """element_appears with no match -> timed_out=True."""

    def test_element_appears_timeout(self) -> None:
        from core.conditions import ConditionChecker

        checker = ConditionChecker(
            condition_type="element_appears",
            params={"description": "Submit button"},
            timeout=0.15,
            poll_interval=0.02,
        )
        result = checker.poll_until()

        assert result.met is False
        assert result.timed_out is True


class TestTimeoutSafety:
    """Any condition without a positive timeout raises ValueError (except n_iterations)."""

    def test_timeout_safety(self) -> None:
        from core.conditions import ConditionChecker

        with pytest.raises(ValueError, match="timeout must be positive"):
            ConditionChecker(
                condition_type="screen_change",
                params={},
                timeout=0,
            )

    def test_n_iterations_allows_zero_timeout(self) -> None:
        """n_iterations is exempt from positive timeout requirement."""
        from core.conditions import ConditionChecker

        # Should not raise -- n_iterations doesn't need timeout
        checker = ConditionChecker(
            condition_type="n_iterations",
            params={"count": 1},
            timeout=0,
            poll_interval=0.01,
        )
        result = checker.poll_until()
        assert result.met is True


# ---------------------------------------------------------------------------
# select_all_extract tests
# ---------------------------------------------------------------------------

class TestSelectAllExtract:
    """select_all_extract reads clipboard after Ctrl+A, Ctrl+C with VLM fallback."""

    @patch("core.executor.hotkey")
    @patch("core.executor._hsleep")
    def test_select_all_extract_clipboard(self, mock_sleep: MagicMock, mock_hotkey: MagicMock) -> None:
        """Clipboard changes after hotkey -> returns new clipboard content."""
        from core.executor import select_all_extract

        call_count = 0

        def mock_paste() -> str:
            nonlocal call_count
            call_count += 1
            return "old text" if call_count <= 1 else "new extracted text"

        with patch("pyperclip.paste", side_effect=mock_paste):
            result = select_all_extract(dry_run=False)

        assert result == "new extracted text"
        # Verify hotkeys were called for Ctrl+A and Ctrl+C
        assert mock_hotkey.call_count == 2

    @patch("core.executor.hotkey")
    @patch("core.executor._hsleep")
    def test_select_all_extract_vlm_fallback(self, mock_sleep: MagicMock, mock_hotkey: MagicMock) -> None:
        """Clipboard unchanged -> falls back to VLM screenshot analysis."""
        from core.executor import select_all_extract

        mock_screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        vlm_result = {"text": "VLM extracted text", "element_type": "unknown"}

        with (
            patch("pyperclip.paste", return_value="same text"),
            patch("core.capture.screenshot_full", return_value=mock_screenshot),
            patch("core.vision.analyze_crop_array", return_value=vlm_result),
        ):
            result = select_all_extract(dry_run=False)

        assert result == "VLM extracted text"

    def test_select_all_extract_dry_run(self) -> None:
        """dry_run=True returns empty string without executing."""
        from core.executor import select_all_extract

        result = select_all_extract(dry_run=True)
        assert result == ""


# ---------------------------------------------------------------------------
# Mini-dialog signal/attribute tests
# ---------------------------------------------------------------------------

class TestWaitDialogSignals:
    """WaitDialog has confirmed and dismissed signal attributes."""

    def test_wait_dialog_signals(self) -> None:
        from recorder.overlay.mini_dialogs import WaitDialog

        assert hasattr(WaitDialog, "confirmed")
        assert hasattr(WaitDialog, "dismissed")

    def test_wait_dialog_condition_types(self) -> None:
        """WaitDialog maps combo text to correct condition_type strings."""
        from recorder.overlay.mini_dialogs import WaitDialog

        expected = {
            "Fixed Timer": "fixed_timer",
            "Element Appears": "element_appears",
            "Screen Change": "screen_change",
            "VLM Check": "vlm_check",
        }
        assert WaitDialog.CONDITION_MAP == expected


class TestPromptDialogSignals:
    """PromptDialog has confirmed and dismissed signal attributes."""

    def test_prompt_dialog_signals(self) -> None:
        from recorder.overlay.mini_dialogs import PromptDialog

        assert hasattr(PromptDialog, "confirmed")
        assert hasattr(PromptDialog, "dismissed")


class TestLoopDialogSignals:
    """LoopDialog has confirmed and dismissed signal attributes."""

    def test_loop_dialog_signals(self) -> None:
        from recorder.overlay.mini_dialogs import LoopDialog

        assert hasattr(LoopDialog, "confirmed")
        assert hasattr(LoopDialog, "dismissed")

    def test_loop_dialog_condition_types(self) -> None:
        """LoopDialog maps combo text to correct condition_type strings."""
        from recorder.overlay.mini_dialogs import LoopDialog

        expected = {
            "N Iterations": "n_iterations",
            "Element Appears": "element_appears",
            "Text Matches": "text_matches",
            "Prompt User": "prompt_user",
        }
        assert LoopDialog.CONDITION_MAP == expected


class TestLoopDefinition:
    """Loop definition dict structure validation."""

    def test_loop_definition(self) -> None:
        """Loop definition with body_step_indices and exit_condition is valid."""
        loop_def = {
            "body_step_indices": [0, 1, 2],
            "exit_condition": {"type": "n_iterations", "count": 5},
        }
        assert loop_def["body_step_indices"] == [0, 1, 2]
        assert loop_def["exit_condition"]["type"] == "n_iterations"
        assert loop_def["exit_condition"]["count"] == 5


# ---------------------------------------------------------------------------
# Wait step format test
# ---------------------------------------------------------------------------

class TestWaitStepFormat:
    """build_v1_step correctly outputs wait step with wait_definition."""

    def test_wait_step_format(self) -> None:
        from routine.format import build_v1_step

        step = {
            "tag_data": {
                "action": "wait",
                "element_type": "unknown",
                "label": "Wait: fixed_timer",
                "caption": "Timeout: 10s",
                "confidence": 1.0,
            },
            "wait_definition": {
                "condition_type": "fixed_timer",
                "timeout": 10.0,
                "params": {"seconds": 10.0},
                "poll_interval": 2.0,
            },
            "bbox": (0, 0, 0, 0),
        }
        result = build_v1_step(step, 0, "node_wait", 1920, 1080)
        assert result["action"] == "wait"
        assert "wait" in result
        assert result["wait"]["condition_type"] == "fixed_timer"
        assert result["wait"]["timeout"] == 10.0


# ---------------------------------------------------------------------------
# prompt_user_blocking tests
# ---------------------------------------------------------------------------

class TestPromptUserBlocking:
    """prompt_user_blocking blocks and resumes on respond_to_prompt."""

    def test_prompt_user_blocking_dry_run(self) -> None:
        """dry_run=True returns empty string without blocking."""
        from core.executor import prompt_user_blocking

        result = prompt_user_blocking("test question", dry_run=True)
        assert result == ""

    def test_respond_to_prompt(self) -> None:
        """Background thread blocks, main thread responds, thread returns answer."""
        import threading
        from core.executor import prompt_user_blocking, respond_to_prompt

        results: list[str] = []

        def bg_worker() -> None:
            answer = prompt_user_blocking("Should I continue?")
            results.append(answer)

        t = threading.Thread(target=bg_worker, daemon=True)
        t.start()

        # Give background thread time to start blocking
        time.sleep(0.1)

        respond_to_prompt("yes")
        t.join(timeout=2.0)

        assert len(results) == 1
        assert results[0] == "yes"
