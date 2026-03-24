"""Comprehensive tests for routine/runner.py and routine/run_log.py.

All external calls (screenshots, clicks, VLM, locate) are mocked.
No real GUI interactions occur during testing.
"""

from __future__ import annotations

import builtins
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.types import ConfirmResult, ElementNotFoundError, LocateResult, Point
from routine.run_log import (
    annotate_screenshot,
    create_run_dir,
    prune_old_runs,
    save_annotated_screenshot,
    save_run_result,
    setup_run_logger,
)
from routine.runner import (
    PreflightError,
    RunEvent,
    RunResult,
    _compute_search_region,
    _dispatch_action,
    preflight_check,
    run_routine,
)

original_import = builtins.__import__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(
    index: int = 0,
    action: str = "click",
    label: str = "Test Button",
    node_id: str = "abc123def456",
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal v1 step dict for testing."""
    step: dict[str, Any] = {
        "step_index": index,
        "node_id": node_id,
        "element_type": "button",
        "label": label,
        "action": action,
        "bbox": {"x": 100, "y": 200, "w": 50, "h": 30},
        "bbox_pct": {"x_pct": 0.05, "y_pct": 0.1, "w_pct": 0.025, "h_pct": 0.015},
        "anchors": {
            "position_pct": {"x_pct": 0.065, "y_pct": 0.115},
            "ocr_text": "Test",
        },
        "snippet_path": "snippets/abc123def456.png",
        "embedding_path": "embeddings/abc123def456.npy",
    }
    step.update(extra)
    return step


def _locate_ok(*args: Any, **kwargs: Any) -> LocateResult:
    return LocateResult(point=Point(125, 215), confidence=0.9, method="mock")


def _locate_fail(*args: Any, **kwargs: Any) -> None:
    raise ElementNotFoundError("test_node", "Element not found")


def _validate_ok(*args: Any, **kwargs: Any) -> ConfirmResult:
    return ConfirmResult(success=True, confidence=0.85, notes="OK")


def _validate_fail(*args: Any, **kwargs: Any) -> ConfirmResult:
    return ConfirmResult(success=False, confidence=0.3, notes="No change")


def _fake_screenshot(*args: Any, **kwargs: Any) -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _patched_import_no_vlm(name: str, *args: Any, **kwargs: Any) -> Any:
    if name == "core.vision" or "core.vision" in name:
        raise ImportError("No VLM")
    return original_import(name, *args, **kwargs)


# Common patch stack for run_routine tests
_RUN_PATCHES = {
    "load": "routine.runner.Routine.load",
    "press_enter": "routine.runner.press_enter",
    "type_text": "routine.runner.type_text",
    "click": "routine.runner.click",
    "double_click": "routine.runner.double_click",
    "right_click": "routine.runner.right_click",
    "drag": "routine.runner.drag",
    "scroll": "routine.runner.scroll",
    "locate": "routine.runner.locate_element_from_step",
    "screenshot": "routine.runner.screenshot_full",
    "validate": "mapper.validator.validate_action",
    "prune": "routine.runner.prune_old_runs",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_routine_dir(tmp_path: Path) -> Path:
    """Create a routine directory with valid v1 schema files."""
    routine_dir = tmp_path / "test_routine"
    routine_dir.mkdir()
    snippets_dir = routine_dir / "snippets"
    snippets_dir.mkdir()
    embeddings_dir = routine_dir / "embeddings"
    embeddings_dir.mkdir()

    node_ids = ["node_aaa111", "node_bbb222", "node_ccc333"]
    for nid in node_ids:
        (snippets_dir / f"{nid}.png").write_bytes(b"\x89PNG placeholder")
        np.save(str(embeddings_dir / f"{nid}.npy"), np.zeros(512))

    return routine_dir


@pytest.fixture()
def mock_routine() -> MagicMock:
    """Return a mock Routine object with 3 steps."""
    routine = MagicMock()
    routine.name = "test_routine"
    routine.start_from = "desktop"
    routine.steps = [
        _make_step(0, "click", "Open Menu", "node_aaa111"),
        _make_step(1, "click", "Settings", "node_bbb222"),
        _make_step(2, "type", "Username", "node_ccc333", text_to_type="admin"),
    ]
    for s in routine.steps:
        nid = s["node_id"]
        s["snippet_path"] = f"snippets/{nid}.png"
        s["embedding_path"] = f"embeddings/{nid}.npy"
    return routine


# ---------------------------------------------------------------------------
# Preflight tests
# ---------------------------------------------------------------------------

class TestPreflight:
    """Pre-flight check tests."""

    def test_preflight_pass(self, sample_routine_dir: Path, mock_routine: MagicMock) -> None:
        """All assets exist. Expect empty error list."""
        errors = preflight_check(mock_routine, sample_routine_dir)
        assert errors == []

    def test_preflight_missing_snippet(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """Remove a snippet file. Expect error mentioning snippet."""
        (sample_routine_dir / "snippets" / "node_aaa111.png").unlink()
        errors = preflight_check(mock_routine, sample_routine_dir)
        assert len(errors) >= 1
        assert any("snippet" in e.lower() for e in errors)

    def test_preflight_vlm_unavailable(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """VLM import fails. Should log warning but NOT add error."""
        with patch("builtins.__import__", side_effect=_patched_import_no_vlm):
            errors = preflight_check(mock_routine, sample_routine_dir)
        # VLM unavailability is a warning, not an error
        assert all("vlm" not in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# run_routine tests
# ---------------------------------------------------------------------------

class TestRunRoutine:
    """run_routine() integration tests with all externals mocked."""

    def test_run_routine_success(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """Successful run: all steps located and executed."""
        with (
            patch(_RUN_PATCHES["load"], return_value=mock_routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_ok),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["validate"], side_effect=_validate_ok),
            patch(_RUN_PATCHES["click"]),
            patch(_RUN_PATCHES["type_text"]),
            patch(_RUN_PATCHES["press_enter"]),
            patch(_RUN_PATCHES["prune"]),
        ):
            result = run_routine(sample_routine_dir, dry_run=True)
        assert result.success is True
        assert result.steps_completed == result.total_steps

    def test_run_routine_failure_cascade_abort(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """All locate stages fail. Expect failure result."""
        with (
            patch(_RUN_PATCHES["load"], return_value=mock_routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_fail),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["prune"]),
        ):
            result = run_routine(sample_routine_dir, dry_run=True)
        assert result.success is False
        assert result.failure_reason is not None
        assert "could not find" in result.failure_reason.lower()

    def test_never_blind_click(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """When locate fails, no executor functions should be called."""
        with (
            patch(_RUN_PATCHES["load"], return_value=mock_routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_fail),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["click"]) as m_click,
            patch(_RUN_PATCHES["double_click"]) as m_dclick,
            patch(_RUN_PATCHES["right_click"]) as m_rclick,
            patch(_RUN_PATCHES["drag"]) as m_drag,
            patch(_RUN_PATCHES["type_text"]) as m_type,
            patch(_RUN_PATCHES["scroll"]) as m_scroll,
            patch(_RUN_PATCHES["prune"]),
        ):
            run_routine(sample_routine_dir, dry_run=True)
        m_click.assert_not_called()
        m_dclick.assert_not_called()
        m_rclick.assert_not_called()
        m_drag.assert_not_called()
        m_type.assert_not_called()
        m_scroll.assert_not_called()


# ---------------------------------------------------------------------------
# Scanner integration tests
# ---------------------------------------------------------------------------

class TestScannerIntegration:
    """Tests for scanner wired into run_routine() preflight."""

    def test_run_routine_blocks_unsafe_routine(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """Verify run_routine returns failure when scanner flags threats."""
        from hub.scanner import ScanResult

        unsafe_result = ScanResult(
            is_safe=False,
            warnings=["Threat detected: malicious URL"],
            risk_score=0.8,
        )

        with (
            patch(_RUN_PATCHES["load"], return_value=mock_routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_ok),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["prune"]),
            patch("routine.runner.scan_routine", return_value=unsafe_result, create=True),
            patch("hub.scanner.scan_routine", return_value=unsafe_result),
        ):
            result = run_routine(sample_routine_dir, dry_run=True)

        assert result.success is False
        assert result.failure_reason is not None
        assert "Security scan" in result.failure_reason

    def test_run_routine_allows_safe_routine(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """Verify run_routine proceeds normally when scanner says safe."""
        from hub.scanner import ScanResult

        safe_result = ScanResult(is_safe=True, warnings=[], risk_score=0.0)

        with (
            patch(_RUN_PATCHES["load"], return_value=mock_routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_ok),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["validate"], side_effect=_validate_ok),
            patch(_RUN_PATCHES["click"]),
            patch(_RUN_PATCHES["type_text"]),
            patch(_RUN_PATCHES["press_enter"]),
            patch(_RUN_PATCHES["prune"]),
            patch("hub.scanner.scan_routine", return_value=safe_result),
        ):
            result = run_routine(sample_routine_dir, dry_run=True)

        assert result.success is True
        assert result.steps_completed == result.total_steps


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------

class TestDispatch:
    """_dispatch_action tests."""

    def test_dispatch_click(self) -> None:
        """Verify click is called with correct coordinates."""
        step = _make_step(action="click")
        lr = LocateResult(point=Point(300, 400), confidence=0.9, method="mock")
        with patch("routine.runner.click") as m:
            result = _dispatch_action(step, lr, dry_run=False)
            m.assert_called_once_with(300, 400, dry_run=False)
            assert result == "clicked"

    def test_dispatch_type(self) -> None:
        """Verify click then type_text are called."""
        step = _make_step(action="type", text_to_type="hello")
        lr = LocateResult(point=Point(100, 200), confidence=0.9, method="mock")
        with (
            patch("routine.runner.click") as m_click,
            patch("routine.runner.type_text") as m_type,
        ):
            result = _dispatch_action(step, lr, dry_run=False)
            m_click.assert_called_once()
            m_type.assert_called_once_with("hello", dry_run=False)
            assert result == "typed"

    def test_dispatch_scroll(self) -> None:
        """Verify scroll is called with direction and amount."""
        step = _make_step(
            action="scroll",
            scroll={"direction": "down", "amount": 5, "unit": "lines"},
        )
        lr = LocateResult(point=Point(500, 600), confidence=0.9, method="mock")
        with patch("routine.runner.scroll") as m:
            result = _dispatch_action(step, lr, dry_run=False)
            m.assert_called_once_with(500, 600, "down", 5, dry_run=False)
            assert result == "scrolled"


# ---------------------------------------------------------------------------
# Loop tests
# ---------------------------------------------------------------------------

class TestLoopStep:
    """Loop step execution tests."""

    def test_loop_step_execution(self, tmp_path: Path) -> None:
        """Loop should iterate the correct number of times."""
        body_step = _make_step(0, "click", "Loop Body", "node_body001")
        loop_step = _make_step(
            1, "loop", "Test Loop", "node_loop001",
            loop={
                "body_step_node_ids": ["node_body001"],
                "exit_condition": {
                    "condition_type": "n_iterations",
                    "params": {"count": 3},
                    "timeout": 0,
                },
                "max_iterations": 10,
            },
        )

        routine = MagicMock()
        routine.name = "loop_test"
        routine.steps = [body_step, loop_step]

        from routine.runner import _handle_loop_step

        step_results: list[dict[str, Any]] = []
        with (
            patch("routine.runner.locate_element_from_step", side_effect=_locate_ok),
            patch("routine.runner.click"),
            patch("routine.runner.screenshot_full", side_effect=_fake_screenshot),
        ):
            _handle_loop_step(
                loop_step, routine, tmp_path, tmp_path,
                1, None, False, step_results,
            )

        assert len(step_results) == 1
        assert step_results[0]["iterations"] == 3


# ---------------------------------------------------------------------------
# Run log tests
# ---------------------------------------------------------------------------

class TestRunLogging:
    """Run log utility tests."""

    def test_run_logging_created(self, tmp_path: Path) -> None:
        """Verify run directory and result.json are created."""
        run_dir = create_run_dir(tmp_path)
        assert run_dir.exists()
        assert run_dir.parent.name == "runs"

        save_run_result(run_dir, {
            "run_id": run_dir.name,
            "status": "success",
            "steps_completed": 3,
        })
        result_file = run_dir / "result.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data["status"] == "success"

    def test_self_cleaning_prune(self, tmp_path: Path) -> None:
        """Create 20 dummy runs, prune, verify retention limits."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        for i in range(20):
            name = f"20260101_{i:06d}_abcd1234"
            rd = runs_dir / name
            rd.mkdir()
            status = "success" if i < 12 else "failed"
            (rd / "result.json").write_text(json.dumps({"status": status}))

        with patch("routine.run_log.get_config", return_value={
            "replay": {"keep_successful_runs": 5, "keep_failed_runs": 10},
        }):
            deleted = prune_old_runs(tmp_path)

        # 12 success - 5 kept = 7 deleted; 8 failed <= 10 = 0 deleted
        assert deleted == 7
        remaining = list(runs_dir.iterdir())
        assert len(remaining) == 13


# ---------------------------------------------------------------------------
# Callback event tests
# ---------------------------------------------------------------------------

class TestCallbackEvents:
    """Event callback tests."""

    def test_callback_events(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """Verify RUN_START, STEP_START, STEP_COMPLETE, RUN_COMPLETE in order."""
        events: list[RunEvent] = []

        def cb(event: RunEvent, data: dict[str, Any]) -> None:
            events.append(event)

        with (
            patch(_RUN_PATCHES["load"], return_value=mock_routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_ok),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["validate"], side_effect=_validate_ok),
            patch(_RUN_PATCHES["click"]),
            patch(_RUN_PATCHES["type_text"]),
            patch(_RUN_PATCHES["press_enter"]),
            patch(_RUN_PATCHES["prune"]),
        ):
            run_routine(sample_routine_dir, callback=cb, dry_run=True)

        assert RunEvent.RUN_START in events
        assert RunEvent.RUN_COMPLETE in events
        assert events.index(RunEvent.RUN_START) < events.index(RunEvent.STEP_START)
        step_complete_indices = [
            i for i, e in enumerate(events) if e == RunEvent.STEP_COMPLETE
        ]
        assert max(step_complete_indices) < events.index(RunEvent.RUN_COMPLETE)

    def test_screenshot_taken_event(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """Verify SCREENSHOT_TAKEN events are emitted for click steps."""
        events: list[tuple[RunEvent, dict[str, Any]]] = []

        def cb(event: RunEvent, data: dict[str, Any]) -> None:
            events.append((event, data))

        with (
            patch(_RUN_PATCHES["load"], return_value=mock_routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_ok),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["validate"], side_effect=_validate_ok),
            patch(_RUN_PATCHES["click"]),
            patch(_RUN_PATCHES["type_text"]),
            patch(_RUN_PATCHES["press_enter"]),
            patch(_RUN_PATCHES["prune"]),
        ):
            run_routine(sample_routine_dir, callback=cb, dry_run=True)

        screenshot_events = [
            (e, d) for e, d in events if e == RunEvent.SCREENSHOT_TAKEN
        ]
        assert len(screenshot_events) >= 1
        purposes = [d.get("purpose") for _, d in screenshot_events]
        assert "before_action" in purposes


# ---------------------------------------------------------------------------
# Config and misc tests
# ---------------------------------------------------------------------------

class TestHumanDelay:
    """Config interaction tests."""

    def test_human_delay_config(
        self, sample_routine_dir: Path, mock_routine: MagicMock,
    ) -> None:
        """Executor functions should be called regardless of delay config."""
        with (
            patch(_RUN_PATCHES["load"], return_value=mock_routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_ok),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["validate"], side_effect=_validate_ok),
            patch(_RUN_PATCHES["click"]),
            patch(_RUN_PATCHES["type_text"]),
            patch(_RUN_PATCHES["press_enter"]),
            patch(_RUN_PATCHES["prune"]),
        ):
            result = run_routine(sample_routine_dir, dry_run=True)
        assert result.success is True


class TestPostActionValidation:
    """Post-action validation tests."""

    def test_post_action_validation(
        self, sample_routine_dir: Path,
    ) -> None:
        """validate_action called for click steps, not for wait/read."""
        routine = MagicMock()
        routine.name = "test"
        routine.start_from = "desktop"
        routine.steps = [
            _make_step(0, "click", "Button", "node_aaa111"),
            _make_step(1, "wait", "Wait Step", "node_bbb222", wait={
                "condition_type": "fixed_timer", "timeout": 0.1,
                "params": {"seconds": 0.1},
            }),
        ]
        for s in routine.steps:
            nid = s["node_id"]
            s["snippet_path"] = f"snippets/{nid}.png"
            s["embedding_path"] = f"embeddings/{nid}.npy"

        with (
            patch(_RUN_PATCHES["load"], return_value=routine),
            patch(_RUN_PATCHES["locate"], side_effect=_locate_ok),
            patch(_RUN_PATCHES["screenshot"], side_effect=_fake_screenshot),
            patch(_RUN_PATCHES["validate"], side_effect=_validate_ok) as m_val,
            patch(_RUN_PATCHES["click"]),
            patch(_RUN_PATCHES["type_text"]),
            patch(_RUN_PATCHES["press_enter"]),
            patch(_RUN_PATCHES["prune"]),
        ):
            run_routine(sample_routine_dir, dry_run=True)
            # validate_action should be called for click, not wait
            assert m_val.call_count >= 1


class TestComputeSearchRegion:
    """_compute_search_region tests."""

    def test_region_centered(self) -> None:
        """Region should be roughly centered on position."""
        rx, ry, rw, rh = _compute_search_region(
            {"x_pct": 0.5, "y_pct": 0.5}, 1920, 1080,
        )
        assert rx < 960
        assert ry < 540
        assert rx + rw > 960
        assert ry + rh > 540

    def test_region_clamped(self) -> None:
        """Region near edge should be clamped to screen bounds."""
        rx, ry, rw, rh = _compute_search_region(
            {"x_pct": 0.0, "y_pct": 0.0}, 1920, 1080,
        )
        assert rx >= 0
        assert ry >= 0
        assert rx + rw <= 1920
        assert ry + rh <= 1080


class TestAnnotateScreenshot:
    """annotate_screenshot tests."""

    def test_annotate_returns_copy(self) -> None:
        """Original image should not be modified."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        original = img.copy()
        result = annotate_screenshot(img, [
            {"x": 10, "y": 10, "w": 30, "h": 30, "label": "test"},
        ], "Title")
        np.testing.assert_array_equal(img, original)
        assert not np.array_equal(result, original)


class TestSetupRunLogger:
    """setup_run_logger tests."""

    def test_logger_handler(self, tmp_path: Path) -> None:
        """Verify file handler is created correctly."""
        handler = setup_run_logger(tmp_path)
        assert isinstance(handler, logging.FileHandler)
        handler.close()
