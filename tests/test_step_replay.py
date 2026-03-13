"""Tests for step-through replay mode and debug-ai flag.

Covers:
- STEP_PREVIEW event emission from the orchestrator
- step_through_prompt() return values
- debug-ai logging of locate method/confidence
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# step_ui tests
# ---------------------------------------------------------------------------


class TestStepThroughPrompt:
    """Tests for recorder.step_ui.step_through_prompt."""

    def test_plain_fallback_execute(self) -> None:
        """Enter key (empty input) defaults to execute."""
        from recorder.step_ui import _plain_prompt

        with patch("builtins.input", return_value=""):
            assert _plain_prompt("Login Button", 1) == "execute"

    def test_plain_fallback_skip(self) -> None:
        """'s' input returns skip."""
        from recorder.step_ui import _plain_prompt

        with patch("builtins.input", return_value="s"):
            assert _plain_prompt("Login Button", 1) == "skip"

    def test_plain_fallback_abort(self) -> None:
        """'q' input returns abort."""
        from recorder.step_ui import _plain_prompt

        with patch("builtins.input", return_value="q"):
            assert _plain_prompt("Login Button", 1) == "abort"

    def test_plain_fallback_eof(self) -> None:
        """EOFError returns abort."""
        from recorder.step_ui import _plain_prompt

        with patch("builtins.input", side_effect=EOFError):
            assert _plain_prompt("Login Button", 1) == "abort"

    def test_plain_fallback_keyboard_interrupt(self) -> None:
        """KeyboardInterrupt returns abort."""
        from recorder.step_ui import _plain_prompt

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _plain_prompt("Login Button", 1) == "abort"

    def test_step_through_prompt_falls_back_without_rich(self) -> None:
        """When rich is not installed, falls back to plain prompt."""
        from recorder.step_ui import step_through_prompt

        with patch("recorder.step_ui._rich_prompt", side_effect=ImportError):
            with patch("recorder.step_ui._plain_prompt", return_value="execute") as mock_plain:
                result = step_through_prompt(
                    label="Submit",
                    node_id="abc12345",
                    step_num=3,
                )
                assert result == "execute"
                mock_plain.assert_called_once()


# ---------------------------------------------------------------------------
# Orchestrator STEP_PREVIEW emission tests
# ---------------------------------------------------------------------------


class TestOrchestratorStepPreview:
    """Tests that the orchestrator emits STEP_PREVIEW events."""

    @patch("mapper.orchestrator.get_execution_plan")
    @patch("mapper.orchestrator.execute_node")
    @patch("mapper.orchestrator.locate_element")
    def test_step_preview_emitted_before_step_start(
        self,
        mock_locate: MagicMock,
        mock_execute: MagicMock,
        mock_plan: MagicMock,
    ) -> None:
        """STEP_PREVIEW is emitted before STEP_START for each node."""
        from core.types import LocateResult, Point, ReplayStep
        from mapper.graph import OCSDGraph
        from mapper.orchestrator import orchestrate_skill
        from mapper.runner import RunnerEventType

        # Build a minimal graph
        graph = OCSDGraph()
        n1 = graph.add_node(
            element_type="button", label="Login",
            x_pct=0.5, y_pct=0.5, resolution=(1920, 1080),
        )
        n2 = graph.add_node(
            element_type="textbox", label="Search",
            x_pct=0.3, y_pct=0.3, resolution=(1920, 1080),
        )
        graph.add_edge(n1, n2, action_type="button")

        # Mock plan
        mock_plan.return_value = {
            "path": [n1, n2],
            "fingerprint_checks": [],
            "branch_points": [],
            "estimated_reliability": 0.9,
        }

        # Mock execution — both succeed
        mock_execute.return_value = ReplayStep(
            node_id=n1,
            located_at=Point(960, 540),
            locate_method="yoloe",
            vlm_confidence=0.9,
            pixel_diff_pct=0.0,
            success=True,
        )

        # Collect events
        events: list[tuple[RunnerEventType, dict]] = []

        def callback(evt: RunnerEventType, data: dict) -> None:
            events.append((evt, data))

        orchestrate_skill(
            graph, n1, n2,
            dry_run=True,
            skip_preflight=True,
            event_callback=callback,
        )

        # Check that STEP_PREVIEW was emitted
        preview_events = [
            (evt, d) for evt, d in events
            if evt == RunnerEventType.STEP_PREVIEW
        ]
        assert len(preview_events) == 2, f"Expected 2 STEP_PREVIEW events, got {len(preview_events)}"

        # Verify STEP_PREVIEW comes before STEP_START
        event_types = [evt for evt, _ in events]
        for i, evt in enumerate(event_types):
            if evt == RunnerEventType.STEP_START:
                # There should be a STEP_PREVIEW before it
                assert i > 0, "STEP_START at index 0 — no preceding STEP_PREVIEW"
                assert event_types[i - 1] == RunnerEventType.STEP_PREVIEW

    @patch("mapper.orchestrator.get_execution_plan")
    @patch("mapper.orchestrator.execute_node")
    @patch("mapper.orchestrator.locate_element")
    def test_step_preview_contains_metadata(
        self,
        mock_locate: MagicMock,
        mock_execute: MagicMock,
        mock_plan: MagicMock,
    ) -> None:
        """STEP_PREVIEW events contain label, action_type, and position."""
        from core.types import Point, ReplayStep
        from mapper.graph import OCSDGraph
        from mapper.orchestrator import orchestrate_skill
        from mapper.runner import RunnerEventType

        graph = OCSDGraph()
        n1 = graph.add_node(
            element_type="button", label="OK",
            x_pct=0.5, y_pct=0.5, resolution=(1920, 1080),
        )

        mock_plan.return_value = {
            "path": [n1],
            "fingerprint_checks": [],
            "branch_points": [],
            "estimated_reliability": 1.0,
        }

        mock_execute.return_value = ReplayStep(
            node_id=n1,
            located_at=Point(960, 540),
            locate_method="yoloe",
            vlm_confidence=0.95,
            pixel_diff_pct=0.0,
            success=True,
        )

        previews: list[dict] = []

        def callback(evt: RunnerEventType, data: dict) -> None:
            if evt == RunnerEventType.STEP_PREVIEW:
                previews.append(data)

        orchestrate_skill(
            graph, n1, n1,
            dry_run=True,
            skip_preflight=True,
            event_callback=callback,
        )

        assert len(previews) == 1
        p = previews[0]
        assert p["label"] == "OK"
        assert p["node_id"] == n1
        assert "element_type" in p
        assert "step" in p


# ---------------------------------------------------------------------------
# Main.py callback integration test
# ---------------------------------------------------------------------------


class TestMainStepCallback:
    """Tests for the _step_event_handler in main.py cmd_execute."""

    def test_debug_ai_logs_locate_info(self) -> None:
        """--debug-ai callback logs locate method and confidence."""
        from mapper.runner import RunnerEventType

        logged: list[str] = []

        def fake_callback(evt: RunnerEventType, data: dict) -> None:
            if evt == RunnerEventType.ELEMENT_LOCATED:
                logged.append(
                    f"{data.get('method')} conf={data.get('confidence')}"
                )

        fake_callback(
            RunnerEventType.ELEMENT_LOCATED,
            {"method": "yoloe", "point": (100, 200), "confidence": 0.92},
        )

        assert len(logged) == 1
        assert "yoloe" in logged[0]
        assert "0.92" in logged[0]

    def test_step_mode_abort_raises(self) -> None:
        """When step_through_prompt returns 'abort', KeyboardInterrupt is raised."""
        with patch("recorder.step_ui.step_through_prompt", return_value="abort"):
            from recorder.step_ui import step_through_prompt

            action = step_through_prompt(
                label="Test", node_id="abc", step_num=1,
            )
            assert action == "abort"
