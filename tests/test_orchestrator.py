"""Tests for mapper.orchestrator — full replay orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.types import (
    ConfirmResult,
    LocateResult,
    Point,
    ReplayLog,
    ReplayStep,
)
from mapper.graph import OCSDGraph
from mapper.orchestrator import (
    _parse_recovery_action,
    diagnose_failure,
    orchestrate_skill,
    preflight_check,
)
from mapper.runner import RunnerEventType


@pytest.fixture
def simple_graph() -> OCSDGraph:
    """Creates a simple A -> B -> C workflow graph."""
    g = OCSDGraph()
    a = g.add_node(element_type="button", label="Login", x_pct=0.5, y_pct=0.3)
    b = g.add_node(element_type="textbox", label="Username", x_pct=0.5, y_pct=0.5)
    c = g.add_node(element_type="button", label="Submit", x_pct=0.5, y_pct=0.7)
    g.add_edge(a, b, action_type="button")
    g.add_edge(b, c, action_type="textbox")
    return g


class TestPreflightCheck:
    """Tests for the pre-flight fingerprint check."""

    def test_skip_vlm_returns_optimistic(self, simple_graph: OCSDGraph) -> None:
        """skip_vlm=True bypasses VLM and returns optimistic result."""
        start_id = simple_graph.nodes[0]
        result = preflight_check(simple_graph, start_id, skip_vlm=True)

        assert result["passed"] is True
        assert result["confidence"] == 0.5
        assert "skipped" in result["notes"].lower()

    def test_missing_node_fails(self, simple_graph: OCSDGraph) -> None:
        """Non-existent start node fails preflight."""
        result = preflight_check(simple_graph, "nonexistent-id")

        assert result["passed"] is False
        assert result["confidence"] == 0.0

    @patch("mapper.orchestrator.locate_element")
    def test_high_confidence_locate_passes(
        self, mock_locate: MagicMock, simple_graph: OCSDGraph
    ) -> None:
        """If locate finds the start element with high confidence, preflight passes."""
        start_id = simple_graph.nodes[0]
        mock_locate.return_value = LocateResult(
            point=Point(500, 300), confidence=0.85, method="yoloe",
        )

        result = preflight_check(simple_graph, start_id)

        assert result["passed"] is True
        assert result["confidence"] == 0.85

    @patch("mapper.orchestrator.locate_element")
    def test_low_confidence_locate_fails(
        self, mock_locate: MagicMock, simple_graph: OCSDGraph
    ) -> None:
        """If locate finds element with low confidence, preflight fails."""
        start_id = simple_graph.nodes[0]
        mock_locate.return_value = LocateResult(
            point=Point(500, 300), confidence=0.3, method="direct",
        )

        result = preflight_check(simple_graph, start_id)

        assert result["passed"] is False
        assert result["confidence"] == 0.3


class TestDiagnoseFailure:
    """Tests for VLM-based failure diagnosis."""

    def test_missing_node_aborts(self, simple_graph: OCSDGraph) -> None:
        """Non-existent node returns abort action."""
        result = diagnose_failure(simple_graph, "nonexistent-id", "not found")

        assert result["action"] == "abort"
        assert result["confidence"] == 0.0

    @patch("mapper.orchestrator.analyze_crop", create=True)
    @patch("mapper.orchestrator.screenshot_full", create=True)
    def test_vlm_unavailable_returns_retry(
        self, mock_ss: MagicMock, mock_crop: MagicMock, simple_graph: OCSDGraph
    ) -> None:
        """When VLM is not available, suggests retry."""
        # Make the import fail by patching at the module level
        node_id = simple_graph.nodes[0]

        with patch.dict("sys.modules", {"core.vision": None}):
            result = diagnose_failure(simple_graph, node_id, "element not found")

        assert result["action"] == "retry"
        assert result["confidence"] == 0.2


class TestParseRecoveryAction:
    """Tests for recovery action parsing."""

    def test_scroll_down(self) -> None:
        assert _parse_recovery_action("The element is below the fold") == "scroll_down"
        assert _parse_recovery_action("Try scroll down") == "scroll_down"

    def test_scroll_up(self) -> None:
        assert _parse_recovery_action("Element is above the viewport") == "scroll_up"

    def test_wait(self) -> None:
        assert _parse_recovery_action("Page is still loading") == "wait"
        assert _parse_recovery_action("I see a spinner") == "wait"

    def test_navigate_back(self) -> None:
        assert _parse_recovery_action("Go back to the previous page") == "navigate_back"

    def test_abort(self) -> None:
        assert _parse_recovery_action("This is a different page, not found") == "abort"
        assert _parse_recovery_action("Wrong page entirely") == "abort"

    def test_default_retry(self) -> None:
        assert _parse_recovery_action("I see a login form") == "retry"
        assert _parse_recovery_action("") == "retry"


class TestOrchestrateSkill:
    """Tests for the full orchestration flow."""

    @patch("mapper.orchestrator.execute_node")
    @patch("mapper.orchestrator.get_execution_plan")
    def test_dry_run_skips_preflight(
        self,
        mock_plan: MagicMock,
        mock_exec: MagicMock,
        simple_graph: OCSDGraph,
    ) -> None:
        """Dry run skips preflight and executes all nodes."""
        nodes = simple_graph.nodes
        mock_plan.return_value = {
            "path": nodes,
            "fingerprint_checks": [],
            "branch_points": [],
            "estimated_reliability": 1.0,
        }
        mock_exec.return_value = ReplayStep(
            node_id=nodes[0],
            located_at=Point(100, 100),
            locate_method="dry_run",
            vlm_confidence=1.0,
            pixel_diff_pct=0.0,
            success=True,
        )

        result = orchestrate_skill(
            simple_graph, nodes[0], nodes[-1], dry_run=True,
        )

        assert result.overall_success is True
        assert len(result.steps) == len(nodes)

    @patch("mapper.orchestrator.execute_node")
    @patch("mapper.orchestrator.get_execution_plan")
    @patch("mapper.orchestrator.preflight_check")
    def test_preflight_failure_aborts(
        self,
        mock_preflight: MagicMock,
        mock_plan: MagicMock,
        mock_exec: MagicMock,
        simple_graph: OCSDGraph,
    ) -> None:
        """Failed preflight aborts before executing any nodes."""
        nodes = simple_graph.nodes
        mock_preflight.return_value = {
            "passed": False,
            "confidence": 0.1,
            "notes": "Wrong screen",
            "screen_state": "",
        }

        result = orchestrate_skill(simple_graph, nodes[0], nodes[-1])

        assert result.overall_success is False
        assert len(result.steps) == 1
        assert "pre-flight" in result.steps[0].error.lower()
        mock_exec.assert_not_called()

    @patch("mapper.orchestrator.diagnose_failure")
    @patch("mapper.orchestrator._execute_recovery")
    @patch("mapper.orchestrator.execute_node")
    @patch("mapper.orchestrator.get_execution_plan")
    @patch("mapper.orchestrator.preflight_check")
    def test_recovery_on_failure(
        self,
        mock_preflight: MagicMock,
        mock_plan: MagicMock,
        mock_exec: MagicMock,
        mock_recovery: MagicMock,
        mock_diagnose: MagicMock,
        simple_graph: OCSDGraph,
    ) -> None:
        """On step failure, orchestrator attempts recovery and retries."""
        nodes = simple_graph.nodes
        mock_preflight.return_value = {
            "passed": True, "confidence": 0.9, "notes": "OK", "screen_state": "",
        }
        mock_plan.return_value = {
            "path": nodes[:1],  # Only one node for simplicity
            "fingerprint_checks": [],
            "branch_points": [],
            "estimated_reliability": 1.0,
        }

        # First call fails, second succeeds (after recovery)
        mock_exec.side_effect = [
            ReplayStep(
                node_id=nodes[0], located_at=None, locate_method="failed",
                vlm_confidence=0.0, pixel_diff_pct=0.0, success=False,
                error="Element not found",
            ),
            ReplayStep(
                node_id=nodes[0], located_at=Point(100, 100), locate_method="ocr",
                vlm_confidence=0.8, pixel_diff_pct=0.0, success=True,
            ),
        ]

        mock_diagnose.return_value = {
            "diagnosis": "Element below fold",
            "suggestion": "scroll down",
            "action": "scroll_down",
            "confidence": 0.7,
        }
        mock_recovery.return_value = True

        result = orchestrate_skill(
            simple_graph, nodes[0], nodes[-1], max_retries=1,
        )

        assert result.overall_success is True
        assert mock_exec.call_count == 2
        mock_diagnose.assert_called_once()
        mock_recovery.assert_called_once_with("scroll_down")

    @patch("mapper.orchestrator.execute_node")
    @patch("mapper.orchestrator.get_execution_plan")
    def test_skip_preflight_flag(
        self,
        mock_plan: MagicMock,
        mock_exec: MagicMock,
        simple_graph: OCSDGraph,
    ) -> None:
        """skip_preflight=True skips the preflight check entirely."""
        nodes = simple_graph.nodes
        mock_plan.return_value = {
            "path": nodes,
            "fingerprint_checks": [],
            "branch_points": [],
            "estimated_reliability": 1.0,
        }
        mock_exec.return_value = ReplayStep(
            node_id=nodes[0],
            located_at=Point(100, 100),
            locate_method="ocr",
            vlm_confidence=0.8,
            pixel_diff_pct=0.0,
            success=True,
        )

        result = orchestrate_skill(
            simple_graph, nodes[0], nodes[-1],
            skip_preflight=True,
        )

        assert result.overall_success is True

    @patch("mapper.orchestrator.execute_node")
    @patch("mapper.orchestrator.get_execution_plan")
    @patch("mapper.orchestrator.preflight_check")
    def test_event_callbacks_fire(
        self,
        mock_preflight: MagicMock,
        mock_plan: MagicMock,
        mock_exec: MagicMock,
        simple_graph: OCSDGraph,
    ) -> None:
        """Event callbacks are fired for step start and complete."""
        nodes = simple_graph.nodes
        mock_preflight.return_value = {
            "passed": True, "confidence": 0.9, "notes": "OK", "screen_state": "",
        }
        mock_plan.return_value = {
            "path": nodes[:1],
            "fingerprint_checks": [],
            "branch_points": [],
            "estimated_reliability": 1.0,
        }
        mock_exec.return_value = ReplayStep(
            node_id=nodes[0],
            located_at=Point(100, 100),
            locate_method="ocr",
            vlm_confidence=0.8,
            pixel_diff_pct=0.0,
            success=True,
        )

        events: list[tuple] = []
        def callback(event_type, data):
            events.append((event_type, data))

        orchestrate_skill(
            simple_graph, nodes[0], nodes[-1],
            event_callback=callback,
        )

        event_types = [e[0] for e in events]
        assert RunnerEventType.STEP_START in event_types
        assert RunnerEventType.STEP_COMPLETE in event_types
        assert RunnerEventType.PATH_COMPLETE in event_types
