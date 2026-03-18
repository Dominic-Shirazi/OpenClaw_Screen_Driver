"""Tests for RecordPhase enum and PipelineBridge signals."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from recorder.overlay.pipeline_bridge import PipelineBridge
from recorder.overlay.record_phase import RecordPhase


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


class TestRecordPhase:
    """Tests for RecordPhase enum members."""

    def test_member_count(self) -> None:
        """RecordPhase has exactly 10 members."""
        assert len(RecordPhase) == 10

    def test_member_names(self) -> None:
        """All 10 expected member names exist."""
        expected = [
            "AWAITING_CLICK",
            "CAPTURING",
            "DETECTING",
            "BBOX_EDITING",
            "VLM_ANALYZING",
            "TAG_DIALOG",
            "COUNTDOWN",
            "EXECUTING",
            "VALIDATING",
            "SUCCESS_FLASH",
        ]
        actual = [m.name for m in RecordPhase]
        assert actual == expected

    def test_members_unique(self) -> None:
        """Each member has a unique value."""
        values = [m.value for m in RecordPhase]
        assert len(values) == len(set(values))


class TestPipelineBridge:
    """Tests for PipelineBridge signal bridge."""

    def test_instantiation(self, qapp: QApplication) -> None:
        """PipelineBridge can be instantiated."""
        bridge = PipelineBridge()
        assert bridge is not None

    def test_has_all_signals(self, qapp: QApplication) -> None:
        """PipelineBridge has all 6 expected signals."""
        expected_signals = [
            "detection_ready",
            "vlm_ready",
            "vlm_failed",
            "execution_complete",
            "save_complete",
            "save_failed",
        ]
        for sig_name in expected_signals:
            assert hasattr(PipelineBridge, sig_name), (
                f"Missing signal: {sig_name}"
            )

    def test_execution_complete_connectable(self, qapp: QApplication) -> None:
        """execution_complete signal can be connected."""
        bridge = PipelineBridge()
        called = []
        bridge.execution_complete.connect(lambda: called.append(True))
        bridge.execution_complete.emit()
        assert called == [True]

    def test_detection_ready_payload(self, qapp: QApplication) -> None:
        """detection_ready signal carries a dict payload."""
        bridge = PipelineBridge()
        received = []
        bridge.detection_ready.connect(lambda d: received.append(d))
        payload = {"bbox": {"x": 0, "y": 0, "w": 100, "h": 50}}
        bridge.detection_ready.emit(payload)
        assert received == [payload]
