"""Tests for hub/scanner.py scan_routine() adapter.

Verifies that the scan_routine() function correctly adapts the v1 routine
format (steps array with action/text_to_type keys) to the legacy scanner
input format (nodes/edges dict) and delegates to scan_skill().
"""

from __future__ import annotations

from hub.scanner import ScanResult, scan_routine


class TestScanRoutineAdapter:
    """Tests for the scan_routine() v1 format adapter."""

    def test_scan_routine_empty_steps_safe(self) -> None:
        """scan_routine with empty steps returns is_safe=True, risk_score=0.0."""
        result = scan_routine({"steps": []})
        assert isinstance(result, ScanResult)
        assert result.is_safe is True
        assert result.risk_score == 0.0
        assert result.warnings == []

    def test_scan_routine_url_in_typed_text(self) -> None:
        """scan_routine with a step typing a URL returns warning with risk > 0."""
        routine_data = {
            "steps": [
                {
                    "action": "type",
                    "text_to_type": "https://evil.example.com/steal",
                    "label": "Address Bar",
                    "element_type": "text_field",
                },
            ],
        }
        result = scan_routine(routine_data)
        assert result.risk_score > 0
        assert len(result.warnings) >= 1
        assert any("url" in w.lower() for w in result.warnings)

    def test_scan_routine_system_path_in_label(self) -> None:
        """scan_routine with a step label referencing ~/.ssh returns warning."""
        routine_data = {
            "steps": [
                {
                    "action": "click",
                    "label": "Open ~/.ssh/id_rsa",
                    "element_type": "file",
                },
            ],
        }
        result = scan_routine(routine_data)
        assert result.risk_score > 0
        assert len(result.warnings) >= 1
        assert any("system path" in w.lower() or "ssh" in w.lower() for w in result.warnings)

    def test_scan_routine_multiple_dangerous_steps_unsafe(self) -> None:
        """scan_routine with URL + password + system path returns is_safe=False (risk >= 0.5)."""
        routine_data = {
            "steps": [
                {
                    "action": "type",
                    "text_to_type": "https://malicious.example.com/exfil",
                    "label": "URL Field",
                    "element_type": "text_field",
                },
                {
                    "action": "type",
                    "text_to_type": "my_password_here",
                    "label": "Open ~/.ssh/config",
                    "element_type": "text_field",
                },
            ],
        }
        result = scan_routine(routine_data)
        assert result.risk_score >= 0.5
        assert result.is_safe is False
        assert len(result.warnings) >= 2

    def test_scan_routine_maps_step_fields_to_edges(self) -> None:
        """scan_routine correctly maps step 'action' to edge 'action_type' and 'text_to_type' to 'action_payload'."""
        routine_data = {
            "steps": [
                {
                    "action": "type",
                    "text_to_type": "hello world",
                    "label": "Text Input",
                    "element_type": "text_field",
                },
                {
                    "action": "click",
                    "label": "Submit",
                    "element_type": "button",
                },
            ],
        }
        # A type step should create an edge; a click step should not.
        # Since "hello world" has no URL/secret/path, it should be safe.
        result = scan_routine(routine_data)
        assert result.is_safe is True
        assert result.risk_score == 0.0

    def test_scan_routine_keystroke_action_creates_edge(self) -> None:
        """scan_routine maps keystroke actions to edges for scanning."""
        routine_data = {
            "steps": [
                {
                    "action": "keystroke",
                    "text_to_type": "https://phishing.example.com",
                    "label": "Shortcut",
                    "element_type": "hotkey",
                },
            ],
        }
        result = scan_routine(routine_data)
        assert result.risk_score > 0
        assert any("url" in w.lower() for w in result.warnings)
