"""Tests for TagDialogPanel.

Covers instantiation, positioning, conditional fields, edit mode,
avoidance rect, and form data extraction.  Uses QT_QPA_PLATFORM=offscreen
for headless CI compatibility.
"""
from __future__ import annotations

import os

# MUST be set before any PyQt6 import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication, QComboBox

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.tag_dialog_panel import TagDialogPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def panel(qapp: QApplication) -> TagDialogPanel:
    """Create a fresh TagDialogPanel for each test."""
    clock = AnimationClock()
    p = TagDialogPanel(clock)
    return p


class TestInstantiation:
    """Verify initial state of the tag dialog panel."""

    def test_z_value(self, panel: TagDialogPanel) -> None:
        """Panel z-value must be 100 (Z_TAG_DIALOG)."""
        assert panel.zValue() == 100

    def test_initial_opacity(self, panel: TagDialogPanel) -> None:
        """Panel starts fully transparent."""
        assert panel._opacity == 0.0

    def test_default_dimensions(self, panel: TagDialogPanel) -> None:
        """Panel defaults to 400x280."""
        assert panel._width == 400.0
        assert panel._height == 280.0

    def test_signals_exist(self, panel: TagDialogPanel) -> None:
        """Confirm and dismiss signals are connectable."""
        handler = MagicMock()
        panel.confirmed.connect(handler)
        panel.dismissed.connect(handler)


class TestPositioning:
    """Verify _compute_position logic for panel placement."""

    def test_prefer_below_right(self, panel: TagDialogPanel) -> None:
        """Panel should appear to the right of the element."""
        element = QRectF(100, 100, 50, 30)
        x, y = panel._compute_position(element, 1920, 1080)
        assert x > element.right()
        assert y >= element.top()

    def test_flip_horizontal_near_right_edge(
        self, panel: TagDialogPanel,
    ) -> None:
        """Panel flips to left when element is near right edge."""
        element = QRectF(1800, 100, 50, 30)
        x, y = panel._compute_position(element, 1920, 1080)
        assert x < element.left()

    def test_flip_vertical_near_bottom(
        self, panel: TagDialogPanel,
    ) -> None:
        """Panel stays on screen when element is near bottom."""
        element = QRectF(100, 900, 50, 30)
        x, y = panel._compute_position(element, 1920, 1080)
        assert y + panel._height + 24 <= 1080

    def test_clamp_to_screen(self, panel: TagDialogPanel) -> None:
        """Panel is clamped within breathing room on all edges."""
        element = QRectF(1850, 1050, 50, 30)
        x, y = panel._compute_position(element, 1920, 1080)
        assert x >= 24
        assert y >= 24
        assert x + panel._width + 24 <= 1920
        assert y + panel._height + 24 <= 1080


class TestConditionalFields:
    """Verify action type dropdown shows/hides conditional fields."""

    def test_type_action_shows_text_field(
        self, panel: TagDialogPanel,
    ) -> None:
        """Selecting 'type' action should show text_to_type field."""
        action_combo = panel._widgets["action_type"]
        assert isinstance(action_combo, QComboBox)
        idx = action_combo.findData("type")
        action_combo.setCurrentIndex(idx)
        assert panel._proxies["text_to_type"].isVisible()
        assert panel._proxies["press_enter"].isVisible()

    def test_click_action_hides_conditional(
        self, panel: TagDialogPanel,
    ) -> None:
        """Default 'click' action should have no conditional fields visible."""
        action_combo = panel._widgets["action_type"]
        assert isinstance(action_combo, QComboBox)
        idx = action_combo.findData("click")
        action_combo.setCurrentIndex(idx)
        assert not panel._proxies["text_to_type"].isVisible()
        assert not panel._proxies["press_enter"].isVisible()
        assert not panel._proxies["direction_amount"].isVisible()

    def test_scroll_shows_direction_field(
        self, panel: TagDialogPanel,
    ) -> None:
        """Selecting 'scroll' action should show direction_amount field."""
        action_combo = panel._widgets["action_type"]
        assert isinstance(action_combo, QComboBox)
        idx = action_combo.findData("scroll")
        action_combo.setCurrentIndex(idx)
        assert panel._proxies["direction_amount"].isVisible()
        assert not panel._proxies["text_to_type"].isVisible()

    def test_panel_height_changes_with_action(
        self, panel: TagDialogPanel,
    ) -> None:
        """Panel target height should change when conditional fields appear."""
        action_combo = panel._widgets["action_type"]
        assert isinstance(action_combo, QComboBox)

        idx_click = action_combo.findData("click")
        action_combo.setCurrentIndex(idx_click)
        height_click = panel._target_height

        idx_type = action_combo.findData("type")
        action_combo.setCurrentIndex(idx_type)
        height_type = panel._target_height

        assert height_type > height_click


class TestEditMode:
    """Verify edit mode behavior."""

    def test_edit_mode_no_typewriter(self, panel: TagDialogPanel) -> None:
        """Edit mode should not activate typewriter."""
        vlm_data = {
            "label": "Submit",
            "caption": "Submit button",
            "action_type": "click",
            "element_type": "button",
        }
        panel.show_dialog(
            QRectF(100, 100, 50, 30),
            vlm_data=vlm_data,
            edit_mode=True,
        )
        assert panel._typewriter.is_active is False

    def test_edit_mode_sets_fields(self, panel: TagDialogPanel) -> None:
        """Edit mode should set field values instantly."""
        vlm_data = {
            "label": "Submit",
            "caption": "Submit button",
            "action_type": "click",
            "element_type": "button",
        }
        panel.show_dialog(
            QRectF(100, 100, 50, 30),
            vlm_data=vlm_data,
            edit_mode=True,
        )
        data = panel.get_form_data()
        assert data["label"] == "Submit"
        assert data["caption"] == "Submit button"


class TestAvoidanceRect:
    """Verify avoidance rect for shimmer layer."""

    def test_returns_qrectf(self, panel: TagDialogPanel) -> None:
        """Avoidance rect should be a QRectF matching panel dimensions."""
        panel.setPos(100, 200)
        rect = panel.get_avoidance_rect()
        assert isinstance(rect, QRectF)
        assert rect.width() >= 400
        assert rect.height() >= 280

    def test_position_matches(self, panel: TagDialogPanel) -> None:
        """Avoidance rect origin should match panel position."""
        panel.setPos(300, 400)
        rect = panel.get_avoidance_rect()
        assert rect.x() == 300.0
        assert rect.y() == 400.0


class TestFormData:
    """Verify form data extraction."""

    def test_returns_dict_with_required_keys(
        self, panel: TagDialogPanel,
    ) -> None:
        """Form data dict must contain all required keys."""
        data = panel.get_form_data()
        assert "label" in data
        assert "caption" in data
        assert "action_type" in data
        assert "element_type" in data

    def test_action_type_default_is_click(
        self, panel: TagDialogPanel,
    ) -> None:
        """Default action type should be 'click'."""
        data = panel.get_form_data()
        assert data["action_type"] == "click"
