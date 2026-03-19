"""Tests for action-type-specific step format fields and executor helpers.

Covers: ACT-01 (click variants), ACT-02 (type action), ACT-03 (scroll action),
ACT-05 (dry-run dispatch), ACT-09 (press_enter helper).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from routine.format import build_v1_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(tag_data: dict, bbox: tuple = (100, 200, 50, 30)) -> dict:
    """Build a minimal internal step dict for testing build_v1_step."""
    return {"tag_data": tag_data, "bbox": bbox}


def _build(tag_data: dict, bbox: tuple = (100, 200, 50, 30)) -> dict:
    """Shortcut: build a v1 step from tag_data."""
    step = _make_step(tag_data, bbox)
    return build_v1_step(step, index=0, node_id="node-test", screen_w=1920, screen_h=1080)


# ---------------------------------------------------------------------------
# ACT-01: Click variants
# ---------------------------------------------------------------------------


def test_click_step_format() -> None:
    """build_v1_step with action='click' produces step with action='click', no extra fields."""
    result = _build({"action": "click"})
    assert result["action"] == "click"
    assert "text_to_type" not in result
    assert "scroll" not in result


def test_double_click_step_format() -> None:
    """build_v1_step with action='double_click' produces step with action='double_click'."""
    result = _build({"action": "double_click"})
    assert result["action"] == "double_click"
    assert "text_to_type" not in result
    assert "scroll" not in result


def test_right_click_step_format() -> None:
    """build_v1_step with action='right_click' produces step with action='right_click'."""
    result = _build({"action": "right_click"})
    assert result["action"] == "right_click"
    assert "text_to_type" not in result
    assert "scroll" not in result


# ---------------------------------------------------------------------------
# ACT-02: Type action
# ---------------------------------------------------------------------------


def test_type_step_format() -> None:
    """build_v1_step with action='type' includes text_to_type and press_enter=True."""
    result = _build({
        "action": "type",
        "text_to_type": "hello",
        "press_enter": True,
    })
    assert result["action"] == "type"
    assert result["text_to_type"] == "hello"
    assert result["press_enter"] is True


def test_type_step_format_no_enter() -> None:
    """build_v1_step with action='type' defaults press_enter to False."""
    result = _build({
        "action": "type",
        "text_to_type": "foo",
    })
    assert result["action"] == "type"
    assert result["text_to_type"] == "foo"
    assert result["press_enter"] is False


# ---------------------------------------------------------------------------
# ACT-03: Scroll action
# ---------------------------------------------------------------------------


def test_scroll_step_format() -> None:
    """build_v1_step with action='scroll' and 'down 5' produces scroll dict."""
    result = _build({
        "action": "scroll",
        "direction_amount": "down 5",
    })
    assert result["action"] == "scroll"
    assert result["scroll"]["direction"] == "down"
    assert result["scroll"]["amount"] == 5


def test_scroll_step_format_left() -> None:
    """build_v1_step with action='scroll' and 'left 10' produces scroll dict."""
    result = _build({
        "action": "scroll",
        "direction_amount": "left 10",
    })
    assert result["action"] == "scroll"
    assert result["scroll"]["direction"] == "left"
    assert result["scroll"]["amount"] == 10


def test_scroll_step_format_with_unit() -> None:
    """build_v1_step with action='scroll' and 'up 3 pages' includes unit."""
    result = _build({
        "action": "scroll",
        "direction_amount": "up 3 pages",
    })
    assert result["action"] == "scroll"
    assert result["scroll"]["direction"] == "up"
    assert result["scroll"]["amount"] == 3
    assert result["scroll"]["unit"] == "pages"


def test_scroll_step_format_defaults() -> None:
    """build_v1_step with action='scroll' and no direction_amount uses defaults."""
    result = _build({"action": "scroll"})
    assert result["action"] == "scroll"
    assert result["scroll"]["direction"] == "down"
    assert result["scroll"]["amount"] == 3
    assert result["scroll"]["unit"] == "lines"


# ---------------------------------------------------------------------------
# ACT-09: press_enter helper
# ---------------------------------------------------------------------------


def test_press_enter() -> None:
    """core.executor.press_enter() calls pyautogui.press('enter')."""
    with patch("core.executor.pyautogui") as mock_pag:
        from core.executor import press_enter
        press_enter()
        mock_pag.press.assert_called_once_with("enter")


def test_press_enter_dry_run() -> None:
    """core.executor.press_enter(dry_run=True) does not call pyautogui."""
    with patch("core.executor.pyautogui") as mock_pag:
        from core.executor import press_enter
        press_enter(dry_run=True)
        mock_pag.press.assert_not_called()


# ---------------------------------------------------------------------------
# _parse_direction_amount helper
# ---------------------------------------------------------------------------


def test_parse_direction_amount_basic() -> None:
    """_parse_direction_amount parses 'down 5' correctly."""
    from routine.format import _parse_direction_amount
    result = _parse_direction_amount("down 5")
    assert result == {"direction": "down", "amount": 5, "unit": "lines"}


def test_parse_direction_amount_with_pages() -> None:
    """_parse_direction_amount parses 'up 3 pages' correctly."""
    from routine.format import _parse_direction_amount
    result = _parse_direction_amount("up 3 pages")
    assert result == {"direction": "up", "amount": 3, "unit": "pages"}


def test_parse_direction_amount_defaults() -> None:
    """_parse_direction_amount with empty string uses defaults."""
    from routine.format import _parse_direction_amount
    result = _parse_direction_amount("")
    assert result == {"direction": "down", "amount": 3, "unit": "lines"}
