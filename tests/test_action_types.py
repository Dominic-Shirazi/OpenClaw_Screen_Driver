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


# ---------------------------------------------------------------------------
# ACT-04: click_drag step format
# ---------------------------------------------------------------------------


def test_click_drag_step_format() -> None:
    """build_v1_step with action='click_drag' and drag_target produces step with drag_target field."""
    step = {
        "tag_data": {"action": "click_drag"},
        "bbox": (100, 200, 50, 30),
        "drag_target": {
            "bbox": {"x": 400, "y": 300, "w": 60, "h": 40},
            "anchors": {"visual_match": "snippets/node-test_target.png"},
        },
    }
    result = build_v1_step(step, index=0, node_id="node-test", screen_w=1920, screen_h=1080)
    assert result["action"] == "click_drag"
    assert result["drag_target"]["bbox"]["x"] == 400
    assert result["drag_target"]["anchors"]["visual_match"] == "snippets/node-test_target.png"


# ---------------------------------------------------------------------------
# ACT-12: prompt_user step format
# ---------------------------------------------------------------------------


def test_prompt_user_step_format() -> None:
    """build_v1_step with action='prompt_user' and question_text produces step with question_text."""
    result = _build({"action": "prompt_user", "question_text": "Are you done?"})
    assert result["action"] == "prompt_user"
    assert result["question_text"] == "Are you done?"


# ---------------------------------------------------------------------------
# ACT-06: read step format
# ---------------------------------------------------------------------------


def test_read_step_format() -> None:
    """build_v1_step with action='read' and vlm_prompt produces step with vlm_prompt."""
    result = _build({"action": "read", "vlm_prompt": "Extract text"})
    assert result["action"] == "read"
    assert result["vlm_prompt"] == "Extract text"


# ---------------------------------------------------------------------------
# ACT-07: snip_and_search step format
# ---------------------------------------------------------------------------


def test_snip_and_search_step_format() -> None:
    """build_v1_step with action='snip_and_search' includes vlm_prompt."""
    result = _build({"action": "snip_and_search", "vlm_prompt": "Find price"})
    assert result["action"] == "snip_and_search"
    assert result["vlm_prompt"] == "Find price"


# ---------------------------------------------------------------------------
# ACT-10: wait step format
# ---------------------------------------------------------------------------


def test_wait_step_format_defaults() -> None:
    """build_v1_step with action='wait' uses default wait definition."""
    result = _build({"action": "wait"})
    assert result["action"] == "wait"
    assert result["wait"]["condition_type"] == "fixed_timer"
    assert result["wait"]["timeout"] == 30.0


# ---------------------------------------------------------------------------
# ACT-11: loop step format
# ---------------------------------------------------------------------------


def test_loop_step_format() -> None:
    """build_v1_step with action='loop' and loop_definition produces step with loop field."""
    step = {
        "tag_data": {"action": "loop"},
        "bbox": (0, 0, 0, 0),
        "loop_definition": {
            "body_step_indices": [0, 1],
            "exit_condition": {"type": "n_iterations", "count": 3},
        },
    }
    result = build_v1_step(step, index=0, node_id="node-test", screen_w=1920, screen_h=1080)
    assert result["action"] == "loop"
    assert result["loop"]["exit_condition"]["type"] == "n_iterations"
    assert result["loop"]["exit_condition"]["count"] == 3
    assert result["loop"]["body_step_indices"] == [0, 1]


def test_resolve_loop_node_ids() -> None:
    """resolve_loop_node_ids converts body_step_indices to body_step_node_ids."""
    from routine.format import resolve_loop_node_ids

    node_ids = ["node-a", "node-b", "node-c"]
    steps = [
        {"action": "click", "node_id": "node-a"},
        {"action": "click", "node_id": "node-b"},
        {
            "action": "loop",
            "node_id": "node-c",
            "loop": {
                "body_step_indices": [0, 1],
                "exit_condition": {"type": "n_iterations", "count": 3},
            },
        },
    ]
    resolve_loop_node_ids(steps, node_ids)

    loop_def = steps[2]["loop"]
    assert "body_step_indices" not in loop_def
    assert loop_def["body_step_node_ids"] == ["node-a", "node-b"]


def test_nested_loop_format() -> None:
    """Nested loop (loop body containing another loop) serializes correctly."""
    from routine.format import resolve_loop_node_ids

    node_ids = ["node-a", "node-b", "node-inner-loop", "node-outer-loop"]
    steps = [
        {"action": "click", "node_id": "node-a"},
        {"action": "click", "node_id": "node-b"},
        {
            "action": "loop",
            "node_id": "node-inner-loop",
            "loop": {
                "body_step_indices": [0, 1],
                "exit_condition": {"type": "n_iterations", "count": 2},
            },
        },
        {
            "action": "loop",
            "node_id": "node-outer-loop",
            "loop": {
                "body_step_indices": [0, 1, 2],
                "exit_condition": {"type": "element_appears", "element_description": "Done", "max_iterations": 10},
            },
        },
    ]
    resolve_loop_node_ids(steps, node_ids)

    inner = steps[2]["loop"]
    assert inner["body_step_node_ids"] == ["node-a", "node-b"]

    outer = steps[3]["loop"]
    assert outer["body_step_node_ids"] == ["node-a", "node-b", "node-inner-loop"]
    assert outer["exit_condition"]["type"] == "element_appears"
    assert outer["exit_condition"]["max_iterations"] == 10


# ---------------------------------------------------------------------------
# Look Here flow: region drag only
# ---------------------------------------------------------------------------


def test_look_here_requires_drag() -> None:
    """RecordSession in AWAITING_REGION_DRAG phase ignores clicks (w=0, h=0)."""
    from unittest.mock import MagicMock, PropertyMock

    from recorder.overlay.record_phase import RecordPhase
    from recorder.record_session import RecordSession

    controller = MagicMock()
    session = RecordSession(controller, "test-routine")
    session._phase = RecordPhase.AWAITING_REGION_DRAG

    # Click (w=0, h=0) should be ignored
    session.on_selection(100, 200, 0, 0)
    assert session._phase == RecordPhase.AWAITING_REGION_DRAG
    assert session._is_look_here is False  # not set because click was ignored
