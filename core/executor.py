"""Mouse and keyboard automation with realistic human-like behavior.

Mouse movement and click targeting are delegated to the ``human_mouse_moves``
library which implements kinematic sub-movement models: Fitts's Law timing,
quartic Bézier paths, skewed-Beta velocity profiles, overshoot/correction,
bivariate-normal click targeting with approach-vector shortfall bias, and
micro-shivers.

Typing uses the bundled ``human_typing`` library for burst cadence, QWERTY
neighbor typos with immediate correction, and fatigue modeling.

The ``human_delay`` config multiplier maps to HumanMouse speed:
  0   → instant (skip all human sim — speed/test mode)
  1.0 → normal human speed
  5.0 → very slow (useful for watching/debugging)

Thread-safe: HumanMouse is not thread-safe, so all calls are serialized
through a lock.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import threading
import time
from typing import Any

import pyautogui

from core.config import get_config

logger = logging.getLogger(__name__)


class PromptTimeoutError(Exception):
    """Raised when prompt_user_blocking times out waiting for a response."""

    pass

# PyAutoGUI safety: moving mouse to (0,0) aborts. Keep this ON.
pyautogui.FAILSAFE = True
# Disable default pause between actions (we handle timing ourselves).
pyautogui.PAUSE = 0

# Thread lock — HumanMouse and PyAutoGUI are NOT thread-safe.
_lock = threading.Lock()
# Separate lock for lazy singleton initialization to avoid races in
# _get_hmm() and _get_human_typer() when called from multiple threads.
_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
# human_mouse_moves integration
# ---------------------------------------------------------------------------

# Add the sibling project to sys.path so we can import it
_HMM_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
                 "..", "human_mouse_moves")
)
if _HMM_DIR not in sys.path:
    sys.path.insert(0, _HMM_DIR)

_hmm_instance: Any = None  # lazy HumanMouse singleton


def _get_hmm() -> Any:
    """Get or create the HumanMouse singleton, configured from human_delay."""
    global _hmm_instance
    hd = _exec_cfg()["human_delay"]
    if _hmm_instance is None or _hmm_instance.speed != hd:
        with _init_lock:
            # Double-check after acquiring lock to avoid duplicate creation.
            if _hmm_instance is None or _hmm_instance.speed != hd:
                from human_mouse_moves import HumanMouse
                _hmm_instance = HumanMouse(speed=max(hd, 0.1))
    return _hmm_instance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _exec_cfg() -> dict[str, Any]:
    """Returns the execution config section with safe defaults."""
    cfg = get_config().get("execution", {})
    return {
        "mouse_duration": float(cfg.get("mouse_duration", 0.15)),
        "type_interval": float(cfg.get("type_interval", 0.03)),
        "human_delay": float(cfg.get("human_delay", 1.0)),
        "typo_chance": float(cfg.get("typo_chance", 0.04)),
    }


def _hsleep(seconds: float) -> None:
    """Sleeps for *seconds* × human_delay. Skips entirely when multiplier is 0."""
    hd = _exec_cfg()["human_delay"]
    if hd <= 0:
        return
    time.sleep(seconds * hd)


def _jitter(value: float, pct: float = 0.15) -> float:
    """Applies ±pct random jitter to *value*."""
    return value * random.uniform(1.0 - pct, 1.0 + pct)


def _instant_move(x: int, y: int) -> None:
    """Instant teleport for speed/test mode (human_delay=0)."""
    with _lock:
        pyautogui.moveTo(x, y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def click(
    x: int,
    y: int,
    button: str = "left",
    radius: int = 8,
    bbox_w: int | None = None,
    bbox_h: int | None = None,
    dry_run: bool = False,
) -> None:
    """Click at (*x*, *y*) with human-like movement and targeting.

    When bbox dimensions are provided, ``human_mouse_moves`` handles
    click targeting (bivariate normal with shortfall bias + edge repulsion)
    and the full kinematic pipeline (ballistic + corrective sub-movements).

    Args:
        x: Target X coordinate (center of element, or bbox left if bbox given).
        y: Target Y coordinate (center of element, or bbox top if bbox given).
        button: ``'left'``, ``'right'``, or ``'middle'``.
        radius: Unused (kept for API compat). Targeting is bbox-driven.
        bbox_w: Element bounding box width.
        bbox_h: Element bounding box height.
        dry_run: Log the action without executing.
    """
    logger.debug(
        "click(%d, %d) button=%s bbox=%s",
        x, y, button,
        f"{bbox_w}x{bbox_h}" if bbox_w else "none",
    )
    if dry_run:
        return

    hd = _exec_cfg()["human_delay"]
    if hd <= 0:
        _instant_move(x, y)
        with _lock:
            pyautogui.click(button=button)
        return

    hmm = _get_hmm()
    with _lock:
        if bbox_w and bbox_h:
            # Pass bbox top-left + dimensions; HumanMouse handles targeting
            bx = x - bbox_w // 2
            by = y - bbox_h // 2
            hmm.click(bx, by, w=bbox_w, h=bbox_h, button=button)
        else:
            hmm.click(x, y, button=button)


def right_click(
    x: int,
    y: int,
    bbox_w: int | None = None,
    bbox_h: int | None = None,
    dry_run: bool = False,
) -> None:
    """Right-clicks at (*x*, *y*) with human-like movement."""
    click(x, y, button="right", bbox_w=bbox_w, bbox_h=bbox_h, dry_run=dry_run)


def double_click(
    x: int,
    y: int,
    bbox_w: int | None = None,
    bbox_h: int | None = None,
    dry_run: bool = False,
) -> None:
    """Double-clicks at (*x*, *y*) with human-like movement."""
    logger.debug("double_click(%d, %d)", x, y)
    if dry_run:
        return

    hd = _exec_cfg()["human_delay"]
    if hd <= 0:
        _instant_move(x, y)
        with _lock:
            pyautogui.doubleClick()
        return

    hmm = _get_hmm()
    with _lock:
        if bbox_w and bbox_h:
            bx = x - bbox_w // 2
            by = y - bbox_h // 2
            hmm.double_click(bx, by, w=bbox_w, h=bbox_h)
        else:
            hmm.double_click(x, y)


def drag(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration: float = 0.3,
    dry_run: bool = False,
) -> None:
    """Drags from (*x1*, *y1*) to (*x2*, *y2*) with human-like movement.

    Args:
        x1, y1: Start coordinates (center of drag source element).
        x2, y2: End coordinates (center of drop target).
        duration: Unused (kept for API compat). Timing is Fitts-driven.
        dry_run: Log without executing.
    """
    logger.debug("drag(%d,%d -> %d,%d)", x1, y1, x2, y2)
    if dry_run:
        return

    hd = _exec_cfg()["human_delay"]
    if hd <= 0:
        _instant_move(x1, y1)
        with _lock:
            pyautogui.mouseDown()
        _instant_move(x2, y2)
        with _lock:
            pyautogui.mouseUp()
        return

    # Use HumanMouse drag with a small synthetic bbox around start/end
    hmm = _get_hmm()
    elem_w, elem_h = 20.0, 20.0  # assume small grab area
    with _lock:
        hmm.drag(
            elem_x=x1 - elem_w / 2, elem_y=y1 - elem_h / 2,
            elem_w=elem_w, elem_h=elem_h,
            drop_x=float(x2), drop_y=float(y2),
        )


def _get_human_typer() -> Any:
    """Lazy-load the HumanTyper from the bundled human_typing package.

    Returns:
        HumanTyper instance configured from human_typing-main/config.yaml.
    """
    global _human_typer
    if _human_typer is not None:
        return _human_typer

    with _init_lock:
        # Double-check after acquiring lock to avoid duplicate creation.
        if _human_typer is not None:
            return _human_typer

        # Add the bundled human_typing package to path
        ht_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "human_typing-main")
        if ht_dir not in sys.path:
            sys.path.insert(0, ht_dir)

        from human_typer import HumanTyper
        _human_typer = HumanTyper(config_path=os.path.join(ht_dir, "config.yaml"))
        return _human_typer


_human_typer: Any = None


def type_text(
    text: str,
    interval: float | None = None,
    dry_run: bool = False,
) -> None:
    """Types *text* using the human_typing library for realistic cadence.

    Uses burst typing (fast within words, pauses between), QWERTY neighbor
    typos with immediate correction, and fatigue modeling.  When human_delay
    is 0, types instantly via pyautogui.

    Args:
        text: The string to type.
        interval: Unused (kept for API compat). Timing is WPM-driven.
        dry_run: Log without executing.
    """
    hd = _exec_cfg()["human_delay"]
    logger.debug("type_text(%r, hd=%.1f)", text[:30], hd)
    if dry_run:
        return

    if hd <= 0:
        # Instant mode — no human simulation
        with _lock:
            pyautogui.write(text, interval=0)
        return

    typer = _get_human_typer()
    with _lock:
        typer.type(text)


def scroll(
    x: int,
    y: int,
    direction: str,
    amount: int,
    dry_run: bool = False,
) -> None:
    """Scrolls at position (*x*, *y*) with human-like movement to target first.

    Args:
        direction: ``'up'``, ``'down'``, ``'left'``, or ``'right'``.
        amount: Number of scroll units.
        dry_run: Log without executing.
    """
    logger.debug("scroll(%d, %d, %s, %d)", x, y, direction, amount)
    if dry_run:
        return

    hd = _exec_cfg()["human_delay"]
    if hd <= 0:
        _instant_move(x, y)
    else:
        hmm = _get_hmm()
        with _lock:
            hmm.move_to(float(x), float(y))

    clicks = amount if direction in ("up", "right") else -amount
    with _lock:
        if direction in ("up", "down"):
            pyautogui.scroll(clicks, x, y)
        else:
            pyautogui.hscroll(clicks, x, y)
    _hsleep(random.uniform(0.08, 0.15))


def press_enter(dry_run: bool = False) -> None:
    """Press the Enter key with human-like timing.

    Args:
        dry_run: Log the action without executing.
    """
    logger.debug("press_enter()")
    if dry_run:
        return
    with _lock:
        pyautogui.press("enter")
    _hsleep(random.uniform(0.05, 0.1))


def hotkey(*keys: str, dry_run: bool = False) -> None:
    """Presses a keyboard shortcut with small inter-key pauses.

    Args:
        keys: Key names, e.g. ``hotkey('ctrl', 'c')``.
        dry_run: Log without executing.
    """
    logger.debug("hotkey(%s)", "+".join(keys))
    if dry_run:
        return

    with _lock:
        pyautogui.hotkey(*keys, interval=_jitter(0.02))
    _hsleep(random.uniform(0.05, 0.1))


def move_to_rest() -> None:
    """Move mouse to rest position at far-right edge of screen.

    After click-based actions the cursor may hover over an element, causing
    tooltips or highlight animations that confuse the next screenshot or
    detection pass.  This parks the cursor 2 px from the right edge at a
    slightly randomised vertical position so it is out of the way.
    """
    import mss

    with mss.mss() as sct:
        # Monitor 0 is the full virtual desktop (all monitors combined)
        mon = sct.monitors[0]
        rest_x = mon["left"] + mon["width"] - 2
        rest_y = mon["top"] + mon["height"] // 2 + random.randint(-50, 50)

    hd = _exec_cfg()["human_delay"]
    if hd <= 0:
        _instant_move(rest_x, rest_y)
    else:
        hmm = _get_hmm()
        with _lock:
            hmm.move_to(float(rest_x), float(rest_y))

    logger.debug("Mouse moved to rest position (%d, %d)", rest_x, rest_y)


# ---------------------------------------------------------------------------
# Prompt-user blocking mechanism
# ---------------------------------------------------------------------------

_prompt_response_event = threading.Event()
_prompt_response_text: str = ""
_prompt_lock = threading.Lock()


def prompt_user_blocking(
    question_text: str,
    screenshot_path: str | None = None,
    dry_run: bool = False,
    timeout: float | None = None,
) -> str:
    """Pause routine execution and wait for user/agent response via API.

    During replay, this blocks the executor thread until respond_to_prompt()
    is called (triggered by POST /respond endpoint).

    Args:
        question_text: The question to surface to the user/agent.
        screenshot_path: Optional screenshot path to include with the prompt.
        dry_run: If True, return empty string without blocking.
        timeout: Max seconds to wait for a response (None = wait forever).

    Returns:
        The user/agent response text.
    """
    global _prompt_response_text
    logger.info("prompt_user_blocking: %s", question_text[:80])
    if dry_run:
        logger.info("prompt_user dry-run: would block waiting for /respond")
        return ""

    # Clear any previous response — lock protects the text/event pair
    with _prompt_lock:
        _prompt_response_event.clear()
        _prompt_response_text = ""

    # Block until respond_to_prompt is called
    logger.info("Routine paused. Waiting for API /respond...")
    signaled = _prompt_response_event.wait(timeout=timeout)
    if not signaled:
        raise PromptTimeoutError(
            f"Prompt timed out after {timeout}s: {question_text[:50]}"
        )

    with _prompt_lock:
        response = _prompt_response_text
    logger.info("Prompt response received: %s", response[:80])
    return response


def respond_to_prompt(response_text: str) -> None:
    """Unblock a waiting prompt_user step with the given response.

    Called by the API /respond endpoint handler.

    Args:
        response_text: The user/agent response text.
    """
    global _prompt_response_text
    with _prompt_lock:
        _prompt_response_text = response_text
        _prompt_response_event.set()
    logger.info("Prompt responded: %s", response_text[:80])


def generate_ai_text(prompt: str, dry_run: bool = False) -> str:
    """Pause execution and request text from the calling agent.

    When a step has ai_generate_text=True, the text_to_type field is a
    description of what text is needed. This function blocks until the
    caller (an AI agent, API client, etc.) provides the actual text
    via the /respond endpoint.

    Args:
        prompt: Description of the text field (shown to caller).
        dry_run: If True, return placeholder text without blocking.

    Returns:
        The text provided by the caller.
    """
    if dry_run:
        logger.info("ai_generate_text dry-run: would block for caller input")
        return prompt  # Use placeholder during dry-run

    # Use the existing prompt_user mechanism to ask the caller
    question = f"Enter text for: {prompt}"
    logger.info("ai_generate_text: blocking for caller input — %s", prompt[:80])
    response = prompt_user_blocking(question)
    if response:
        logger.info("ai_generate_text: received %d chars from caller", len(response))
        return response

    logger.warning("ai_generate_text: empty response, using placeholder")
    return prompt


def select_all_extract(dry_run: bool = False) -> str:
    """Select all text and extract via clipboard, with VLM fallback.

    Sends Ctrl+A, Ctrl+C, reads clipboard. If clipboard unchanged,
    falls back to VLM screenshot analysis.

    Args:
        dry_run: Log without executing.

    Returns:
        Extracted text content (empty string on dry_run).
    """
    logger.debug("select_all_extract(dry_run=%s)", dry_run)
    if dry_run:
        return ""

    import pyperclip

    old_clipboard = ""
    try:
        old_clipboard = pyperclip.paste()
    except Exception:
        pass

    _mod = "command" if sys.platform == "darwin" else "ctrl"
    hotkey(_mod, "a")
    _hsleep(0.15)
    hotkey(_mod, "c")
    _hsleep(0.3)  # Wait for clipboard update

    new_clipboard = ""
    try:
        new_clipboard = pyperclip.paste()
    except Exception:
        pass

    if new_clipboard and new_clipboard != old_clipboard:
        return new_clipboard

    # Fallback: VLM screenshot analysis
    logger.info("Clipboard empty/unchanged, falling back to VLM")
    try:
        from core.capture import screenshot_full
        from core.vision import analyze_crop_array

        screenshot = screenshot_full()
        result = analyze_crop_array(
            screenshot, "Extract all visible text from this screen"
        )
        return result.get("text", "") if isinstance(result, dict) else ""
    except Exception as e:
        logger.warning("VLM fallback failed: %s", e)
        return ""
