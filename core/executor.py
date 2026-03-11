"""Mouse and keyboard automation with anti-bot measures.

All mouse movements use easing curves with randomized parameters.
Timing includes ±15% jitter to simulate human behavior.
Thread-safe: all PyAutoGUI calls are serialized through a lock.
"""

from __future__ import annotations

import random
import threading
import time
import logging

import pyautogui

from core.config import get_config

logger = logging.getLogger(__name__)

# PyAutoGUI safety: moving mouse to (0,0) aborts. Keep this ON.
pyautogui.FAILSAFE = True
# Disable default pause between actions (we handle timing ourselves)
pyautogui.PAUSE = 0

# Thread lock — PyAutoGUI is NOT thread-safe
_lock = threading.Lock()

# Available easing functions for randomization
_EASING_FUNCTIONS = [
    pyautogui.easeInOutQuad,
    pyautogui.easeInOutCubic,
    pyautogui.easeInOutSine,
]


def _jitter(value: float, pct: float = 0.15) -> float:
    """Apply ±pct random jitter to a value."""
    factor = random.uniform(1.0 - pct, 1.0 + pct)
    return value * factor


def _get_easing(easing: str = "easeInOutQuad"):
    """Resolve easing string to function. 'random' picks randomly."""
    if easing == "random":
        return random.choice(_EASING_FUNCTIONS)
    return getattr(pyautogui, easing, pyautogui.easeInOutQuad)


def click(
    x: int,
    y: int,
    button: str = "left",
    easing: str = "easeInOutQuad",
    duration: float | None = None,
    dry_run: bool = False,
) -> None:
    """Click at (x, y) with human-like mouse movement.

    Args:
        x, y: Screen coordinates.
        button: 'left', 'right', or 'middle'.
        easing: Easing function name or 'random'.
        duration: Mouse move duration in seconds (jittered). None uses config default.
        dry_run: Log without executing.
    """
    if duration is None:
        duration = get_config()["execution"]["mouse_duration"]
    duration = _jitter(duration)
    ease_fn = _get_easing(easing)

    logger.debug("click(%d, %d, button=%s, duration=%.3f)", x, y, button, duration)
    if dry_run:
        return

    with _lock:
        pyautogui.click(x, y, button=button, duration=duration, tween=ease_fn)


def right_click(x: int, y: int, dry_run: bool = False) -> None:
    """Right-click at (x, y)."""
    click(x, y, button="right", dry_run=dry_run)


def double_click(x: int, y: int, dry_run: bool = False) -> None:
    """Double-click at (x, y) with human-like movement."""
    cfg = get_config()["execution"]
    duration = _jitter(cfg["mouse_duration"])
    ease_fn = _get_easing("random")

    logger.debug("double_click(%d, %d)", x, y)
    if dry_run:
        return

    with _lock:
        pyautogui.doubleClick(x, y, duration=duration, tween=ease_fn)


def drag(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration: float = 0.3,
    dry_run: bool = False,
) -> None:
    """Drag from (x1, y1) to (x2, y2)."""
    duration = _jitter(duration)
    ease_fn = _get_easing("random")

    logger.debug("drag(%d,%d → %d,%d, duration=%.3f)", x1, y1, x2, y2, duration)
    if dry_run:
        return

    with _lock:
        pyautogui.moveTo(x1, y1, duration=_jitter(0.1), tween=ease_fn)
        time.sleep(_jitter(0.05))
        pyautogui.drag(
            x2 - x1, y2 - y1,
            duration=duration,
            button="left",
            tween=ease_fn,
        )


def type_text(
    text: str,
    interval: float | None = None,
    dry_run: bool = False,
) -> None:
    """Type text with human-like per-character timing variation.

    Each character gets its own jittered interval to avoid
    uniform typing patterns that trigger bot detection.
    """
    if interval is None:
        interval = get_config()["execution"]["type_interval"]

    logger.debug("type_text(%r, interval=%.3f)", text[:30], interval)
    if dry_run:
        return

    with _lock:
        for char in text:
            pyautogui.write(char, interval=0)
            time.sleep(_jitter(interval))


def scroll(
    x: int,
    y: int,
    direction: str,
    amount: int,
    dry_run: bool = False,
) -> None:
    """Scroll at position (x, y).

    Args:
        direction: 'up', 'down', 'left', or 'right'.
        amount: Number of scroll units.
    """
    clicks = amount if direction in ("up", "right") else -amount

    logger.debug("scroll(%d, %d, %s, %d)", x, y, direction, amount)
    if dry_run:
        return

    with _lock:
        pyautogui.moveTo(x, y, duration=_jitter(0.08))
        if direction in ("up", "down"):
            pyautogui.scroll(clicks, x, y)
        else:
            pyautogui.hscroll(clicks, x, y)
        time.sleep(_jitter(0.05))


def hotkey(*keys: str, dry_run: bool = False) -> None:
    """Press a keyboard shortcut with small pauses between keys.

    Args:
        keys: Key names, e.g. hotkey('ctrl', 'c').
    """
    logger.debug("hotkey(%s)", "+".join(keys))
    if dry_run:
        return

    with _lock:
        pyautogui.hotkey(*keys, interval=_jitter(0.02))
