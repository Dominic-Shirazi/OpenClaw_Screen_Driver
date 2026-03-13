"""Mouse and keyboard automation with realistic human-like behavior.

All mouse movements follow cubic Bézier curves with random control points.
Click targets get a "doughnut" offset (never dead center, never outside
element bounds). Typing simulates per-word speed variation, rhythm jitter,
and occasional typos with immediate backspace correction.

All timing is scaled by the ``human_delay`` multiplier from config:
  0   → instant (skip all sleeps — speed/test mode)
  1.0 → normal human speed
  5.0 → very slow (useful for watching/debugging)

Thread-safe: every PyAutoGUI call is serialized through a lock.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Any

import pyautogui

from core.config import get_config

logger = logging.getLogger(__name__)

# PyAutoGUI safety: moving mouse to (0,0) aborts. Keep this ON.
pyautogui.FAILSAFE = True
# Disable default pause between actions (we handle timing ourselves).
pyautogui.PAUSE = 0

# Thread lock — PyAutoGUI is NOT thread-safe.
_lock = threading.Lock()

# QWERTY physical-neighbor map for typo simulation.
# Each key maps to adjacent keys a human finger might accidentally hit.
_ADJACENT_KEYS: dict[str, str] = {
    "q": "wa12", "w": "qeas23", "e": "wrsd34", "r": "etdf45",
    "t": "ryfg56", "y": "tugh67", "u": "yijh78", "i": "uokj89",
    "o": "iplk90", "p": "ol0",
    "a": "qwsz", "s": "wedaxz", "d": "erfscx", "f": "rtgdcv",
    "g": "tyhfvb", "h": "yujgbn", "j": "uikhbn", "k": "ioljm",
    "l": "opk",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
    "1": "2q", "2": "13wq", "3": "24ew", "4": "35re", "5": "46tr",
    "6": "57yt", "7": "68uy", "8": "79iu", "9": "80oi", "0": "9p",
}


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


def _doughnut_offset(radius: int = 8) -> tuple[int, int]:
    """Returns a (dx, dy) offset inside a fuzzy doughnut around (0, 0).

    The distribution peaks at ~40 % of the radius, never hits dead center
    (min 1 px away), and never exceeds *radius*.

    Args:
        radius: Maximum distance from center in pixels.

    Returns:
        Integer (dx, dy) offset to add to the click target.
    """
    angle = random.uniform(0, 2 * math.pi)
    dist = random.gauss(radius * 0.4, radius * 0.25)
    dist = max(1.0, min(float(radius), abs(dist)))
    dx = int(round(dist * math.cos(angle)))
    dy = int(round(dist * math.sin(angle)))
    # Guarantee at least 1 px offset
    if dx == 0 and dy == 0:
        dx = random.choice([-1, 1])
    return dx, dy


def _bezier_point(
    t: float,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> tuple[float, float]:
    """Evaluates a cubic Bézier curve at parameter *t* ∈ [0, 1]."""
    u = 1.0 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def _bezier_move(target_x: int, target_y: int, duration: float | None = None) -> None:
    """Moves the mouse to (*target_x*, *target_y*) along a cubic Bézier path.

    Two random control points are placed ±10–30 % of the travel distance
    away from the straight line, producing a natural arc.

    Args:
        target_x: Destination X.
        target_y: Destination Y.
        duration: Total movement time in seconds (before human_delay scaling).
    """
    cfg = _exec_cfg()
    if duration is None:
        duration = cfg["mouse_duration"]
    hd = cfg["human_delay"]

    with _lock:
        sx, sy = pyautogui.position()

    dist = math.hypot(target_x - sx, target_y - sy)
    if dist < 2:
        with _lock:
            pyautogui.moveTo(target_x, target_y)
        return

    # Random control-point offset perpendicular to the line
    def _cp_off() -> float:
        return random.uniform(0.1, 0.3) * dist * random.choice([-1, 1])

    p0 = (float(sx), float(sy))
    p1 = (sx + _cp_off(), sy + _cp_off())
    p2 = (target_x + _cp_off(), target_y + _cp_off())
    p3 = (float(target_x), float(target_y))

    steps = random.randint(20, 40)
    step_sleep = (duration * hd) / steps if hd > 0 else 0

    for i in range(1, steps + 1):
        t = i / steps
        bx, by = _bezier_point(t, p0, p1, p2, p3)
        with _lock:
            pyautogui.moveTo(int(bx), int(by))
        if step_sleep > 0:
            time.sleep(step_sleep)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def click(
    x: int,
    y: int,
    button: str = "left",
    radius: int = 8,
    duration: float | None = None,
    dry_run: bool = False,
) -> None:
    """Clicks at (*x*, *y*) with human-like Bézier movement and doughnut offset.

    Args:
        x: Target X coordinate (center of element).
        y: Target Y coordinate (center of element).
        button: ``'left'``, ``'right'``, or ``'middle'``.
        radius: Doughnut offset radius in pixels. Derive from element size
                when available (e.g. min(w, h) // 4).
        duration: Mouse-move duration in seconds (pre-human_delay scaling).
        dry_run: Log the action without executing.
    """
    dx, dy = _doughnut_offset(radius)
    tx, ty = x + dx, y + dy
    logger.debug(
        "click(%d, %d) → offset(%d, %d) button=%s", x, y, tx, ty, button,
    )
    if dry_run:
        return

    _bezier_move(tx, ty, duration)
    with _lock:
        pyautogui.click(button=button)
    _hsleep(random.uniform(0.03, 0.08))


def right_click(x: int, y: int, dry_run: bool = False) -> None:
    """Right-clicks at (*x*, *y*) with human-like movement."""
    click(x, y, button="right", dry_run=dry_run)


def double_click(x: int, y: int, dry_run: bool = False) -> None:
    """Double-clicks at (*x*, *y*) with human-like movement."""
    logger.debug("double_click(%d, %d)", x, y)
    if dry_run:
        return

    dx, dy = _doughnut_offset(8)
    _bezier_move(x + dx, y + dy)
    with _lock:
        pyautogui.doubleClick()
    _hsleep(random.uniform(0.03, 0.08))


def drag(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration: float = 0.3,
    dry_run: bool = False,
) -> None:
    """Drags from (*x1*, *y1*) to (*x2*, *y2*) using Bézier paths.

    Args:
        x1, y1: Start coordinates.
        x2, y2: End coordinates.
        duration: Drag travel time in seconds (pre-human_delay scaling).
        dry_run: Log without executing.
    """
    logger.debug("drag(%d,%d → %d,%d)", x1, y1, x2, y2)
    if dry_run:
        return

    _bezier_move(x1, y1)
    with _lock:
        pyautogui.mouseDown()
    _hsleep(random.uniform(0.08, 0.15))
    _bezier_move(x2, y2, duration)
    with _lock:
        pyautogui.mouseUp()
    _hsleep(random.uniform(0.03, 0.08))


def type_text(
    text: str,
    interval: float | None = None,
    dry_run: bool = False,
) -> None:
    """Types *text* with human-like rhythm, speed variation, and typo simulation.

    Each word gets a slightly different overall speed. Within a word,
    per-character timing varies with Gaussian jitter. Spaces get an extra
    pause. With probability ``typo_chance``, a wrong adjacent key is typed,
    immediately backspaced, and the correct key follows.

    Args:
        text: The string to type.
        interval: Base per-character interval (overrides config).
        dry_run: Log without executing.
    """
    cfg = _exec_cfg()
    base = interval if interval is not None else cfg["type_interval"]
    typo_chance = cfg["typo_chance"]
    hd = cfg["human_delay"]

    logger.debug("type_text(%r, interval=%.3f, typo=%.2f)", text[:30], base, typo_chance)
    if dry_run:
        return

    words = text.split(" ")
    for wi, word in enumerate(words):
        # Per-word speed variation
        word_speed = base * max(0.3, random.gauss(1.0, 0.15))

        for char in word:
            # Typo simulation
            lower = char.lower()
            if (
                hd > 0
                and random.random() < typo_chance
                and lower in _ADJACENT_KEYS
            ):
                wrong = random.choice(_ADJACENT_KEYS[lower])
                with _lock:
                    pyautogui.write(wrong, interval=0)
                _hsleep(random.uniform(0.05, 0.15))
                with _lock:
                    pyautogui.press("backspace")
                _hsleep(random.uniform(0.02, 0.08))

            with _lock:
                pyautogui.write(char, interval=0)

            # Per-character rhythm jitter
            char_sleep = word_speed * max(0.1, random.gauss(1.0, 0.2))
            _hsleep(char_sleep)

        # Space between words (except after last word)
        if wi < len(words) - 1:
            with _lock:
                pyautogui.write(" ", interval=0)
            _hsleep(random.uniform(0.05, 0.15))


def scroll(
    x: int,
    y: int,
    direction: str,
    amount: int,
    dry_run: bool = False,
) -> None:
    """Scrolls at position (*x*, *y*) with Bézier movement to target first.

    Args:
        direction: ``'up'``, ``'down'``, ``'left'``, or ``'right'``.
        amount: Number of scroll units.
        dry_run: Log without executing.
    """
    logger.debug("scroll(%d, %d, %s, %d)", x, y, direction, amount)
    if dry_run:
        return

    _bezier_move(x, y)
    clicks = amount if direction in ("up", "right") else -amount
    with _lock:
        if direction in ("up", "down"):
            pyautogui.scroll(clicks, x, y)
        else:
            pyautogui.hscroll(clicks, x, y)
    _hsleep(random.uniform(0.08, 0.15))


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
