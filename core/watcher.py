"""Background watcher for window title, URL, and pixel-diff changes.

Runs as a daemon thread. Emits WatcherEvent instances via a callback
when the active window title changes, the browser URL changes, or
a pixel-diff threshold is exceeded between poll cycles.

Thread-safe start/stop via threading.Event. The callback is invoked
without any held locks to prevent deadlocks.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Callable

from core.capture import get_window_title, pixel_diff, screenshot_full
from core.config import get_config
from core.types import WatcherEvent

logger = logging.getLogger(__name__)

# Type alias for the callback function
WatcherCallback = Callable[[WatcherEvent], Any]

# Module-level state
_watcher_thread: threading.Thread | None = None
_stop_event: threading.Event = threading.Event()
_running_lock: threading.Lock = threading.Lock()


def _get_active_url_safe() -> str | None:
    """Best-effort URL retrieval that never raises.

    Only works on Windows. Returns None on other platforms or on failure.
    """
    if sys.platform != "win32":
        return None
    try:
        from core.capture import get_active_url
        return get_active_url()
    except Exception:
        return None


def _poll_loop(
    callback: WatcherCallback,
    diff_threshold: float,
    poll_ms: int,
) -> None:
    """Main polling loop running in the daemon thread.

    Args:
        callback: Function to call with WatcherEvent on state change.
        diff_threshold: Pixel diff percentage (0.0-1.0) to trigger event.
        poll_ms: Milliseconds between poll cycles.
    """
    poll_interval = poll_ms / 1000.0

    # Initialize baseline state
    prev_title: str = ""
    prev_url: str | None = None
    prev_screenshot = None

    try:
        prev_title = get_window_title()
    except Exception as e:
        logger.debug("Initial window title fetch failed: %s", e)

    try:
        prev_url = _get_active_url_safe()
    except Exception as e:
        logger.debug("Initial URL fetch failed: %s", e)

    try:
        prev_screenshot = screenshot_full()
    except Exception as e:
        logger.debug("Initial screenshot failed: %s", e)

    logger.info(
        "Watcher started: poll_ms=%d, diff_threshold=%.2f",
        poll_ms,
        diff_threshold,
    )

    while not _stop_event.is_set():
        try:
            _poll_once(
                callback,
                diff_threshold,
                prev_title,
                prev_url,
                prev_screenshot,
            )
        except Exception as e:
            logger.warning("Watcher poll error (continuing): %s", e)

        # Update baseline for next cycle
        try:
            new_title = get_window_title()
            if new_title != prev_title:
                prev_title = new_title
        except Exception:
            pass

        try:
            new_url = _get_active_url_safe()
            if new_url != prev_url:
                prev_url = new_url
        except Exception:
            pass

        try:
            prev_screenshot = screenshot_full()
        except Exception:
            pass

        # Sleep in small increments for responsive shutdown
        _stop_event.wait(timeout=poll_interval)

    logger.info("Watcher stopped")


def _poll_once(
    callback: WatcherCallback,
    diff_threshold: float,
    prev_title: str,
    prev_url: str | None,
    prev_screenshot: Any,
) -> None:
    """Execute a single poll cycle, emitting events as needed.

    Events are emitted OUTSIDE any held locks to prevent deadlocks.

    Args:
        callback: Function to call with WatcherEvent.
        diff_threshold: Pixel diff threshold.
        prev_title: Previous window title.
        prev_url: Previous browser URL.
        prev_screenshot: Previous screenshot numpy array.
    """
    events_to_emit: list[WatcherEvent] = []

    # Check window title change
    try:
        current_title = get_window_title()
        if current_title and current_title != prev_title:
            events_to_emit.append(
                WatcherEvent(
                    event_type="window_changed",
                    old_value=prev_title,
                    new_value=current_title,
                )
            )
    except Exception as e:
        logger.debug("Window title check failed: %s", e)

    # Check URL change (browser only, best-effort)
    try:
        current_url = _get_active_url_safe()
        if current_url and current_url != prev_url:
            events_to_emit.append(
                WatcherEvent(
                    event_type="url_changed",
                    old_value=prev_url,
                    new_value=current_url,
                )
            )
    except Exception as e:
        logger.debug("URL check failed: %s", e)

    # Check pixel diff
    if prev_screenshot is not None:
        try:
            current_screenshot = screenshot_full()
            diff_pct = pixel_diff(prev_screenshot, current_screenshot)

            if diff_pct > diff_threshold:
                events_to_emit.append(
                    WatcherEvent(
                        event_type="pixel_diff_exceeded",
                        old_value=None,
                        new_value=None,
                        diff_pct=diff_pct,
                    )
                )
        except Exception as e:
            logger.debug("Pixel diff check failed: %s", e)

    # Emit all collected events OUTSIDE any locks
    for event in events_to_emit:
        try:
            callback(event)
        except Exception as e:
            logger.error("Watcher callback raised exception: %s", e)


def start_watching(
    callback: WatcherCallback,
    diff_threshold: float | None = None,
    poll_ms: int | None = None,
) -> None:
    """Starts the background watcher thread.

    Args:
        callback: Function called with a WatcherEvent on each state change.
                 Must be thread-safe and non-blocking.
        diff_threshold: Pixel diff percentage to trigger pixel_diff_exceeded event.
                       Defaults to config execution.pixel_diff_threshold.
        poll_ms: Milliseconds between poll cycles. Defaults to 200.

    Raises:
        RuntimeError: If the watcher is already running.
    """
    global _watcher_thread

    with _running_lock:
        if _watcher_thread is not None and _watcher_thread.is_alive():
            raise RuntimeError("Watcher is already running. Call stop_watching() first.")

        # Resolve defaults from config
        if diff_threshold is None:
            config = get_config()
            diff_threshold = config.get("execution", {}).get(
                "pixel_diff_threshold", 0.08
            )
        if poll_ms is None:
            poll_ms = 200

        _stop_event.clear()

        _watcher_thread = threading.Thread(
            target=_poll_loop,
            args=(callback, diff_threshold, poll_ms),
            name="ocsd-watcher",
            daemon=True,
        )
        _watcher_thread.start()


def stop_watching() -> None:
    """Stops the background watcher thread.

    Blocks until the thread exits (up to 2 seconds).
    Safe to call even if the watcher is not running.
    """
    global _watcher_thread

    with _running_lock:
        if _watcher_thread is None or not _watcher_thread.is_alive():
            _watcher_thread = None
            return

        _stop_event.set()

    # Join outside the lock so the thread can finish cleanly
    if _watcher_thread is not None:
        _watcher_thread.join(timeout=2.0)
        if _watcher_thread.is_alive():
            logger.warning("Watcher thread did not exit cleanly within 2s")

    with _running_lock:
        _watcher_thread = None


def is_watching() -> bool:
    """Returns True if the watcher thread is currently running."""
    with _running_lock:
        return _watcher_thread is not None and _watcher_thread.is_alive()
