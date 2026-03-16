"""Overlay package -- transparent fullscreen overlay for recording sessions."""

from __future__ import annotations

from recorder.overlay.capture_guard import capture_guard
from recorder.overlay.controller import OverlayController
from recorder.overlay.platform_linux import ensure_xcb_platform
from recorder.overlay.state import TRANSITIONS, STATE_COLORS, OverlayState
