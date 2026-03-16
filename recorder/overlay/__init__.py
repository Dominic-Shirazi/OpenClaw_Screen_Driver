"""Overlay package -- transparent fullscreen overlay for recording sessions."""

from __future__ import annotations

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.capture_guard import capture_guard
from recorder.overlay.controller import OverlayController
from recorder.overlay.donut_cloud_layer import DonutCloudLayer
from recorder.overlay.platform_linux import ensure_xcb_platform
from recorder.overlay.scan_layer import ScanLayer, ScanPhase
from recorder.overlay.shimmer_layer import ShimmerLayer
from recorder.overlay.state import TRANSITIONS, STATE_COLORS, OverlayState
