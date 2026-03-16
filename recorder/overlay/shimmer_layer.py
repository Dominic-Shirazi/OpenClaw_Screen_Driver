"""Animated border shimmer glow replacing the static BorderLayer.

Renders soft lights that shine INWARD from behind the screen bezel.
Each light source is positioned off-screen with its gradient center
outside the visible area, so only the inner spill is seen.  Lights
vary in size, overlap additively via QPainter.CompositionMode_Plus
so they blend rather than stack, and sweep continuously around the
perimeter.
"""

from __future__ import annotations

import logging
import math
import random

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QRadialGradient
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

from recorder.overlay.state import STATE_COLORS, OverlayState

logger = logging.getLogger(__name__)

# How far inward the glow bleeds (base radius before per-light variance)
_GLOW_READY: float = 140.0
_GLOW_RECORDING: float = 120.0
_RETREAT_RADIUS: float = 250.0

# How far off-screen the light centers sit (behind the bezel)
_OFFSCREEN_DEPTH: float = 80.0

# Number of light sources around the perimeter
_LIGHT_COUNT: int = 48
# Fraction of perimeter that is lit at once
_ACTIVE_SPAN: float = 0.35

# Seed for deterministic per-light size variance
_SIZE_SEED: int = 42


class ShimmerLayer(QGraphicsObject):
    """Animated border shimmer glow indicating overlay state.

    Renders soft radial gradient blobs along the screen perimeter,
    creating a wide feathered glow that bleeds inward from the edges.
    The glow sweep rotates continuously, with speed and color determined
    by the current OverlayState.  Intensity is attenuated near the mouse
    cursor and registered avoidance rects using smoothstep falloff.

    Args:
        screen_w: Logical screen width in pixels.
        screen_h: Logical screen height in pixels.
    """

    def __init__(self, screen_w: int, screen_h: int) -> None:
        super().__init__()
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._phase: float = 0.0  # 0.0 to 1.0, maps to position around perimeter
        self._loop_duration: float = 5.0  # seconds for one full rotation
        self._glow_radius: float = _GLOW_READY
        self._target_glow_radius: float = _GLOW_READY
        self._base_color: QColor = QColor(50, 200, 50)  # Green default
        self._alpha_mult: float = 0.40  # Peak alpha multiplier
        self._mouse_pos: QPointF = QPointF(-1000, -1000)  # Offscreen initially
        self._avoidance_rects: list[QRectF] = []
        self.setZValue(10)  # Same as old BorderLayer
        self.setFlag(
            QGraphicsObject.GraphicsItemFlag.ItemHasNoContents, False
        )

    def boundingRect(self) -> QRectF:
        """Return full screen rect as bounding box.

        Returns:
            QRectF covering the entire screen area.
        """
        return QRectF(0, 0, self._screen_w, self._screen_h)

    def set_state(self, state: OverlayState) -> None:
        """Update shimmer appearance based on overlay state.

        Args:
            state: The current overlay state.
        """
        r, g, b, _a = STATE_COLORS[state]
        self._base_color = QColor(r, g, b)
        if state == OverlayState.RECORDING:
            self._loop_duration = 2.0
            self._target_glow_radius = _GLOW_RECORDING
            self._alpha_mult = 0.55
        else:
            self._loop_duration = 5.0
            self._target_glow_radius = _GLOW_READY
            self._alpha_mult = 0.40

    def set_mouse_pos(self, x: float, y: float) -> None:
        """Store the current mouse position for retreat calculation.

        Args:
            x: Mouse X coordinate in scene space.
            y: Mouse Y coordinate in scene space.
        """
        self._mouse_pos = QPointF(x, y)

    def set_avoidance_rects(self, rects: list[QRectF]) -> None:
        """Set UI element rects that the shimmer should avoid.

        Args:
            rects: List of QRectF bounding boxes for active UI elements.
        """
        self._avoidance_rects = list(rects)

    def tick(self, dt: float) -> None:
        """Advance the shimmer animation by one frame.

        Called by AnimationClock each tick.  Advances the sweep
        position and smoothly transitions glow radius.

        Args:
            dt: Elapsed seconds since last tick.
        """
        self._phase = (self._phase + dt / self._loop_duration) % 1.0
        # Smooth glow radius transition
        self._glow_radius += (
            (self._target_glow_radius - self._glow_radius)
            * min(1.0, dt * 4.0)
        )
        self.update()  # Schedule repaint (safe: called from tick, NOT from paint)

    def _retreat_factor(self, px: float, py: float) -> float:
        """Calculate retreat factor at a point based on mouse/rect distance.

        Returns 1.0 far from mouse/avoidance rects, 0.0 at the mouse
        position, with smoothstep falloff in between.

        Args:
            px: Point X coordinate.
            py: Point Y coordinate.

        Returns:
            Factor from 0.0 (fully retreated) to 1.0 (full glow).
        """
        dist = math.hypot(px - self._mouse_pos.x(), py - self._mouse_pos.y())

        for rect in self._avoidance_rects:
            cx = max(rect.left(), min(px, rect.right()))
            cy = max(rect.top(), min(py, rect.bottom()))
            rect_dist = math.hypot(px - cx, py - cy)
            dist = min(dist, rect_dist)

        if dist >= _RETREAT_RADIUS:
            return 1.0
        t = dist / _RETREAT_RADIUS
        return t * t * (3.0 - 2.0 * t)

    def _build_lights(self) -> list[tuple[float, float, float, float, float]]:
        """Build the fixed light source layout around the perimeter.

        Each light has its center pushed OFF-SCREEN by _OFFSCREEN_DEPTH
        so only the inner spill is visible (light shining inward).
        Sizes are varied deterministically per light.

        Returns:
            List of (center_x, center_y, radius, edge_x, edge_y, frac)
            tuples.  center is off-screen, edge is the on-screen point.
        """
        lights: list[tuple[float, float, float, float, float]] = []
        w = float(self._screen_w)
        h = float(self._screen_h)
        perimeter = 2.0 * (w + h)
        rng = random.Random(_SIZE_SEED)

        for i in range(_LIGHT_COUNT):
            frac = i / _LIGHT_COUNT
            d = frac * perimeter

            # Point on the screen edge and the outward normal direction
            if d < w:
                ex, ey = d, 0.0
                nx, ny = 0.0, -1.0  # points up (off top edge)
            elif d < w + h:
                ex, ey = w, d - w
                nx, ny = 1.0, 0.0   # points right (off right edge)
            elif d < 2 * w + h:
                ex, ey = 2 * w + h - d, h
                nx, ny = 0.0, 1.0   # points down (off bottom edge)
            else:
                ex, ey = 0.0, perimeter - d
                nx, ny = -1.0, 0.0  # points left (off left edge)

            # Push center off-screen along the outward normal
            cx = ex + nx * _OFFSCREEN_DEPTH
            cy = ey + ny * _OFFSCREEN_DEPTH

            # Vary size: 0.6x to 1.5x base radius
            size_mult = 0.6 + rng.random() * 0.9

            lights.append((cx, cy, size_mult, ex, ey, frac))
        return lights

    def _sweep_brightness(self, frac: float) -> float:
        """Calculate brightness at a perimeter position based on sweep phase.

        Args:
            frac: Position around perimeter (0.0 to 1.0).

        Returns:
            Brightness factor 0.0 to 1.0.
        """
        delta = abs(frac - self._phase)
        if delta > 0.5:
            delta = 1.0 - delta

        if delta > _ACTIVE_SPAN:
            return 0.0
        t = delta / _ACTIVE_SPAN
        return 0.5 * (1.0 + math.cos(t * math.pi))

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint lights shining inward from behind the screen bezel.

        Light centers are off-screen; only the inner spill is visible.
        CompositionMode_Plus blends overlapping lights additively so
        they merge rather than stack.

        CRITICAL: No self.update() call in this method.

        Args:
            painter: The QPainter to draw with.
            option: Style options (unused).
            widget: Target widget (unused).
        """
        glow_r = self._glow_radius
        if glow_r <= 0:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        # Additive blending: overlapping lights merge into brighter glow
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        for cx, cy, size_mult, ex, ey, frac in self._build_lights():
            brightness = self._sweep_brightness(frac)
            if brightness < 0.01:
                continue

            # Retreat check at the on-screen edge point
            retreat = self._retreat_factor(ex, ey)
            if retreat < 0.01:
                continue

            # Effective radius for this light (varied size)
            r = glow_r * size_mult + _OFFSCREEN_DEPTH

            # Combined intensity — capped lower for additive blending
            peak_alpha = brightness * retreat * self._alpha_mult
            if peak_alpha < 0.01:
                continue

            center = QPointF(cx, cy)
            gradient = QRadialGradient(center, r)

            core = QColor(self._base_color)
            core.setAlphaF(min(peak_alpha, 1.0))
            mid = QColor(self._base_color)
            mid.setAlphaF(min(peak_alpha * 0.35, 1.0))
            edge = QColor(self._base_color)
            edge.setAlphaF(0.0)

            gradient.setColorAt(0.0, core)
            gradient.setColorAt(0.35, mid)
            gradient.setColorAt(1.0, edge)

            painter.setBrush(gradient)
            painter.drawEllipse(center, r, r)

        painter.restore()
