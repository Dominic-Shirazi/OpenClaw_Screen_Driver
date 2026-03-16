"""Animated border shimmer glow replacing the static BorderLayer.

Renders a rotating conical gradient sweep along the screen border
that changes color and speed with overlay state.  The shimmer wave
retreats from the mouse cursor and registered avoidance rects with
a smooth organic falloff (smoothstep).
"""

from __future__ import annotations

import logging
import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QConicalGradient, QPainter, QPainterPath
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

from recorder.overlay.state import STATE_COLORS, OverlayState

logger = logging.getLogger(__name__)

# Border width per state (user decision: 12-20px range)
_BORDER_READY: int = 16
_BORDER_RECORDING: int = 12
_RETREAT_RADIUS: float = 200.0

# Number of segments to divide the border into for mouse-retreat alpha modulation
_SEGMENT_COUNT: int = 32


class ShimmerLayer(QGraphicsObject):
    """Animated border shimmer glow indicating overlay state.

    Renders a rotating QConicalGradient clipped to the screen border
    strip.  The gradient rotates continuously, with speed and color
    determined by the current OverlayState.  The shimmer intensity
    is attenuated near the mouse cursor and registered avoidance rects
    using a smoothstep distance falloff.

    Args:
        screen_w: Logical screen width in pixels.
        screen_h: Logical screen height in pixels.
    """

    def __init__(self, screen_w: int, screen_h: int) -> None:
        super().__init__()
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._phase: float = 0.0  # 0.0 to 1.0, maps to 0-360 degrees
        self._loop_duration: float = 5.0  # seconds for one full rotation
        self._border_width: float = _BORDER_READY
        self._target_border_width: float = _BORDER_READY
        self._base_color: QColor = QColor(50, 200, 50)  # Green default
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
            self._target_border_width = _BORDER_RECORDING
        else:
            self._loop_duration = 5.0
            self._target_border_width = _BORDER_READY

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

        Called by AnimationClock each tick.  Advances the gradient
        rotation phase and smoothly transitions border width.

        Args:
            dt: Elapsed seconds since last tick.
        """
        self._phase = (self._phase + dt / self._loop_duration) % 1.0
        # Smooth border width transition
        self._border_width += (
            (self._target_border_width - self._border_width)
            * min(1.0, dt * 4.0)
        )
        self.update()  # Schedule repaint (safe: called from tick, NOT from paint)

    def _shimmer_intensity_at(self, px: float, py: float) -> float:
        """Calculate shimmer intensity at a point based on mouse distance.

        Returns 1.0 far from mouse/avoidance rects, 0.0 at the mouse
        position, with smoothstep falloff in between.

        Args:
            px: Point X coordinate.
            py: Point Y coordinate.

        Returns:
            Intensity from 0.0 (fully retreated) to 1.0 (full shimmer).
        """
        # Distance from mouse
        dist = math.hypot(px - self._mouse_pos.x(), py - self._mouse_pos.y())

        # Distance from avoidance rects
        for rect in self._avoidance_rects:
            # Closest point on rect to (px, py)
            cx = max(rect.left(), min(px, rect.right()))
            cy = max(rect.top(), min(py, rect.bottom()))
            rect_dist = math.hypot(px - cx, py - cy)
            dist = min(dist, rect_dist)

        if dist >= _RETREAT_RADIUS:
            return 1.0
        # Smoothstep: t * t * (3 - 2t)
        t = dist / _RETREAT_RADIUS
        return t * t * (3.0 - 2.0 * t)

    def _build_border_path(self) -> QPainterPath:
        """Build a QPainterPath representing the border strip.

        Returns:
            Path covering only the border region (outer - inner rect).
        """
        bw = self._border_width
        outer = QPainterPath()
        outer.addRect(QRectF(0, 0, self._screen_w, self._screen_h))
        inner = QPainterPath()
        inner.addRect(QRectF(bw, bw, self._screen_w - 2 * bw, self._screen_h - 2 * bw))
        return outer - inner

    def _sample_border_points(self) -> list[tuple[float, float]]:
        """Generate sample points around the border perimeter.

        Returns:
            List of (x, y) tuples at evenly spaced positions around
            the screen perimeter.
        """
        points: list[tuple[float, float]] = []
        w = float(self._screen_w)
        h = float(self._screen_h)
        bw = self._border_width / 2.0  # Sample at mid-border

        # Distribute _SEGMENT_COUNT points around perimeter
        perimeter = 2.0 * (w + h)
        for i in range(_SEGMENT_COUNT):
            frac = i / _SEGMENT_COUNT
            dist_along = frac * perimeter
            if dist_along < w:
                # Top edge
                points.append((dist_along, bw))
            elif dist_along < w + h:
                # Right edge
                points.append((w - bw, dist_along - w))
            elif dist_along < 2 * w + h:
                # Bottom edge
                points.append((2 * w + h - dist_along, h - bw))
            else:
                # Left edge
                points.append((bw, perimeter - dist_along))
        return points

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the shimmer border with rotating gradient and mouse retreat.

        CRITICAL: No self.update() call in this method.

        Args:
            painter: The QPainter to draw with.
            option: Style options (unused).
            widget: Target widget (unused).
        """
        bw = self._border_width
        if bw <= 0:
            return

        rect = QRectF(0, 0, self._screen_w, self._screen_h)
        cx = rect.center().x()
        cy = rect.center().y()

        # Build the rotating conical gradient
        angle = self._phase * 360.0
        gradient = QConicalGradient(QPointF(cx, cy), angle)

        bright = QColor(self._base_color)
        bright.setAlpha(220)
        dim = QColor(self._base_color)
        dim.setAlpha(40)

        gradient.setColorAt(0.0, bright)
        gradient.setColorAt(0.25, dim)
        gradient.setColorAt(0.5, bright)
        gradient.setColorAt(0.75, dim)
        gradient.setColorAt(1.0, bright)

        # Clip to border strip
        border_path = self._build_border_path()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipPath(border_path)
        painter.fillRect(rect, gradient)

        # Mouse retreat: darken segments near mouse/avoidance rects
        # Paint semi-transparent black over low-intensity regions
        points = self._sample_border_points()
        perimeter = 2.0 * (self._screen_w + self._screen_h)
        seg_len = perimeter / _SEGMENT_COUNT

        for i, (px, py) in enumerate(points):
            intensity = self._shimmer_intensity_at(px, py)
            if intensity >= 0.99:
                continue  # Full shimmer, no darkening needed
            # Determine the darkening alpha (higher = more retreat)
            dark_alpha = int((1.0 - intensity) * 200)
            if dark_alpha < 5:
                continue

            # Build a small rect around this segment point
            # The rect is aligned to the border edge
            half_seg = seg_len / 2.0 + 2.0  # slight overlap
            seg_rect = QRectF(px - half_seg, py - half_seg, half_seg * 2, half_seg * 2)

            dark = QColor(0, 0, 0, dark_alpha)
            painter.fillRect(seg_rect, dark)

        painter.restore()
