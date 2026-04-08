"""Animated edge glow: ocean waves lapping at the screen edges.

Uses OpenSimplex noise for organic, natural wave shapes instead of
sine waves.  Multiple noise octaves create realistic grouping where
small waves merge into larger swells.  Waves surge inward/outward
(not laterally) because time drives depth, not position.

Mouse position is polled via Win32 GetCursorPos on Windows (works even
when the overlay is click-through) and via QCursor.pos() on macOS/Linux,
so waves properly retreat from the cursor on all platforms.
"""

from __future__ import annotations

import logging
import math
import sys

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QLinearGradient,
    QPainter,
    QPainterPath,
)
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

from opensimplex import OpenSimplex
from recorder.overlay.state import STATE_COLORS, OverlayState

logger = logging.getLogger(__name__)

# ---- Wave geometry ----
_BASE_DEPTH: float = 14.0      # calm water line (px from edge)
_PERMANENT_DEPTH: float = 5.0  # always-visible shoreline
_MAX_DEPTH: float = 32.0       # gradient extent
_SAMPLE_STEP: int = 8          # px between wave samples (perf: 8 is smooth enough)

# ---- Noise octaves: (spatial_scale, time_speed, amplitude) ----
# Large scale = big grouped swells, small scale = tiny ripples
_NOISE_OCTAVES: list[tuple[float, float, float]] = [
    (0.003,  0.25,  8.0),   # large slow swells
    (0.010,  0.50,  4.5),   # medium waves
    (0.030,  0.90,  2.5),   # small choppy waves
    (0.080,  1.50,  1.2),   # tiny ripples
]

# Per-edge seed offsets so edges look different
_EDGE_SEEDS: list[int] = [0, 1, 2, 3]

# ---- Mouse retreat ----
_RETREAT_START: float = 200.0   # waves start retreating
_RETREAT_GONE: float = 35.0     # wave amplitude fully gone
_BASE_FADE_DIST: float = 20.0   # permanent base starts fading
_BASE_MIN_ALPHA: float = 0.25   # base never below this
_MOUSE_SMOOTH: float = 6.0     # lerp speed for mouse tracking (higher = faster)

# ---- Win32 cursor polling ----
if sys.platform == "win32":
    _user32 = ctypes.windll.user32

    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.wintypes.LONG), ("y", ctypes.wintypes.LONG)]


def _get_cursor_pos() -> tuple[int, int]:
    """Get global cursor position in a cross-platform manner.

    Win32: Uses GetCursorPos (works even when overlay is click-through).
    macOS/Linux: Uses QCursor.pos() from PyQt6 (requires a running
    QApplication, which is always present when the overlay is active).

    Returns:
        (x, y) screen coordinates.
    """
    if sys.platform == "win32":
        pt = _POINT()
        _user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
    # Cross-platform fallback via Qt — works on macOS and Linux
    pos = QCursor.pos()
    return pos.x(), pos.y()


class ShimmerLayer(QGraphicsObject):
    """Animated wave-edge glow indicating overlay state.

    Uses OpenSimplex noise for organic wave shapes that surge
    inward/outward.  Polls cursor position directly so waves
    retreat from mouse even in click-through mode.

    Args:
        screen_w: Logical screen width in pixels.
        screen_h: Logical screen height in pixels.
    """

    def __init__(self, screen_w: int, screen_h: int) -> None:
        super().__init__()
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # Updated: decorative layer, pass clicks through
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._time: float = 0.0
        self._base_color: QColor = QColor(50, 200, 50)
        self._alpha_mult: float = 0.90
        # Raw polled position and smoothed position for gradual retreat
        self._raw_mouse_x: float = -1000.0
        self._raw_mouse_y: float = -1000.0
        self._mouse_x: float = -1000.0
        self._mouse_y: float = -1000.0
        self._avoidance_rects: list[QRectF] = []
        self.setZValue(10)
        self.setFlag(
            QGraphicsObject.GraphicsItemFlag.ItemHasNoContents, False
        )

        # Create separate noise generators per edge for variety
        self._noise_gens: list[OpenSimplex] = [
            OpenSimplex(seed=42 + s) for s in _EDGE_SEEDS
        ]

    def boundingRect(self) -> QRectF:
        """Return full screen rect as bounding box."""
        return QRectF(0, 0, self._screen_w, self._screen_h)

    def set_state(self, state: OverlayState) -> None:
        """Update wave appearance based on overlay state.

        Args:
            state: The current overlay state.
        """
        r, g, b, _a = STATE_COLORS[state]
        self._base_color = QColor(r, g, b)
        if state == OverlayState.RECORDING:
            self._alpha_mult = 1.0
        elif state == OverlayState.REPLAYING:
            self._alpha_mult = 0.85
        else:
            self._alpha_mult = 0.90

    def set_mouse_pos(self, x: float, y: float) -> None:
        """Store mouse position (called from view, but we also poll directly).

        Args:
            x: Mouse X coordinate in scene space.
            y: Mouse Y coordinate in scene space.
        """
        self._raw_mouse_x = x
        self._raw_mouse_y = y

    def set_avoidance_rects(self, rects: list[QRectF]) -> None:
        """Set UI element rects that waves should avoid.

        Args:
            rects: List of QRectF bounding boxes for active UI elements.
        """
        self._avoidance_rects = list(rects)

    def tick(self, dt: float) -> None:
        """Advance wave animation and poll cursor position.

        Args:
            dt: Elapsed seconds since last tick.
        """
        self._time += dt

        # Poll cursor directly — works even in click-through mode
        mx, my = _get_cursor_pos()
        self._raw_mouse_x = float(mx)
        self._raw_mouse_y = float(my)

        # Smooth mouse position — waves pull back gradually, not jump
        k = min(1.0, _MOUSE_SMOOTH * dt)
        self._mouse_x += (self._raw_mouse_x - self._mouse_x) * k
        self._mouse_y += (self._raw_mouse_y - self._mouse_y) * k

        self.update()

    # ------------------------------------------------------------------
    # Wave noise
    # ------------------------------------------------------------------

    def _wave_depth(self, pos: float, edge_idx: int) -> float:
        """Calculate wave depth at a position using layered simplex noise.

        Args:
            pos: Position along the edge in pixels.
            edge_idx: Which edge (0=top, 1=right, 2=bottom, 3=left).

        Returns:
            Depth in pixels from edge inward.
        """
        noise_gen = self._noise_gens[edge_idx]
        t = self._time
        depth = _BASE_DEPTH

        for spatial_scale, time_speed, amplitude in _NOISE_OCTAVES:
            # noise2 returns values in [-1, 1]
            n = noise_gen.noise2(pos * spatial_scale, t * time_speed)
            depth += amplitude * n

        return max(_PERMANENT_DEPTH, min(depth, _MAX_DEPTH))

    # ------------------------------------------------------------------
    # Mouse retreat
    # ------------------------------------------------------------------

    def _mouse_retreat(self, edge_x: float, edge_y: float) -> tuple[float, float]:
        """Calculate wave and base retreat from mouse proximity.

        Args:
            edge_x: X coordinate of point on screen edge.
            edge_y: Y coordinate of point on screen edge.

        Returns:
            (wave_factor, base_factor) — both 0.0 to 1.0.
        """
        dist = math.hypot(edge_x - self._mouse_x, edge_y - self._mouse_y)

        for rect in self._avoidance_rects:
            cx = max(rect.left(), min(edge_x, rect.right()))
            cy = max(rect.top(), min(edge_y, rect.bottom()))
            rect_dist = math.hypot(edge_x - cx, edge_y - cy)
            dist = min(dist, rect_dist)

        # Wave amplitude retreat
        if dist >= _RETREAT_START:
            wave_f = 1.0
        elif dist <= _RETREAT_GONE:
            wave_f = 0.0
        else:
            t = (dist - _RETREAT_GONE) / (_RETREAT_START - _RETREAT_GONE)
            wave_f = t * t * (3.0 - 2.0 * t)

        # Base opacity retreat (never fully gone)
        if dist >= _RETREAT_START:
            base_f = 1.0
        elif dist <= _BASE_FADE_DIST:
            base_f = _BASE_MIN_ALPHA
        else:
            t = (dist - _BASE_FADE_DIST) / (_RETREAT_START - _BASE_FADE_DIST)
            base_f = _BASE_MIN_ALPHA + (1.0 - _BASE_MIN_ALPHA) * t * t * (3.0 - 2.0 * t)

        return wave_f, base_f

    # ------------------------------------------------------------------
    # Edge painting
    # ------------------------------------------------------------------

    def _make_gradient(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        base_alpha: float,
    ) -> QLinearGradient:
        """Create edge-to-inward gradient (water thinning on sand).

        Args:
            x0, y0: Start point (at screen edge).
            x1, y1: End point (max wave depth inward).
            base_alpha: Overall alpha multiplier.

        Returns:
            QLinearGradient fading from bright at edge to transparent.
        """
        grad = QLinearGradient(QPointF(x0, y0), QPointF(x1, y1))

        c0 = QColor(self._base_color)
        c0.setAlphaF(min(0.75 * base_alpha, 1.0))

        c1 = QColor(self._base_color)
        c1.setAlphaF(min(0.50 * base_alpha, 1.0))

        c2 = QColor(self._base_color)
        c2.setAlphaF(min(0.18 * base_alpha, 1.0))

        c3 = QColor(self._base_color)
        c3.setAlphaF(0.0)

        grad.setColorAt(0.0, c0)
        grad.setColorAt(0.15, c1)
        grad.setColorAt(0.50, c2)
        grad.setColorAt(1.0, c3)

        return grad

    def _paint_top_edge(self, painter: QPainter) -> None:
        """Paint waves along the top screen edge."""
        w = self._screen_w
        path = QPainterPath()
        path.moveTo(0, 0)

        pos = 0
        while pos <= w:
            wave_f, base_f = self._mouse_retreat(float(pos), 0.0)
            raw_depth = self._wave_depth(float(pos), 0)
            depth = _PERMANENT_DEPTH + (raw_depth - _PERMANENT_DEPTH) * wave_f
            depth *= base_f
            path.lineTo(pos, depth)
            pos += _SAMPLE_STEP

        path.lineTo(w, 0)
        path.closeSubpath()

        grad = self._make_gradient(0, 0, 0, _MAX_DEPTH, self._alpha_mult)
        painter.setBrush(grad)
        painter.drawPath(path)

    def _paint_bottom_edge(self, painter: QPainter) -> None:
        """Paint waves along the bottom screen edge."""
        w = self._screen_w
        h = float(self._screen_h)
        path = QPainterPath()
        path.moveTo(0, h)

        pos = 0
        while pos <= w:
            wave_f, base_f = self._mouse_retreat(float(pos), h)
            raw_depth = self._wave_depth(float(pos), 2)
            depth = _PERMANENT_DEPTH + (raw_depth - _PERMANENT_DEPTH) * wave_f
            depth *= base_f
            path.lineTo(pos, h - depth)
            pos += _SAMPLE_STEP

        path.lineTo(w, h)
        path.closeSubpath()

        grad = self._make_gradient(0, h, 0, h - _MAX_DEPTH, self._alpha_mult)
        painter.setBrush(grad)
        painter.drawPath(path)

    def _paint_left_edge(self, painter: QPainter) -> None:
        """Paint waves along the left screen edge."""
        h = self._screen_h
        path = QPainterPath()
        path.moveTo(0, 0)

        pos = 0
        while pos <= h:
            wave_f, base_f = self._mouse_retreat(0.0, float(pos))
            raw_depth = self._wave_depth(float(pos), 3)
            depth = _PERMANENT_DEPTH + (raw_depth - _PERMANENT_DEPTH) * wave_f
            depth *= base_f
            path.lineTo(depth, pos)
            pos += _SAMPLE_STEP

        path.lineTo(0, h)
        path.closeSubpath()

        grad = self._make_gradient(0, 0, _MAX_DEPTH, 0, self._alpha_mult)
        painter.setBrush(grad)
        painter.drawPath(path)

    def _paint_right_edge(self, painter: QPainter) -> None:
        """Paint waves along the right screen edge."""
        w = float(self._screen_w)
        h = self._screen_h
        path = QPainterPath()
        path.moveTo(w, 0)

        pos = 0
        while pos <= h:
            wave_f, base_f = self._mouse_retreat(w, float(pos))
            raw_depth = self._wave_depth(float(pos), 1)
            depth = _PERMANENT_DEPTH + (raw_depth - _PERMANENT_DEPTH) * wave_f
            depth *= base_f
            path.lineTo(w - depth, pos)
            pos += _SAMPLE_STEP

        path.lineTo(w, h)
        path.closeSubpath()

        grad = self._make_gradient(w, 0, w - _MAX_DEPTH, 0, self._alpha_mult)
        painter.setBrush(grad)
        painter.drawPath(path)

    # ------------------------------------------------------------------
    # Main paint
    # ------------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint wave edges on all four screen sides.

        Args:
            painter: The QPainter to draw with.
            option: Style options (unused).
            widget: Target widget (unused).
        """
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)

        self._paint_top_edge(painter)
        self._paint_bottom_edge(painter)
        self._paint_left_edge(painter)
        self._paint_right_edge(painter)

        painter.restore()
