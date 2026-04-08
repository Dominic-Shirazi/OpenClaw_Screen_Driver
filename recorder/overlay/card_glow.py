"""Card border glow painting helper.

Renders a continuous ambient underglow around a rectangular perimeter.
All edges glow simultaneously — no sweep or trailing.  Organic motion
comes from layered sine waves that vary each point's intensity/reach
over time, like fire or aurora.

The effect: a panel sitting on a glowing surface, light leaking from
all edges at once.  Some spots reach further than others, and those
spots shift over time for a living, breathing quality.

IMPORTANT — callers MUST:
    1. Clip painting to OUTSIDE the card rect (QPainterPath subtraction)
    2. Paint glow BEFORE the card body
"""
from __future__ import annotations

import logging
import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QRadialGradient

from recorder.overlay.hud_common import ACCENT_GREEN

logger = logging.getLogger(__name__)

# How far outside the card edge blob centers sit
_OUTWARD_OFFSET: float = 4.0


def _edge_point(
    distance: float, rect: QRectF,
) -> tuple[float, float, float, float]:
    """Compute edge position and outward normal for a perimeter distance.

    Args:
        distance: Distance along the perimeter (0 to perimeter length).
        rect: The rectangle to walk around.

    Returns:
        Tuple of (edge_x, edge_y, normal_x, normal_y).
    """
    w = rect.width()
    h = rect.height()
    x0 = rect.x()
    y0 = rect.y()

    d = distance
    if d < w:
        return x0 + d, y0, 0.0, -1.0
    d -= w
    if d < h:
        return x0 + w, y0 + d, 1.0, 0.0
    d -= h
    if d < w:
        return x0 + w - d, y0 + h, 0.0, 1.0
    d -= w
    return x0, y0 + h - d, -1.0, 0.0


def _flicker(frac: float, phase: float) -> float:
    """Compute organic flicker intensity for a perimeter position.

    Layers multiple sine waves at different frequencies to create
    non-repeating, natural variation.  Every point always has a base
    glow — the flicker just modulates how far the light reaches.

    Args:
        frac: Position around perimeter (0.0 to 1.0).
        phase: Time-based animation phase (increments continuously).

    Returns:
        Intensity multiplier from 0.4 (dim) to 1.0 (bright).
    """
    # Three sine waves at different frequencies for organic feel
    # Each uses a different prime multiplier to avoid repeating patterns
    wave1 = math.sin(frac * 7.0 * math.pi + phase * 2.1)
    wave2 = math.sin(frac * 13.0 * math.pi + phase * 3.7 + 1.3)
    wave3 = math.sin(frac * 19.0 * math.pi + phase * 1.3 + 2.7)

    # Combine: weighted average normalized to 0..1
    combined = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2 + 1.0) / 2.0

    # Map to 0.4..1.0 range so nothing ever goes fully dark
    return 0.4 + combined * 0.6


def paint_card_glow(
    painter: QPainter,
    rect: QRectF,
    brightness: float = 1.0,
    phase: float = 0.0,
    light_count: int = 40,
    glow_radius: float = 45.0,
    color: QColor | None = None,
) -> None:
    """Paint continuous ambient underglow around a rectangle.

    All edges glow simultaneously.  Organic variation in intensity
    and reach creates living, breathing motion without any directional
    sweep.  Each blob is a large, soft ellipse stretched along the
    edge tangent, fading smoothly to transparent.

    Args:
        painter: The QPainter to draw with (must be active).
        rect: The card rectangle to glow around.
        brightness: Overall brightness multiplier (0.0 to ~2.0).
        phase: Time phase for organic flicker animation.
        light_count: Number of blobs around the perimeter.
        glow_radius: How far the glow reaches outward.
        color: Glow color.  Defaults to ACCENT_GREEN.
    """
    if brightness < 0.01:
        return

    base_color = color if color is not None else ACCENT_GREEN
    perimeter = 2.0 * (rect.width() + rect.height())

    if perimeter < 1.0:
        return

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setCompositionMode(
        QPainter.CompositionMode.CompositionMode_Plus,
    )

    for i in range(light_count):
        frac = i / light_count
        d = frac * perimeter

        ex, ey, nx, ny = _edge_point(d, rect)

        # Organic flicker: varies reach/intensity per position over time
        flick = _flicker(frac, phase)

        # Push center outside the edge
        cx = ex + nx * _OUTWARD_OFFSET
        cy = ey + ny * _OUTWARD_OFFSET

        center = QPointF(cx, cy)

        # Scale radius by flicker — brighter spots reach further
        this_radius = glow_radius * (0.7 + flick * 0.5)

        # Alpha: soft base, modulated by flicker and brightness
        per_alpha = min(0.08 * flick * brightness, 1.0)

        # Stretched ellipse along the edge tangent
        # Tangent stretch ensures neighbors overlap into continuous band
        r_tangent = this_radius * 1.6
        r_outward = this_radius

        # Gradient: very soft falloff — 5 stops for smooth fade to nothing
        grad_r = max(r_outward, r_tangent)
        gradient = QRadialGradient(center, grad_r)

        c0 = QColor(base_color)
        c0.setAlphaF(per_alpha)

        c1 = QColor(base_color)
        c1.setAlphaF(per_alpha * 0.7)

        c2 = QColor(base_color)
        c2.setAlphaF(per_alpha * 0.35)

        c3 = QColor(base_color)
        c3.setAlphaF(per_alpha * 0.1)

        c4 = QColor(base_color)
        c4.setAlphaF(0.0)

        gradient.setColorAt(0.0, c0)
        gradient.setColorAt(0.15, c1)
        gradient.setColorAt(0.35, c2)
        gradient.setColorAt(0.65, c3)
        gradient.setColorAt(1.0, c4)

        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)

        if abs(nx) > abs(ny):
            rx = r_outward
            ry = r_tangent
        else:
            rx = r_tangent
            ry = r_outward

        painter.drawEllipse(center, rx, ry)

    painter.restore()
