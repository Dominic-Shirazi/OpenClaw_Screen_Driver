"""Card border glow painting helper.

Renders soft radial gradient lights around a rectangular perimeter
using the same additive blending technique (CompositionMode_Plus)
as ShimmerLayer.  Used by both the tag dialog and toolbar panels.
"""
from __future__ import annotations

import logging
import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QRadialGradient

from recorder.overlay.hud_common import ACCENT_GREEN

logger = logging.getLogger(__name__)

# Fraction of perimeter that is lit at once (same as ShimmerLayer)
_ACTIVE_SPAN: float = 0.35

# How far outside the card edge light centers are pushed
_OUTWARD_DEPTH: float = 15.0


def _edge_point(
    distance: float, rect: QRectF,
) -> tuple[float, float, float, float]:
    """Compute edge position and outward normal for a perimeter distance.

    Traversal order matches ShimmerLayer: top -> right -> bottom -> left.

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
        # Top edge: left to right
        return x0 + d, y0, 0.0, -1.0
    d -= w
    if d < h:
        # Right edge: top to bottom
        return x0 + w, y0 + d, 1.0, 0.0
    d -= h
    if d < w:
        # Bottom edge: right to left
        return x0 + w - d, y0 + h, 0.0, 1.0
    d -= w
    # Left edge: bottom to top
    return x0, y0 + h - d, -1.0, 0.0


def _sweep_brightness(frac: float, phase: float) -> float:
    """Calculate brightness at a perimeter position based on sweep phase.

    Uses the same algorithm as ShimmerLayer._sweep_brightness.

    Args:
        frac: Position around perimeter (0.0 to 1.0).
        phase: Current sweep phase (0.0 to 1.0).

    Returns:
        Brightness factor from 0.0 to 1.0.
    """
    delta = abs(frac - phase)
    if delta > 0.5:
        delta = 1.0 - delta

    if delta > _ACTIVE_SPAN:
        return 0.0
    t = delta / _ACTIVE_SPAN
    return 0.5 * (1.0 + math.cos(t * math.pi))


def paint_card_glow(
    painter: QPainter,
    rect: QRectF,
    brightness: float = 1.0,
    phase: float = 0.0,
    light_count: int = 24,
    glow_radius: float = 30.0,
    color: QColor | None = None,
) -> None:
    """Paint additive-blended radial gradient lights around a rectangle.

    Each light is a radial gradient positioned just outside the card
    edge, with only the inner spill visible.  Uses CompositionMode_Plus
    for the same additive blending as ShimmerLayer.

    Args:
        painter: The QPainter to draw with (must be active).
        rect: The card rectangle to glow around.
        brightness: Overall brightness multiplier (0.0 to 1.0).
        phase: Sweep position around perimeter (0.0 to 1.0).
        light_count: Number of light sources around the perimeter.
        glow_radius: Radius of each radial gradient.
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

        sweep = _sweep_brightness(frac, phase)
        if sweep < 0.01:
            continue

        # Push center outside the card edge
        cx = ex + nx * _OUTWARD_DEPTH
        cy = ey + ny * _OUTWARD_DEPTH

        # Core alpha: sweep * brightness * 0.5, capped at 1.0
        peak_alpha = min(sweep * brightness * 0.5, 1.0)
        if peak_alpha < 0.01:
            continue

        center = QPointF(cx, cy)
        gradient = QRadialGradient(center, glow_radius)

        core = QColor(base_color)
        core.setAlphaF(peak_alpha)
        edge = QColor(base_color)
        edge.setAlphaF(0.0)

        gradient.setColorAt(0.0, core)
        gradient.setColorAt(1.0, edge)

        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, glow_radius, glow_radius)

    painter.restore()
