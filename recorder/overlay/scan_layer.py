"""Multi-phase element scan animation layer.

After a click capture, the scan animation provides cinematic visual
feedback while the AI processes the bounding box.  The animation
sequences through corner glow, counter-clockwise line draw, fill,
vertical laser sweep, and horizontal laser sweep phases.  Laser
sweeps repeat until ``receive_fitted_bbox()`` is called with the
AI result, at which point the scan snaps to the fitted bbox.

Each ``ScanLayer`` instance has a ``tick(dt)`` method designed to
be registered with the shared ``AnimationClock``.
"""

from __future__ import annotations

import logging
import math
from enum import Enum, auto

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

logger = logging.getLogger(__name__)


class ScanPhase(Enum):
    """Animation phase for the element scan sequence."""

    IDLE = auto()
    CORNER_GLOW = auto()
    LINE_DRAW = auto()
    FILL_INWARD = auto()
    LASER_VERTICAL = auto()
    LASER_HORIZONTAL = auto()
    WAITING_AI = auto()
    SNAP_TO_FITTED = auto()
    DONE = auto()


_PHASE_DURATIONS: dict[ScanPhase, float] = {
    ScanPhase.CORNER_GLOW: 0.3,
    ScanPhase.LINE_DRAW: 0.8,
    ScanPhase.FILL_INWARD: 0.4,
    ScanPhase.LASER_VERTICAL: 1.0,
    ScanPhase.LASER_HORIZONTAL: 1.0,
    ScanPhase.SNAP_TO_FITTED: 0.5,
}
"""Duration in seconds for each timed phase."""

_GLOW_OVERFLOW: float = 20.0
"""Extra pixels beyond scan rect for glow overflow in boundingRect."""

_CORNER_RADIUS: float = 6.0
"""Radius of corner glow circles."""


class ScanLayer(QGraphicsObject):
    """Multi-phase element scan animation overlay.

    Args:
        x: Left edge of the rough snip boundary.
        y: Top edge of the rough snip boundary.
        w: Width of the rough snip boundary.
        h: Height of the rough snip boundary.
    """

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        super().__init__()
        self._scan_rect = QRectF(x, y, w, h)
        self._fitted_rect: QRectF | None = None
        self._phase: ScanPhase = ScanPhase.IDLE
        self._phase_progress: float = 0.0
        self._ai_ready: bool = False
        self.setZValue(60)

        # Red glow colour from STATE_COLORS[RECORDING]
        self._glow_color = QColor(255, 50, 50)

    def boundingRect(self) -> QRectF:
        """Return bounding rect with glow overflow margin."""
        return self._scan_rect.adjusted(
            -_GLOW_OVERFLOW, -_GLOW_OVERFLOW,
            _GLOW_OVERFLOW, _GLOW_OVERFLOW,
        )

    @property
    def phase(self) -> ScanPhase:
        """Return the current scan phase."""
        return self._phase

    def start_scan(self) -> None:
        """Begin the scan animation sequence.

        Transitions from IDLE to CORNER_GLOW and resets progress.
        """
        self._phase = ScanPhase.CORNER_GLOW
        self._phase_progress = 0.0
        self._ai_ready = False
        self._fitted_rect = None
        self.update()

    def receive_fitted_bbox(self, x: int, y: int, w: int, h: int) -> None:
        """Provide the AI-fitted bounding box result.

        If the scan is currently in a laser phase or WAITING_AI,
        immediately transitions to SNAP_TO_FITTED.

        Args:
            x: Left edge of fitted bbox.
            y: Top edge of fitted bbox.
            w: Width of fitted bbox.
            h: Height of fitted bbox.
        """
        self._fitted_rect = QRectF(x, y, w, h)
        self._ai_ready = True
        snap_phases = {
            ScanPhase.WAITING_AI,
            ScanPhase.LASER_VERTICAL,
            ScanPhase.LASER_HORIZONTAL,
        }
        if self._phase in snap_phases:
            self._phase = ScanPhase.SNAP_TO_FITTED
            self._phase_progress = 0.0
            self.update()

    def reset(self) -> None:
        """Reset to IDLE state, clearing all animation progress."""
        self._phase = ScanPhase.IDLE
        self._phase_progress = 0.0
        self._fitted_rect = None
        self._ai_ready = False
        self.update()

    def tick(self, dt: float) -> None:
        """Advance the animation by *dt* seconds.

        Called by the shared ``AnimationClock`` each frame.

        Args:
            dt: Time elapsed since last tick in seconds.
        """
        if self._phase in (ScanPhase.IDLE, ScanPhase.DONE):
            return

        # WAITING_AI is an instant transition phase (no duration)
        if self._phase == ScanPhase.WAITING_AI:
            self._advance_phase()
            self.update()
            return

        duration = _PHASE_DURATIONS.get(self._phase, 1.0)
        self._phase_progress += dt / duration

        if self._phase_progress >= 1.0:
            self._advance_phase()

        self.update()

    def _advance_phase(self) -> None:
        """Transition to the next phase in the sequence."""
        self._phase_progress = 0.0

        transitions: dict[ScanPhase, ScanPhase] = {
            ScanPhase.CORNER_GLOW: ScanPhase.LINE_DRAW,
            ScanPhase.LINE_DRAW: ScanPhase.FILL_INWARD,
            ScanPhase.FILL_INWARD: ScanPhase.LASER_VERTICAL,
            ScanPhase.LASER_VERTICAL: ScanPhase.LASER_HORIZONTAL,
            ScanPhase.SNAP_TO_FITTED: ScanPhase.DONE,
        }

        if self._phase == ScanPhase.LASER_HORIZONTAL:
            if self._ai_ready:
                self._phase = ScanPhase.SNAP_TO_FITTED
            else:
                self._phase = ScanPhase.WAITING_AI
        elif self._phase == ScanPhase.WAITING_AI:
            self._phase = ScanPhase.LASER_VERTICAL
        elif self._phase in transitions:
            self._phase = transitions[self._phase]
        else:
            logger.warning("No transition from phase %s", self._phase)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Render the current scan phase.

        IMPORTANT: This method must NOT call ``self.update()``.
        """
        if self._phase in (ScanPhase.IDLE, ScanPhase.DONE):
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._phase == ScanPhase.CORNER_GLOW:
            self._paint_corner_glow(painter)
        elif self._phase == ScanPhase.LINE_DRAW:
            self._paint_line_draw(painter)
        elif self._phase == ScanPhase.FILL_INWARD:
            self._paint_fill_inward(painter)
        elif self._phase == ScanPhase.LASER_VERTICAL:
            self._paint_laser_vertical(painter)
        elif self._phase == ScanPhase.LASER_HORIZONTAL:
            self._paint_laser_horizontal(painter)
        elif self._phase == ScanPhase.WAITING_AI:
            # Draw residual glow while waiting
            self._paint_fill_inward(painter)
        elif self._phase == ScanPhase.SNAP_TO_FITTED:
            self._paint_snap_to_fitted(painter)

        painter.restore()

    # ------------------------------------------------------------------
    # Phase-specific paint helpers
    # ------------------------------------------------------------------

    def _paint_corner_glow(self, painter: QPainter) -> None:
        """Draw four glowing red circles at corners, fading in."""
        r = self._scan_rect
        alpha = int(220 * min(self._phase_progress, 1.0))
        color = QColor(self._glow_color)
        color.setAlpha(alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)

        corners = [
            QPointF(r.left(), r.top()),
            QPointF(r.right(), r.top()),
            QPointF(r.left(), r.bottom()),
            QPointF(r.right(), r.bottom()),
        ]
        for c in corners:
            painter.drawEllipse(c, _CORNER_RADIUS, _CORNER_RADIUS)

    def _paint_line_draw(self, painter: QPainter) -> None:
        """Draw CCW lines from each corner, revealing with progress."""
        r = self._scan_rect
        t = min(self._phase_progress, 1.0)

        # Edge segments: CCW from top-left
        # top-left -> bottom-left, bottom-left -> bottom-right,
        # bottom-right -> top-right, top-right -> top-left
        edges = [
            (QPointF(r.left(), r.top()), QPointF(r.left(), r.bottom())),
            (QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom())),
            (QPointF(r.right(), r.bottom()), QPointF(r.right(), r.top())),
            (QPointF(r.right(), r.top()), QPointF(r.left(), r.top())),
        ]

        # Glow pen (wider, semi-transparent)
        glow_pen = QPen(QColor(255, 50, 50, 50))
        glow_pen.setWidthF(6.0)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        # Core pen (thin, full colour)
        core_pen = QPen(QColor(255, 50, 50, 200))
        core_pen.setWidthF(2.0)
        core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        for start, end in edges:
            mid = QPointF(
                start.x() + (end.x() - start.x()) * t,
                start.y() + (end.y() - start.y()) * t,
            )
            # Glow pass
            painter.setPen(glow_pen)
            painter.drawLine(start, mid)
            # Core pass
            painter.setPen(core_pen)
            painter.drawLine(start, mid)

    def _paint_fill_inward(self, painter: QPainter) -> None:
        """Draw semi-transparent red fill growing inward from edges."""
        r = self._scan_rect
        t = min(self._phase_progress, 1.0)

        half_min = min(r.width(), r.height()) / 2.0
        inset = half_min * (1.0 - t)

        fill_rect = r.adjusted(inset, inset, -inset, -inset)
        if fill_rect.width() <= 0 or fill_rect.height() <= 0:
            fill_rect = r

        # Outer region fill (35% opacity)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 50, 50, 90))

        # Draw a frame: outer minus inner
        outer_path = QPainterPath()
        outer_path.addRect(r)
        inner_path = QPainterPath()
        inner_path.addRect(fill_rect)
        frame_path = outer_path - inner_path
        painter.drawPath(frame_path)

    def _paint_laser_vertical(self, painter: QPainter) -> None:
        """Draw horizontal laser line sweeping top to bottom."""
        r = self._scan_rect
        t = min(self._phase_progress, 1.0)
        y_pos = r.top() + r.height() * t

        self._draw_laser_line(
            painter,
            QPointF(r.left(), y_pos),
            QPointF(r.right(), y_pos),
        )

    def _paint_laser_horizontal(self, painter: QPainter) -> None:
        """Draw vertical laser line sweeping left to right."""
        r = self._scan_rect
        t = min(self._phase_progress, 1.0)
        x_pos = r.left() + r.width() * t

        self._draw_laser_line(
            painter,
            QPointF(x_pos, r.top()),
            QPointF(x_pos, r.bottom()),
        )

    def _draw_laser_line(
        self, painter: QPainter, p1: QPointF, p2: QPointF,
    ) -> None:
        """Draw a laser scan line with white core and red glow halo.

        Args:
            painter: Active QPainter.
            p1: Line start point.
            p2: Line end point.
        """
        # Red glow halo
        glow_pen = QPen(QColor(255, 50, 50, 80))
        glow_pen.setWidthF(6.0)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(glow_pen)
        painter.drawLine(p1, p2)

        # White core
        core_pen = QPen(QColor(255, 255, 255, 220))
        core_pen.setWidthF(2.0)
        core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(core_pen)
        painter.drawLine(p1, p2)

        # Gradient leading edge for cinematic feel
        if p1.y() == p2.y():
            # Horizontal line (vertical sweep) - gradient above/below
            grad = QLinearGradient(
                QPointF(p1.x(), p1.y() - 10),
                QPointF(p1.x(), p1.y() + 10),
            )
        else:
            # Vertical line (horizontal sweep) - gradient left/right
            grad = QLinearGradient(
                QPointF(p1.x() - 10, p1.y()),
                QPointF(p1.x() + 10, p1.y()),
            )
        grad.setColorAt(0.0, QColor(255, 50, 50, 0))
        grad.setColorAt(0.5, QColor(255, 50, 50, 40))
        grad.setColorAt(1.0, QColor(255, 50, 50, 0))

    def _paint_snap_to_fitted(self, painter: QPainter) -> None:
        """Interpolate corners from scan_rect to fitted_rect."""
        if self._fitted_rect is None:
            return

        t = min(self._phase_progress, 1.0)
        # Ease-in-out cubic
        t = self._ease_in_out_cubic(t)

        src = self._scan_rect
        dst = self._fitted_rect
        interp = QRectF(
            src.left() + (dst.left() - src.left()) * t,
            src.top() + (dst.top() - src.top()) * t,
            src.width() + (dst.width() - src.width()) * t,
            src.height() + (dst.height() - src.height()) * t,
        )

        # Outline with fading glow
        glow_alpha = int(120 * (1.0 - t))
        glow_pen = QPen(QColor(255, 50, 50, glow_alpha))
        glow_pen.setWidthF(4.0)
        painter.setPen(glow_pen)
        painter.setBrush(QColor(255, 50, 50, 20))
        painter.drawRect(interp)

        # Core outline
        core_pen = QPen(QColor(255, 50, 50, 200))
        core_pen.setWidthF(1.5)
        painter.setPen(core_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(interp)

    @staticmethod
    def _ease_in_out_cubic(t: float) -> float:
        """Apply cubic ease-in-out to progress value.

        Args:
            t: Linear progress 0.0 to 1.0.

        Returns:
            Eased progress value.
        """
        if t < 0.5:
            return 4.0 * t * t * t
        return 1.0 - math.pow(-2.0 * t + 2.0, 3) / 2.0
