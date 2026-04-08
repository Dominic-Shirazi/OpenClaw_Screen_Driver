"""Gaussian probability density visualizer with raindrop ripple effect.

Renders a soft radial gradient heat map centered on the click target
that fades in from center outward.  Concurrent raindrop ripples
(expanding/fading circles) animate within the cloud.  Color transitions
from red (editing) to green (accepted) when the user confirms.

Each ``DonutCloudLayer`` instance has a ``tick(dt)`` method designed
to be registered with the shared ``AnimationClock``.
"""

from __future__ import annotations

import logging
import math
import random

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

logger = logging.getLogger(__name__)


class _Raindrop:
    """Expanding/fading ripple circle within the donut cloud."""

    __slots__ = ("center", "max_radius", "progress", "speed")

    def __init__(self, center: QPointF, max_radius: float) -> None:
        self.center = center
        self.max_radius = max_radius
        self.progress: float = 0.0
        self.speed: float = 0.3 + random.random() * 0.4

    def tick(self, dt: float) -> bool:
        """Advance ripple. Returns False when complete.

        Args:
            dt: Time elapsed since last tick in seconds.

        Returns:
            True if the raindrop is still active, False when complete.
        """
        self.progress += self.speed * dt
        return self.progress < 1.0

    @property
    def radius(self) -> float:
        """Current radius based on progress."""
        return self.max_radius * self.progress

    @property
    def opacity(self) -> float:
        """Current opacity (1.0 at start, 0.0 at completion)."""
        return max(0.0, 1.0 - self.progress)


class DonutCloudLayer(QGraphicsObject):
    """Gaussian probability density cloud with raindrop ripple animation.

    Renders a radial gradient heat map blob centered on the target
    position.  The blob fades in from center outward.  Concurrent
    raindrop ripples (3--5 expanding/fading circles) animate within
    the cloud, concentrated in the high-probability center.

    Color starts red (editing) and transitions to green on ``accept()``.

    Args:
        center_x: X coordinate of the cloud center.
        center_y: Y coordinate of the cloud center.
        radius_x: Horizontal radius of the ellipse.
        radius_y: Vertical radius of the ellipse.
    """

    def __init__(
        self,
        center_x: float,
        center_y: float,
        radius_x: float = 60.0,
        radius_y: float = 60.0,
    ) -> None:
        super().__init__()
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # Updated: decorative layer, pass clicks through
        self._center = QPointF(center_x, center_y)
        self._radius_x: float = radius_x
        self._radius_y: float = radius_y
        self._color = QColor(255, 50, 50)  # Red = editing
        self._target_color = QColor(255, 50, 50)
        self._accepted: bool = False
        self._fade_in_progress: float = 0.0  # 0.0 to 1.0
        self._raindrops: list[_Raindrop] = []
        self._max_raindrops: int = 5
        self._raindrop_spawn_timer: float = 0.0
        self.setZValue(45)  # Below BboxLayer at 50

    def boundingRect(self) -> QRectF:
        """Return bounding rect covering the full cloud area with padding.

        Returns:
            QRectF enclosing the cloud with overflow for raindrop drawing.
        """
        r = max(self._radius_x, self._radius_y) + 20.0  # Padding for overflow
        return QRectF(
            self._center.x() - r,
            self._center.y() - r,
            r * 2,
            r * 2,
        )

    def accept(self) -> None:
        """Transition cloud color from red (editing) to green (accepted)."""
        self._accepted = True
        self._target_color = QColor(50, 200, 50)

    def set_radius(self, rx: float, ry: float) -> None:
        """Reshape the cloud ellipse dimensions.

        Args:
            rx: New horizontal radius.
            ry: New vertical radius.
        """
        self._radius_x = rx
        self._radius_y = ry
        self.prepareGeometryChange()

    def set_center(self, x: float, y: float) -> None:
        """Move the cloud center to a new position.

        Args:
            x: New center X coordinate.
            y: New center Y coordinate.
        """
        self._center = QPointF(x, y)
        self.prepareGeometryChange()

    @property
    def fade_in_progress(self) -> float:
        """Current fade-in progress (0.0 to 1.0)."""
        return self._fade_in_progress

    def tick(self, dt: float) -> None:
        """Advance the cloud animation by one frame.

        Called by the shared ``AnimationClock`` each tick.  Advances
        fade-in, smoothly transitions color, manages raindrop lifecycle.

        Args:
            dt: Elapsed seconds since last tick.
        """
        # Advance fade-in (0.5s full fade)
        self._fade_in_progress = min(1.0, self._fade_in_progress + dt * 2.0)

        # Smooth color transition: lerp each channel toward target
        lerp_factor = min(1.0, dt * 3.0)
        self._color = QColor(
            int(self._color.red() + (self._target_color.red() - self._color.red()) * lerp_factor),
            int(self._color.green() + (self._target_color.green() - self._color.green()) * lerp_factor),
            int(self._color.blue() + (self._target_color.blue() - self._color.blue()) * lerp_factor),
        )

        # Advance existing raindrops, remove completed ones
        self._raindrops = [r for r in self._raindrops if r.tick(dt)]

        # Spawn new raindrops
        self._raindrop_spawn_timer += dt
        if self._raindrop_spawn_timer > 0.3 and len(self._raindrops) < self._max_raindrops:
            radius = max(self._radius_x, self._radius_y)
            angle = random.uniform(0, 2 * math.pi)
            dist = abs(random.gauss(0, radius * 0.3))
            pos = QPointF(
                self._center.x() + math.cos(angle) * dist,
                self._center.y() + math.sin(angle) * dist,
            )
            self._raindrops.append(
                _Raindrop(pos, max_radius=15 + random.random() * 20)
            )
            self._raindrop_spawn_timer = 0.0

        self.update()  # Schedule repaint (safe: from tick, NOT from paint)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Render the Gaussian heat map blob and raindrop ripples.

        CRITICAL: This method must NOT call ``self.update()``.

        Args:
            painter: The QPainter to draw with.
            option: Style options (unused).
            widget: Target widget (unused).
        """
        if self._fade_in_progress <= 0.0:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # --- Main heat map blob (QRadialGradient) ---
        max_r = max(self._radius_x, self._radius_y)
        gradient = QRadialGradient(self._center, max_r)

        core_alpha = int(0.8 * self._fade_in_progress * 0.6 * 255)
        mid_alpha = int(0.4 * self._fade_in_progress * 0.6 * 255)

        core_color = QColor(self._color.red(), self._color.green(), self._color.blue(), core_alpha)
        mid_color = QColor(self._color.red(), self._color.green(), self._color.blue(), mid_alpha)
        edge_color = QColor(self._color.red(), self._color.green(), self._color.blue(), 0)

        gradient.setColorAt(0.0, core_color)
        gradient.setColorAt(0.4, mid_color)
        gradient.setColorAt(1.0, edge_color)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(
            self._center,
            self._radius_x * self._fade_in_progress,
            self._radius_y * self._fade_in_progress,
        )

        # --- Raindrop ripples ---
        for raindrop in self._raindrops:
            ripple_pen = QPen(
                QColor(
                    self._color.red(),
                    self._color.green(),
                    self._color.blue(),
                    int(raindrop.opacity * 120),
                ),
                1.5,
            )
            painter.setPen(ripple_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(raindrop.center, raindrop.radius, raindrop.radius)

        painter.restore()
