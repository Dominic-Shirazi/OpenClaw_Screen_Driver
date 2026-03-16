"""Interactive bounding box components for the recording overlay.

Pure Qt graphics primitives — no business logic. Contains the visual
representation of detected UI elements with draggable corner handles
and click probability donut visualization.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPen, QRadialGradient
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

# Color map for element types (RGBA)
_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    # Interactive
    "textbox": (0, 150, 255, 140),
    "button": (0, 200, 0, 140),
    "button_nav": (0, 255, 100, 140),
    "toggle": (255, 165, 0, 140),
    "tab": (128, 0, 128, 140),
    "dropdown": (255, 200, 0, 140),
    "scrollbar": (128, 128, 128, 100),
    "link": (0, 180, 230, 140),
    "icon": (180, 180, 0, 140),
    "drag_source": (0, 255, 255, 140),
    "drag_target": (0, 200, 200, 140),
    # Structural regions
    "region_chrome": (180, 120, 60, 80),
    "region_menu": (160, 80, 160, 80),
    "region_sidebar": (60, 140, 130, 80),
    "region_content": (100, 140, 200, 60),
    "region_form": (200, 160, 80, 80),
    "region_header": (140, 100, 180, 80),
    "region_footer": (100, 120, 100, 80),
    "region_toolbar": (160, 140, 100, 80),
    "region_modal": (200, 80, 80, 80),
    "region_custom": (120, 120, 180, 80),
    "landmark": (255, 200, 0, 120),
    # Static / read-only
    "read_here": (255, 0, 0, 140),
    "image": (200, 200, 200, 80),
    "modal": (255, 100, 100, 100),
    "notification": (255, 255, 0, 140),
    # Meta
    "unknown": (100, 100, 100, 100),
}

_DEFAULT_COLOR = (100, 100, 100, 100)

# Border width for the screen-edge indicator
_BORDER_WIDTH = 3

# Corner handle radius in pixels
_HANDLE_RADIUS = 5


class _HandleItem(QGraphicsEllipseItem):
    """Draggable corner handle for resizing element bounding boxes.

    Small circle at a bbox corner. Drag to resize the parent box.
    The cursor changes on hover to indicate resize direction.
    """

    def __init__(
        self,
        corner: str,
        box_group: _ElementBoxGroup,
        color: QColor,
    ) -> None:
        r = _HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self._corner = corner
        self._box_group = box_group

        self.setPen(QPen(color, 1.5))
        self.setBrush(QBrush(QColor(255, 255, 255, 200)))
        self.setZValue(50)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        if corner in ("tl", "br"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        self.setAcceptHoverEvents(True)

    def itemChange(
        self, change: QGraphicsItem.GraphicsItemChange, value: Any,
    ) -> Any:
        """Notifies the parent box group when this handle moves."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._box_group.handle_moved(self._corner)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event: Any) -> None:
        """Enlarges handle on hover for easier grabbing."""
        self.setBrush(QBrush(QColor(255, 255, 100, 240)))
        self.setScale(1.4)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: Any) -> None:
        """Restores handle size when mouse leaves."""
        self.setBrush(QBrush(QColor(255, 255, 255, 200)))
        self.setScale(1.0)
        super().hoverLeaveEvent(event)


class _ElementBoxGroup:
    """Manages a bounding box with corner handles, label, and click donut.

    Contains:
    - Main rect item (colored border + light fill)
    - 4 corner handles (draggable to resize)
    - Label text above the box
    - Radial gradient "donut" showing click probability distribution
    - Center dot marking the precise click target
    """

    def __init__(
        self,
        scene: QGraphicsScene,
        x: float,
        y: float,
        w: float,
        h: float,
        color_rgba: tuple[int, int, int, int],
        label: str,
        confidence: float,
    ) -> None:
        self._scene = scene
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        r, g, b, a = color_rgba
        self._color = QColor(r, g, b)
        self._alpha = a

        # Main rect
        pen = QPen(QColor(r, g, b, min(a + 60, 255)))
        pen.setWidth(2)
        brush = QBrush(QColor(r, g, b, a // 4))
        self._rect_item = scene.addRect(QRectF(x, y, w, h), pen, brush)
        self._rect_item.setZValue(10)

        # Donut gradient (click probability visualization)
        self._donut_item = self._create_donut(x, y, w, h)

        # Center dot — precise click target indicator
        dot_r = 3.0
        self._center_dot = scene.addEllipse(
            x + w / 2 - dot_r, y + h / 2 - dot_r, dot_r * 2, dot_r * 2,
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(255, 255, 255, 180)),
        )
        self._center_dot.setZValue(15)

        # Label text
        self._label_item: QGraphicsSimpleTextItem | None = None
        if label:
            text_str = f"{label} ({confidence:.0%})"
            self._label_item = scene.addSimpleText(text_str)
            self._label_item.setPos(x, max(0, y - 16))
            self._label_item.setBrush(QBrush(QColor(r, g, b, 230)))
            font = QFont("Segoe UI", 9)
            font.setBold(True)
            self._label_item.setFont(font)
            self._label_item.setZValue(20)

        # Corner handles
        handle_color = QColor(r, g, b, 220)
        self._handles: dict[str, _HandleItem] = {}
        for corner in ("tl", "tr", "bl", "br"):
            handle = _HandleItem(corner, self, handle_color)
            scene.addItem(handle)
            self._handles[corner] = handle
        self._position_handles()

    def _create_donut(
        self, x: float, y: float, w: float, h: float,
    ) -> QGraphicsEllipseItem:
        """Creates the radial gradient donut showing click probability."""
        cx = x + w / 2
        cy = y + h / 2
        radius = max(w, h) / 2
        if radius < 1:
            radius = 1.0

        gradient = QRadialGradient(cx, cy, radius)
        gradient.setColorAt(0.0, QColor(80, 220, 80, 100))
        gradient.setColorAt(0.4, QColor(80, 220, 80, 60))
        gradient.setColorAt(0.8, QColor(80, 220, 80, 20))
        gradient.setColorAt(1.0, QColor(80, 220, 80, 0))

        donut = self._scene.addEllipse(
            x, y, w, h,
            QPen(Qt.PenStyle.NoPen),
            QBrush(gradient),
        )
        donut.setZValue(5)
        return donut

    def _position_handles(self) -> None:
        """Sets handle positions to match current rect corners."""
        x, y, w, h = self._x, self._y, self._w, self._h
        positions = {
            "tl": (x, y),
            "tr": (x + w, y),
            "bl": (x, y + h),
            "br": (x + w, y + h),
        }
        for corner, (hx, hy) in positions.items():
            handle = self._handles[corner]
            # Disable geometry signals to prevent recursion
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
                False,
            )
            handle.setPos(hx, hy)
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
                True,
            )

    def handle_moved(self, corner: str) -> None:
        """Called when a corner handle is dragged. Updates rect and siblings.

        The opposite corner stays fixed; the moved corner defines the new
        edge positions. Adjacent corners are repositioned to match.
        """
        moved = self._handles[corner].pos()

        if corner == "tl":
            fixed = self._handles["br"].pos()
            new_x, new_y = moved.x(), moved.y()
            new_w = fixed.x() - moved.x()
            new_h = fixed.y() - moved.y()
        elif corner == "tr":
            fixed = self._handles["bl"].pos()
            new_x = fixed.x()
            new_y = moved.y()
            new_w = moved.x() - fixed.x()
            new_h = fixed.y() - moved.y()
        elif corner == "bl":
            fixed = self._handles["tr"].pos()
            new_x = moved.x()
            new_y = fixed.y()
            new_w = fixed.x() - moved.x()
            new_h = moved.y() - fixed.y()
        elif corner == "br":
            fixed = self._handles["tl"].pos()
            new_x = fixed.x()
            new_y = fixed.y()
            new_w = moved.x() - fixed.x()
            new_h = moved.y() - fixed.y()
        else:
            return

        # Enforce minimum size — don't update if too small
        if new_w < 10 or new_h < 10:
            return

        self._x = new_x
        self._y = new_y
        self._w = new_w
        self._h = new_h

        # Update rect
        self._rect_item.setRect(QRectF(new_x, new_y, new_w, new_h))

        # Rebuild donut
        self._scene.removeItem(self._donut_item)
        self._donut_item = self._create_donut(new_x, new_y, new_w, new_h)

        # Update center dot
        dot_r = 3.0
        self._center_dot.setRect(QRectF(
            new_x + new_w / 2 - dot_r, new_y + new_h / 2 - dot_r,
            dot_r * 2, dot_r * 2,
        ))

        # Update label position
        if self._label_item is not None:
            self._label_item.setPos(new_x, max(0, new_y - 16))

        # Reposition sibling handles
        self._position_handles()

    def _all_items(self) -> list[Any]:
        """Returns all scene items owned by this group."""
        items: list[Any] = [
            self._rect_item, self._donut_item, self._center_dot,
        ]
        if self._label_item:
            items.append(self._label_item)
        items.extend(self._handles.values())
        return items

    def highlight(self, active: bool) -> None:
        """Highlights (active review) or dims this element box."""
        if active:
            pen = QPen(QColor(255, 255, 0, 255))
            pen.setWidth(3)
            self._rect_item.setPen(pen)
            for item in self._all_items():
                item.setOpacity(1.0)
        else:
            for item in self._all_items():
                item.setOpacity(0.3)

    def reset_highlight(self) -> None:
        """Restores normal appearance after review."""
        r = self._color.red()
        g = self._color.green()
        b = self._color.blue()
        pen = QPen(QColor(r, g, b, min(self._alpha + 60, 255)))
        pen.setWidth(2)
        self._rect_item.setPen(pen)
        for item in self._all_items():
            item.setOpacity(1.0)

    def get_rect(self) -> tuple[int, int, int, int]:
        """Returns the current bbox as (x, y, w, h) after any resizing."""
        return (int(self._x), int(self._y), int(self._w), int(self._h))
