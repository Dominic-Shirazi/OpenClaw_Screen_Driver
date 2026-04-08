"""Bounding box rendering with corner resize handles and morph animation.

Each ``BboxLayer`` represents a single detected UI element drawn on
the overlay scene.  Corner handles are included for future resize
support (actual drag logic is Phase 2+).

The ``morph_to()`` method smoothly animates the bbox from its current
position to a new target using InOutCubic easing over 500ms, driven
by a ``QPropertyAnimation`` on a helper ``QObject``.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QRectF, Qt, pyqtProperty
from PyQt6.QtGui import QBrush, QColor, QCursor, QFont, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

logger = logging.getLogger(__name__)

_HANDLE_SIZE: int = 8
"""Side length of corner handle squares in pixels."""


class _MorphHelper(QObject):
    """Helper QObject that owns the morph progress property.

    ``QPropertyAnimation`` requires a ``QObject`` target, but
    ``QGraphicsItemGroup`` is not a ``QObject``.  This helper bridges
    the gap by exposing a ``progress`` property that drives the
    bbox layer's interpolation.

    Args:
        bbox_layer: The ``BboxLayer`` to drive.
    """

    def __init__(self, bbox_layer: BboxLayer) -> None:
        super().__init__()
        self._bbox = bbox_layer
        self._progress = 0.0

    @pyqtProperty(float)  # type: ignore[misc]
    def progress(self) -> float:
        """Current morph progress (0.0 to 1.0)."""
        return self._progress

    @progress.setter  # type: ignore[attr-defined]
    def progress(self, value: float) -> None:
        self._progress = value
        self._bbox._apply_morph_progress(value)


class BboxLayer(QGraphicsItemGroup):
    """Visual bounding box with corner handles and optional label.

    Args:
        x: Left edge in scene coordinates.
        y: Top edge in scene coordinates.
        w: Width in pixels.
        h: Height in pixels.
        color_rgba: Border colour as ``(r, g, b, a)`` tuple.
        label: Optional text label shown above the box.
        confidence: Detection confidence (0.0 -- 1.0), shown in label.
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color_rgba: tuple[int, int, int, int],
        label: str = "",
        confidence: float = 0.0,
    ) -> None:
        super().__init__()
        self.setZValue(50)

        self._x = x
        self._y = y
        self._w = w
        self._h = h

        # Morph animation state
        self._morph_start: tuple[int, int, int, int] = (x, y, w, h)
        self._morph_end: tuple[int, int, int, int] = (x, y, w, h)
        self._is_morphing: bool = False
        self._morph_helper: _MorphHelper | None = None
        self._morph_anim: QPropertyAnimation | None = None

        r, g, b, a = color_rgba

        # Main border rectangle
        pen = QPen(QColor(r, g, b, min(a + 60, 255)))
        pen.setWidth(2)
        brush = QBrush(QColor(r, g, b, a // 4))
        self._rect = QGraphicsRectItem(QRectF(x, y, w, h), self)
        self._rect.setPen(pen)
        self._rect.setBrush(brush)

        # Optional label text above the box
        self._label: QGraphicsSimpleTextItem | None = None
        if label:
            text_str = f"{label} ({confidence:.0%})"
            self._label = QGraphicsSimpleTextItem(text_str, self)
            self._label.setPos(x, max(0, y - 16))
            self._label.setBrush(QBrush(QColor(r, g, b, 230)))
            font = QFont("Segoe UI", 9)
            font.setBold(True)
            self._label.setFont(font)

        # Resize handles (small white squares at corners and edge midpoints)
        self._handles: list[QGraphicsRectItem] = []
        self._editing: bool = False
        handle_brush = QBrush(QColor(255, 255, 255, 200))
        handle_pen = QPen(QColor(r, g, b, 220))
        handle_pen.setWidth(1)

        # Cursor hints for each handle position
        # Order: TL, TC, TR, RC, BR, BC, BL, LC
        _handle_cursors = [
            Qt.CursorShape.SizeFDiagCursor,   # top-left
            Qt.CursorShape.SizeVerCursor,      # top-center
            Qt.CursorShape.SizeBDiagCursor,    # top-right
            Qt.CursorShape.SizeHorCursor,      # right-center
            Qt.CursorShape.SizeFDiagCursor,    # bottom-right
            Qt.CursorShape.SizeVerCursor,      # bottom-center
            Qt.CursorShape.SizeBDiagCursor,    # bottom-left
            Qt.CursorShape.SizeHorCursor,      # left-center
        ]

        for (hx, hy), cursor_shape in zip(
            self._handle_positions(x, y, w, h), _handle_cursors,
        ):
            handle = QGraphicsRectItem(
                QRectF(hx, hy, _HANDLE_SIZE, _HANDLE_SIZE),
                self,
            )
            handle.setPen(handle_pen)
            handle.setBrush(handle_brush)
            handle.setCursor(QCursor(cursor_shape))
            # Handles start non-movable; enable_editing() activates them
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False,
            )
            self._handles.append(handle)

    @staticmethod
    def _handle_positions(
        x: int, y: int, w: int, h: int,
    ) -> list[tuple[float, float]]:
        """Return top-left positions for all 8 handle rects.

        Order: TL, TC, TR, RC, BR, BC, BL, LC (clockwise from top-left).

        Args:
            x: Box left edge.
            y: Box top edge.
            w: Box width.
            h: Box height.

        Returns:
            Eight ``(hx, hy)`` tuples for corners and edge midpoints.
        """
        hs = _HANDLE_SIZE
        half = hs / 2
        return [
            (x - half, y - half),                # top-left
            (x + w / 2 - half, y - half),        # top-center
            (x + w - half, y - half),            # top-right
            (x + w - half, y + h / 2 - half),   # right-center
            (x + w - half, y + h - half),        # bottom-right
            (x + w / 2 - half, y + h - half),   # bottom-center
            (x - half, y + h - half),            # bottom-left
            (x - half, y + h / 2 - half),       # left-center
        ]

    # Backward-compatible alias
    _corner_positions = _handle_positions

    def get_rect(self) -> tuple[int, int, int, int]:
        """Return the current bounding box as ``(x, y, w, h)``.

        Returns:
            Tuple of ``(x, y, width, height)`` accounting for any
            handle dragging.
        """
        return (self._x, self._y, self._w, self._h)

    def highlight(self, active: bool) -> None:
        """Highlight or dim this bounding box.

        Args:
            active: If True, render at full opacity.  If False, dim to
                40% opacity for de-emphasis.
        """
        self.setOpacity(1.0 if active else 0.4)

    def reset_highlight(self) -> None:
        """Restore the bounding box to default (full) opacity."""
        self.setOpacity(1.0)

    # ------------------------------------------------------------------
    # Interactive editing
    # ------------------------------------------------------------------

    _MIN_BBOX_SIZE: int = 20
    """Minimum bbox width/height during interactive editing."""

    def enable_editing(self, enabled: bool = True) -> None:
        """Toggle interactive drag-editing on all handles.

        Args:
            enabled: If True, handles become movable. If False, locked.
        """
        self._editing = enabled
        for handle in self._handles:
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable, enabled,
            )
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
                enabled,
            )
        logger.debug("BboxLayer editing %s", "enabled" if enabled else "disabled")

    def get_edited_rect(self) -> tuple[int, int, int, int]:
        """Compute the bbox rect from current handle positions.

        Uses the top-left (handle 0) and bottom-right (handle 4) to
        determine the edited rectangle, clamped to minimum size.

        Returns:
            Tuple of ``(x, y, w, h)`` in scene coordinates.
        """
        if len(self._handles) < 8:
            return (self._x, self._y, self._w, self._h)

        half = _HANDLE_SIZE / 2

        # Read corner positions from TL and BR handles.
        # handle.rect() is the LOCAL bounding rect (never changes on drag).
        # handle.pos() is the offset applied by Qt when the item is dragged.
        # The actual scene-level rect is rect translated by pos.
        tl = self._handles[0]
        br = self._handles[4]
        tl_rect = tl.rect().translated(tl.pos())
        br_rect = br.rect().translated(br.pos())

        x1 = tl_rect.x() + half
        y1 = tl_rect.y() + half
        x2 = br_rect.x() + half
        y2 = br_rect.y() + half

        # Ensure correct ordering
        x = int(round(min(x1, x2)))
        y = int(round(min(y1, y2)))
        w = int(round(abs(x2 - x1)))
        h = int(round(abs(y2 - y1)))

        # Clamp to minimum size
        w = max(w, self._MIN_BBOX_SIZE)
        h = max(h, self._MIN_BBOX_SIZE)

        return (x, y, w, h)

    def accept_edit(self) -> None:
        """Finalize the edited bbox and disable editing mode.

        Updates internal coordinates to match the current handle positions.
        """
        self._x, self._y, self._w, self._h = self.get_edited_rect()
        self._rect.setRect(QRectF(self._x, self._y, self._w, self._h))
        self._sync_handles_to_rect()
        self.enable_editing(False)
        logger.debug(
            "BboxLayer edit accepted: (%d,%d,%d,%d)",
            self._x, self._y, self._w, self._h,
        )

    def reject_edit(self, original: tuple[int, int, int, int]) -> None:
        """Revert handles to the given original rect and disable editing.

        Args:
            original: The ``(x, y, w, h)`` to revert to.
        """
        self._x, self._y, self._w, self._h = original
        self._rect.setRect(QRectF(self._x, self._y, self._w, self._h))
        self._sync_handles_to_rect()
        self.enable_editing(False)
        logger.debug(
            "BboxLayer edit rejected, reverted to: (%d,%d,%d,%d)",
            *original,
        )

    def _sync_handles_to_rect(self) -> None:
        """Reposition all handles to match current _x, _y, _w, _h."""
        positions = self._handle_positions(self._x, self._y, self._w, self._h)
        for handle, (hx, hy) in zip(self._handles, positions):
            # Reset any drag offset so rect coordinates are authoritative
            handle.setPos(0, 0)
            handle.setRect(QRectF(hx, hy, _HANDLE_SIZE, _HANDLE_SIZE))
        if self._label is not None:
            self._label.setPos(self._x, max(0, self._y - 16))

    # ------------------------------------------------------------------
    # Morph animation
    # ------------------------------------------------------------------

    @property
    def is_morphing(self) -> bool:
        """Return True if a morph animation is in progress."""
        return self._is_morphing

    def morph_to(
        self, x: int, y: int, w: int, h: int, duration_ms: int = 500,
    ) -> None:
        """Animate the bbox from its current rect to a new target.

        Uses ``QPropertyAnimation`` with ``InOutCubic`` easing on a
        helper ``QObject`` that drives interpolation via
        ``_apply_morph_progress()``.

        Args:
            x: Target left edge.
            y: Target top edge.
            w: Target width.
            h: Target height.
            duration_ms: Animation duration in milliseconds (default 500,
                within the 400--600ms range specified in CONTEXT.md).
        """
        self._morph_start = (self._x, self._y, self._w, self._h)
        self._morph_end = (x, y, w, h)

        if self._morph_helper is None:
            self._morph_helper = _MorphHelper(self)

        # Reset helper progress
        self._morph_helper._progress = 0.0

        self._morph_anim = QPropertyAnimation(
            self._morph_helper, b"progress",
        )
        self._morph_anim.setDuration(duration_ms)
        self._morph_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._morph_anim.setStartValue(0.0)
        self._morph_anim.setEndValue(1.0)
        self._morph_anim.finished.connect(self._on_morph_done)

        self._is_morphing = True
        self._morph_anim.start()
        logger.debug(
            "Morph started: (%d,%d,%d,%d) -> (%d,%d,%d,%d) over %dms",
            *self._morph_start, *self._morph_end, duration_ms,
        )

    def _apply_morph_progress(self, t: float) -> None:
        """Interpolate rect and children to morph progress *t*.

        Args:
            t: Progress value from 0.0 (start) to 1.0 (end).
        """
        sx, sy, sw, sh = self._morph_start
        ex, ey, ew, eh = self._morph_end

        ix = int(round(sx + (ex - sx) * t))
        iy = int(round(sy + (ey - sy) * t))
        iw = int(round(sw + (ew - sw) * t))
        ih = int(round(sh + (eh - sh) * t))

        # Update stored coordinates
        self._x = ix
        self._y = iy
        self._w = iw
        self._h = ih

        # Update main rect item
        self._rect.setRect(QRectF(ix, iy, iw, ih))

        # Update handle positions
        positions = self._corner_positions(ix, iy, iw, ih)
        for handle, (hx, hy) in zip(self._handles, positions):
            handle.setRect(QRectF(hx, hy, _HANDLE_SIZE, _HANDLE_SIZE))

        # Update label position if present
        if self._label is not None:
            self._label.setPos(ix, max(0, iy - 16))

    def _on_morph_done(self) -> None:
        """Finalise morph: snap to exact target and update styling.

        Ensures no floating-point drift in final values and applies
        the post-morph visual style (thin red outline, faint fill).
        """
        self._is_morphing = False

        # Snap to exact target values
        ex, ey, ew, eh = self._morph_end
        self._x = ex
        self._y = ey
        self._w = ew
        self._h = eh
        self._rect.setRect(QRectF(ex, ey, ew, eh))

        positions = self._corner_positions(ex, ey, ew, eh)
        for handle, (hx, hy) in zip(self._handles, positions):
            handle.setRect(QRectF(hx, hy, _HANDLE_SIZE, _HANDLE_SIZE))

        if self._label is not None:
            self._label.setPos(ex, max(0, ey - 16))

        # Post-morph styling: thin red outline, glow retreats
        pen = QPen(QColor(255, 50, 50, 180))
        pen.setWidth(1)
        self._rect.setPen(pen)
        self._rect.setBrush(QBrush(QColor(255, 50, 50, 20)))

        logger.debug("Morph complete: final rect (%d,%d,%d,%d)", ex, ey, ew, eh)
