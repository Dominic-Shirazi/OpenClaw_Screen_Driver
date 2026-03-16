"""PyQt6 side-by-side bbox comparison dialog for YOLOE refinement.

Shows the user's original bounding box alongside the YOLOE suggestion
with a click-probability overlay and adjustable edges, so the user can
accept, tweak, or reject the machine-refined bbox.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.types import Rect

if TYPE_CHECKING:
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QMouseEvent, QPaintEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
_BG = "#1a1a2e"
_FONT_FAMILY = "Segoe UI"
_ACCEPT_BG = "#2d5a2d"
_ACCEPT_HOVER = "#3a7a3a"
_SUBTITLE_COLOR = "#888"
_TIP_COLOR = "#aaa"
_GREEN_BORDER = QColor(80, 200, 80, 200)
_GREEN_FILL = QColor(80, 200, 80)

_MAX_PREVIEW = 320  # max width/height for each side image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_pixmap(img: np.ndarray) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap.

    Args:
        img: BGR uint8 image array (H, W, 3).

    Returns:
        A QPixmap created from the image.
    """
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        # BGR -> RGB
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    # Copy so the data outlives the numpy array
    return QPixmap.fromImage(qimg.copy())


def _scale_to_fit(pixmap: QPixmap, max_size: int) -> QPixmap:
    """Scale a pixmap to fit within max_size while keeping aspect ratio.

    Args:
        pixmap: Source pixmap.
        max_size: Maximum width or height.

    Returns:
        Scaled QPixmap.
    """
    if pixmap.width() <= max_size and pixmap.height() <= max_size:
        return pixmap
    return pixmap.scaled(
        max_size,
        max_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


# ---------------------------------------------------------------------------
# _AdjustableImageLabel — right-side image with draggable bbox edges
# ---------------------------------------------------------------------------

class _AdjustableImageLabel(QLabel):
    """QLabel that displays an image with a draggable green bbox overlay.

    The user can grab any edge of the bbox rectangle and drag it to
    adjust the YOLOE suggestion.  A radial-gradient click-probability
    overlay is painted on top of the image.
    """

    _EDGE_GRAB = 8  # pixels from edge to trigger grab

    def __init__(
        self,
        pixmap: QPixmap,
        bbox_rect: QRect,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the adjustable label.

        Args:
            pixmap: The YOLOE crop image.
            bbox_rect: Initial bbox rectangle in *pixmap* coordinates.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._base_pixmap = pixmap
        self._bbox = QRect(bbox_rect)
        self._dragging: str | None = None  # "left" | "right" | "top" | "bottom"
        self._drag_origin: QPoint | None = None
        self._adjusted = False
        self.setMouseTracking(True)
        self.setFixedSize(pixmap.size())

    @property
    def adjusted(self) -> bool:
        """Whether the user has moved any edge."""
        return self._adjusted

    @property
    def bbox(self) -> QRect:
        """Current bbox in pixmap coordinates."""
        return QRect(self._bbox)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint image, radial gradient overlay, and bbox border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Base image
        painter.drawPixmap(0, 0, self._base_pixmap)

        # Radial gradient click-probability overlay
        cx = self._bbox.center().x()
        cy = self._bbox.center().y()
        grad_radius = max(self._bbox.width(), self._bbox.height()) / 2.0
        if grad_radius < 1:
            grad_radius = 1.0

        gradient = QRadialGradient(float(cx), float(cy), grad_radius)
        gradient.setColorAt(0.0, QColor(80, 200, 80, 200))   # ~80% opacity centre
        gradient.setColorAt(0.5, QColor(80, 200, 80, 100))
        gradient.setColorAt(1.0, QColor(80, 200, 80, 0))      # fully transparent

        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self._bbox)

        # Green bbox border
        pen = QPen(_GREEN_BORDER, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._bbox)

        painter.end()

    # -- mouse interaction -------------------------------------------------

    def _edge_at(self, pos: QPoint) -> str | None:
        """Return which bbox edge (if any) the mouse is near.

        Args:
            pos: Mouse position in widget coordinates.

        Returns:
            One of "left", "right", "top", "bottom", or *None*.
        """
        g = self._EDGE_GRAB
        r = self._bbox
        if r.top() - g <= pos.y() <= r.bottom() + g:
            if abs(pos.x() - r.left()) <= g:
                return "left"
            if abs(pos.x() - r.right()) <= g:
                return "right"
        if r.left() - g <= pos.x() <= r.right() + g:
            if abs(pos.y() - r.top()) <= g:
                return "top"
            if abs(pos.y() - r.bottom()) <= g:
                return "bottom"
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start dragging an edge if the click is near one."""
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self._edge_at(event.pos())
            if edge is not None:
                self._dragging = edge
                self._drag_origin = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Update the bbox edge position while dragging."""
        if self._dragging and self._drag_origin:
            delta = event.pos() - self._drag_origin
            r = QRect(self._bbox)

            if self._dragging == "left":
                r.setLeft(max(0, r.left() + delta.x()))
            elif self._dragging == "right":
                r.setRight(min(self._base_pixmap.width() - 1, r.right() + delta.x()))
            elif self._dragging == "top":
                r.setTop(max(0, r.top() + delta.y()))
            elif self._dragging == "bottom":
                r.setBottom(min(self._base_pixmap.height() - 1, r.bottom() + delta.y()))

            if r.width() > 4 and r.height() > 4:
                self._bbox = r
                self._adjusted = True
                self._drag_origin = event.pos()
                self.update()
        else:
            # Change cursor when near an edge
            edge = self._edge_at(event.pos())
            if edge in ("left", "right"):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif edge in ("top", "bottom"):
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = None
            self._drag_origin = None
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# RefineDialog
# ---------------------------------------------------------------------------

class RefineDialog(QDialog):
    """Side-by-side bbox comparison dialog.

    Shows the user's original crop on the left and the YOLOE-refined
    suggestion on the right with a click-probability gradient overlay
    and adjustable edges.
    """

    def __init__(
        self,
        user_crop: np.ndarray,
        yoloe_crop: np.ndarray,
        yoloe_rect: Rect,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the refine dialog.

        Args:
            user_crop: BGR image of the user's original bbox crop.
            yoloe_crop: BGR image of the YOLOE search region crop.
            yoloe_rect: The YOLOE-detected bbox in *yoloe_crop* coordinates.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Refine Bounding Box")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self._yoloe_rect = yoloe_rect
        self._result: tuple[str, Rect | None] = ("rejected", None)

        self.setStyleSheet(f"background: {_BG}; color: white; font-family: '{_FONT_FAMILY}';")

        root = QVBoxLayout(self)
        root.setSpacing(10)

        # -- Tip line --
        tip = QLabel(
            "Tight boxes improve click accuracy \u2014 the box center "
            "is where clicks land during replay"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {_TIP_COLOR}; font-size: 12px; padding: 4px;")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(tip)

        # -- Side-by-side images --
        images_layout = QHBoxLayout()
        images_layout.setSpacing(16)

        # Left: user's original crop
        left_layout = QVBoxLayout()
        left_label = QLabel("Your box")
        left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_label.setStyleSheet(f"color: {_SUBTITLE_COLOR}; font-size: 11px;")
        left_layout.addWidget(left_label)

        user_pm = _scale_to_fit(_ndarray_to_pixmap(user_crop), _MAX_PREVIEW)
        user_img = QLabel()
        user_img.setPixmap(user_pm)
        user_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        user_img.setStyleSheet("border: 1px solid #444; padding: 2px;")
        left_layout.addWidget(user_img, alignment=Qt.AlignmentFlag.AlignCenter)
        images_layout.addLayout(left_layout)

        # Right: YOLOE suggestion (adjustable)
        right_layout = QVBoxLayout()
        right_label = QLabel("YOLOE suggestion")
        right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_label.setStyleSheet(f"color: {_SUBTITLE_COLOR}; font-size: 11px;")
        right_layout.addWidget(right_label)

        yoloe_pm = _ndarray_to_pixmap(yoloe_crop)

        # Scale factor for display
        scale = 1.0
        if yoloe_pm.width() > _MAX_PREVIEW or yoloe_pm.height() > _MAX_PREVIEW:
            scale = min(
                _MAX_PREVIEW / yoloe_pm.width(),
                _MAX_PREVIEW / yoloe_pm.height(),
            )

        display_pm = _scale_to_fit(yoloe_pm, _MAX_PREVIEW)

        # Scale the bbox rect accordingly
        display_bbox = QRect(
            int(yoloe_rect.x * scale),
            int(yoloe_rect.y * scale),
            int(yoloe_rect.w * scale),
            int(yoloe_rect.h * scale),
        )

        self._scale = scale
        self._adjustable = _AdjustableImageLabel(display_pm, display_bbox)
        right_layout.addWidget(self._adjustable, alignment=Qt.AlignmentFlag.AlignCenter)
        images_layout.addLayout(right_layout)

        root.addLayout(images_layout)

        # -- Buttons --
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        # Accept button
        accept_outer = QVBoxLayout()
        accept_outer.setSpacing(0)
        self._accept_btn = QPushButton("Accept")
        self._accept_btn.setFixedSize(120, 36)
        self._accept_btn.setStyleSheet(
            f"QPushButton {{ background: {_ACCEPT_BG}; color: white; "
            f"padding: 6px 14px; border-radius: 4px; font-weight: bold; "
            f"font-size: 13px; }}"
            f"QPushButton:hover {{ background: {_ACCEPT_HOVER}; }}"
        )
        self._accept_btn.clicked.connect(self._on_accept)
        accept_outer.addWidget(self._accept_btn)

        self._accept_sub = QLabel("")
        self._accept_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._accept_sub.setStyleSheet(f"color: {_SUBTITLE_COLOR}; font-size: 10px;")
        self._accept_sub.setFixedHeight(14)
        accept_outer.addWidget(self._accept_sub)

        btn_layout.addLayout(accept_outer)

        # Reject button
        reject_btn = QPushButton("Reject")
        reject_btn.setFixedSize(120, 36)
        reject_btn.setStyleSheet(
            "QPushButton { padding: 6px 14px; border-radius: 4px; "
            "font-size: 13px; color: white; background: #444; }"
            "QPushButton:hover { background: #555; }"
        )
        reject_btn.clicked.connect(self._on_reject)
        btn_layout.addWidget(reject_btn)

        btn_layout.addStretch()
        root.addLayout(btn_layout)

        self.adjustSize()
        logger.debug("RefineDialog created with yoloe_rect=%s", yoloe_rect)

    # -- slots -------------------------------------------------------------

    def _on_accept(self) -> None:
        """Handle the Accept button."""
        if self._adjustable.adjusted:
            self._accept_sub.setText("with changes")

        # Convert display-space bbox back to original crop coordinates
        qr = self._adjustable.bbox
        inv = 1.0 / self._scale if self._scale > 0 else 1.0
        final_rect = Rect(
            x=int(qr.x() * inv),
            y=int(qr.y() * inv),
            w=int(qr.width() * inv),
            h=int(qr.height() * inv),
        )
        self._result = ("accepted", final_rect)
        logger.info("RefineDialog accepted rect=%s adjusted=%s", final_rect, self._adjustable.adjusted)
        self.accept()

    def _on_reject(self) -> None:
        """Handle the Reject button."""
        self._result = ("rejected", None)
        logger.info("RefineDialog rejected")
        self.reject()

    # -- public API --------------------------------------------------------

    def get_result(self) -> tuple[str, Rect | None]:
        """Return the dialog result.

        Returns:
            A tuple of ``("accepted", Rect)`` with the (possibly
            user-adjusted) bbox, or ``("rejected", None)`` if the user
            chose to keep their original box.
        """
        return self._result
