"""Mini-dialog panels for Wait, Prompt, and Loop toolbar actions.

Lightweight frosted-glass QGraphicsObject dialogs for configuring wait
conditions, prompt-user questions, and loop definitions during recording.
Follows the same visual language as TagDialogPanel (frosted glass, card
glow, proxy widgets).
"""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QComboBox,
    QGraphicsObject,
    QGraphicsProxyWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyleOptionGraphicsItem,
    QWidget,
)

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.card_glow import paint_card_glow
from recorder.overlay.hud_common import (
    CONFIRM_BG,
    CORNER_RADIUS,
    DISMISS_BG,
    DISMISS_TEXT,
    FIELD_STYLESHEET,
    FONT_FAMILY,
    FONT_SIZE_INPUT,
    FONT_SIZE_LABEL,
    FONT_WEIGHT_LIGHT,
    FROST_BG,
    HIGHLIGHT_TOP,
    SPACING,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    Z_TAG_DIALOG,
)

logger = logging.getLogger(__name__)


class _TopComboBox(QComboBox):
    """QComboBox whose popup renders above a fullscreen transparent overlay.

    Standard QComboBox popups appear behind always-on-top overlay windows.
    This subclass applies WindowStaysOnTopHint to the native popup so it
    stays visible.
    """

    def showPopup(self) -> None:
        """Open the dropdown popup above the overlay."""
        super().showPopup()
        popup = self.view()
        if popup:
            popup_window = popup.window()
            if popup_window:
                popup_window.setWindowFlag(
                    Qt.WindowType.WindowStaysOnTopHint, True,
                )
                popup_window.raise_()
                popup_window.show()  # re-show required after flag change

    def hidePopup(self) -> None:
        """Close the dropdown popup and remove the top-hint flag."""
        popup = self.view()
        if popup:
            popup_window = popup.window()
            if popup_window:
                popup_window.setWindowFlag(
                    Qt.WindowType.WindowStaysOnTopHint, False,
                )
        super().hidePopup()

# Condition type display labels -> internal condition_type strings
_CONDITION_MAP: dict[str, str] = {
    "Fixed Timer": "fixed_timer",
    "Element Appears": "element_appears",
    "Screen Change": "screen_change",
    "VLM Check": "vlm_check",
}

# Param field config per condition type: (label, default, visible)
_PARAM_CONFIG: dict[str, tuple[str, str, bool]] = {
    "Fixed Timer": ("", "", False),
    "Element Appears": ("Element description", "", True),
    "Screen Change": ("Change threshold (0-1)", "0.05", True),
    "VLM Check": ("Condition prompt", "", True),
}

# ---------------------------------------------------------------------------
# Shared button stylesheet
# ---------------------------------------------------------------------------

_CONFIRM_BTN_STYLE: str = (
    f"background: rgba({CONFIRM_BG.red()}, {CONFIRM_BG.green()}, "
    f"{CONFIRM_BG.blue()}, {CONFIRM_BG.alpha()}); "
    f"color: rgba(255, 255, 255, 230); border-radius: 4px; "
    f"padding: 4px 12px; font-weight: 400; font-size: 12px;"
)

_DISMISS_BTN_STYLE: str = (
    f"background: rgba({DISMISS_BG.red()}, {DISMISS_BG.green()}, "
    f"{DISMISS_BG.blue()}, {DISMISS_BG.alpha()}); "
    f"color: rgba({DISMISS_TEXT.red()}, {DISMISS_TEXT.green()}, "
    f"{DISMISS_TEXT.blue()}, {DISMISS_TEXT.alpha()}); "
    f"border-radius: 4px; padding: 4px 12px; font-weight: 400; font-size: 12px;"
)


def _make_label(text: str) -> QLabel:
    """Create a styled QLabel for dialog fields.

    Args:
        text: Label text.

    Returns:
        Configured QLabel widget.
    """
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: rgba({TEXT_SECONDARY.red()}, {TEXT_SECONDARY.green()}, "
        f"{TEXT_SECONDARY.blue()}, {TEXT_SECONDARY.alpha()}); "
        f"font-size: {FONT_SIZE_LABEL}px; font-weight: {FONT_WEIGHT_LIGHT}; "
        f"background: transparent;"
    )
    if FONT_FAMILY:
        lbl.setFont(QFont(FONT_FAMILY, FONT_SIZE_LABEL, FONT_WEIGHT_LIGHT))
    return lbl


def _make_proxy(
    widget: QWidget,
    parent: QGraphicsObject,
    x: float,
    y: float,
    width: float | None = None,
) -> QGraphicsProxyWidget:
    """Create a QGraphicsProxyWidget and position it.

    Args:
        widget: The Qt widget to embed.
        parent: Parent QGraphicsObject.
        x: X position relative to parent.
        y: Y position relative to parent.
        width: Optional fixed width for the widget.

    Returns:
        The configured proxy widget.
    """
    proxy = QGraphicsProxyWidget(parent)
    if width is not None:
        widget.setFixedWidth(int(width))
    proxy.setWidget(widget)
    proxy.setPos(x, y)
    return proxy


# ============================================================================
# WaitDialog
# ============================================================================


class WaitDialog(QGraphicsObject):
    """Mini-dialog for configuring wait step conditions.

    Presents a condition type dropdown, timeout field, and optional
    condition-specific parameter field with confirm/cancel buttons.
    Emits ``confirmed(dict)`` with the wait configuration or
    ``dismissed()`` on cancel.
    """

    confirmed = pyqtSignal(dict)
    dismissed = pyqtSignal()

    # Condition type mapping (class-level for testability)
    CONDITION_MAP = _CONDITION_MAP

    def __init__(
        self,
        clock: AnimationClock,
        screen_w: int,
        screen_h: int,
        parent: QGraphicsObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._width: float = 320.0
        self._height: float = 240.0
        self._glow_phase: float = 0.0

        self.setZValue(Z_TAG_DIALOG)

        pad = SPACING.md
        field_w = self._width - 2 * pad
        y_cursor = pad

        # Title
        title_lbl = _make_label("Wait Condition")
        title_lbl.setStyleSheet(
            f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
            f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
            f"font-size: {FONT_SIZE_INPUT}px; font-weight: 400; "
            f"background: transparent;"
        )
        _make_proxy(title_lbl, self, pad, y_cursor, field_w)
        y_cursor += 22

        # Condition type combo
        _make_proxy(_make_label("Condition type"), self, pad, y_cursor, field_w)
        y_cursor += 18
        self._combo = _TopComboBox()
        self._combo.setStyleSheet(FIELD_STYLESHEET)
        for display_text in _CONDITION_MAP:
            self._combo.addItem(display_text)
        self._combo_proxy = _make_proxy(self._combo, self, pad, y_cursor, field_w)
        self._combo.currentTextChanged.connect(self._on_condition_changed)
        y_cursor += 30

        # Timeout field
        _make_proxy(_make_label("Timeout (seconds)"), self, pad, y_cursor, field_w)
        y_cursor += 18
        self._timeout_input = QLineEdit("30")
        self._timeout_input.setStyleSheet(FIELD_STYLESHEET)
        _make_proxy(self._timeout_input, self, pad, y_cursor, field_w)
        y_cursor += 30

        # Param label + field (conditionally visible)
        self._param_label = _make_label("")
        self._param_label_proxy = _make_proxy(
            self._param_label, self, pad, y_cursor, field_w,
        )
        y_cursor += 18
        self._param_input = QLineEdit()
        self._param_input.setStyleSheet(FIELD_STYLESHEET)
        self._param_proxy = _make_proxy(
            self._param_input, self, pad, y_cursor, field_w,
        )
        y_cursor += 30

        # Apply initial condition state
        self._on_condition_changed(self._combo.currentText())

        # Buttons
        btn_w = (field_w - SPACING.sm) / 2

        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.setStyleSheet(_CONFIRM_BTN_STYLE)
        self._confirm_btn.clicked.connect(self._on_confirm)
        _make_proxy(self._confirm_btn, self, pad, y_cursor, btn_w)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(_DISMISS_BTN_STYLE)
        self._cancel_btn.clicked.connect(self._on_cancel)
        _make_proxy(
            self._cancel_btn, self, pad + btn_w + SPACING.sm, y_cursor, btn_w,
        )

        # Register for glow animation
        self._clock.register(self._tick)

    def boundingRect(self) -> QRectF:
        """Return bounding rectangle with glow margin.

        Returns:
            Bounding rect including glow overshoot.
        """
        margin = 50.0
        return QRectF(
            -margin, -margin,
            self._width + 2 * margin,
            self._height + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint frosted glass background with card glow.

        Args:
            painter: The QPainter to draw with.
            option: Style options (unused).
            widget: Target widget (unused).
        """
        card = QRectF(0, 0, self._width, self._height)

        # Card glow (paint outside card rect)
        outer = self.boundingRect()
        clip = QPainterPath()
        clip.addRect(outer)
        inner = QPainterPath()
        inner.addRoundedRect(card, CORNER_RADIUS, CORNER_RADIUS)
        clip = clip - inner
        painter.setClipPath(clip)
        paint_card_glow(painter, card, brightness=0.7, phase=self._glow_phase)
        painter.setClipping(False)

        # Frosted glass body
        path = QPainterPath()
        path.addRoundedRect(card, CORNER_RADIUS, CORNER_RADIUS)
        painter.fillPath(path, FROST_BG)

        # Top highlight
        painter.setPen(Qt.PenStyle.NoPen)
        highlight = QRectF(2, 2, self._width - 4, 6)
        painter.setBrush(HIGHLIGHT_TOP)
        painter.drawRoundedRect(highlight, 3, 3)

    def _tick(self, dt: float) -> None:
        """Advance glow phase animation.

        Args:
            dt: Time delta in seconds.
        """
        self._glow_phase += dt * 0.5
        self.update()

    def _on_condition_changed(self, text: str) -> None:
        """Show/hide param field based on condition type selection.

        Args:
            text: Display text from the combo box.
        """
        config = _PARAM_CONFIG.get(text, ("", "", False))
        label_text, default_val, visible = config
        self._param_label.setText(label_text)
        self._param_input.setText(default_val)
        self._param_label_proxy.setVisible(visible)
        self._param_proxy.setVisible(visible)

    def _on_confirm(self) -> None:
        """Validate inputs and emit confirmed signal with config dict."""
        # Validate timeout
        try:
            timeout = float(self._timeout_input.text())
            if timeout <= 0:
                raise ValueError("Timeout must be positive")
        except (ValueError, TypeError):
            logger.warning("Invalid timeout value: %s", self._timeout_input.text())
            return

        display_text = self._combo.currentText()
        condition_type = _CONDITION_MAP.get(display_text, "fixed_timer")

        # Build params based on condition type
        params: dict[str, Any] = {}
        if condition_type == "fixed_timer":
            params["seconds"] = timeout
        elif condition_type == "element_appears":
            params["description"] = self._param_input.text()
        elif condition_type == "screen_change":
            try:
                params["threshold"] = float(self._param_input.text())
            except (ValueError, TypeError):
                params["threshold"] = 0.05
        elif condition_type == "vlm_check":
            params["prompt"] = self._param_input.text()

        config = {
            "condition_type": condition_type,
            "timeout": timeout,
            "params": params,
            "poll_interval": 2.0,
        }

        logger.info("Wait dialog confirmed: %s", condition_type)
        self.confirmed.emit(config)

    def _on_cancel(self) -> None:
        """Emit dismissed signal."""
        logger.info("Wait dialog cancelled")
        self.dismissed.emit()


# ============================================================================
# PromptDialog
# ============================================================================


class PromptDialog(QGraphicsObject):
    """Mini-dialog for configuring prompt_user step questions.

    Presents a single text input for the question to ask the user/agent
    during replay, with confirm/cancel buttons.  Emits
    ``confirmed(dict)`` with ``{"question_text": str}`` or
    ``dismissed()`` on cancel.
    """

    confirmed = pyqtSignal(dict)
    dismissed = pyqtSignal()

    def __init__(
        self,
        clock: AnimationClock,
        screen_w: int,
        screen_h: int,
        parent: QGraphicsObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._width: float = 320.0
        self._height: float = 160.0
        self._glow_phase: float = 0.0

        self.setZValue(Z_TAG_DIALOG)

        pad = SPACING.md
        field_w = self._width - 2 * pad
        y_cursor = pad

        # Title
        title_lbl = _make_label("Prompt User")
        title_lbl.setStyleSheet(
            f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
            f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
            f"font-size: {FONT_SIZE_INPUT}px; font-weight: 400; "
            f"background: transparent;"
        )
        _make_proxy(title_lbl, self, pad, y_cursor, field_w)
        y_cursor += 22

        # Question text field
        _make_proxy(_make_label("Question text"), self, pad, y_cursor, field_w)
        y_cursor += 18
        self._question_input = QLineEdit()
        self._question_input.setStyleSheet(FIELD_STYLESHEET)
        self._question_input.setPlaceholderText(
            "e.g., Should I continue with this file?"
        )
        _make_proxy(self._question_input, self, pad, y_cursor, field_w)
        y_cursor += 30

        # Buttons
        btn_w = (field_w - SPACING.sm) / 2

        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.setStyleSheet(_CONFIRM_BTN_STYLE)
        self._confirm_btn.clicked.connect(self._on_confirm)
        _make_proxy(self._confirm_btn, self, pad, y_cursor, btn_w)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(_DISMISS_BTN_STYLE)
        self._cancel_btn.clicked.connect(self._on_cancel)
        _make_proxy(
            self._cancel_btn, self, pad + btn_w + SPACING.sm, y_cursor, btn_w,
        )

        # Register for glow animation
        self._clock.register(self._tick)

    def boundingRect(self) -> QRectF:
        """Return bounding rectangle with glow margin.

        Returns:
            Bounding rect including glow overshoot.
        """
        margin = 50.0
        return QRectF(
            -margin, -margin,
            self._width + 2 * margin,
            self._height + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint frosted glass background with card glow.

        Args:
            painter: The QPainter to draw with.
            option: Style options (unused).
            widget: Target widget (unused).
        """
        card = QRectF(0, 0, self._width, self._height)

        # Card glow
        outer = self.boundingRect()
        clip = QPainterPath()
        clip.addRect(outer)
        inner = QPainterPath()
        inner.addRoundedRect(card, CORNER_RADIUS, CORNER_RADIUS)
        clip = clip - inner
        painter.setClipPath(clip)
        paint_card_glow(painter, card, brightness=0.7, phase=self._glow_phase)
        painter.setClipping(False)

        # Frosted glass body
        path = QPainterPath()
        path.addRoundedRect(card, CORNER_RADIUS, CORNER_RADIUS)
        painter.fillPath(path, FROST_BG)

        # Top highlight
        painter.setPen(Qt.PenStyle.NoPen)
        highlight = QRectF(2, 2, self._width - 4, 6)
        painter.setBrush(HIGHLIGHT_TOP)
        painter.drawRoundedRect(highlight, 3, 3)

    def _tick(self, dt: float) -> None:
        """Advance glow phase animation.

        Args:
            dt: Time delta in seconds.
        """
        self._glow_phase += dt * 0.5
        self.update()

    def _on_confirm(self) -> None:
        """Validate and emit confirmed signal with question text."""
        question = self._question_input.text().strip()
        if not question:
            logger.warning("Prompt dialog: question text is empty")
            return
        logger.info("Prompt dialog confirmed: %s", question[:50])
        self.confirmed.emit({"question_text": question})

    def _on_cancel(self) -> None:
        """Emit dismissed signal."""
        logger.info("Prompt dialog cancelled")
        self.dismissed.emit()


# ============================================================================
# LoopDialog
# ============================================================================

# Exit condition display labels -> internal type strings
_LOOP_CONDITION_MAP: dict[str, str] = {
    "N Iterations": "n_iterations",
    "Element Appears": "element_appears",
    "Text Matches": "text_matches",
    "Prompt User": "prompt_user",
}


class LoopDialog(QGraphicsObject):
    """Mini-dialog for defining a loop step during recording.

    Presents a step range selector (from/to dropdowns) populated with
    previously recorded steps, and an exit condition section with four
    condition types.  Emits ``confirmed(dict)`` with the loop config
    or ``dismissed()`` on cancel.
    """

    confirmed = pyqtSignal(dict)
    dismissed = pyqtSignal()

    # Class-level for testability
    CONDITION_MAP = _LOOP_CONDITION_MAP

    def __init__(
        self,
        clock: AnimationClock,
        steps: list[dict[str, Any]],
        screen_w: int,
        screen_h: int,
        parent: QGraphicsObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._steps = steps
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._width: float = 380.0
        self._height: float = 300.0
        self._glow_phase: float = 0.0

        self.setZValue(Z_TAG_DIALOG)

        pad = SPACING.md
        field_w = self._width - 2 * pad
        y_cursor = pad

        # Title
        title_lbl = _make_label("Define Loop")
        title_lbl.setStyleSheet(
            f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
            f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
            f"font-size: {FONT_SIZE_INPUT}px; font-weight: 400; "
            f"background: transparent;"
        )
        _make_proxy(title_lbl, self, pad, y_cursor, field_w)
        y_cursor += 22

        # ---- Section 1: Step Range Selector ----
        _make_proxy(_make_label("From step"), self, pad, y_cursor, field_w / 2 - 4)
        _make_proxy(
            _make_label("To step"), self, pad + field_w / 2 + 4, y_cursor,
            field_w / 2 - 4,
        )
        y_cursor += 18

        step_labels = [
            f"{i + 1}. {s.get('tag_data', {}).get('label', f'Step {i + 1}')}"
            for i, s in enumerate(steps)
        ]

        half_w = field_w / 2 - 4

        self._from_combo = _TopComboBox()
        self._from_combo.setStyleSheet(FIELD_STYLESHEET)
        for lbl in step_labels:
            self._from_combo.addItem(lbl)
        _make_proxy(self._from_combo, self, pad, y_cursor, half_w)

        self._to_combo = _TopComboBox()
        self._to_combo.setStyleSheet(FIELD_STYLESHEET)
        for lbl in step_labels:
            self._to_combo.addItem(lbl)
        if step_labels:
            self._to_combo.setCurrentIndex(len(step_labels) - 1)
        _make_proxy(self._to_combo, self, pad + half_w + 8, y_cursor, half_w)
        y_cursor += 30

        # ---- Section 2: Exit Condition ----
        _make_proxy(_make_label("Exit condition"), self, pad, y_cursor, field_w)
        y_cursor += 18

        self._condition_combo = _TopComboBox()
        self._condition_combo.setStyleSheet(FIELD_STYLESHEET)
        for display_text in _LOOP_CONDITION_MAP:
            self._condition_combo.addItem(display_text)
        _make_proxy(self._condition_combo, self, pad, y_cursor, field_w)
        self._condition_combo.currentTextChanged.connect(self._on_condition_changed)
        y_cursor += 30

        # Primary param field (count / description / target text / question)
        self._param_label = _make_label("Count")
        self._param_label_proxy = _make_proxy(
            self._param_label, self, pad, y_cursor, field_w,
        )
        y_cursor += 18
        self._param_input = QLineEdit("5")
        self._param_input.setStyleSheet(FIELD_STYLESHEET)
        self._param_proxy = _make_proxy(
            self._param_input, self, pad, y_cursor, field_w,
        )
        y_cursor += 30

        # Max iterations field (hidden for N Iterations)
        self._max_label = _make_label("Max iterations (safety limit)")
        self._max_label_proxy = _make_proxy(
            self._max_label, self, pad, y_cursor, field_w,
        )
        y_cursor += 18
        self._max_input = QLineEdit("20")
        self._max_input.setStyleSheet(FIELD_STYLESHEET)
        self._max_proxy = _make_proxy(
            self._max_input, self, pad, y_cursor, field_w,
        )
        y_cursor += 30

        # Apply initial condition state (N Iterations hides max)
        self._on_condition_changed(self._condition_combo.currentText())

        # Buttons
        btn_w = (field_w - SPACING.sm) / 2

        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.setStyleSheet(_CONFIRM_BTN_STYLE)
        self._confirm_btn.clicked.connect(self._on_confirm)
        _make_proxy(self._confirm_btn, self, pad, y_cursor, btn_w)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(_DISMISS_BTN_STYLE)
        self._cancel_btn.clicked.connect(self._on_cancel)
        _make_proxy(
            self._cancel_btn, self, pad + btn_w + SPACING.sm, y_cursor, btn_w,
        )

        # Register for glow animation
        self._clock.register(self._tick)

    def boundingRect(self) -> QRectF:
        """Return bounding rectangle with glow margin.

        Returns:
            Bounding rect including glow overshoot.
        """
        margin = 50.0
        return QRectF(
            -margin, -margin,
            self._width + 2 * margin,
            self._height + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint frosted glass background with card glow.

        Args:
            painter: The QPainter to draw with.
            option: Style options (unused).
            widget: Target widget (unused).
        """
        card = QRectF(0, 0, self._width, self._height)

        # Card glow
        outer = self.boundingRect()
        clip = QPainterPath()
        clip.addRect(outer)
        inner = QPainterPath()
        inner.addRoundedRect(card, CORNER_RADIUS, CORNER_RADIUS)
        clip = clip - inner
        painter.setClipPath(clip)
        paint_card_glow(painter, card, brightness=0.7, phase=self._glow_phase)
        painter.setClipping(False)

        # Frosted glass body
        path = QPainterPath()
        path.addRoundedRect(card, CORNER_RADIUS, CORNER_RADIUS)
        painter.fillPath(path, FROST_BG)

        # Top highlight
        painter.setPen(Qt.PenStyle.NoPen)
        highlight = QRectF(2, 2, self._width - 4, 6)
        painter.setBrush(HIGHLIGHT_TOP)
        painter.drawRoundedRect(highlight, 3, 3)

    def _tick(self, dt: float) -> None:
        """Advance glow phase animation.

        Args:
            dt: Time delta in seconds.
        """
        self._glow_phase += dt * 0.5
        self.update()

    def _on_condition_changed(self, text: str) -> None:
        """Show/hide fields based on exit condition type selection.

        Args:
            text: Display text from the condition combo box.
        """
        ctype = _LOOP_CONDITION_MAP.get(text, "n_iterations")

        if ctype == "n_iterations":
            self._param_label.setText("Count")
            self._param_input.setText("5")
            self._param_label_proxy.setVisible(True)
            self._param_proxy.setVisible(True)
            self._max_label_proxy.setVisible(False)
            self._max_proxy.setVisible(False)
        elif ctype == "element_appears":
            self._param_label.setText("Element description")
            self._param_input.setText("")
            self._param_label_proxy.setVisible(True)
            self._param_proxy.setVisible(True)
            self._max_label_proxy.setVisible(True)
            self._max_proxy.setVisible(True)
            self._max_input.setText("20")
        elif ctype == "text_matches":
            self._param_label.setText("Target text")
            self._param_input.setText("")
            self._param_label_proxy.setVisible(True)
            self._param_proxy.setVisible(True)
            self._max_label_proxy.setVisible(True)
            self._max_proxy.setVisible(True)
            self._max_input.setText("20")
        elif ctype == "prompt_user":
            self._param_label.setText("Question")
            self._param_input.setText("")
            self._param_label_proxy.setVisible(True)
            self._param_proxy.setVisible(True)
            self._max_label_proxy.setVisible(True)
            self._max_proxy.setVisible(True)
            self._max_input.setText("10")

    def _on_confirm(self) -> None:
        """Validate inputs and emit confirmed signal with loop config dict."""
        # Validate step range
        start_idx = self._from_combo.currentIndex()
        end_idx = self._to_combo.currentIndex()

        if not self._steps:
            logger.warning("Loop dialog: no steps to loop over")
            return

        if start_idx > end_idx:
            logger.warning(
                "Loop dialog: invalid range (%d > %d)", start_idx, end_idx,
            )
            return

        display_text = self._condition_combo.currentText()
        ctype = _LOOP_CONDITION_MAP.get(display_text, "n_iterations")

        exit_condition: dict[str, Any] = {"type": ctype}

        if ctype == "n_iterations":
            try:
                count = int(self._param_input.text())
                if count <= 0:
                    raise ValueError("Count must be positive")
            except (ValueError, TypeError):
                logger.warning("Loop dialog: invalid count: %s", self._param_input.text())
                return
            exit_condition["count"] = count
        elif ctype == "element_appears":
            exit_condition["element_description"] = self._param_input.text()
            try:
                max_iter = int(self._max_input.text())
                if max_iter <= 0:
                    raise ValueError("Max iterations must be positive")
            except (ValueError, TypeError):
                logger.warning("Loop dialog: invalid max_iterations")
                return
            exit_condition["max_iterations"] = max_iter
        elif ctype == "text_matches":
            exit_condition["target_text"] = self._param_input.text()
            try:
                max_iter = int(self._max_input.text())
                if max_iter <= 0:
                    raise ValueError("Max iterations must be positive")
            except (ValueError, TypeError):
                logger.warning("Loop dialog: invalid max_iterations")
                return
            exit_condition["max_iterations"] = max_iter
        elif ctype == "prompt_user":
            exit_condition["question_text"] = self._param_input.text()
            try:
                max_iter = int(self._max_input.text())
                if max_iter <= 0:
                    raise ValueError("Max iterations must be positive")
            except (ValueError, TypeError):
                logger.warning("Loop dialog: invalid max_iterations")
                return
            exit_condition["max_iterations"] = max_iter

        config = {
            "body_step_range": (start_idx, end_idx),
            "exit_condition": exit_condition,
        }

        logger.info(
            "Loop dialog confirmed: steps %d-%d, exit=%s",
            start_idx + 1, end_idx + 1, ctype,
        )
        self.confirmed.emit(config)

    def _on_cancel(self) -> None:
        """Emit dismissed signal."""
        logger.info("Loop dialog cancelled")
        self.dismissed.emit()
