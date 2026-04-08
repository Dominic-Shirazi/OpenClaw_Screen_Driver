"""Frosted-glass tag dialog panel for the overlay HUD.

QGraphicsObject with embedded form fields via QGraphicsProxyWidget,
typewriter VLM fill, conditional action-type fields, and card border
glow animation.  This is the primary data capture UI during recording.
"""
from __future__ import annotations

import logging
from typing import Callable

from PyQt6.QtCore import QObject, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsItem,
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
    FADE_IN_MS,
    FADE_OUT_MS,
    FIELD_BG,
    FIELD_STYLESHEET,
    FONT_FAMILY,
    FONT_SIZE_HELPER,
    FONT_SIZE_INPUT,
    FONT_SIZE_LABEL,
    FONT_WEIGHT_LIGHT,
    FONT_WEIGHT_REGULAR,
    FROST_BG,
    HIGHLIGHT_TOP,
    SPACING,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    Z_TAG_DIALOG,
)
from recorder.overlay.typewriter_engine import TypewriterEngine

logger = logging.getLogger(__name__)


class _ProxyComboBox(QComboBox):
    """QComboBox that raises its proxy z-value when popup is shown."""

    def __init__(self, proxy_key: str, parent: QGraphicsObject | None = None) -> None:
        super().__init__()
        self._proxy_key = proxy_key
        self._dialog_ref = parent

    # ------------------------------------------------------------------
    # Popup lifecycle
    # ------------------------------------------------------------------

    def showPopup(self) -> None:
        """Open the dropdown popup above the overlay.

        1. Raise proxy z-value so the popup renders above siblings.
        2. Raise the native popup window with WindowStaysOnTopHint.
        """
        if self._dialog_ref is not None:
            proxy = self._dialog_ref._proxies.get(self._proxy_key)
            if proxy is not None:
                proxy.setZValue(50)

        super().showPopup()

        # Raise the native popup window above the always-on-top overlay.
        popup = self.view()
        if popup:
            popup_window = popup.window()
            if popup_window:
                popup_window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
                popup_window.raise_()
                popup_window.show()  # re-show required after flag change

    def hidePopup(self) -> None:
        """Close the dropdown popup and restore proxy z-value.

        1. Remove WindowStaysOnTopHint from the popup.
        2. Close the popup via super().
        3. Reset proxy z-value.
        """
        # Remove WindowStaysOnTopHint before closing to avoid side effects.
        popup = self.view()
        if popup:
            popup_window = popup.window()
            if popup_window:
                popup_window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)

        super().hidePopup()

        if self._dialog_ref is not None:
            proxy = self._dialog_ref._proxies.get(self._proxy_key)
            if proxy is not None:
                proxy.setZValue(10)


# ---------------------------------------------------------------------------
# Action type choices for the dropdown
# ---------------------------------------------------------------------------

_ACTION_TYPES: list[str] = [
    "click",
    "double_click",
    "right_click",
    "click_drag",
    "type",
    "scroll",
    "wait",
    "read",
    "snip_and_search",
    "select_all_extract",
    "loop",
    "prompt_user",
]

# Element type groups (mirrors recorder/dialog.py _TYPE_GROUPS)
_TYPE_GROUPS: list[tuple[str, list[str]]] = [
    ("-- Interactive --", [
        "textbox", "button", "button_nav", "link", "icon",
        "toggle", "tab", "dropdown", "scrollbar",
        "drag_source", "drag_target",
    ]),
    ("-- Regions (visual grounding) --", [
        "region_chrome", "region_menu", "region_sidebar",
        "region_content", "region_form", "region_header",
        "region_footer", "region_toolbar", "region_modal",
        "region_custom", "landmark",
    ]),
    ("-- Static / Read-only --", [
        "image", "read_here", "notification", "modal",
    ]),
    ("-- Meta / Flow control --", [
        "destination", "branch_point", "fingerprint", "unknown",
    ]),
]

# Conditional field visibility rules: action_type -> list of field keys
_CONDITIONAL_FIELDS: dict[str, list[str]] = {
    "type": ["text_to_type", "press_enter"],
    "scroll": ["direction_amount"],
    "wait": ["condition_timeout"],
    "click_drag": ["drag_target_hint"],
    "read": ["vlm_prompt"],
    "snip_and_search": ["vlm_prompt"],
    "prompt_user": ["question_text"],
}

# Helper tips per conditional field key
_HELPER_TIPS: dict[str, str] = {
    "text_to_type": "I'll click here before typing -- no need for a separate click step",
    "press_enter": "Press Enter to send? Common for search boxes and chat inputs",
    "direction_amount": "Direction and pixel amount to scroll",
    "condition_timeout": "Condition to wait for, or timeout in seconds",
    "drag_target_hint": "I'll capture the drag destination after you confirm this step",
    "vlm_prompt": "Describe what you want to extract from this area",
    "question_text": "This question will be shown to the user or agent during replay",
}


class TagDialogPanel(QGraphicsObject):
    """Frosted-glass tag dialog with form fields, typewriter fill, and card glow.

    Signals:
        confirmed: Emitted with form data dict when user confirms.
        dismissed: Emitted when user dismisses the dialog.
    """

    confirmed = pyqtSignal(dict)
    dismissed = pyqtSignal(dict)

    def __init__(
        self,
        clock: AnimationClock,
        parent: QGraphicsObject | None = None,
    ) -> None:
        """Initialize the tag dialog panel.

        Args:
            clock: AnimationClock for tick-driven animations.
            parent: Optional parent QGraphicsObject.
        """
        super().__init__(parent)
        self.setZValue(Z_TAG_DIALOG)
        self.setFlag(
            QGraphicsObject.GraphicsItemFlag.ItemIsFocusable, True,
        )
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton,
        )

        self._width: float = 400.0
        self._height: float = 280.0
        self._target_height: float = 280.0
        self._corner_radius: float = CORNER_RADIUS

        self._opacity: float = 0.0
        self._target_opacity: float = 0.0

        self._glow_phase: float = 0.0
        self._glow_brightness: float = 1.0
        self._glow_pulsing: bool = False

        self._clock = clock
        self._typewriter = TypewriterEngine(clock, parent=self)
        self._typewriter.char_inserted.connect(self._on_char_inserted)
        self._typewriter.finished.connect(self._on_typewriter_done)

        self._fading_out: bool = False

        # Field index mapping for typewriter interruption
        self._typewriter_field_map: dict[QLineEdit, int] = {}

        # Create all form widgets
        self._proxies: dict[str, QGraphicsProxyWidget] = {}
        self._labels: dict[str, QGraphicsProxyWidget] = {}
        self._tips: dict[str, QGraphicsProxyWidget] = {}
        self._widgets: dict[str, QWidget] = {}
        self._create_fields()

        # Register tick callback
        self._clock.register(self._tick)

        logger.debug("TagDialogPanel created (z=%d)", Z_TAG_DIALOG)

    # ------------------------------------------------------------------
    # Field creation helpers
    # ------------------------------------------------------------------

    def _make_font(self, size: int, weight: int = FONT_WEIGHT_LIGHT) -> QFont:
        """Create a QFont with project typography settings.

        Args:
            size: Font size in pixels.
            weight: Font weight (300=light, 400=regular).

        Returns:
            Configured QFont instance.
        """
        font = QFont()
        if FONT_FAMILY:
            font.setFamily(FONT_FAMILY)
        font.setPixelSize(size)
        font.setWeight(QFont.Weight(weight))
        return font

    def _create_field(
        self, widget: QWidget, x: float, y: float, width: float,
    ) -> QGraphicsProxyWidget:
        """Embed a widget as a QGraphicsProxyWidget child of this item.

        Args:
            widget: The QWidget to embed.
            x: X position within the panel.
            y: Y position within the panel.
            width: Fixed width for the widget.

        Returns:
            The created proxy widget.
        """
        widget.setStyleSheet(FIELD_STYLESHEET)
        widget.setFixedWidth(int(width))
        widget.setFont(self._make_font(FONT_SIZE_INPUT))
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(widget)
        proxy.setPos(x, y)
        proxy.setFlag(
            QGraphicsProxyWidget.GraphicsItemFlag.ItemIsFocusable, True,
        )
        # Combo popups need higher z to render above sibling proxies
        if isinstance(widget, QComboBox):
            proxy.setZValue(10)
        return proxy

    def _create_label(
        self, text: str, x: float, y: float,
    ) -> QGraphicsProxyWidget:
        """Create a label proxy widget above a field.

        Args:
            text: Label text.
            x: X position.
            y: Y position.

        Returns:
            The created proxy widget.
        """
        label = QLabel(text)
        label.setFont(self._make_font(FONT_SIZE_LABEL, FONT_WEIGHT_REGULAR))
        label.setStyleSheet(
            f"color: rgba({TEXT_SECONDARY.red()}, {TEXT_SECONDARY.green()}, "
            f"{TEXT_SECONDARY.blue()}, {TEXT_SECONDARY.alpha()}); "
            "background: transparent;"
        )
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(label)
        proxy.setPos(x, y)
        return proxy

    def _create_tip(
        self, text: str, x: float, y: float, width: float,
    ) -> QGraphicsProxyWidget:
        """Create a helper tip label.

        Args:
            text: Tip text.
            x: X position.
            y: Y position.
            width: Fixed width.

        Returns:
            The created proxy widget.
        """
        label = QLabel(text)
        label.setFont(self._make_font(FONT_SIZE_HELPER))
        label.setWordWrap(True)
        label.setFixedWidth(int(width))
        label.setStyleSheet(
            f"color: rgba({TEXT_SECONDARY.red()}, {TEXT_SECONDARY.green()}, "
            f"{TEXT_SECONDARY.blue()}, {TEXT_SECONDARY.alpha()}); "
            "background: transparent;"
        )
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(label)
        proxy.setPos(x, y)
        return proxy

    def _create_fields(self) -> None:
        """Create all form widgets and embed as proxy children."""
        pad = float(SPACING.md)  # 16px internal padding
        field_w = self._width - 2 * pad
        y = pad

        # -- Label field --
        self._labels["label"] = self._create_label("Label", pad, y)
        y += 18
        label_edit = QLineEdit()
        label_edit.setPlaceholderText("Element label...")
        self._widgets["label"] = label_edit
        self._proxies["label"] = self._create_field(label_edit, pad, y, field_w)
        label_edit.textEdited.connect(lambda _: self._on_user_edit(label_edit))
        y += 34

        # -- Caption field --
        self._labels["caption"] = self._create_label("Caption", pad, y)
        y += 18
        caption_edit = QLineEdit()
        caption_edit.setPlaceholderText("What this element does...")
        self._widgets["caption"] = caption_edit
        self._proxies["caption"] = self._create_field(
            caption_edit, pad, y, field_w,
        )
        caption_edit.textEdited.connect(
            lambda _: self._on_user_edit(caption_edit),
        )
        y += 34

        # -- Action Type combo --
        self._labels["action_type"] = self._create_label(
            "Action Type", pad, y,
        )
        y += 18
        action_combo = _ProxyComboBox("action_type", self)
        for at in _ACTION_TYPES:
            action_combo.addItem(at, at)
        self._widgets["action_type"] = action_combo
        self._proxies["action_type"] = self._create_field(
            action_combo, pad, y, field_w,
        )
        action_combo.currentIndexChanged.connect(self._on_action_type_changed)
        action_combo.currentIndexChanged.connect(
            lambda idx: logger.info(
                "ACTION COMBO: index changed to %d = '%s'",
                idx, action_combo.itemData(idx),
            )
        )
        y += 34

        # -- Element Type combo (grouped with separator headers) --
        self._labels["element_type"] = self._create_label(
            "Element Type", pad, y,
        )
        y += 18
        elem_combo = _ProxyComboBox("element_type", self)
        elem_combo.setMaxVisibleItems(20)
        self._populate_element_type_combo(elem_combo)
        self._widgets["element_type"] = elem_combo
        self._proxies["element_type"] = self._create_field(
            elem_combo, pad, y, field_w,
        )
        # Select "unknown" as default
        self._select_element_type(elem_combo, "unknown")
        y += 34

        # -- Conditional fields --
        # Text to type (action_type == "type")
        self._labels["text_to_type"] = self._create_label(
            "Text to type", pad, y,
        )
        text_edit = QLineEdit()
        text_edit.setPlaceholderText("Text to enter...")
        self._widgets["text_to_type"] = text_edit
        self._proxies["text_to_type"] = self._create_field(
            text_edit, pad, y + 18, field_w,
        )
        self._tips["text_to_type"] = self._create_tip(
            _HELPER_TIPS["text_to_type"], pad, y + 52, field_w,
        )

        # Press Enter checkbox (action_type == "type")
        press_enter = QCheckBox("Press Enter to send?")
        press_enter.setStyleSheet(
            f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
            f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
            "background: transparent;"
        )
        press_enter.setFont(self._make_font(FONT_SIZE_INPUT))
        self._widgets["press_enter"] = press_enter
        proxy_enter = QGraphicsProxyWidget(self)
        proxy_enter.setWidget(press_enter)
        proxy_enter.setFlag(
            QGraphicsProxyWidget.GraphicsItemFlag.ItemIsFocusable, True,
        )
        self._proxies["press_enter"] = proxy_enter
        self._tips["press_enter"] = self._create_tip(
            _HELPER_TIPS["press_enter"], pad, y + 80, field_w,
        )

        # Direction/Amount (action_type == "scroll")
        self._labels["direction_amount"] = self._create_label(
            "Direction/Amount", pad, y,
        )
        dir_edit = QLineEdit()
        dir_edit.setPlaceholderText("Direction and pixel amount to scroll")
        self._widgets["direction_amount"] = dir_edit
        self._proxies["direction_amount"] = self._create_field(
            dir_edit, pad, y + 18, field_w,
        )
        self._tips["direction_amount"] = self._create_tip(
            _HELPER_TIPS["direction_amount"], pad, y + 52, field_w,
        )

        # Condition/Timeout (action_type == "wait")
        self._labels["condition_timeout"] = self._create_label(
            "Condition/Timeout", pad, y,
        )
        cond_edit = QLineEdit()
        cond_edit.setPlaceholderText(
            "Condition to wait for, or timeout in seconds",
        )
        self._widgets["condition_timeout"] = cond_edit
        self._proxies["condition_timeout"] = self._create_field(
            cond_edit, pad, y + 18, field_w,
        )
        self._tips["condition_timeout"] = self._create_tip(
            _HELPER_TIPS["condition_timeout"], pad, y + 52, field_w,
        )

        # Drag target hint (action_type == "click_drag")
        self._labels["drag_target_hint"] = self._create_label(
            "Drag target", pad, y,
        )
        drag_edit = QLineEdit()
        drag_edit.setPlaceholderText(
            "I'll capture the drag destination after you confirm this step",
        )
        self._widgets["drag_target_hint"] = drag_edit
        self._proxies["drag_target_hint"] = self._create_field(
            drag_edit, pad, y + 18, field_w,
        )
        self._tips["drag_target_hint"] = self._create_tip(
            _HELPER_TIPS["drag_target_hint"], pad, y + 52, field_w,
        )

        # VLM prompt (action_type in ("read", "snip_and_search"))
        self._labels["vlm_prompt"] = self._create_label(
            "VLM prompt", pad, y,
        )
        vlm_edit = QLineEdit()
        vlm_edit.setPlaceholderText("What to look for...")
        self._widgets["vlm_prompt"] = vlm_edit
        self._proxies["vlm_prompt"] = self._create_field(
            vlm_edit, pad, y + 18, field_w,
        )
        self._tips["vlm_prompt"] = self._create_tip(
            _HELPER_TIPS["vlm_prompt"], pad, y + 52, field_w,
        )

        # Question text (action_type == "prompt_user")
        self._labels["question_text"] = self._create_label(
            "Question", pad, y,
        )
        q_edit = QLineEdit()
        q_edit.setPlaceholderText("Question for the user...")
        self._widgets["question_text"] = q_edit
        self._proxies["question_text"] = self._create_field(
            q_edit, pad, y + 18, field_w,
        )
        self._tips["question_text"] = self._create_tip(
            _HELPER_TIPS["question_text"], pad, y + 52, field_w,
        )

        # -- Buttons --
        confirm_btn = QPushButton("Confirm")
        confirm_btn.setFixedWidth(int(field_w / 2 - SPACING.sm))
        confirm_btn.setStyleSheet(
            f"background: rgba({CONFIRM_BG.red()}, {CONFIRM_BG.green()}, "
            f"{CONFIRM_BG.blue()}, {CONFIRM_BG.alpha()}); "
            f"color: rgba({TEXT_PRIMARY.red()}, {TEXT_PRIMARY.green()}, "
            f"{TEXT_PRIMARY.blue()}, {TEXT_PRIMARY.alpha()}); "
            "border-radius: 4px; padding: 6px 12px; "
            f"font-size: {FONT_SIZE_INPUT}px; font-weight: 400;"
        )
        confirm_btn.clicked.connect(self.confirm)
        self._widgets["confirm_btn"] = confirm_btn
        proxy_confirm = QGraphicsProxyWidget(self)
        proxy_confirm.setWidget(confirm_btn)
        self._proxies["confirm_btn"] = proxy_confirm

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setFixedWidth(int(field_w / 2 - SPACING.sm))
        dismiss_btn.setStyleSheet(
            f"background: rgba({DISMISS_BG.red()}, {DISMISS_BG.green()}, "
            f"{DISMISS_BG.blue()}, {DISMISS_BG.alpha()}); "
            f"color: rgba({DISMISS_TEXT.red()}, {DISMISS_TEXT.green()}, "
            f"{DISMISS_TEXT.blue()}, {DISMISS_TEXT.alpha()}); "
            "border-radius: 4px; padding: 6px 12px; "
            f"font-size: {FONT_SIZE_INPUT}px; font-weight: 400;"
        )
        dismiss_btn.clicked.connect(self.dismiss)
        self._widgets["dismiss_btn"] = dismiss_btn
        proxy_dismiss = QGraphicsProxyWidget(self)
        proxy_dismiss.setWidget(dismiss_btn)
        self._proxies["dismiss_btn"] = proxy_dismiss

        # Initial layout: hide all conditional fields, position buttons
        self._on_action_type_changed(0)

    def _populate_element_type_combo(self, combo: QComboBox) -> None:
        """Fill element type combo with grouped items and separator headers.

        Args:
            combo: The QComboBox to populate.
        """
        for group_label, values in _TYPE_GROUPS:
            combo.addItem(group_label, None)
            idx = combo.count() - 1
            model = combo.model()
            if model is not None:
                item = model.item(idx)
                if item is not None:
                    item.setEnabled(False)
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(8)
                    item.setFont(font)
            for val in values:
                combo.addItem(val, val)

    def _select_element_type(self, combo: QComboBox, value: str) -> None:
        """Select an element type in the combo by value string.

        Args:
            combo: The QComboBox to search.
            value: The element type string to select.
        """
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    # ------------------------------------------------------------------
    # Action type change handler
    # ------------------------------------------------------------------

    def _on_action_type_changed(self, index: int) -> None:
        """Handle action type dropdown selection change.

        Shows/hides conditional fields and relayouts the panel.

        Args:
            index: New combo index (unused, reads current data).
        """
        action_combo = self._widgets["action_type"]
        if not isinstance(action_combo, QComboBox):
            return
        current_action = action_combo.currentData()

        # Determine which conditional field keys to show
        visible_keys: set[str] = set()
        if current_action in _CONDITIONAL_FIELDS:
            visible_keys = set(_CONDITIONAL_FIELDS[current_action])

        # All conditional field keys
        all_conditional = {
            "text_to_type", "press_enter", "direction_amount",
            "condition_timeout", "drag_target_hint", "vlm_prompt",
            "question_text",
        }

        for key in all_conditional:
            visible = key in visible_keys
            if key in self._proxies:
                self._proxies[key].setVisible(visible)
            if key in self._labels:
                self._labels[key].setVisible(visible)
            if key in self._tips:
                self._tips[key].setVisible(visible)

        self._relayout_fields()

    def _relayout_fields(self) -> None:
        """Reposition all visible proxy widgets and resize panel height."""
        pad = float(SPACING.md)
        field_w = self._width - 2 * pad
        y = pad

        # Always-visible fields: label, caption, action_type, element_type
        always_visible = ["label", "caption", "action_type", "element_type"]
        for key in always_visible:
            if key in self._labels:
                self._labels[key].setPos(pad, y)
            y += 18  # label height
            if key in self._proxies:
                self._proxies[key].setPos(pad, y)
            y += 34  # field + gap

        # Conditional fields
        conditional_order = [
            "text_to_type", "press_enter", "direction_amount",
            "condition_timeout", "drag_target_hint", "vlm_prompt",
            "question_text",
        ]

        for key in conditional_order:
            proxy = self._proxies.get(key)
            if proxy is None or not proxy.isVisible():
                continue

            # Label
            label_proxy = self._labels.get(key)
            if label_proxy is not None and label_proxy.isVisible():
                label_proxy.setPos(pad, y)
                y += 18

            # Field
            if key == "press_enter":
                # Checkbox — no separate label above
                proxy.setPos(pad, y)
                y += 28
            else:
                proxy.setPos(pad, y)
                y += 30

            # Tip
            tip_proxy = self._tips.get(key)
            if tip_proxy is not None and tip_proxy.isVisible():
                tip_proxy.setPos(pad, y)
                y += 20

            y += float(SPACING.sm)

        # Buttons row
        y += float(SPACING.sm)
        btn_w = field_w / 2 - SPACING.sm
        self._proxies["confirm_btn"].setPos(pad, y)
        self._proxies["dismiss_btn"].setPos(pad + btn_w + 2 * SPACING.sm, y)
        y += 40

        self._target_height = y + pad

    # ------------------------------------------------------------------
    # Show / dismiss
    # ------------------------------------------------------------------

    def show_dialog(
        self,
        element_rect: QRectF,
        vlm_data: dict | None = None,
        edit_mode: bool = False,
        screen_w: float = 1920.0,
        screen_h: float = 1080.0,
    ) -> None:
        """Show the tag dialog near the captured element.

        Args:
            element_rect: Bounding rect of the captured element in scene coords.
            vlm_data: Optional VLM-predicted field values dict.
            edit_mode: If True, pre-fill fields instantly without typewriter.
            screen_w: Screen width for positioning clamping.
            screen_h: Screen height for positioning clamping.
        """
        self.setVisible(True)
        x, y = self._compute_position(element_rect, screen_w, screen_h)
        self.setPos(x, y)

        self._fading_out = False

        # Re-register tick callback (unregistered on dismiss fade-out completion).
        # Unregister first to avoid duplicates if show_dialog is called while visible.
        self._clock.unregister(self._tick)
        self._clock.register(self._tick)

        # Re-enable mouse acceptance and focus (dismiss() disables them)
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton,
        )
        self.setFlag(
            QGraphicsObject.GraphicsItemFlag.ItemIsFocusable, True,
        )

        if vlm_data:
            # Set combo fields instantly (dropdowns don't typewrite)
            if "action_type" in vlm_data:
                combo = self._widgets["action_type"]
                if isinstance(combo, QComboBox):
                    idx = combo.findData(vlm_data["action_type"])
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

            if "element_type" in vlm_data:
                combo = self._widgets["element_type"]
                if isinstance(combo, QComboBox):
                    self._select_element_type(combo, vlm_data["element_type"])

            # Set conditional field values if present
            for key in (
                "text_to_type", "direction_amount", "condition_timeout",
                "drag_target_hint", "vlm_prompt", "question_text",
            ):
                if key in vlm_data and key in self._widgets:
                    w = self._widgets[key]
                    if isinstance(w, QLineEdit):
                        w.setText(vlm_data[key])

            if "press_enter" in vlm_data and "press_enter" in self._widgets:
                w = self._widgets["press_enter"]
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(vlm_data["press_enter"]))

            if edit_mode:
                # Pre-fill label/caption instantly, no typewriter
                label_w = self._widgets["label"]
                caption_w = self._widgets["caption"]
                if isinstance(label_w, QLineEdit):
                    label_w.setText(vlm_data.get("label", ""))
                if isinstance(caption_w, QLineEdit):
                    caption_w.setText(vlm_data.get("caption", ""))
            else:
                # Start typewriter for label and caption
                tw_fields: list[tuple[QLineEdit, str]] = []
                label_w = self._widgets["label"]
                caption_w = self._widgets["caption"]
                if isinstance(label_w, QLineEdit):
                    label_w.clear()
                    tw_fields.append(
                        (label_w, vlm_data.get("label", "")),
                    )
                    self._typewriter_field_map[label_w] = 0
                if isinstance(caption_w, QLineEdit):
                    caption_w.clear()
                    tw_fields.append(
                        (caption_w, vlm_data.get("caption", "")),
                    )
                    self._typewriter_field_map[caption_w] = 1
                if tw_fields:
                    self._typewriter.start(tw_fields)
        else:
            # No VLM data — show empty fields
            for key in ("label", "caption"):
                w = self._widgets.get(key)
                if isinstance(w, QLineEdit):
                    w.clear()

        # Restore proxy visibility (hidden on dismiss)
        for proxy in self._proxies.values():
            if proxy is not None:
                proxy.setVisible(True)
        for proxy in self._labels.values():
            if proxy is not None:
                proxy.setVisible(True)
        # Re-apply conditional field visibility
        action_combo = self._widgets.get("action_type")
        if isinstance(action_combo, QComboBox):
            self._on_action_type_changed(action_combo.currentIndex())

        # Fade in
        self._opacity = 0.0
        self._target_opacity = 1.0

        logger.debug(
            "TagDialogPanel shown at (%.0f, %.0f), edit_mode=%s",
            x, y, edit_mode,
        )

    def _compute_position(
        self,
        element_rect: QRectF,
        screen_w: float,
        screen_h: float,
    ) -> tuple[float, float]:
        """Compute dialog position relative to captured element.

        Prefers below-right of the element rect, flipping if near screen
        edges, with 24px breathing room clamping.

        Args:
            element_rect: Bounding rect of the captured element.
            screen_w: Screen width.
            screen_h: Screen height.

        Returns:
            Tuple of (x, y) for panel position.
        """
        breathing = float(SPACING.lg)  # 24px
        gap = float(SPACING.md)  # 16px

        # Prefer below-right
        x = element_rect.right() + gap
        y = element_rect.top()

        # Flip horizontal if overflows right
        if x + self._width + breathing > screen_w:
            x = element_rect.left() - gap - self._width

        # Flip vertical if overflows bottom
        if y + self._height + breathing > screen_h:
            y = screen_h - self._height - breathing

        # Clamp
        x = max(breathing, min(x, screen_w - self._width - breathing))
        y = max(breathing, min(y, screen_h - self._height - breathing))

        return x, y

    def dismiss(self) -> None:
        """Fade out and emit dismissed signal."""
        self._target_opacity = 0.0
        self._fading_out = True
        # Updated: immediately stop accepting mouse/keyboard during fade-out
        # so the invisible panel doesn't swallow clicks (fixes dry-run double-click)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
        self._typewriter.stop()
        # Hide all proxy widgets immediately so they don't linger
        for proxy in self._proxies.values():
            if proxy is not None:
                proxy.setVisible(False)
        for proxy in self._labels.values():
            if proxy is not None:
                proxy.setVisible(False)
        for proxy in self._tips.values():
            if proxy is not None:
                proxy.setVisible(False)
        logger.debug("TagDialogPanel dismissing")

    def confirm(self) -> None:
        """Collect form data, emit confirmed signal, and dismiss."""
        data = self.get_form_data()
        self.confirmed.emit(data)
        self.dismiss()

    def get_form_data(self) -> dict:
        """Return current form field values as a dict.

        Returns:
            Dict with label, caption, action_type, element_type, and
            any visible conditional field values.
        """
        data: dict[str, object] = {}

        label_w = self._widgets.get("label")
        data["label"] = label_w.text() if isinstance(label_w, QLineEdit) else ""

        caption_w = self._widgets.get("caption")
        data["caption"] = (
            caption_w.text() if isinstance(caption_w, QLineEdit) else ""
        )

        action_w = self._widgets.get("action_type")
        if isinstance(action_w, QComboBox):
            logger.info(
                "GET_FORM_DATA: action_type currentIndex=%d, currentText='%s', currentData='%s'",
                action_w.currentIndex(), action_w.currentText(), action_w.currentData(),
            )
        data["action_type"] = (
            action_w.currentData()
            if isinstance(action_w, QComboBox)
            else "click"
        )

        elem_w = self._widgets.get("element_type")
        data["element_type"] = (
            elem_w.currentData()
            if isinstance(elem_w, QComboBox)
            else "unknown"
        )

        # Conditional fields — include if widget is visible
        for key in (
            "text_to_type", "direction_amount", "condition_timeout",
            "drag_target_hint", "vlm_prompt", "question_text",
        ):
            proxy = self._proxies.get(key)
            w = self._widgets.get(key)
            if proxy is not None and proxy.isVisible() and isinstance(w, QLineEdit):
                data[key] = w.text()

        press_proxy = self._proxies.get("press_enter")
        press_w = self._widgets.get("press_enter")
        if (
            press_proxy is not None
            and press_proxy.isVisible()
            and isinstance(press_w, QCheckBox)
        ):
            data["press_enter"] = press_w.isChecked()

        return data

    def get_avoidance_rect(self) -> QRectF:
        """Return the scene bounding rect for shimmer avoidance.

        Returns:
            QRectF in scene coordinates covering this panel.
        """
        pos = self.pos()
        return QRectF(pos.x(), pos.y(), self._width, self._height)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        """Return bounding rect with extra padding for card glow overflow.

        Returns:
            QRectF with 30px padding on all sides.
        """
        return QRectF(-80, -80, self._width + 160, self._height + 160)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint frosted glass background, highlight gradient, and card glow.

        Args:
            painter: Active QPainter.
            option: Style option (unused).
            widget: Target widget (unused).
        """
        if self._opacity < 0.01:
            return

        painter.save()
        painter.setOpacity(self._opacity)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(0, 0, self._width, self._height)

        # Card border glow — clip to OUTSIDE the card so nothing bleeds through
        card_path = QPainterPath()
        card_path.addRoundedRect(rect, self._corner_radius, self._corner_radius)

        outer = QPainterPath()
        margin = 80.0  # glow reach
        outer.addRect(rect.adjusted(-margin, -margin, margin, margin))
        glow_clip = outer - card_path

        painter.save()
        painter.setClipPath(glow_clip)
        paint_card_glow(
            painter,
            rect,
            brightness=self._glow_brightness,
            phase=self._glow_phase,
        )
        painter.restore()
        painter.setOpacity(self._opacity)

        # Frosted glass background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(FROST_BG)
        painter.drawPath(card_path)

        # Inner highlight gradient at top
        gradient = QLinearGradient(0, 0, 0, self._height * 0.3)
        gradient.setColorAt(0.0, HIGHLIGHT_TOP)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(gradient)
        painter.drawPath(card_path)

        painter.restore()

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def set_glow_pulsing(self, enabled: bool) -> None:
        """Enable or disable glow pulsing as a loading indicator.

        When enabled, the glow brightness oscillates between 0.5 and 2.0
        to indicate background processing (detection, VLM analysis).

        Args:
            enabled: Whether to pulse the glow.
        """
        self._glow_pulsing = enabled
        if not enabled:
            self._glow_brightness = 1.0
        logger.debug("TagDialogPanel glow pulsing: %s", enabled)

    def _tick(self, dt: float) -> None:
        """Advance animations by delta-time.

        Args:
            dt: Elapsed seconds since last tick.
        """
        # Continuous time for organic flicker (sine waves handle periodicity)
        self._glow_phase += dt * 0.8

        # Glow brightness: pulse mode or decay mode
        if getattr(self, "_glow_pulsing", False):
            import math
            self._glow_brightness = 1.25 + 0.75 * math.sin(self._glow_phase * 3.0)
        else:
            # Glow brightness decay
            self._glow_brightness += (
                (1.0 - self._glow_brightness) * min(1.0, dt * 8.0)
            )

        # Opacity interpolation
        speed = 1000.0 / FADE_IN_MS if self._target_opacity > 0.5 else 1000.0 / FADE_OUT_MS
        self._opacity += (
            (self._target_opacity - self._opacity) * min(1.0, dt * speed)
        )

        # Height interpolation
        self._height += (
            (self._target_height - self._height) * min(1.0, dt * 8.0)
        )

        # Check if fade-out complete
        if self._fading_out and self._opacity < 0.01:
            self._opacity = 0.0
            self._fading_out = False
            self._clock.unregister(self._tick)
            self.dismissed.emit({})

        self.update()

    # ------------------------------------------------------------------
    # Typewriter callbacks
    # ------------------------------------------------------------------

    def _on_char_inserted(self, field_index: int) -> None:
        """Flash glow brightness on each character insertion.

        Args:
            field_index: Index of the field being typed into.
        """
        self._glow_brightness = 2.0

    def _on_typewriter_done(self) -> None:
        """Handle typewriter completion — all fields now editable."""
        logger.debug("Typewriter fill complete, fields editable")

    def _on_user_edit(self, widget: QLineEdit) -> None:
        """Handle user editing a field — interrupt typewriter for it.

        Args:
            widget: The QLineEdit being edited by the user.
        """
        field_idx = self._typewriter_field_map.get(widget)
        if field_idx is not None and self._typewriter.is_active:
            self._typewriter.interrupt_field(field_idx)

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: object) -> None:
        """Handle Enter (confirm) and Escape (dismiss) shortcuts.

        Args:
            event: The key event.
        """
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        if isinstance(event, QKeyEvent):
            if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                self.confirm()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape:
                self.dismiss()
                event.accept()
                return
        super().keyPressEvent(event)
