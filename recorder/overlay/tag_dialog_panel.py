"""Tag dialog panel for the overlay HUD.

QWidget-based floating panel with direct form fields (no proxy widgets),
typewriter VLM fill, conditional action-type fields, and card border
glow animation.  This is the primary data capture UI during recording.

Uses a custom _DropdownButton widget instead of QComboBox to avoid the
Windows transparency bug where QComboBox popups are always transparent
when the parent widget tree has had any transparency attributes.  The card
glow paints outside the rounded-rect clip region; fade-in/out uses
setWindowOpacity() which works independently of translucency attributes.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from PyQt6.QtCore import QPoint, QRectF, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
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

# All conditional field keys
_ALL_CONDITIONAL: set[str] = {
    "text_to_type", "press_enter", "direction_amount",
    "condition_timeout", "drag_target_hint", "vlm_prompt",
    "question_text",
}

# ---------------------------------------------------------------------------
# QSS styling
# ---------------------------------------------------------------------------

_DIALOG_QSS: str = """
QWidget#TagDialog {
    background: rgb(20, 22, 28);
    border-radius: 12px;
}
QLineEdit {
    background-color: rgba(20, 20, 35, 220);
    color: #ffffff;
    border: 1px solid rgba(50, 200, 50, 60);
    border-radius: 4px;
    padding: 4px 6px;
    font-weight: 400;
    font-size: 14px;
}
QLabel {
    color: #c0c8d0;
    background: transparent;
    font-size: 14px;
    font-weight: 500;
}
QCheckBox {
    color: #ffffff;
    background: transparent;
    font-size: 14px;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid rgba(50, 200, 50, 60);
    border-radius: 3px;
    background-color: rgba(20, 20, 35, 220);
}
QCheckBox::indicator:checked {
    background-color: rgba(50, 200, 50, 180);
}
"""


# ---------------------------------------------------------------------------
# Custom dropdown widget — replaces QComboBox to avoid Windows transparency bug
# ---------------------------------------------------------------------------

class _DropdownPopup(QWidget):
    """Frameless popup containing a QListWidget for item selection.

    Uses its own top-level window flags with a solid background so it
    is never affected by parent widget transparency attributes.
    """

    item_selected = pyqtSignal(int)  # index of selected item

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the popup widget.

        Args:
            parent: Optional parent widget (used for positioning only).
        """
        super().__init__(parent=None)  # No Qt parent — independent window
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Popup
        )
        # Explicitly do NOT set WA_TranslucentBackground
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAutoFillBackground(True)

        # Force solid background via palette
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(30, 34, 42))
        self.setPalette(pal)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget {"
            "  background-color: rgb(30, 34, 42);"
            "  color: #ffffff;"
            "  border: 1px solid rgba(100, 200, 255, 120);"
            "  font-size: 14px;"
            "  padding: 4px;"
            "  outline: none;"
            "}"
            "QListWidget::item {"
            "  padding: 4px 8px;"
            "}"
            "QListWidget::item:selected {"
            "  background-color: rgb(60, 140, 200);"
            "  color: #ffffff;"
            "}"
            "QListWidget::item:hover {"
            "  background-color: rgba(60, 140, 200, 120);"
            "}"
        )
        # Force solid background on list too
        list_pal = self._list.palette()
        list_pal.setColor(self._list.backgroundRole(), QColor(30, 34, 42))
        self._list.setPalette(list_pal)
        self._list.setAutoFillBackground(True)

        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

    @property
    def list_widget(self) -> QListWidget:
        """Return the internal QListWidget."""
        return self._list

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle click on a list item.

        Args:
            item: The clicked QListWidgetItem.
        """
        # Skip disabled (header) items
        if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        row = self._list.row(item)
        self.item_selected.emit(row)
        self.hide()


class _DropdownButton(QWidget):
    """Custom dropdown replacing QComboBox to avoid Windows transparency bugs.

    Presents a QPushButton showing the current selection with a down-arrow
    indicator.  When clicked, shows a _DropdownPopup with a QListWidget
    for item selection.

    Provides a QComboBox-compatible API subset.
    """

    currentIndexChanged = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the dropdown button.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._items: list[tuple[str, Any]] = []  # (text, data)
        self._current_index: int = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._button = QPushButton()
        self._button.setStyleSheet(
            "QPushButton {"
            "  background-color: rgb(40, 44, 52);"
            "  color: #ffffff;"
            "  border: 1px solid rgba(100, 200, 255, 60);"
            "  border-radius: 6px;"
            "  padding: 4px 8px;"
            "  font-size: 14px;"
            "  text-align: left;"
            "}"
            "QPushButton:hover {"
            "  border: 1px solid rgba(100, 200, 255, 120);"
            "}"
        )
        self._button.clicked.connect(self._show_popup)
        layout.addWidget(self._button)

        self._popup = _DropdownPopup(self)
        self._popup.item_selected.connect(self._on_popup_selected)

        self._update_button_text()

    def addItem(self, text: str, data: Any = None) -> None:
        """Add an item to the dropdown.

        Args:
            text: Display text.
            data: Associated data (defaults to None).
        """
        self._items.append((text, data))
        list_item = QListWidgetItem(text)
        self._popup.list_widget.addItem(list_item)
        if self._current_index < 0 and data is not None:
            # Auto-select first enabled item
            self.setCurrentIndex(len(self._items) - 1)

    def addItems(self, texts: list[str]) -> None:
        """Add multiple items to the dropdown.

        Args:
            texts: List of display text strings (data = text).
        """
        for t in texts:
            self.addItem(t, t)

    def addGroupHeader(self, text: str) -> None:
        """Add a disabled header item for visual grouping.

        Args:
            text: Header text.
        """
        self._items.append((text, None))
        list_item = QListWidgetItem(text)
        list_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Disabled
        font = QFont()
        font.setBold(True)
        font.setPointSize(8)
        list_item.setFont(font)
        list_item.setForeground(QColor(140, 150, 160))
        self._popup.list_widget.addItem(list_item)

    def setMaxVisibleItems(self, count: int) -> None:
        """Set max visible items hint (controls popup height).

        Args:
            count: Number of visible items.
        """
        # Store for popup sizing
        self._max_visible = count

    def currentText(self) -> str:
        """Return the display text of the current selection.

        Returns:
            Current item text, or empty string if nothing selected.
        """
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][0]
        return ""

    def currentData(self) -> Any:
        """Return the data of the current selection.

        Returns:
            Current item data, or None if nothing selected.
        """
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def currentIndex(self) -> int:
        """Return the current selection index.

        Returns:
            Zero-based index, or -1 if nothing selected.
        """
        return self._current_index

    def setCurrentIndex(self, index: int) -> None:
        """Set the current selection by index.

        Args:
            index: Zero-based index to select.
        """
        if 0 <= index < len(self._items):
            old = self._current_index
            self._current_index = index
            self._update_button_text()
            if old != index:
                self.currentIndexChanged.emit(index)

    def setCurrentText(self, text: str) -> None:
        """Set the current selection by matching display text.

        Args:
            text: Text to search for.
        """
        for i, (t, _d) in enumerate(self._items):
            if t == text:
                self.setCurrentIndex(i)
                return

    def findData(self, data: Any) -> int:
        """Find the index of an item by its data value.

        Args:
            data: Data value to search for.

        Returns:
            Index of matching item, or -1 if not found.
        """
        for i, (_t, d) in enumerate(self._items):
            if d == data:
                return i
        return -1

    def count(self) -> int:
        """Return the total number of items.

        Returns:
            Item count.
        """
        return len(self._items)

    def itemData(self, index: int) -> Any:
        """Return the data for an item at a given index.

        Args:
            index: Zero-based index.

        Returns:
            Item data, or None if index is out of range.
        """
        if 0 <= index < len(self._items):
            return self._items[index][1]
        return None

    def model(self) -> None:
        """Compatibility stub — returns None (no Qt model).

        Returns:
            None.
        """
        return None

    def _update_button_text(self) -> None:
        """Update the button label to show current selection + arrow."""
        text = self.currentText() or "(select)"
        self._button.setText(f"  {text}  \u25BC")

    def _show_popup(self) -> None:
        """Show the dropdown popup below the button."""
        # Compute popup size
        popup_w = max(self.width(), 200)
        max_vis = getattr(self, "_max_visible", 12)
        item_h = 28  # approximate per-item height
        popup_h = min(len(self._items), max_vis) * item_h + 8
        popup_h = max(popup_h, 60)

        self._popup.setFixedSize(popup_w, popup_h)
        self._popup.list_widget.setFixedSize(popup_w - 2, popup_h - 2)

        # Position below the button in global coords
        global_pos = self._button.mapToGlobal(self._button.rect().bottomLeft())
        self._popup.move(global_pos)

        # Highlight current selection
        if 0 <= self._current_index < self._popup.list_widget.count():
            self._popup.list_widget.setCurrentRow(self._current_index)

        self._popup.show()

    def _on_popup_selected(self, index: int) -> None:
        """Handle selection from the popup list.

        Args:
            index: Index of the selected item.
        """
        self.setCurrentIndex(index)


class TagDialogPanel(QWidget):
    """Tag dialog with form fields, typewriter fill, and card glow.

    This is a top-level QWidget (frameless, opaque dark background,
    always-on-top) that floats over the QGraphicsView overlay.  Form
    fields are direct QWidget children -- no QGraphicsProxyWidget wrappers.

    Signals:
        confirmed: Emitted with form data dict when user confirms.
        dismissed: Emitted when user dismisses the dialog.
    """

    confirmed = pyqtSignal(dict)
    dismissed = pyqtSignal(dict)

    def __init__(
        self,
        clock: AnimationClock,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the tag dialog panel.

        Args:
            clock: AnimationClock for tick-driven animations.
            parent: Optional parent QWidget.
        """
        super().__init__(parent)
        self.setObjectName("TagDialog")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setStyleSheet(_DIALOG_QSS)

        self._panel_width: int = 400
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
        self._loading: bool = False

        # Field index mapping for typewriter interruption
        self._typewriter_field_map: dict[QLineEdit, int] = {}

        # Build form layout
        self._widgets: dict[str, QWidget] = {}
        self._field_labels: dict[str, QLabel] = {}
        self._field_tips: dict[str, QLabel] = {}
        self._field_rows: dict[str, QWidget] = {}
        self._build_ui()

        # Register tick callback
        self._clock.register(self._tick)

        # Size the widget — fixed width, height managed via _settle_size()
        self.setFixedWidth(self._panel_width)

        # Drag state for frameless window movement
        self._drag_pos: QPoint | None = None
        self._drag_zone_height: int = 40  # pixels from top edge

        # Show move cursor when hovering the drag zone
        self.setMouseTracking(True)

        logger.debug("TagDialogPanel created (QWidget-based)")

    # ------------------------------------------------------------------
    # UI construction
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

    def _make_field_row(
        self,
        key: str,
        label_text: str,
        widget: QWidget,
        tip_text: str | None = None,
    ) -> QWidget:
        """Create a vertical container: label + field + optional tip.

        Args:
            key: Unique key for lookup.
            label_text: Label text above the field.
            widget: The form widget (QLineEdit, _DropdownButton, etc.).
            tip_text: Optional helper tip below the field.

        Returns:
            Container QWidget holding the row.
        """
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label = QLabel(label_text)
        label.setFont(self._make_font(FONT_SIZE_LABEL, FONT_WEIGHT_REGULAR))
        layout.addWidget(label)
        self._field_labels[key] = label

        widget.setFont(self._make_font(FONT_SIZE_INPUT))
        layout.addWidget(widget)
        self._widgets[key] = widget

        if tip_text:
            tip = QLabel(tip_text)
            tip.setFont(self._make_font(FONT_SIZE_HELPER))
            tip.setWordWrap(True)
            layout.addWidget(tip)
            self._field_tips[key] = tip

        self._field_rows[key] = container
        return container

    def _build_ui(self) -> None:
        """Build the full form layout with all fields."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        main_layout.setSpacing(SPACING.sm)

        # -- Label field --
        label_edit = QLineEdit()
        label_edit.setPlaceholderText("Element label...")
        label_edit.textEdited.connect(lambda _: self._on_user_edit(label_edit))
        main_layout.addWidget(
            self._make_field_row("label", "Label", label_edit),
        )

        # -- Caption field --
        caption_edit = QLineEdit()
        caption_edit.setPlaceholderText("What this element does...")
        caption_edit.textEdited.connect(lambda _: self._on_user_edit(caption_edit))
        main_layout.addWidget(
            self._make_field_row("caption", "Caption", caption_edit),
        )

        # -- Action Type dropdown --
        action_dropdown = _DropdownButton()
        for at in _ACTION_TYPES:
            action_dropdown.addItem(at, at)
        action_dropdown.currentIndexChanged.connect(self._on_action_type_changed)
        main_layout.addWidget(
            self._make_field_row("action_type", "Action Type", action_dropdown),
        )

        # -- Element Type dropdown (grouped with separator headers) --
        elem_dropdown = _DropdownButton()
        elem_dropdown.setMaxVisibleItems(20)
        self._populate_element_type_dropdown(elem_dropdown)
        main_layout.addWidget(
            self._make_field_row("element_type", "Element Type", elem_dropdown),
        )
        # Select "unknown" as default
        self._select_element_type(elem_dropdown, "unknown")

        # -- Conditional fields --

        # Text to type (action_type == "type")
        text_edit = QLineEdit()
        text_edit.setPlaceholderText("Text to enter...")
        main_layout.addWidget(
            self._make_field_row(
                "text_to_type", "Text to type", text_edit,
                tip_text=_HELPER_TIPS["text_to_type"],
            ),
        )

        # Press Enter checkbox (action_type == "type")
        press_enter = QCheckBox("Press Enter to send?")
        press_enter.setFont(self._make_font(FONT_SIZE_INPUT))
        # No label row for checkbox — it's self-labeling
        self._widgets["press_enter"] = press_enter
        press_enter_container = QWidget()
        press_enter_container.setStyleSheet("background: transparent;")
        pe_layout = QVBoxLayout(press_enter_container)
        pe_layout.setContentsMargins(0, 0, 0, 0)
        pe_layout.setSpacing(2)
        pe_layout.addWidget(press_enter)
        tip_pe = QLabel(_HELPER_TIPS["press_enter"])
        tip_pe.setFont(self._make_font(FONT_SIZE_HELPER))
        tip_pe.setWordWrap(True)
        pe_layout.addWidget(tip_pe)
        self._field_tips["press_enter"] = tip_pe
        self._field_rows["press_enter"] = press_enter_container
        main_layout.addWidget(press_enter_container)

        # Direction/Amount (action_type == "scroll")
        dir_edit = QLineEdit()
        dir_edit.setPlaceholderText("Direction and pixel amount to scroll")
        main_layout.addWidget(
            self._make_field_row(
                "direction_amount", "Direction/Amount", dir_edit,
                tip_text=_HELPER_TIPS["direction_amount"],
            ),
        )

        # Condition/Timeout (action_type == "wait")
        cond_edit = QLineEdit()
        cond_edit.setPlaceholderText("Condition to wait for, or timeout in seconds")
        main_layout.addWidget(
            self._make_field_row(
                "condition_timeout", "Condition/Timeout", cond_edit,
                tip_text=_HELPER_TIPS["condition_timeout"],
            ),
        )

        # Drag target hint (action_type == "click_drag")
        drag_edit = QLineEdit()
        drag_edit.setPlaceholderText(
            "I'll capture the drag destination after you confirm this step",
        )
        main_layout.addWidget(
            self._make_field_row(
                "drag_target_hint", "Drag target", drag_edit,
                tip_text=_HELPER_TIPS["drag_target_hint"],
            ),
        )

        # VLM prompt (action_type in ("read", "snip_and_search"))
        vlm_edit = QLineEdit()
        vlm_edit.setPlaceholderText("What to look for...")
        main_layout.addWidget(
            self._make_field_row(
                "vlm_prompt", "VLM prompt", vlm_edit,
                tip_text=_HELPER_TIPS["vlm_prompt"],
            ),
        )

        # Question text (action_type == "prompt_user")
        q_edit = QLineEdit()
        q_edit.setPlaceholderText("Question for the user...")
        main_layout.addWidget(
            self._make_field_row(
                "question_text", "Question", q_edit,
                tip_text=_HELPER_TIPS["question_text"],
            ),
        )

        # -- Buttons row --
        btn_row = QWidget()
        btn_row.setStyleSheet("background: transparent;")
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 4, 0, 0)
        btn_layout.setSpacing(SPACING.sm)

        confirm_btn = QPushButton("Confirm")
        confirm_btn.setStyleSheet(
            f"background: rgba({CONFIRM_BG.red()}, {CONFIRM_BG.green()}, "
            f"{CONFIRM_BG.blue()}, {CONFIRM_BG.alpha()}); "
            "color: #ffffff; "
            "border-radius: 4px; padding: 6px 12px; "
            f"font-size: {FONT_SIZE_INPUT}px; font-weight: 500;"
        )
        confirm_btn.clicked.connect(self.confirm)
        self._widgets["confirm_btn"] = confirm_btn
        btn_layout.addWidget(confirm_btn)

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setStyleSheet(
            f"background: rgba({DISMISS_BG.red()}, {DISMISS_BG.green()}, "
            f"{DISMISS_BG.blue()}, {DISMISS_BG.alpha()}); "
            "color: #e8e8e8; "
            "border-radius: 4px; padding: 6px 12px; "
            f"font-size: {FONT_SIZE_INPUT}px; font-weight: 500;"
        )
        dismiss_btn.clicked.connect(self.dismiss)
        self._widgets["dismiss_btn"] = dismiss_btn
        btn_layout.addWidget(dismiss_btn)

        main_layout.addWidget(btn_row)

        # Initial layout: hide all conditional fields
        self._on_action_type_changed(0)

    def _populate_element_type_dropdown(self, dropdown: _DropdownButton) -> None:
        """Fill element type dropdown with grouped items and separator headers.

        Args:
            dropdown: The _DropdownButton to populate.
        """
        for group_label, values in _TYPE_GROUPS:
            dropdown.addGroupHeader(group_label)
            for val in values:
                dropdown.addItem(val, val)

    def _select_element_type(
        self, dropdown: _DropdownButton, value: str,
    ) -> None:
        """Select an element type in the dropdown by value string.

        Args:
            dropdown: The _DropdownButton to search.
            value: The element type string to select.
        """
        idx = dropdown.findData(value)
        if idx >= 0:
            dropdown.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _settle_size(self) -> None:
        """Collapse layout to final size in one step to avoid resize spam.

        Unlocks the height constraint, lets the layout engine compute
        the ideal size, then locks it with setFixedSize so Qt never
        requests an intermediate geometry that Windows would adjust by
        ~19 px (the DPI-rounding delta that triggers
        QWindowsWindow::setGeometry warnings).
        """
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        size = self.sizeHint()
        self.setFixedSize(self._panel_width, size.height())

    # ------------------------------------------------------------------
    # Action type change handler
    # ------------------------------------------------------------------

    def _on_action_type_changed(self, index: int) -> None:
        """Handle action type dropdown selection change.

        Shows/hides conditional fields and relayouts the panel.

        Args:
            index: New combo index (unused, reads current data).
        """
        action_w = self._widgets.get("action_type")
        if not isinstance(action_w, _DropdownButton):
            return
        current_action = action_w.currentData()

        # Determine which conditional field keys to show
        visible_keys: set[str] = set()
        if current_action in _CONDITIONAL_FIELDS:
            visible_keys = set(_CONDITIONAL_FIELDS[current_action])

        for key in _ALL_CONDITIONAL:
            row = self._field_rows.get(key)
            if row is not None:
                row.setVisible(key in visible_keys)

        # Collapse to final size in one step (avoids DPI resize spam)
        self._settle_size()

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
            element_rect: Bounding rect of the captured element in screen coords.
            vlm_data: Optional VLM-predicted field values dict.
            edit_mode: If True, pre-fill fields instantly without typewriter.
            screen_w: Screen width for positioning clamping.
            screen_h: Screen height for positioning clamping.
        """
        self._fading_out = False

        # Re-register tick callback (unregistered on dismiss fade-out completion).
        self._clock.unregister(self._tick)
        self._clock.register(self._tick)

        # Clear fields for fresh dialog
        for key in ("label", "caption"):
            w = self._widgets.get(key)
            if isinstance(w, QLineEdit):
                w.clear()

        # Clear conditional text fields too
        for key in ("text_to_type", "direction_amount", "condition_timeout",
                     "drag_target_hint", "vlm_prompt", "question_text"):
            w = self._widgets.get(key)
            if isinstance(w, QLineEdit):
                w.clear()

        if vlm_data is None:
            # Loading state — show spinner, hide all form rows
            self._loading = True
            self._set_form_visible(False)
        else:
            # Data available — populate immediately
            self._loading = False
            self._set_form_visible(True)
            self._populate_fields(vlm_data, edit_mode)
            # Re-apply conditional field visibility
            action_w = self._widgets.get("action_type")
            if isinstance(action_w, _DropdownButton):
                self._on_action_type_changed(action_w.currentIndex())

        # Compute position and show
        self._settle_size()
        x, y = self._compute_position(element_rect, screen_w, screen_h)
        self.move(int(x), int(y))

        # Fade in
        self._opacity = 0.0
        self._target_opacity = 1.0
        self.show()
        self.raise_()
        self.activateWindow()

        logger.debug(
            "TagDialogPanel shown at (%d, %d), edit_mode=%s, loading=%s",
            int(x), int(y), edit_mode, self._loading,
        )

    def _set_form_visible(self, visible: bool) -> None:
        """Show or hide all form field rows.

        Args:
            visible: Whether form fields should be visible.
        """
        for key in list(self._field_rows.keys()):
            self._field_rows[key].setVisible(visible)
        # Also toggle buttons
        for key in ("confirm_btn", "dismiss_btn"):
            w = self._widgets.get(key)
            if w is not None:
                p = w.parentWidget()
                if p is not None:
                    p.setVisible(visible)

    def _populate_fields(
        self,
        vlm_data: dict,
        edit_mode: bool = False,
    ) -> None:
        """Fill form fields from VLM data dict.

        Args:
            vlm_data: VLM-predicted field values dict.
            edit_mode: If True, pre-fill instantly without typewriter.
        """
        if not vlm_data:
            return

        # Set dropdown fields instantly (dropdowns don't typewrite)
        if "action_type" in vlm_data:
            dropdown = self._widgets.get("action_type")
            if isinstance(dropdown, _DropdownButton):
                idx = dropdown.findData(vlm_data["action_type"])
                if idx >= 0:
                    dropdown.setCurrentIndex(idx)

        if "element_type" in vlm_data:
            dropdown = self._widgets.get("element_type")
            if isinstance(dropdown, _DropdownButton):
                self._select_element_type(dropdown, vlm_data["element_type"])

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
            label_w = self._widgets.get("label")
            caption_w = self._widgets.get("caption")
            if isinstance(label_w, QLineEdit):
                label_w.setText(vlm_data.get("label", ""))
            if isinstance(caption_w, QLineEdit):
                caption_w.setText(vlm_data.get("caption", ""))
        else:
            # Start typewriter for label and caption
            tw_fields: list[tuple[QLineEdit, str]] = []
            label_w = self._widgets.get("label")
            caption_w = self._widgets.get("caption")
            if isinstance(label_w, QLineEdit):
                label_w.clear()
                tw_fields.append((label_w, vlm_data.get("label", "")))
                self._typewriter_field_map[label_w] = 0
            if isinstance(caption_w, QLineEdit):
                caption_w.clear()
                tw_fields.append((caption_w, vlm_data.get("caption", "")))
                self._typewriter_field_map[caption_w] = 1
            if tw_fields:
                self._typewriter.start(tw_fields)

    def populate_data(self, vlm_data: dict) -> None:
        """Transition from loading state to populated fields.

        Called when VLM results arrive after the dialog was shown in
        loading mode.

        Args:
            vlm_data: VLM analysis result dict (may be empty for manual entry).
        """
        self._loading = False

        # Populate field values
        self._populate_fields(vlm_data, edit_mode=False)

        # Show all form rows
        self._set_form_visible(True)

        # Re-apply conditional field visibility
        action_w = self._widgets.get("action_type")
        if isinstance(action_w, _DropdownButton):
            self._on_action_type_changed(action_w.currentIndex())

        self._settle_size()
        self.update()

        logger.debug("TagDialogPanel populated with VLM data, loading=False")

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
            Tuple of (x, y) for widget position in screen coords.
        """
        breathing = float(SPACING.lg)  # 24px
        gap = float(SPACING.md)  # 16px
        w = self.width()
        h = self.height()

        # Prefer below-right
        x = element_rect.right() + gap
        y = element_rect.top()

        # Flip horizontal if overflows right
        if x + w + breathing > screen_w:
            x = element_rect.left() - gap - w

        # Flip vertical if overflows bottom
        if y + h + breathing > screen_h:
            y = screen_h - h - breathing

        # Clamp
        x = max(breathing, min(x, screen_w - w - breathing))
        y = max(breathing, min(y, screen_h - h - breathing))

        return x, y

    def dismiss(self) -> None:
        """Fade out and emit dismissed signal."""
        self._target_opacity = 0.0
        self._fading_out = True
        self._loading = False
        self._typewriter.stop()
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
        data["action_type"] = (
            action_w.currentData()
            if isinstance(action_w, _DropdownButton)
            else "click"
        )

        elem_w = self._widgets.get("element_type")
        data["element_type"] = (
            elem_w.currentData()
            if isinstance(elem_w, _DropdownButton)
            else "unknown"
        )

        # Conditional fields — always include values (not gated on visibility
        # because dismiss() may have already hidden things by the time toolbar
        # confirm reads form data).
        for key in (
            "text_to_type", "direction_amount", "condition_timeout",
            "drag_target_hint", "vlm_prompt", "question_text",
        ):
            w = self._widgets.get(key)
            if isinstance(w, QLineEdit):
                data[key] = w.text()

        press_w = self._widgets.get("press_enter")
        if isinstance(press_w, QCheckBox):
            data["press_enter"] = press_w.isChecked()

        return data

    def get_avoidance_rect(self) -> QRectF:
        """Return the screen bounding rect for shimmer avoidance.

        Returns:
            QRectF in screen coordinates covering this panel.
        """
        geo = self.geometry()
        return QRectF(geo.x(), geo.y(), geo.width(), geo.height())

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Frameless window drag support
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Begin drag if the click is in the title/header drag zone.

        Args:
            event: The mouse press event.
        """
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().y() < self._drag_zone_height
        ):
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Move the dialog when dragging; update cursor in drag zone.

        Args:
            event: The mouse move event.
        """
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            # Update cursor based on whether the pointer is in the drag zone
            if event.position().y() < self._drag_zone_height:
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            else:
                self.unsetCursor()
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Clear drag state on mouse release.

        Args:
            event: The mouse release event.
        """
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: Any) -> None:
        """Paint card glow, highlight gradient, and loading spinner.

        Args:
            event: The paint event.
        """
        if self._opacity < 0.01:
            return

        painter = QPainter(self)
        painter.setOpacity(self._opacity)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        frect = QRectF(rect)

        # Card border glow — clip to OUTSIDE the card so nothing bleeds through
        card_path = QPainterPath()
        card_path.addRoundedRect(frect, self._corner_radius, self._corner_radius)

        outer = QPainterPath()
        margin = 80.0  # glow reach
        outer.addRect(frect.adjusted(-margin, -margin, margin, margin))
        glow_clip = outer - card_path

        painter.save()
        painter.setClipPath(glow_clip)
        paint_card_glow(
            painter,
            frect,
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
        gradient = QLinearGradient(0, 0, 0, frect.height() * 0.3)
        gradient.setColorAt(0.0, HIGHLIGHT_TOP)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(gradient)
        painter.drawPath(card_path)

        # Loading spinner
        if self._loading:
            self._paint_spinner(painter, frect)

        painter.end()

    def _paint_spinner(self, painter: QPainter, rect: QRectF) -> None:
        """Paint a spinning arc loader in the center of the dialog.

        Args:
            painter: Active QPainter (already has opacity set).
            rect: The panel content rect.
        """
        radius = 40.0
        stroke = 3.0
        cx = rect.width() / 2.0
        cy = rect.height() / 2.0 - 12.0

        spinner_rect = QRectF(
            cx - radius, cy - radius,
            radius * 2.0, radius * 2.0,
        )

        rotation_deg = self._glow_phase * 360.0 * 1.875

        accent = QColor(0, 200, 220)

        # Subtle glow behind spinner
        glow_pen = QPen(QColor(0, 200, 220, 50))
        glow_pen.setWidthF(stroke + 4.0)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        start_angle_16 = int(rotation_deg * 16.0) % (360 * 16)
        span_16 = 270 * 16
        painter.drawArc(spinner_rect, start_angle_16, span_16)

        # Main spinner arc
        arc_pen = QPen(accent)
        arc_pen.setWidthF(stroke)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        painter.drawArc(spinner_rect, start_angle_16, span_16)

        # "Analyzing element..." text below spinner
        painter.setPen(QColor(200, 200, 210, 180))
        font = QFont()
        if FONT_FAMILY:
            font.setFamily(FONT_FAMILY)
        font.setPixelSize(FONT_SIZE_HELPER)
        font.setWeight(QFont.Weight(FONT_WEIGHT_LIGHT))
        painter.setFont(font)
        text_rect = QRectF(0, cy + radius + 12.0, rect.width(), 20.0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter, "Analyzing element...")

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def set_glow_pulsing(self, enabled: bool) -> None:
        """Enable or disable glow pulsing as a loading indicator.

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
        self._glow_phase += dt * 0.8

        # Glow brightness: pulse mode or decay mode
        if self._glow_pulsing:
            self._glow_brightness = 1.25 + 0.75 * math.sin(self._glow_phase * 3.0)
        else:
            self._glow_brightness += (
                (1.0 - self._glow_brightness) * min(1.0, dt * 8.0)
            )

        # Opacity interpolation
        speed = 1000.0 / FADE_IN_MS if self._target_opacity > 0.5 else 1000.0 / FADE_OUT_MS
        self._opacity += (
            (self._target_opacity - self._opacity) * min(1.0, dt * speed)
        )

        # Check if fade-out complete
        if self._fading_out and self._opacity < 0.01:
            self._opacity = 0.0
            self._fading_out = False
            self._clock.unregister(self._tick)
            self.hide()
            self.dismissed.emit({})

        self.setWindowOpacity(self._opacity)
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

    def keyPressEvent(self, event: Any) -> None:
        """Handle Enter (confirm) and Escape (dismiss) shortcuts.

        Args:
            event: The key event.
        """
        if isinstance(event, QKeyEvent):
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.confirm()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape:
                self.dismiss()
                event.accept()
                return
        super().keyPressEvent(event)
