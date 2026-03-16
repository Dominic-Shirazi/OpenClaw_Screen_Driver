"""PyQt6 element tagging dialog for recording sessions.

Modal dialog that opens when the user clicks/boxes a UI element during
recording. Allows tagging the element with type, label, layer, and notes.

The element type dropdown is grouped by category (Interactive, Structural,
Static, Meta) with separator headers and a live description panel that
shows examples for the currently selected type.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from recorder.element_types import ElementType

logger = logging.getLogger(__name__)


# ── Element type metadata ──────────────────────────────────────────────
# Maps each ElementType value to (short_desc, example_text).
# short_desc shows in the dropdown item.  example_text shows in the
# explainer panel below the dropdown when that item is selected.

_TYPE_INFO: dict[str, tuple[str, str]] = {
    # Interactive
    "textbox": (
        "Input field",
        'e.g. "Username", "Search bar", "Email address"',
    ),
    "button": (
        "Click — stays on page",
        'e.g. "Save", "Apply", "Cancel", "Add to cart"',
    ),
    "button_nav": (
        "Click — navigates away",
        'e.g. "Login", "Submit", "Next page", "Go to dashboard"',
    ),
    "toggle": (
        "On/off switch",
        'e.g. checkbox, radio button, "Dark mode" toggle',
    ),
    "tab": (
        "Tab within page",
        'e.g. "General | Advanced | Privacy" tab bar',
    ),
    "dropdown": (
        "Expands a list",
        'e.g. "Select country ▾", "Sort by ▾", combobox',
    ),
    "scrollbar": (
        "Scroll target",
        "Vertical or horizontal scrollbar track",
    ),
    "link": (
        "Hyperlink",
        'e.g. "Forgot password?", "Terms of Service", nav link',
    ),
    "icon": (
        "Clickable icon",
        'e.g. ☰ hamburger, ⚙ gear, ✕ close, 🔔 notification bell',
    ),
    "drag_source": (
        "Drag starts here",
        "e.g. draggable card, file icon, slider thumb",
    ),
    "drag_target": (
        "Drag ends here",
        "e.g. drop zone, trash area, folder target",
    ),
    # Structural regions
    "region_chrome": (
        "App/browser chrome",
        "Title bar, tab strip, address bar, nav buttons — NOT page content",
    ),
    "region_menu": (
        "Menu area",
        "Top menu bar, hamburger menu panel, context menu",
    ),
    "region_sidebar": (
        "Sidebar / nav panel",
        "Left/right nav panel, folder tree, chat list",
    ),
    "region_content": (
        "Main content area",
        "Article body, feed, dashboard grid — the big center block",
    ),
    "region_form": (
        "Form group",
        "Login form, signup fields, settings section — inputs as a unit",
    ),
    "region_header": (
        "Page/section header",
        "Hero banner, page title row, section heading area",
    ),
    "region_footer": (
        "Page/section footer",
        "Footer links, copyright bar, pagination strip",
    ),
    "region_toolbar": (
        "Toolbar / action bar",
        "Editor toolbar, formatting buttons, bulk-action bar",
    ),
    "region_modal": (
        "Popup/dialog box",
        "Modal overlay, confirmation dialog, lightbox",
    ),
    "region_custom": (
        "Custom region",
        "Describe in Notes — any named zone not covered above",
    ),
    "landmark": (
        "Visual anchor",
        "Logo, distinctive icon, unique badge — helps YOLO-E orient",
    ),
    # Static / read-only
    "image": (
        "Non-interactive image",
        "Photo, illustration, chart — may contain text (OCR target)",
    ),
    "read_here": (
        "Data extraction point",
        'e.g. "Order total: $42.50" — read this value during replay',
    ),
    "notification": (
        "Toast / alert",
        '"Saved!", "Error: invalid email", status badge',
    ),
    "modal": (
        "Popup (legacy)",
        "Use region_modal for new recordings",
    ),
    # Meta
    "destination": (
        "Success state",
        "The page/screen that means 'we made it'",
    ),
    "branch_point": (
        "Conditional fork",
        "If logged in → dashboard, else → login page",
    ),
    "fingerprint": (
        "App identity check",
        "YOLO-E verifies we're in the right app/page state",
    ),
    "unknown": (
        "Unclassified",
        "Tag it manually — AI couldn't determine the type",
    ),
}

# Ordered groups for the dropdown (label, list of ElementType values)
_TYPE_GROUPS: list[tuple[str, list[str]]] = [
    ("── Interactive ──", [
        "textbox", "button", "button_nav", "link", "icon",
        "toggle", "tab", "dropdown", "scrollbar",
        "drag_source", "drag_target",
    ]),
    ("── Regions (visual grounding) ──", [
        "region_chrome", "region_menu", "region_sidebar",
        "region_content", "region_form", "region_header",
        "region_footer", "region_toolbar", "region_modal",
        "region_custom", "landmark",
    ]),
    ("── Static / Read-only ──", [
        "image", "read_here", "notification", "modal",
    ]),
    ("── Meta / Flow control ──", [
        "destination", "branch_point", "fingerprint", "unknown",
    ]),
]


class TagDialog(QDialog):
    """Modal dialog for tagging UI elements during a recording session.

    Allows the user to specify the type, label, layer, and notes for a
    detected element. Pre-fills fields from VLM/OCR/UIA guesses.

    The element type dropdown is grouped by category with separator
    headers and a live explainer panel.
    """

    def __init__(
        self,
        element_type_guess: str | ElementType,
        label_guess: str,
        ocr_text: str | None,
        layer_guess: str,
        uia_hint: str | None,
        x: int,
        y: int,
        parent: QWidget | None = None,
        is_bbox: bool = False,
    ) -> None:
        """Initializes the TagDialog.

        Args:
            element_type_guess: AI-predicted element type (string or ElementType).
            label_guess: AI-predicted label for the element.
            ocr_text: Text detected via OCR on the element.
            layer_guess: Predicted UI layer (os_ui, app_persistent, page_specific).
            uia_hint: Raw hint from the accessibility tree.
            x: X-coordinate for dialog positioning.
            y: Y-coordinate for dialog positioning.
            parent: Parent widget.
            is_bbox: If True, element was selected via bounding box (shows tip).
        """
        super().__init__(parent)
        self.setWindowTitle("Tag Element")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self._result_data: dict[str, Any] | None = None
        self._ocr_text = ocr_text
        self._uia_hint = uia_hint

        # Layout setup
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        form_layout = QFormLayout()
        form_layout.setSpacing(4)

        # ── Element Type dropdown (grouped with separators) ──
        self.type_combo = QComboBox()
        self.type_combo.setMaxVisibleItems(20)
        self._populate_type_combo()

        # Set initial type guess
        guess_val = (
            element_type_guess.value
            if isinstance(element_type_guess, ElementType)
            else str(element_type_guess)
        )
        self._select_type_value(guess_val)

        form_layout.addRow("Type:", self.type_combo)

        # ── Explainer label (updates on selection change) ──
        self._explainer = QLabel()
        self._explainer.setWordWrap(True)
        self._explainer.setStyleSheet(
            "color: #888; font-size: 11px; padding: 2px 4px; "
            "background: #1a1a2e; border-radius: 3px;"
        )
        self._explainer.setMinimumHeight(28)
        layout.addLayout(form_layout)
        layout.addWidget(self._explainer)
        self.type_combo.currentIndexChanged.connect(self._update_explainer)
        self._update_explainer()  # Set initial text

        # ── Label text input ──
        form2 = QFormLayout()
        form2.setSpacing(4)
        self.label_input = QLineEdit(label_guess)
        self.label_input.setPlaceholderText("Human-readable name for this element")
        form2.addRow("Label:", self.label_input)

        # ── Layer dropdown ──
        self.layer_combo = QComboBox()
        _LAYER_ITEMS = [
            ("page_specific", "Page-specific — unique to this page/state"),
            ("app_persistent", "App-persistent — appears across pages (navbar, sidebar)"),
            ("os_ui", "OS UI — taskbar, system tray, window controls"),
        ]
        for value, display in _LAYER_ITEMS:
            self.layer_combo.addItem(display, value)
        # Select the guessed layer
        for i in range(self.layer_combo.count()):
            if self.layer_combo.itemData(i) == layer_guess:
                self.layer_combo.setCurrentIndex(i)
                break
        form2.addRow("Layer:", self.layer_combo)

        # ── Notes free text ──
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText(
            "Additional context, instructions, or describe custom regions..."
        )
        self.notes_input.setMaximumHeight(60)
        form2.addRow("Notes:", self.notes_input)

        layout.addLayout(form2)

        # ── Input Spec section (visible when type is textbox) ──
        self._input_spec_widget = QWidget()
        input_spec_layout = QVBoxLayout(self._input_spec_widget)
        input_spec_layout.setContentsMargins(0, 4, 0, 4)
        input_spec_layout.setSpacing(4)

        spec_header = QLabel("Input Spec — what gets typed during replay?")
        spec_header.setStyleSheet(
            "color: #aaa; font-size: 11px; font-weight: bold;"
        )
        input_spec_layout.addWidget(spec_header)

        spec_form = QFormLayout()
        spec_form.setSpacing(4)

        self._input_mode_combo = QComboBox()
        self._input_mode_combo.addItem(
            "Variable — agent fills at runtime", "variable"
        )
        self._input_mode_combo.addItem(
            "Literal — same text every replay", "literal"
        )
        spec_form.addRow("Mode:", self._input_mode_combo)

        self._input_value_edit = QLineEdit()
        self._input_value_edit.setPlaceholderText("enter_text")
        spec_form.addRow("Value:", self._input_value_edit)

        self._input_hint_label = QLabel()
        self._input_hint_label.setWordWrap(True)
        self._input_hint_label.setStyleSheet("color: #666; font-size: 10px;")
        input_spec_layout.addLayout(spec_form)
        input_spec_layout.addWidget(self._input_hint_label)

        self._input_mode_combo.currentIndexChanged.connect(
            self._update_input_spec_hint
        )
        self._update_input_spec_hint()

        layout.addWidget(self._input_spec_widget)
        # Start hidden — shown when type changes to textbox
        self._input_spec_widget.setVisible(False)
        self.type_combo.currentIndexChanged.connect(self._toggle_input_spec)
        self._toggle_input_spec()  # Set initial visibility

        # ── Action buttons ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        confirm_btn = QPushButton("✓ Confirm")
        confirm_btn.setDefault(True)
        confirm_btn.setStyleSheet(
            "QPushButton { background: #2d5a2d; color: white; "
            "padding: 6px 14px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #3a7a3a; }"
        )
        confirm_btn.clicked.connect(self._on_confirm)

        skip_btn = QPushButton("Skip")
        skip_btn.setStyleSheet(
            "QPushButton { padding: 6px 10px; border-radius: 4px; }"
        )
        skip_btn.clicked.connect(self.reject)

        branch_btn = QPushButton("🔀 Branch")
        branch_btn.setToolTip("Mark as a conditional fork (if/else)")
        branch_btn.setStyleSheet(
            "QPushButton { padding: 6px 10px; border-radius: 4px; }"
        )
        branch_btn.clicked.connect(self._on_branch)

        dest_btn = QPushButton("🎯 Destination")
        dest_btn.setToolTip("Mark as success state — 'we made it here'")
        dest_btn.setStyleSheet(
            "QPushButton { padding: 6px 10px; border-radius: 4px; }"
        )
        dest_btn.clicked.connect(self._on_destination)

        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(skip_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(branch_btn)
        btn_layout.addWidget(dest_btn)

        layout.addLayout(btn_layout)

        # ── Bbox tip (visible only when element was selected via bbox) ──
        if is_bbox:
            bbox_tip = QLabel(
                "Tip: tight boxes improve replay accuracy. "
                "Drag edges close to the element."
            )
            bbox_tip.setStyleSheet("color: #666; font-size: 10px;")
            bbox_tip.setWordWrap(True)
            layout.addWidget(bbox_tip)

        # ── Voice Input button (stubbed for Stage 1) ──
        voice_btn = QPushButton("🎤 Voice Input (Stage 2)")
        voice_btn.setEnabled(False)
        voice_btn.setMaximumHeight(24)
        font = voice_btn.font()
        font.setPointSize(8)
        voice_btn.setFont(font)
        layout.addWidget(voice_btn)

        # Dialog size — taller to fit explainer
        self.setFixedSize(520, 420)

        # Position near the clicked element, clamped to screen bounds
        self._clamp_to_screen(x + 20, y + 20)

        # Focus the label input so user can start typing immediately
        self.label_input.setFocus()
        self.label_input.selectAll()

        logger.debug(
            "TagDialog initialized at (%d, %d) with guess '%s'",
            x + 20,
            y + 20,
            element_type_guess,
        )

    def _clamp_to_screen(self, target_x: int, target_y: int) -> None:
        """Moves the dialog to (target_x, target_y), clamped to screen bounds.

        Ensures the dialog is fully visible — even if the element is near
        the taskbar, screen edges, or on a secondary monitor.

        Args:
            target_x: Desired X position.
            target_y: Desired Y position.
        """
        from PyQt6.QtWidgets import QApplication

        screen = QApplication.screenAt(self.pos())
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            self.move(target_x, target_y)
            return

        geom = screen.availableGeometry()  # excludes taskbar
        dw, dh = self.width(), self.height()

        # Clamp so the full dialog fits within available screen area
        final_x = max(geom.x(), min(target_x, geom.x() + geom.width() - dw))
        final_y = max(geom.y(), min(target_y, geom.y() + geom.height() - dh))

        self.move(final_x, final_y)

    def _populate_type_combo(self) -> None:
        """Fills the type dropdown with grouped items and separator headers."""
        for group_label, values in _TYPE_GROUPS:
            # Add group header as a disabled separator item
            self.type_combo.addItem(group_label, None)
            idx = self.type_combo.count() - 1
            # Make it non-selectable and visually distinct
            model = self.type_combo.model()
            if model is not None:
                item = model.item(idx)
                if item is not None:
                    item.setEnabled(False)
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(8)
                    item.setFont(font)

            # Add actual items with short description
            for val in values:
                et = ElementType(val)
                short_desc = _TYPE_INFO.get(val, ("", ""))[0]
                if short_desc:
                    display = f"{val}  —  {short_desc}"
                else:
                    display = val
                self.type_combo.addItem(display, et)

    def _select_type_value(self, value: str) -> None:
        """Selects a type in the combo by its ElementType value string."""
        for i in range(self.type_combo.count()):
            data = self.type_combo.itemData(i)
            if isinstance(data, ElementType) and data.value == value:
                self.type_combo.setCurrentIndex(i)
                return
        # Fallback: try to find "unknown"
        self._select_type_value("unknown") if value != "unknown" else None

    def _update_explainer(self) -> None:
        """Updates the explainer label based on the current dropdown selection."""
        data = self.type_combo.currentData()
        if isinstance(data, ElementType):
            val = data.value
            info = _TYPE_INFO.get(val)
            if info:
                short, example = info
                self._explainer.setText(f"💡 {example}")
            else:
                self._explainer.setText("")
        else:
            self._explainer.setText("")

    def _toggle_input_spec(self) -> None:
        """Shows/hides the Input Spec section based on element type."""
        data = self.type_combo.currentData()
        is_textbox = isinstance(data, ElementType) and data.value == "textbox"
        self._input_spec_widget.setVisible(is_textbox)
        # Adjust dialog height when input spec is shown
        if is_textbox:
            self.setFixedSize(520, 520)
        else:
            self.setFixedSize(520, 420)

    def _update_input_spec_hint(self) -> None:
        """Updates the hint text below the input spec value field."""
        mode = self._input_mode_combo.currentData()
        if mode == "variable":
            self._input_value_edit.setPlaceholderText("enter_text")
            self._input_hint_label.setText(
                "Variable name — the agent passes the value at runtime.\n"
                "Stored as {{variable_name}} in the skill JSON."
            )
        elif mode == "literal":
            self._input_value_edit.setPlaceholderText("https://example.com")
            self._input_hint_label.setText(
                "This exact text will be typed every replay. Use for fixed "
                "values only (URLs, app names, search terms) — NOT for "
                "credentials or anything that changes."
            )

    def _collect_data(self) -> dict[str, Any]:
        """Collects current field values into a dict.

        Returns:
            Dict with element_type, label, layer, notes, is_branch,
            ocr_text, uia_hint, and optionally input_spec.
        """
        data: dict[str, Any] = {
            "element_type": self.type_combo.currentData(),
            "label": self.label_input.text().strip(),
            "layer": self.layer_combo.currentData(),
            "notes": self.notes_input.toPlainText().strip(),
            "is_branch": False,
            "ocr_text": self._ocr_text,
            "uia_hint": self._uia_hint,
        }

        # Add input_spec for textbox elements
        et = self.type_combo.currentData()
        if isinstance(et, ElementType) and et.value == "textbox":
            mode = self._input_mode_combo.currentData()
            raw_value = self._input_value_edit.text().strip()
            if mode == "variable":
                # Default to "enter_text" if user left it blank
                var_name = raw_value or "enter_text"
                data["input_spec"] = {
                    "type": "variable",
                    "value": f"{{{{{var_name}}}}}",
                }
            elif mode == "literal":
                data["input_spec"] = {
                    "type": "literal",
                    "value": raw_value,
                }

        return data

    def _on_confirm(self) -> None:
        """Handles the Confirm button click."""
        self._result_data = self._collect_data()
        logger.info(
            "TagDialog confirmed: %s (%s)",
            self._result_data["label"],
            self._result_data["element_type"],
        )
        self.accept()

    def _on_branch(self) -> None:
        """Handles the Branch Point button click."""
        self._result_data = self._collect_data()
        self._result_data["is_branch"] = True
        logger.info("TagDialog branch point: %s", self._result_data["label"])
        self.accept()

    def _on_destination(self) -> None:
        """Handles the 'This is a destination' button click."""
        self._result_data = self._collect_data()
        self._result_data["is_destination"] = True
        logger.info("TagDialog destination marked: %s", self._result_data["label"])
        self.accept()

    def get_result(self) -> dict[str, Any] | None:
        """Gets the result of the dialog interaction.

        Returns:
            A dict containing the element tag data if accepted, or None
            if the user skipped/cancelled.
        """
        if self.result() == QDialog.DialogCode.Accepted:
            return self._result_data
        logger.info("TagDialog skipped/cancelled.")
        return None
