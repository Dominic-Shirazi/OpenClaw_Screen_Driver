"""PyQt6 element tagging dialog for recording sessions.

Modal dialog that opens when the user clicks on a UI element during
recording. Allows tagging the element with type, label, layer, and notes.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from recorder.element_types import ElementType

logger = logging.getLogger(__name__)


class TagDialog(QDialog):
    """Modal dialog for tagging UI elements during a recording session.

    Allows the user to specify the type, label, layer, and notes for a
    detected element. Pre-fills fields from VLM/OCR/UIA guesses.
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
        form_layout = QFormLayout()

        # Element Type dropdown
        self.type_combo = QComboBox()
        for et in ElementType:
            self.type_combo.addItem(et.value, et)

        # Set initial type guess
        if isinstance(element_type_guess, ElementType):
            index = self.type_combo.findData(element_type_guess)
        else:
            index = self.type_combo.findText(element_type_guess)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

        form_layout.addRow("Element Type:", self.type_combo)

        # Label text input
        self.label_input = QLineEdit(label_guess)
        form_layout.addRow("Label:", self.label_input)

        # Layer dropdown
        self.layer_combo = QComboBox()
        layers = ["os_ui", "app_persistent", "page_specific"]
        self.layer_combo.addItems(layers)
        if layer_guess in layers:
            self.layer_combo.setCurrentText(layer_guess)
        else:
            self.layer_combo.setCurrentText("page_specific")
        form_layout.addRow("Layer:", self.layer_combo)

        # Notes free text
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Additional context or instructions...")
        self.notes_input.setMaximumHeight(80)
        form_layout.addRow("Notes:", self.notes_input)

        layout.addLayout(form_layout)

        # Voice Input button (stubbed for Stage 1)
        voice_btn = QPushButton("Voice Input")
        voice_btn.setEnabled(False)
        voice_btn.setToolTip("Voice input coming in Stage 2")
        layout.addWidget(voice_btn)

        # Action buttons
        btn_layout = QHBoxLayout()

        confirm_btn = QPushButton("Confirm")
        confirm_btn.setDefault(True)
        confirm_btn.clicked.connect(self._on_confirm)

        skip_btn = QPushButton("Skip")
        skip_btn.clicked.connect(self.reject)

        branch_btn = QPushButton("Branch Point")
        branch_btn.clicked.connect(self._on_branch)

        dest_btn = QPushButton("This is a destination")
        dest_btn.clicked.connect(self._on_destination)

        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(skip_btn)
        btn_layout.addWidget(branch_btn)
        btn_layout.addWidget(dest_btn)

        layout.addLayout(btn_layout)

        # Compact size
        self.setFixedSize(450, 350)

        # Position near the clicked element (offset so we don't cover it)
        self.move(x + 20, y + 20)

        logger.debug(
            "TagDialog initialized at (%d, %d) with guess '%s'",
            x + 20,
            y + 20,
            element_type_guess,
        )

    def _collect_data(self) -> dict[str, Any]:
        """Collects current field values into a dict.

        Returns:
            Dict with element_type, label, layer, notes, is_branch,
            ocr_text, and uia_hint.
        """
        return {
            "element_type": self.type_combo.currentData(),
            "label": self.label_input.text().strip(),
            "layer": self.layer_combo.currentText(),
            "notes": self.notes_input.toPlainText().strip(),
            "is_branch": False,
            "ocr_text": self._ocr_text,
            "uia_hint": self._uia_hint,
        }

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
        self._result_data["element_type"] = ElementType.READ_HERE
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
