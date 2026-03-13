"""Shared test configuration and fixtures.

Pre-mocks heavy optional dependencies (faiss, torch, transformers,
ultralytics, pywinauto, pytesseract) so that tests can import core
modules without requiring GPU libraries or model weights.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Pre-mock heavy optional deps BEFORE any test imports core modules.
# This allows `import core.embeddings` etc. to succeed without installing
# multi-GB packages like torch/transformers/faiss/ultralytics.
# ---------------------------------------------------------------------------

_MOCK_MODULES = [
    "faiss",
    "torch",
    "transformers",
    "transformers.CLIPModel",
    "transformers.CLIPProcessor",
    "ultralytics",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.yoloe",
    "pywinauto",
    "pywinauto.application",
    "pytesseract",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# pytesseract needs specific attributes that core/ocr.py uses
_pytess_mock = sys.modules["pytesseract"]
_pytess_mock.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
_pytess_mock.Output = MagicMock()
_pytess_mock.Output.DICT = "dict"
