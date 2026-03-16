# Coding Conventions

**Analysis Date:** 2026-03-16

## Naming Patterns

**Files:**
- Module files use `snake_case`: `config.py`, `executor.py`, `pathfinder.py`
- No special prefixes or suffixes
- Test files follow `test_<module>.py` pattern in `/tests` directory

**Functions:**
- All function names use `snake_case`: `screenshot_full()`, `locate_element()`, `execute_node()`
- Private/internal functions prefixed with single underscore: `_exec_cfg()`, `_deep_merge()`, `_hsleep()`
- Type aliases and callbacks use `PascalCase`: `EventCallback`, `RunnerEventType`

**Variables:**
- Global state variables use `snake_case` with leading underscore: `_config_cache`, `_lock`, `_ADJACENT_KEYS` (constants in UPPER_SNAKE_CASE)
- Instance attributes use `snake_case`: `self.session_id`, `self.skill_id`
- Loop variables use single letters when appropriate: `for k, v in`, `for i in range`

**Types:**
- Dataclasses use `PascalCase`: `Point`, `Rect`, `LocateResult`, `ElementNotFoundError`
- Protocol classes (abstract interfaces) use `PascalCase`: `DetectionProvider`
- Enums use `PascalCase` with `auto()` for values: `RunnerEventType`, `class RunnerEventType(Enum)`

## Code Style

**Formatting:**
- Line length: 100 characters (enforced by Ruff in `pyproject.toml`)
- Python version: 3.11+ (enforced in `pyproject.toml`)
- All Python 3.11+ type syntax uses `from __future__ import annotations` at top of every module
- No trailing commas in single-line dicts/lists
- Two blank lines between top-level definitions
- One blank line between methods

**Linting:**
- Ruff is configured with `line-length = 100` and `target-version = "py311"`
- No additional linting config (`.eslintrc`, `.flake8`, `.pylintrc`) — Ruff is the standard
- Code must pass Ruff without errors

**Import Organization:**
Standard library imports first, then third-party, then local (implicit grouping):

```python
from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import mss
import numpy as np
import pyautogui
from dotenv import load_dotenv

from core.config import get_config
from core.types import Point, Rect, LocateResult
from mapper.graph import OCSDGraph
```

**Path Aliases:**
- No path aliases configured. All imports are relative to project root:
  - `from core.config import get_config`
  - `from mapper.graph import OCSDGraph`
  - `from recorder.session import RecordingSession`

## Error Handling

**Patterns:**
- Catch specific exceptions only, never bare `except:` or `except Exception:`
- Platform-specific code guarded with `try ImportError:` for optional dependencies:

```python
try:
    from core.omniparser import OmniParserProvider
except ImportError:
    OmniParserProvider = None  # type: ignore[assignment,misc]
```

- Raise custom exception types from `core.types`:
  - `ElementNotFoundError(node_id, message)` — element could not be located
  - `LowConfidenceError(node_id, confidence, threshold)` — VLM confidence too low
  - `PathNotFoundError(start_id, end_id)` — no graph path exists
- Raise `ValueError` for invalid config: `raise ValueError("Invalid model")`
- Raise `RuntimeError` for system-level failures (binary not found, etc.):

```python
raise RuntimeError(
    "Tesseract OCR binary not found. "
    "Install Tesseract and ensure it's on PATH, or set TESSERACT_CMD "
    "in your .env file."
)
```

## Logging

**Framework:** `logging.getLogger(__name__)`

**Patterns:**
- Every module that does significant work has `logger = logging.getLogger(__name__)` at module top
- Log levels used correctly:
  - `logger.debug()` — detailed flow info (e.g., "Model loaded from cache")
  - `logger.info()` — important state changes (e.g., "Recording session started")
  - `logger.warning()` — recoverable issues (e.g., "Tesseract not found, will skip OCR")
  - `logger.error()` — errors that prevent operation (e.g., "Failed to locate element")
- Use format strings with `%s`, `%d`, `%f` (not f-strings in logging):

```python
logger.info("Recording session %s started in '%s' mode", self.session_id[:8], self.current_mode)
logger.debug("Tesseract binary: %s", resolved)
```

- Never use `print()` for diagnostics

## Comments

**When to Comment:**
- Explain *why* something works, not *what* (code should be clear enough for what)
- Document non-obvious algorithms (e.g., "Gaussian distribution peaks at 40% of radius")
- Document platform-specific code (e.g., "Monitor 1 is usually the primary monitor in mss")
- Explain config/constant decisions

**Example:**
```python
# Absolute difference between the two images
diff = cv2.absdiff(img_a, img_b)

# Count pixels that changed by more than threshold
changed_pixels = np.sum(pixel_diffs > threshold)
```

**JSDoc/Type Hints:**
- Every public function has a Google-style docstring:

```python
def locate_element(
    node_id: str,
    position_hint: Point | None = None,
) -> LocateResult:
    """Locates a UI element on screen using the locate cascade.

    Args:
        node_id: The ID of the element node to locate.
        position_hint: Optional expected position to narrow search scope.

    Returns:
        LocateResult with the detected point and confidence.

    Raises:
        ElementNotFoundError: If no method could locate the element.
    """
```

- Private functions have brief single-line docstrings or no docstring if self-explanatory
- Type hints are **mandatory** on all function signatures (enforced by convention, not linter)

## Function Design

**Size:**
- Target 20-40 lines per function (soft guideline)
- Break long functions into helpers with `_` prefix
- If a function is >80 lines, consider splitting it

**Parameters:**
- Positional parameters for required data
- Keyword-only parameters (after `*`) for config/optional values:

```python
def add_node(
    self,
    element_type: str,
    label: str,
    *,
    node_id: str | None = None,
    layer: str = "page_specific",
) -> str:
```

**Return Values:**
- Use `None` for "no result" (not empty strings or zero)
- Use dataclass return types for structured results:

```python
def execute_node(...) -> ReplayStep:
    return ReplayStep(
        node_id=node_id,
        located_at=point,
        success=True,
        error=None,
    )
```

- Use `|` for union types, not `Optional[]`: `str | None` instead of `Optional[str]`

## Module Design

**Exports:**
- No `__all__` exports (use conventions to signal public API)
- Public functions/classes have full docstrings
- Private functions start with `_`
- Common pattern: module has 3-5 public functions + internal helpers

**Barrel Files:**
- `core/__init__.py` is minimal (just sets up logger namespace if needed)
- No re-exports of entire modules to reduce circular dependencies
- Import directly from submodules: `from core.config import get_config`

**Module Responsibilities:**
- `core/` — Low-level primitives (capture, executor, config, types)
- `mapper/` — Graph, pathfinding, orchestration, validation
- `recorder/` — Recording UI, element detection, hotkeys
- `api/` — FastAPI server endpoints
- `hub/` — Skill registry/manifest
- `tests/` — All test files with conftest fixtures

---

*Convention analysis: 2026-03-16*
