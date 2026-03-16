# Surgical Split Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split overlay.py, main.py, and runner.py into focused modules so changes to rendering, recording, and execution don't break each other.

**Architecture:** Extract-don't-rewrite. Move existing code into new files with clear boundaries. Keep all public APIs stable. Add centralized GPU memory management. Clean up stale artifacts.

**Tech Stack:** Python 3.11+, PyQt6, torch (optional), ultralytics, networkx

**Spec:** `docs/superpowers/specs/2026-03-13-surgical-split-design.md`

**Test command:** `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`

**Important:** This is a pure extraction refactor. No new behavior. No new tests needed. Existing tests validate correctness after each extraction. Run full suite after every task.

---

## Chunk 1: Foundation + Overlay Split

### Task 1: Create `core/gpu.py`

**Files:**
- Create: `core/gpu.py`

- [ ] **Step 1: Create core/gpu.py**

```python
"""Centralized GPU memory management.

Provides cleanup() and get_device() so individual modules don't need
scattered try/except torch blocks for memory management.
"""

from __future__ import annotations

import logging

from core.config import get_config

logger = logging.getLogger(__name__)


def cleanup() -> None:
    """Free GPU memory caches. Safe to call even without torch."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        pass


def get_device(role: str) -> str:
    """Returns the configured torch device string for a role.

    Args:
        role: One of "vlm" (detection models) or "embeddings" (CLIP, Florence-2).

    Returns:
        Device string like "cuda:0", "cuda:1", or "cpu".
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
    except ImportError:
        return "cpu"

    cfg = get_config()
    hw = cfg.get("hardware", {})

    if role == "vlm":
        gpu_id = hw.get("gpu_vlm", 0)
    elif role == "embeddings":
        gpu_id = hw.get("gpu_embeddings", 1)
    else:
        gpu_id = 0

    return f"cuda:{gpu_id}"
```

- [ ] **Step 2: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All tests pass (no imports changed yet)

- [ ] **Step 3: Commit**

```bash
git add core/gpu.py
git commit -m "feat(core): add gpu.py for centralized GPU memory management"
```

---

### Task 2: Extract `recorder/hotkeys.py`

**Files:**
- Create: `recorder/hotkeys.py`
- Modify: `recorder/overlay.py` (remove hotkey classes, add import)

- [ ] **Step 1: Create recorder/hotkeys.py**

Extract these from `recorder/overlay.py` verbatim (lines 102-265):
- The comment block `# Global hotkey backends`
- `_Win32PollingHotkeyListener` class (lines 106-188)
- `_PynputHotkeyListener` class (lines 191-255)
- `_create_hotkey_listener()` factory function (lines 258-265)

The file needs these imports at the top:
```python
"""Global hotkey listeners for the recording overlay.

Win32: Polls GetAsyncKeyState on a QTimer (only approach that works
alongside PyQt6's event loop).
Non-Windows: pynput GlobalHotKeys fallback.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

from PyQt6.QtCore import QTimer

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Update recorder/overlay.py**

Remove lines 102-265 (the hotkey section). Replace with:
```python
from recorder.hotkeys import _create_hotkey_listener
```

Also remove unused imports that were only needed by hotkeys: none — the hotkey classes only used `QTimer`, `Callable`, `sys`, `logging`, which overlay.py also uses.

- [ ] **Step 3: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add recorder/hotkeys.py recorder/overlay.py
git commit -m "refactor(recorder): extract hotkeys.py from overlay.py"
```

---

### Task 3: Extract `recorder/overlay_items.py`

**Files:**
- Create: `recorder/overlay_items.py`
- Modify: `recorder/overlay.py` (remove item classes, add import)

- [ ] **Step 1: Create recorder/overlay_items.py**

Extract from `recorder/overlay.py`:
- Color map `_TYPE_COLORS` and `_DEFAULT_COLOR` (lines 52-87 after hotkey removal — adjust for actual line numbers after Task 2)
- Constants `_BORDER_WIDTH`, `_HANDLE_RADIUS` (lines 89-93)
- `_HandleItem` class
- `_ElementBoxGroup` class (includes `_create_donut`, `_position_handles`, `handle_moved`, `highlight`, `reset_highlight`, `get_rect`)

The file needs these imports:
```python
"""Interactive bounding box components for the recording overlay.

Pure Qt graphics primitives — no business logic. Contains the visual
representation of detected UI elements with draggable corner handles
and click probability donut visualization.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPen, QRadialGradient
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)
```

**Note:** `_HandleItem` forward-references `_ElementBoxGroup` in its constructor type hint. Since both are in the same file and `from __future__ import annotations` is used, this works.

- [ ] **Step 2: Update recorder/overlay.py**

Remove the extracted classes and constants. Add import:
```python
from recorder.overlay_items import (
    _BORDER_WIDTH,
    _DEFAULT_COLOR,
    _HANDLE_RADIUS,
    _TYPE_COLORS,
    _ElementBoxGroup,
    _HandleItem,
)
```

Remove from overlay.py's imports any that are now only used by overlay_items (check carefully — `QRadialGradient`, `QFont`, `QGraphicsSimpleTextItem`, `QGraphicsEllipseItem`, `QGraphicsItem` may only be needed by overlay_items, but `_OverlayView` also uses `QGraphicsSimpleTextItem` and `QGraphicsRectItem`). Keep imports that `_OverlayView` or `OverlayController` still needs.

- [ ] **Step 3: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add recorder/overlay_items.py recorder/overlay.py
git commit -m "refactor(recorder): extract overlay_items.py — graphics primitives"
```

---

### Task 4: Extract `recorder/overlay_view.py`

**Files:**
- Create: `recorder/overlay_view.py`
- Modify: `recorder/overlay.py` (remove _OverlayView, add import)

- [ ] **Step 1: Create recorder/overlay_view.py**

Extract `_OverlayView(QGraphicsView)` class and all its methods from overlay.py. This is the Qt window with mouse events, scene management, border drawing, etc.

Imports needed:
```python
"""PyQt6 overlay window for recording sessions.

Transparent fullscreen window with interactive bounding box overlays.
Handles mouse events for click/drag selection and corner handle resizing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from recorder.overlay_items import (
    _BORDER_WIDTH,
    _DEFAULT_COLOR,
    _TYPE_COLORS,
    _ElementBoxGroup,
    _HandleItem,
)

if TYPE_CHECKING:
    from recorder.overlay import OverlayController, OverlayMode

logger = logging.getLogger(__name__)

# Win32 constants (needed for layered window setup)
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
```

**Critical:** The `_OverlayView` references `OverlayController` and `OverlayMode` — use the `TYPE_CHECKING` guard shown above. At runtime, `controller` is passed as a parameter, so no actual import is needed. For `OverlayMode` comparisons at runtime (e.g., `if mode == OverlayMode.RECORD`), import it conditionally or access via `self._controller.mode` comparison. Best approach: import `OverlayMode` normally from overlay.py since it's a simple enum with no circular dependency risk — only `OverlayController` needs the TYPE_CHECKING guard.

Update to:
```python
from recorder.overlay import OverlayMode

if TYPE_CHECKING:
    from recorder.overlay import OverlayController
```

This works because `overlay.py` will import from `overlay_view.py`, but `overlay_view.py` only imports the `OverlayMode` enum from `overlay.py` (not `OverlayController`), and `OverlayMode` is defined before `OverlayController` in `overlay.py`. No circular dependency.

- [ ] **Step 2: Update recorder/overlay.py**

Remove the `_OverlayView` class entirely. Remove Win32 constants (GWL_EXSTYLE, etc.) — they now live in overlay_view.py. Add import:
```python
from recorder.overlay_view import _OverlayView
```

The remaining overlay.py should contain only:
- `OverlayMode` enum
- `OverlayController` class
- Imports from `overlay_view` and `hotkeys`

- [ ] **Step 3: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add recorder/overlay_view.py recorder/overlay.py
git commit -m "refactor(recorder): extract overlay_view.py — Qt window"
```

---

### Task 5: Verify overlay split is clean

- [ ] **Step 1: Verify overlay.py is now ~250 lines**

Run: `wc -l recorder/overlay.py` (from repo root)
Expected: ~250 lines (OverlayMode + OverlayController only)

- [ ] **Step 2: Verify public API unchanged**

The following import must still work:
```python
from recorder.overlay import OverlayController, OverlayMode
```

- [ ] **Step 3: Full test suite**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit if any fixups needed**

---

## Chunk 2: Runner Split

### Task 6: Extract `core/locate.py`

**Files:**
- Create: `core/locate.py`
- Modify: `mapper/runner.py` (remove locate functions, add import)

- [ ] **Step 1: Create core/locate.py**

Extract from `mapper/runner.py`:
- `_resolve_position_hint()` function (lines 63-78)
- `locate_element()` function (lines 81-279)
- Any private helpers called only by locate_element (check for `_try_` prefixed functions — there may not be separate `_try_*` functions, the stages may be inline in `locate_element`)

The file needs the imports that `locate_element` uses. Read the function carefully and copy only the imports it needs:
```python
"""5-stage locate cascade for finding UI elements during replay.

Stages (cheapest first):
1. OmniParser detect+match — saved snippet → detect boxes → CLIP match
2. CLIP embedding — compare saved embedding against detected candidates
3. OCR text match — scoped to region around expected position
4. VLM full scan — screenshot → LiteLLM → match by label
5. Position fallback — blind click at recorded coordinates
"""

from __future__ import annotations

import logging
from typing import Any

# Lazy imports for optional deps happen inside stage functions
from core.config import get_config
from core.types import LocateResult

logger = logging.getLogger(__name__)
```

Add additional imports as needed based on what `locate_element` actually imports at the top of runner.py (e.g., `from core.capture import screenshot_full`, `from core.ocr import find_text_on_screen`). Use lazy imports inside stage blocks where the original code does.

- [ ] **Step 2: Update mapper/runner.py**

Remove `_resolve_position_hint` and `locate_element` functions. Add:
```python
from core.locate import locate_element
```

Remove imports from runner.py's top that are now only used by locate_element (e.g., capture, detection, embeddings, ocr, vision imports). Keep imports used by `run_skill`, `execute_node`, etc.

- [ ] **Step 3: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass. Tests that call `locate_element` via `mapper.runner` may need the import updated if they import it directly — check `tests/test_cascade.py`.

- [ ] **Step 4: Fix any broken test imports**

If tests import `from mapper.runner import locate_element`, update to `from core.locate import locate_element`. Also add a re-export in runner.py as a safety net:
```python
# Re-export for backwards compatibility
from core.locate import locate_element  # noqa: F401
```

- [ ] **Step 5: Run tests again**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add core/locate.py mapper/runner.py tests/
git commit -m "refactor(core): extract locate.py — 5-stage locate cascade"
```

---

## Chunk 3: main.py Split

### Task 7: Extract `recorder/record_controller.py`

**Files:**
- Create: `recorder/record_controller.py`
- Modify: `main.py` (remove recording functions, add import)

- [ ] **Step 1: Create recorder/record_controller.py**

Extract from `main.py` these functions (in order of definition):
- `_try_refine_bbox()` (line 80)
- `_auto_snip()` (line 226)
- `_trigger_smart_detect()` (line 318)
- `_start_review()` (line 367)
- `cmd_record()` (line 454) — includes the `on_element_clicked` and `on_mode_changed` closures defined inside it
- `_save_snippets_and_embeddings()` (line 640)
- `_save_recording()` (line 733)
- `cmd_compose()` (line 987)

Module docstring and imports:
```python
"""Recording session controller.

Manages the record/diagram/compose workflows: overlay setup, element
click handling, auto-snip, smart detection, review flow, and skill
saving.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core.config import get_config

logger = logging.getLogger(__name__)
```

Additional imports should be lazy (inside functions) matching the pattern already used in main.py — the recording functions use lazy imports for `core.capture`, `core.detection`, `recorder.overlay`, `recorder.dialog`, etc.

- [ ] **Step 2: Update main.py**

Remove all extracted functions. Add lazy import in the dispatch:
```python
def main() -> int:
    ...
    if args.record or getattr(args, "diagram", False):
        from recorder.record_controller import cmd_record
        return cmd_record(args)
    elif getattr(args, "execute", None):
        from mapper.execute_controller import cmd_execute
        return cmd_execute(args)
    elif getattr(args, "compose", None):
        from recorder.record_controller import cmd_compose
        return cmd_compose(args)
    ...
```

- [ ] **Step 3: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Fix any broken test imports**

Check `tests/test_main_refine.py` — if it imports `_try_refine_bbox` from main, update to import from `recorder.record_controller`.

- [ ] **Step 5: Commit**

```bash
git add recorder/record_controller.py main.py tests/
git commit -m "refactor: extract record_controller.py from main.py"
```

---

### Task 8: Extract `mapper/execute_controller.py`

**Files:**
- Create: `mapper/execute_controller.py`
- Modify: `main.py` (remove cmd_execute)

- [ ] **Step 1: Create mapper/execute_controller.py**

Extract from `main.py`:
- `cmd_execute()` (line 842)

```python
"""Skill execution controller.

Loads a saved skill graph, plans the execution path, and replays it
using the locate cascade and action executor.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from core.config import get_config

logger = logging.getLogger(__name__)
```

Copy `cmd_execute` with all its internal imports intact (it already uses lazy imports).

- [ ] **Step 2: Update main.py**

Remove `cmd_execute` function. The dispatch in `main()` already imports it lazily from Task 7.

- [ ] **Step 3: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add mapper/execute_controller.py main.py
git commit -m "refactor: extract execute_controller.py from main.py"
```

---

### Task 9: Verify main.py is thin

- [ ] **Step 1: Check main.py line count**

Run: `wc -l main.py`
Expected: ~150 lines. Should contain only: imports, `_setup_logging`, `_setup_dpi_awareness`, `_ensure_dirs`, `_setup_signal_handler`, `_has_mode_arg`, `_run_tui`, `build_parser`, `main`.

- [ ] **Step 2: Full test suite**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

---

## Chunk 4: Cleanup + README

### Task 10: Delete stale files

**Files to delete:**
- `Dockerfile`
- `docker-entrypoint.sh`
- `scripts/benchmark_yoloe_ui.py`
- `Claude_CLI_Guide.md`
- `Gemini_CLI_Guide.md`
- `PROGRESS.md`

- [ ] **Step 1: Delete files**

```bash
git rm Dockerfile docker-entrypoint.sh scripts/benchmark_yoloe_ui.py Claude_CLI_Guide.md Gemini_CLI_Guide.md PROGRESS.md
```

If `scripts/` directory is now empty, remove it too:
```bash
rmdir scripts 2>/dev/null || true
```

- [ ] **Step 2: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass (none of these files are imported)

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove stale Docker, YOLOE benchmark, and CLI guide files"
```

---

### Task 11: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Remove dead yoloe dependency group**

Remove:
```toml
# YOLO-E visual grounding (dormant — kept for reference)
yoloe = [
    "ultralytics>=8.3",
]
```

The `omniparser` group already has `ultralytics>=8.3` so nothing is lost.

- [ ] **Step 2: Run tests**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: remove dead yoloe dependency group from pyproject.toml"
```

---

### Task 12: Rewrite README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README**

Key changes:
- Replace all YOLOE references with OmniParser
- Update locate cascade: OmniParser → CLIP → OCR → VLM → position fallback
- Remove `executor/` from project structure (doesn't exist)
- Add new files: `core/gpu.py`, `core/locate.py`, `recorder/hotkeys.py`, `recorder/overlay_view.py`, `recorder/overlay_items.py`, `recorder/record_controller.py`, `mapper/execute_controller.py`
- Remove Docker section
- Update optional deps: remove `yoloe` group, show `detection` group
- Update test count to match actual
- Remove references to `core/yoloe.py` (deleted)
- Update "What's Next" section to reflect V1 priorities (not YOLOE training)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README to reflect current architecture"
```

---

### Task 13: Final verification

- [ ] **Step 1: Full test suite**

Run: `C:\Users\persi\Documents\OpenClaw_Screen_Driver\.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify no stale imports**

Run: `grep -r "from core.yoloe" --include="*.py" .` (from repo root)
Expected: No matches

Run: `grep -r "yoloe" --include="*.py" . | grep -v ".venv" | grep -v __pycache__ | grep -v test_`
Expected: No matches (except possibly comments that can be cleaned up)

- [ ] **Step 3: Verify file counts**

```bash
wc -l main.py recorder/overlay.py recorder/overlay_view.py recorder/overlay_items.py recorder/hotkeys.py recorder/record_controller.py mapper/runner.py mapper/execute_controller.py core/locate.py core/gpu.py
```

Expected approximate counts:
- main.py: ~150
- recorder/overlay.py: ~250
- recorder/overlay_view.py: ~350
- recorder/overlay_items.py: ~200
- recorder/hotkeys.py: ~170
- recorder/record_controller.py: ~750
- mapper/runner.py: ~350
- mapper/execute_controller.py: ~200
- core/locate.py: ~300
- core/gpu.py: ~50
