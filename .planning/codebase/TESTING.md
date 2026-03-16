# Testing Patterns

**Analysis Date:** 2026-03-16

## Test Framework

**Runner:**
- pytest 7.0+ (configured in `pyproject.toml`)
- Config: `pyproject.toml` → `[tool.pytest.ini_options]`
  - `testpaths = ["tests"]`
  - `pythonpath = ["."]` (allows `from core.X import` in tests)

**Assertion Library:**
- pytest built-in assertions (`assert x == y`, `assert x is None`)
- `pytest.raises()` for exception testing

**Run Commands:**
```bash
pytest tests/                          # Run all tests
pytest tests/ -v                       # Verbose output
pytest tests/test_cascade.py -v        # Single file
pytest tests/ -k "TestFindPath"        # Filter by class name
pytest tests/ --co                     # List all tests without running
```

Watch mode is not configured. Use `pytest-watch` if needed (not in dependencies).

## Test File Organization

**Location:**
- Co-located in `/tests` directory (separate from source)
- One test file per module: `test_config.py` tests `core/config.py`, etc.

**Naming:**
- Test files: `test_<module>.py`
- Test classes: `Test<Feature>` (PascalCase, no underscore prefix)
- Test methods: `def test_<condition>_<expected_outcome>()`

**Structure:**
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and pre-mocks
├── test_cascade.py          # Locate cascade (ocr, clip, embedding)
├── test_config.py
├── test_detection_provider.py
├── test_export.py
├── test_florence.py
├── test_model_cache.py
├── test_orchestrator.py     # Full orchestration tests
├── test_pathfinder.py
├── test_step_replay.py
└── ...
```

## Test Structure

**Suite Organization:**

Most test files use class-based organization with setup/teardown:

```python
from __future__ import annotations

import pytest
from core.types import Point, Rect


class TestFindPath:
    """Tests for pathfinder.find_path() function."""

    def test_linear_path(self):
        """Finds path in simple A -> B -> C graph."""
        g, a, b, c = _linear_graph()
        path = find_path(g, a, c)
        assert path == [a, b, c]

    def test_same_node(self):
        """Returns single-node path when start == end."""
        g, a, _, _ = _linear_graph()
        path = find_path(g, a, a)
        assert path == [a]

    def test_no_path_raises(self):
        """Raises PathNotFoundError when no path exists."""
        g, a, _, c = _linear_graph()
        with pytest.raises(PathNotFoundError):
            find_path(g, c, a)


class TestDescribeElement:
    """Tests for florence.describe_element()."""

    def setup_method(self):
        """Reset module state before each test."""
        import core.florence as fl
        fl._model = None
        fl._processor = None

    def teardown_method(self):
        """Clean up after each test."""
        import core.florence as fl
        fl._model = None
        fl._processor = None

    def test_returns_structured_dict(self):
        """Returns dict with type, label, confidence keys."""
        # ... test implementation
```

**Patterns:**

1. **Helper functions** (module-level, `_` prefix):
```python
def _linear_graph() -> tuple[OCSDGraph, str, str, str]:
    """Creates A -> B -> C linear graph."""
    g = OCSDGraph(skill_id="test")
    a = g.add_node("textbox", "Username", x_pct=0.3, y_pct=0.3)
    b = g.add_node("textbox", "Password", x_pct=0.3, y_pct=0.5)
    c = g.add_node("button_nav", "Login Button", x_pct=0.3, y_pct=0.7)
    g.add_edge(a, b, "textbox", action_payload="tab")
    g.add_edge(b, c, "textbox", action_payload="tab")
    return g, a, b, c
```

2. **Fixtures** (pytest fixtures for parameterized/reused data):
```python
@pytest.fixture
def simple_graph() -> OCSDGraph:
    """Creates a simple A -> B -> C workflow graph."""
    g = OCSDGraph()
    a = g.add_node(element_type="button", label="Login", x_pct=0.5, y_pct=0.3)
    b = g.add_node(element_type="textbox", label="Username", x_pct=0.5, y_pct=0.5)
    c = g.add_node(element_type="button", label="Submit", x_pct=0.5, y_pct=0.7)
    g.add_edge(a, b, action_type="button")
    g.add_edge(b, c, action_type="textbox")
    return g
```

3. **Autouse fixtures** (applied to all tests in class):
```python
class TestOCRRegionScoping:
    @pytest.fixture(autouse=True)
    def _patch_tesseract(self, mocker: Any) -> None:
        """Disable the real Tesseract binary check for every test."""
        import core.ocr
        mocker.patch("core.ocr._ensure_tesseract_installed")
```

## Mocking

**Framework:** `unittest.mock` (Python stdlib) + `pytest-mock` plugin

**Patterns:**

1. **Pre-mocking heavy dependencies** in `conftest.py`:
```python
# tests/conftest.py
from unittest.mock import MagicMock
import sys

_MOCK_MODULES = [
    "faiss",
    "torch",
    "transformers",
    "ultralytics",
    "pywinauto",
    "pytesseract",
    "huggingface_hub",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# pytesseract needs specific attributes
_pytess_mock = sys.modules["pytesseract"]
_pytess_mock.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
_pytess_mock.Output = MagicMock()
_pytess_mock.Output.DICT = "dict"
```

This allows `import core.embeddings` (which imports `torch`, `faiss`, `transformers`) without requiring GPU libraries.

2. **Mocking with mocker fixture**:
```python
def test_get_cache_dir_returns_config_value(self):
    from core.model_cache import get_cache_dir

    with patch(
        "core.model_cache.get_config",
        return_value={"paths": {"model_cache": "~/.custom/models"}},
    ):
        result = get_cache_dir()
        assert result == Path.home() / ".custom" / "models"
```

3. **Patch with context manager**:
```python
mock_model = MagicMock()
mock_proc = MagicMock()

with patch("core.florence.model_cache") as mock_cache, \
     patch("core.florence.AutoModelForCausalLM") as MockModel, \
     patch("core.florence.AutoProcessor") as MockProc, \
     patch("core.florence.get_config", return_value={...}):
    mock_cache.ensure_model.return_value = Path("/fake/florence")
    MockModel.from_pretrained.return_value = mock_model
    MockProc.from_pretrained.return_value = mock_proc
    load_model()

assert fl._model is mock_model
```

4. **Mocking return values with MagicMock**:
```python
mock_download = MagicMock(return_value="/fake/path/model.pt")
with patch("huggingface_hub.hf_hub_download", mock_download):
    result = ensure_model("microsoft/OmniParser-v2.0", "icon_detect/model.pt")

assert result == Path("/fake/path/model.pt")
mock_download.assert_called_once_with(
    repo_id="microsoft/OmniParser-v2.0",
    filename="icon_detect/model.pt",
    cache_dir=Path("/cache"),
)
```

**What to Mock:**
- External services (Ollama, HuggingFace, AWS)
- System calls (screenshot, mouse movement)
- Heavy ML libraries (torch, transformers, faiss)
- Binary dependencies (Tesseract, system APIs)

**What NOT to Mock:**
- Graph construction (`OCSDGraph` — use helpers like `_linear_graph()`)
- Core data types (`Point`, `Rect`, `LocateResult`)
- Config loading (use `patch("get_config", return_value={...})`)
- Business logic (pathfinding, locate cascade) — test with real graph objects

## Fixtures and Factories

**Test Data:**
- Graphs: `_linear_graph()`, `_diamond_graph()`, `simple_graph` fixture
- Images: `_bgr_image(h, w)`, `_full_screen_image()` (return numpy BGR arrays)
- Dummy numpy arrays:

```python
def _bgr_image(h: int = 100, w: int = 200, channels: int = 3) -> np.ndarray:
    """Creates a dummy BGR image filled with random pixel values."""
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)

def _full_screen_image() -> np.ndarray:
    """Creates a 1080p dummy BGR screenshot."""
    return _bgr_image(h=1080, w=1920)
```

**Location:**
- Helpers at module top (above test classes)
- Fixtures in same file or in `conftest.py` if shared
- `conftest.py` handles pre-mocking and global fixtures

## Coverage

**Requirements:** None enforced (no coverage threshold in config)

**View Coverage:**
```bash
pytest tests/ --cov=core --cov=mapper --cov=recorder --cov-report=html
```

Note: `pytest-cov` not in dev dependencies; install separately if needed.

## Test Types

**Unit Tests:**
- Scope: Single module/function (e.g., `test_pathfinder.py` tests graph pathfinding)
- Approach: Create minimal fixtures (graphs), mock external I/O, assert on return values
- Examples: `test_find_path()`, `test_get_cache_dir()`, `test_ocr_region_scoping()`

**Integration Tests:**
- Scope: Multiple modules working together (e.g., recorder → graph → pathfinder)
- Approach: Create real graph via helpers, mock I/O (screenshot, locate), verify end-to-end
- Examples: `test_orchestrator.py` (full skill execution simulation)
- Run: Same `pytest tests/` command (no separate marker)

**E2E Tests:**
- Status: Not implemented in current codebase
- When used: `@pytest.mark.e2e` + `pytest -m e2e`
- Would test: Full UI automation against test HTML page (not production sites)

## Common Patterns

**Async Testing:**
Not used (codebase is synchronous). If needed:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_func()
    assert result
```

**Error Testing:**
```python
def test_low_confidence_raises_error(self):
    """Raises LowConfidenceError when confidence below threshold."""
    from core.types import LowConfidenceError

    with pytest.raises(LowConfidenceError) as exc_info:
        locate_element(node_id="bad", threshold=0.9)

    assert exc_info.value.node_id == "bad"
    assert exc_info.value.confidence < 0.9
```

**Parametrized Tests:**
Not commonly used in current codebase, but supported:
```python
@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("world", 5),
    ("", 0),
])
def test_string_length(input, expected):
    assert len(input) == expected
```

**Temporary test disable:**
```python
@pytest.mark.skip(reason="WIP — waiting for model cache PR")
def test_florence_batch_caption():
    pass

@pytest.mark.xfail(reason="OCR fails on blurry images")
def test_ocr_blurry_screenshot():
    pass
```

## Test Quality Guidelines

**Failing tests should be specific:**
- Bad: `test_locate_element()` — too broad
- Good: `test_locate_element_falls_back_to_ocr_when_omniparser_fails()`

**Test one thing per test:**
- Bad: Test that locate → execute → validate all work together in one test
- Good: Separate tests for locate, execute, validate; integration test that combines them

**Use descriptive assertion messages:**
```python
result = find_path(g, a, d)
assert b in path, f"Expected path to use node B for high-success branch, got {path}"
```

**Mock at the boundary:**
- Mock external services (HuggingFace, Ollama)
- Don't mock core business logic

---

*Testing analysis: 2026-03-16*
