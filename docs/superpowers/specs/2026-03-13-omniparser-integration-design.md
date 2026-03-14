# OmniParser + Florence-2 Integration Design

## Goal

Replace the underperforming YOLOE text-prompt detection with OmniParser's UI-trained YOLOv8 for element detection and Florence-2 for element captioning, while keeping Qwen3-VL (via Ollama/LiteLLM) for deep reasoning. This creates a three-tier vision stack: fast detection (OmniParser), fast labeling (Florence-2), and deep understanding (Qwen3-VL).

## Context

Wave 1 proved that YOLOE text-prompt detection is insufficient for UI elements — it detected only 4 elements on a Windows desktop that should have 20+. YOLOE was trained on COCO objects, not UI elements. Microsoft's OmniParser v2 includes a YOLOv8 model fine-tuned specifically for UI element detection (icons, buttons, text fields, etc.) that solves this problem.

### Decisions Made

- **Detection:** OmniParser YOLOv8 icon_detect replaces YOLOE in all active code paths
- **Labeling:** Florence-2-large (default, configurable to base) labels all detected elements, then its output feeds into Qwen3-VL as context
- **Reasoning:** Qwen3-VL via Ollama/LiteLLM for deep element analysis (on-demand, per-click)
- **Matching:** CLIP/FAISS remains unchanged for visual similarity
- **YOLOE code:** `core/yoloe.py` remains in the repo dormant (not imported in active paths)
- **Weights:** Auto-download from HuggingFace on first use to `~/.cache/ocsd/models/`
- **Future:** Train MIT-licensed YOLO fine-tune on UI datasets before commercial release to eliminate AGPL dependency

### Licensing

| Component | License | Status |
|-----------|---------|--------|
| OmniParser YOLOv8 weights | AGPL-3.0 (via Ultralytics) | OK for prototype; replace before commercial |
| Florence-2 | MIT | Clear for any use |
| Qwen3-VL | Apache 2.0 | Clear for any use |
| CLIP | MIT | Clear for any use |
| Future MIT YOLO fine-tune | MIT | Planned replacement for AGPL weights |

---

## Architecture

### Approach: Detection Provider Abstraction

Create clean module boundaries with a `DetectionProvider` protocol that allows swapping detection backends. OmniParser is the first (and currently only) provider. When the MIT YOLO fine-tune is ready, it slots in as a second provider with zero changes to consuming code.

### New Modules

```
core/
├── model_cache.py      # Shared HuggingFace weight download + cache management
├── detection.py         # DetectionProvider protocol + factory
├── omniparser.py        # OmniParser YOLOv8 icon_detect wrapper
└── florence.py           # Florence-2 captioning module
```

### Module Dependency Graph

```
main.py
  ├── core/detection.py (factory: get_detector())
  │     └── core/omniparser.py (OmniParserProvider)
  │           └── core/model_cache.py (weight download)
  ├── core/florence.py (captioning)
  │     └── core/model_cache.py (weight download)
  ├── core/vision.py (Qwen3-VL, accepts florence_context)
  ├── core/embeddings.py (CLIP/FAISS, unchanged)
  ├── recorder/smart_detect.py (uses detection + florence)
  └── mapper/runner.py (uses detection + embeddings for replay)
```

---

## Module Specifications

### `core/model_cache.py` — Weight Management

**Purpose:** Single utility for downloading and caching model weights from HuggingFace Hub.

**Public API:**

```python
def ensure_model(repo_id: str, filename: str | None = None,
                 cache_dir: Path | None = None) -> Path:
    """Download model file from HuggingFace if not cached. Return local path.

    Args:
        repo_id: HuggingFace repo (e.g., "microsoft/OmniParser-v2.0")
        filename: Specific file within repo (e.g., "icon_detect/model.pt").
                  If None, downloads entire repo snapshot.
        cache_dir: Override default cache directory.

    Returns:
        Path to local cached file or directory.
    """

def get_cache_dir() -> Path:
    """Return model cache directory from config or default ~/.cache/ocsd/models/."""

def clear_cache(repo_id: str | None = None) -> None:
    """Remove cached weights. If repo_id given, only that repo."""
```

**Implementation details:**
- Uses `huggingface_hub.hf_hub_download()` for single files, `snapshot_download()` for repos
- Thread-safe via file-level locking (huggingface_hub handles this)
- Default cache: `~/.cache/ocsd/models/`, overridable via `paths.model_cache` in config.yaml
- `get_cache_dir()` must call `Path.expanduser()` to resolve `~` in the config path
- Logs download progress via `logging.getLogger(__name__)`
- No GPU, no torch — pure download utility

**Dependencies:** `huggingface-hub>=0.20`

---

### `core/detection.py` — Provider Protocol & Factory

**Purpose:** Abstract detection interface so backends can be swapped.

**Public API:**

```python
from typing import Protocol

class DetectionProvider(Protocol):
    def detect(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Detect all UI elements in screenshot.

        Returns list of candidate dicts matching the existing smart_detect format:
            {
                "rect": {"x": int, "y": int, "w": int, "h": int},
                "type_guess": str,   # YOLOv8 class name
                "label_guess": str,  # empty, Florence-2 fills later
                "confidence": float,
            }

        Note: We use dicts (not CandidateElement dataclass) because
        downstream consumers (overlay, vision.first_pass_map) already
        expect dicts with nested rect dicts. The CandidateElement dataclass
        in types.py uses Rect objects — a future cleanup can unify these,
        but this integration preserves the existing contract.
        """
        ...

    def detect_and_match(
        self, screenshot: np.ndarray, saved_snippet: np.ndarray,
        hint_x: int, hint_y: int, match_threshold: float = 0.7,
        search_radius: int = 400,
    ) -> LocateResult | None:
        """Detect all elements, then find best CLIP match to saved_snippet.

        Used in replay cascade Stage 1.
        Returns LocateResult if match above threshold, else None.
        The LocateResult.point is set to the center of the matched bbox.
        """
        ...

def get_detector(config: dict | None = None) -> DetectionProvider:
    """Factory that returns the configured detection provider.

    Reads config['models']['detector'] to select backend.
    Currently only 'omniparser' is supported.
    """
```

**Implementation details:**
- `get_detector()` reads `config['models']['detector']` (default: `"omniparser"`)
- Returns singleton instance (lazy-loaded, thread-safe via `threading.Lock()` on first creation)
- Raises `ValueError` for unknown detector names

---

### `core/omniparser.py` — OmniParser YOLOv8 Wrapper

**Purpose:** Wrap OmniParser's fine-tuned YOLOv8 icon_detect model for UI element detection.

**Public API:**

```python
class OmniParserProvider:
    """DetectionProvider implementation using OmniParser's YOLOv8 icon_detect."""

    def __init__(self, confidence_threshold: float = 0.3):
        """Initialize. Model loads lazily on first detect() call."""

    def detect(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLOv8 icon_detect on screenshot.

        Returns list of candidate dicts:
          - rect: {"x": int, "y": int, "w": int, "h": int}
          - type_guess: YOLOv8 class name (icon, text, etc.)
          - confidence: detection confidence
          - label_guess: empty string (Florence-2 fills this later)
        """

    def detect_and_match(
        self, screenshot: np.ndarray, saved_snippet: np.ndarray,
        hint_x: int, hint_y: int, match_threshold: float = 0.7,
        search_radius: int = 400,
    ) -> LocateResult | None:
        """Detect all elements, CLIP-match saved snippet against crops.

        1. detect(screenshot) → all bboxes
        2. Filter to bboxes within search_radius of (hint_x, hint_y)
        3. Generate CLIP embedding of saved_snippet
        4. Generate CLIP embedding of each candidate crop
        5. Return best match above match_threshold as LocateResult
           with point set to center of matched bbox
        """
```

**Implementation details:**
- Lazy model load: `model_cache.ensure_model("microsoft/OmniParser-v2.0", "icon_detect/model.pt")`
- Uses `ultralytics.YOLO(model_path)` to load and predict
- Prediction call: `model.predict(screenshot, conf=confidence_threshold, verbose=False)`
- Maps YOLOv8 class IDs to type_guess strings
- Thread-safe: model load behind `threading.Lock()`
- `detect_and_match()` uses `core.embeddings.generate_embedding()` for CLIP comparison — reuses existing CLIP infrastructure
- `search_radius` defaults to `config['detection']['search_radius']` (400px)
- Device: configurable, defaults to `config['hardware']['gpu_vlm']` (GPU 0), falls back to CPU if GPU unavailable

**Dependencies:** `ultralytics>=8.0` (AGPL — prototype only)

---

### `core/florence.py` — Florence-2 Captioning

**Purpose:** Fast element labeling using Microsoft Florence-2 vision model.

**Public API:**

```python
def load_model(variant: str | None = None) -> None:
    """Pre-load Florence-2 model. Called at startup for warm cache.

    Args:
        variant: Model name override. Default from config
                 (florence-2-large or florence-2-base).
    """

def caption_crop(image: np.ndarray, task: str = "<CAPTION>") -> str:
    """Caption a single image crop.

    Args:
        image: RGB numpy array of element crop.
        task: Florence-2 task token. Options:
              "<CAPTION>" — short caption
              "<DETAILED_CAPTION>" — detailed description
              "<MORE_DETAILED_CAPTION>" — very detailed

    Returns:
        Caption string (e.g., "a blue submit button with white text").
    """

def caption_batch(images: list[np.ndarray],
                  task: str = "<CAPTION>") -> list[str]:
    """Caption multiple crops. More efficient than calling caption_crop in a loop.

    Returns list of caption strings, one per input image.
    If a crop is degenerate (empty, 0-dim), returns empty string for that
    position to maintain 1:1 correspondence with input list.
    """

def describe_element(image: np.ndarray) -> dict[str, str]:
    """Structured element description for UI labeling.

    Returns:
        {
            "type_guess": "button",
            "label_guess": "Submit",
            "description": "A blue rectangular button with white 'Submit' text"
        }

    Uses <DETAILED_CAPTION> internally, then parses the response
    to extract type and label.
    """
```

**Implementation details:**
- Lazy model load via `model_cache.ensure_model()`
- Default variant: `microsoft/Florence-2-large` (configurable to `microsoft/Florence-2-base`)
- Uses `transformers.AutoModelForCausalLM` + `AutoProcessor`
- Device: `config['hardware']['gpu_embeddings']` (GPU 1), falls back to CPU
- Thread-safe: model load behind `threading.Lock()`
- `describe_element()` extracts type/label from caption text using:
  1. Keyword matching against `ElementType` enum values (button, textbox, toggle, etc.)
  2. If a known type keyword is found, the preceding adjectives become the label
  3. Example: "a blue submit button with white text" → type="button", label="Submit"
  4. Example: "a text input field labeled Email" → type="textbox", label="Email"
  5. If no type keyword found, type="unknown" and full caption becomes the label
- `caption_batch()` processes sequentially (Florence-2 doesn't support true batching easily), but each crop is fast (~100-300ms)

**Dependencies:** `transformers>=4.40`, `torch>=2.0`

---

## Pipeline Changes

### Recording Pipeline (`recorder/smart_detect.py`)

**Current flow:**
```
detect_ui_elements(screenshot, use_yoloe=True)
  → _detect_via_yoloe(screenshot)      # YOLOE text-prompt
  → fallback: _detect_via_ocr(screenshot)  # Tesseract OCR
```

**New flow:**
```
detect_ui_elements(screenshot)
  → detection.get_detector().detect(screenshot)      # OmniParser YOLOv8
  → florence.caption_batch(crops_from_detections)     # Florence-2 labels
  → merge labels into CandidateElement list
  → fallback: _detect_via_ocr(screenshot)             # OCR if no detections
```

**Changes:**
- Remove `use_yoloe` parameter from `detect_ui_elements()` (always use OmniParser)
- Remove `use_yoloe` parameter from `detect_ui_elements_async()` (same)
- Remove `_detect_via_yoloe()` function
- Add `_enrich_with_florence(candidates, screenshot)` function that:
  1. Crops each candidate bbox from screenshot
  2. Calls `florence.caption_batch(crops)`
  3. Updates each candidate's `label_guess` with Florence-2 caption
  4. If a crop is degenerate (0 width or height), sets label_guess to empty string (skip, don't raise)
- OCR fallback remains for cases where OmniParser finds nothing

### Recording Click Handler (`main.py`)

**Current flow:**
```
on_click(x, y):
  → crop element at (x, y)
  → vision.analyze_crop(crop)  # Qwen3-VL via LiteLLM
  → show TagDialog with VLM result
```

**New flow:**
```
on_click(x, y):
  → find nearest candidate from OmniParser detections
  → get Florence-2 label (already computed in batch)
  → build context_prompt that includes Florence-2 label
  → vision.analyze_crop(crop_path, context_prompt)  # Qwen3-VL with Florence-2 context
  → show TagDialog with enriched result
```

**Changes to `core/vision.py`:**

The existing `analyze_crop(img_path: str, context_prompt: str)` signature already supports passing context. No signature change needed. Instead, the caller in `main.py` builds a richer `context_prompt` that includes the Florence-2 label:

```python
# In main.py click handler, before calling analyze_crop:
florence_label = candidate.get("florence_caption", "")
if florence_label:
    context_prompt = (
        f'A fast vision model identified this element as: "{florence_label}". '
        "Confirm or correct the identification. "
        "Return JSON: {\"element_type\": ..., \"label_guess\": ..., "
        "\"confidence\": 0.0-1.0, \"ocr_text\": ...}"
    )
else:
    context_prompt = "Identify this UI element..."

result = analyze_crop(crop_path, context_prompt)
```

There is also `analyze_crop_array(img: np.ndarray, context_prompt: str)` which works with numpy arrays instead of file paths — the same context_prompt approach applies to both.

This gives Qwen3-VL better context, improving label accuracy, without any changes to vision.py's interface.

### Replay Cascade (`mapper/runner.py`)

**Current Stage 1:** `_locate_via_yoloe(snippet, screenshot, hint_x, hint_y)`

**New Stage 1:** `_locate_via_omniparser(snippet, screenshot, hint_x, hint_y)`

```python
def _locate_via_omniparser(snippet, screenshot, hint_x, hint_y):
    detector = detection.get_detector()
    cfg = get_config()
    result = detector.detect_and_match(
        screenshot, snippet, hint_x, hint_y,
        match_threshold=cfg["detection"]["match_threshold"],
        search_radius=cfg["detection"]["search_radius"],
    )
    # detect_and_match() returns LocateResult | None directly
    # with method="omniparser" and point set to bbox center
    return result
```

**Stages 2-5:** Unchanged (CLIP/FAISS, OCR, VLM, position fallback).

---

## Configuration Changes

### `config.yaml` changes:

The existing `detection:` section has `model: "yolo-e"`, `confidence_threshold: 0.4`, and `crop_buffer_pct: 0.30`. These are replaced/updated:

```yaml
models:
  # Existing (unchanged)
  vlm: "vision"
  clip: "openai/clip-vit-base-patch32"
  # Removed (was: yoloe: "yoloe-26s-seg.pt")
  # New
  detector: "omniparser"                        # detection backend selector
  florence: "microsoft/Florence-2-large"         # or "microsoft/Florence-2-base"
  omniparser_weights: "microsoft/OmniParser-v2.0"

detection:
  # CHANGED: model key removed (replaced by models.detector above)
  # CHANGED: confidence_threshold lowered from 0.4 to 0.3 (OmniParser's
  #          UI-trained model is more precise, lower threshold catches more elements)
  confidence_threshold: 0.3
  match_threshold: 0.7         # NEW: CLIP cosine similarity for replay matching
  # CHANGED: crop_buffer_pct reduced from 0.30 to 0.10 (OmniParser's bboxes
  #          are tighter than YOLOE's, less buffer needed)
  crop_buffer_pct: 0.10
  search_radius: 400           # Unchanged

paths:
  # Existing paths unchanged
  model_cache: "~/.cache/ocsd/models"  # NEW: auto-download destination
```

### `pyproject.toml` changes:

```toml
[project.optional-dependencies]
# Existing (unchanged)
embeddings = ["transformers>=4.36", "faiss-cpu>=1.7", "torch>=2.1"]
vlm = ["openai>=1.0"]
windows = ["pywinauto>=0.6"]
api = ["fastapi>=0.100", "uvicorn>=0.20"]
tui = ["rich>=13.0"]
dev = ["pytest>=7.0", "pytest-mock>=3.10", "ruff>=0.1"]

# Deprecated (dormant, kept for reference)
yoloe = ["ultralytics>=8.0"]

# New
omniparser = ["ultralytics>=8.0"]
florence = ["transformers>=4.36", "torch>=2.1"]
model-cache = ["huggingface-hub>=0.20"]
detection = ["ocsd[omniparser,florence,model-cache]"]

# Updated
all = ["ocsd[detection,embeddings,vlm,windows,api,tui,dev]"]
```

---

## Testing Strategy

### New test files:

1. **`tests/test_model_cache.py`**
   - Mock `huggingface_hub.hf_hub_download` and `snapshot_download`
   - Test `ensure_model()` downloads when not cached
   - Test `ensure_model()` returns cached path when already downloaded
   - Test `get_cache_dir()` with default and config override
   - Test `clear_cache()` removes correct files

2. **`tests/test_omniparser.py`**
   - Mock `ultralytics.YOLO` model
   - Test `detect()` returns `list[CandidateElement]` with correct fields
   - Test `detect()` filters by confidence threshold
   - Test `detect_and_match()` returns best CLIP match above threshold
   - Test `detect_and_match()` returns None when no match above threshold
   - Test `detect_and_match()` filters by search radius
   - Test lazy model loading (model not loaded until first call)
   - Test thread safety (concurrent detect calls)

3. **`tests/test_florence.py`**
   - Mock `transformers.AutoModelForCausalLM` and `AutoProcessor`
   - Test `caption_crop()` returns string caption
   - Test `caption_batch()` returns list of captions
   - Test `describe_element()` returns structured dict with type_guess, label_guess
   - Test lazy model loading
   - Test variant selection (large vs base)

4. **`tests/test_detection_provider.py`**
   - Test `get_detector()` returns `OmniParserProvider` for config `"omniparser"`
   - Test `get_detector()` raises `ValueError` for unknown detector
   - Test singleton behavior (same instance returned)

### Updated test files:

5. **`tests/test_smart_detect.py`** — Update to mock OmniParser instead of YOLOE
6. **`tests/test_cascade.py`** — Update Stage 1 to use OmniParser mock

### Existing YOLOE test files (left as-is):

7. **`tests/test_yoloe_text_prompt.py`** — Tests for `core/yoloe.py`. Left unchanged since the module is dormant but still importable. These tests continue to pass via existing conftest mocks.
8. **`tests/test_refine.py`** — Tests for `refine_bbox()` and `infer_bbox_at_point()`. Left unchanged — these test `core/yoloe.py` directly. Once `_try_refine_bbox()` in main.py is migrated to use OmniParser, equivalent tests should be added to `test_omniparser.py`.

### Test principles:
- All tests mock the actual models (no GPU needed, no weight download)
- Tests run fast (<1s each)
- `conftest.py` updated with OmniParser and Florence-2 fixtures

---

## YOLOE Deprecation

`core/yoloe.py` remains in the repo but is no longer imported by any active module:

- `recorder/smart_detect.py` — remove YOLOE import, use `detection.get_detector()`
- `mapper/runner.py` — remove YOLOE import, use `detection.get_detector()`
- `main.py` — three YOLOE touchpoints to update:
  1. Line ~354: Remove `from core.yoloe import init_text_embeddings` and the call. Replace with OmniParser/Florence-2 model pre-loading.
  2. Line ~111: Remove `from core.yoloe import infer_bbox_at_point, refine_bbox`. The `_try_refine_bbox()` function (line ~80) uses these for bbox refinement on user click/draw. Replace with OmniParser: after detecting all elements, find the detection whose bbox best overlaps the user's click point or drawn rect. This is simpler and more reliable than YOLOE's one-shot visual grounding.
  3. Line ~200: `detect_ui_elements_async` call — no change needed (smart_detect.py handles the swap internally).

The `[yoloe]` optional dependency group stays in `pyproject.toml` for reference but is not included in `[all]`.

---

## Hardware Requirements

| Component | VRAM | Time per call |
|-----------|------|---------------|
| OmniParser YOLOv8 | ~200MB | ~20-50ms |
| Florence-2-large | ~1.5GB | ~200-300ms/crop |
| Florence-2-base | ~460MB | ~100ms/crop |
| CLIP (existing) | ~400MB | ~10ms/embedding |
| Qwen3-VL (external) | Depends on variant | 1-3s |

**Total VRAM for default config (Florence-2-large):** ~2.1GB across two GPUs
- GPU 0: OmniParser YOLOv8 (~200MB)
- GPU 1: Florence-2-large (~1.5GB) + CLIP (~400MB)

Fits comfortably on dual RTX 2070 (8GB each).

**CPU fallback:** All components work on CPU with degraded speed. Florence-2-large on CPU is ~1-2s/crop instead of ~200-300ms.

---

## Future: MIT YOLO Fine-tune

When the MIT-licensed YOLO fine-tune is ready:

1. Create `core/mit_yolo.py` implementing `DetectionProvider`
2. Add `mit-yolo` option to `config['models']['detector']`
3. Update `get_detector()` factory to return `MitYoloProvider` when selected
4. Remove `ultralytics` dependency from `[omniparser]` group
5. Set `detector: "mit-yolo"` as default in `config.yaml`

**No other code changes needed** — the `DetectionProvider` abstraction handles the swap.

Training data sources for future fine-tune:
- [Rico](https://interactionmining.org/rico) — 66k mobile UI screenshots with annotations
- [WebUI](https://huggingface.co/datasets/biglab/webui-7k) — web UI screenshots
- Custom: OmniParser detections on real recordings (bootstrap from prototype usage)

---

## Summary of Changes

| File | Change |
|------|--------|
| `core/model_cache.py` | **NEW** — HuggingFace weight download utility |
| `core/detection.py` | **NEW** — DetectionProvider protocol + factory |
| `core/omniparser.py` | **NEW** — OmniParser YOLOv8 wrapper |
| `core/florence.py` | **NEW** — Florence-2 captioning module |
| `core/vision.py` | **UNCHANGED** — callers pass Florence-2 context via existing `context_prompt` param |
| `recorder/smart_detect.py` | **MODIFY** — swap YOLOE for OmniParser + Florence-2 |
| `mapper/runner.py` | **MODIFY** — swap YOLOE Stage 1 for OmniParser |
| `main.py` | **MODIFY** — replace YOLOE init with OmniParser/Florence-2 init |
| `config.yaml` | **MODIFY** — add new model config keys |
| `pyproject.toml` | **MODIFY** — add new dependency groups |
| `core/yoloe.py` | **DORMANT** — no changes, no longer imported |
| `tests/conftest.py` | **MODIFY** — add OmniParser/Florence-2 fixtures |
| `tests/test_model_cache.py` | **NEW** |
| `tests/test_omniparser.py` | **NEW** |
| `tests/test_florence.py` | **NEW** |
| `tests/test_detection_provider.py` | **NEW** |
| `tests/test_smart_detect.py` | **MODIFY** — update mocks |
| `tests/test_cascade.py` | **MODIFY** — update Stage 1 |
