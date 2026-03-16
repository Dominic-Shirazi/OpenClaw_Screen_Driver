# OCSD — OpenClaw Screen Driver

AI-powered screen automation framework. Show it how to do something once (record), it learns the path, and replays it autonomously — finding elements even if they moved, self-verifying each step, and recovering from failures.

Reduces token usage by ~99% compared to full computer-use models by recording reusable "edges" (UI interaction paths) that replay with cheap vision matching instead of expensive LLM calls.

---

## Quick Start

### Prerequisites

| Dependency | Install |
|------------|---------|
| **Python 3.11+** | [python.org](https://www.python.org/downloads/) |
| **Tesseract OCR** | `winget install UB-Mannheim.TesseractOCR` (Win) / `brew install tesseract` (Mac) / `sudo apt install tesseract-ocr` (Linux) |
| **Ollama** (optional) | For local VLM features — [ollama.com](https://ollama.com/) |

### Install

```bash
git clone https://github.com/openclaw/ocsd.git
cd ocsd

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Core only:
pip install -e .

# With detection stack (OmniParser + Florence-2):
pip install -e ".[detection]"

# Everything:
pip install -e ".[all,dev]"
```

### Optional Feature Groups

```bash
pip install -e ".[embeddings]"   # CLIP + FAISS visual similarity search
pip install -e ".[detection]"    # OmniParser (YOLOv8) + Florence-2 captioning
pip install -e ".[vlm]"          # VLM analysis via LiteLLM proxy
pip install -e ".[api]"          # FastAPI REST server
pip install -e ".[windows]"      # Windows accessibility bridge (pywinauto)
pip install -e ".[tui]"          # Rich TUI launcher
pip install -e ".[dev]"          # pytest, ruff
pip install -e ".[all]"          # Everything above
```

---

## Usage

### Record a workflow

```bash
python main.py --record
```

1. A transparent overlay appears (green border = passthrough mode)
2. Press **Ctrl+R** to enter record mode (red border)
3. OmniParser auto-detects UI elements; review each one via dialog
4. Click or drag-to-box missed elements manually
5. Press **Ctrl+Q** to save and exit

### Record a diagram (unconnected nodes)

```bash
python main.py --diagram --name my_page
```

Same as `--record` but creates nodes without edges — for page layout mapping.

### Execute a saved skill

```bash
python main.py --execute path/to/skill.json
python main.py --execute path/to/skill.json --to "Submit"   # stop at label
python main.py --dry-run --execute path/to/skill.json       # simulate
```

### Compose edges on a diagram

```bash
python main.py --compose path/to/diagram.json
```

Click source node → click target node to draw edges interactively.

### Locate Cascade (replay)

During execution, elements are found using a 5-stage cascade (cheapest first):

1. **OmniParser detect+match** (~50ms) — YOLOv8 detects UI boxes, CLIP matches saved snippet
2. **CLIP embedding** (~100ms) — cosine similarity against saved recording embedding
3. **OCR text match** (~200ms) — scoped Tesseract search near expected position
4. **VLM full scan** (1-3s) — LiteLLM vision model identifies element by label
5. **Position fallback** — blind click at recorded coordinates

### API Server

```bash
pip install -e ".[api]"
uvicorn api.server:app --port 8420 --reload
```

---

## Project Structure

```
ocsd/
├── main.py                          # CLI entry point + compose command
├── config.yaml                      # All configuration
├── pyproject.toml                   # Package config + optional deps
│
├── core/                            # Shared infrastructure
│   ├── types.py                     # Data classes (Point, Rect, LocateResult, etc.)
│   ├── config.py                    # YAML config loader
│   ├── capture.py                   # Screenshots, pixel diff, snippet loading
│   ├── locate.py                    # 5-stage element location cascade
│   ├── ocr.py                       # Tesseract OCR with scoped region search
│   ├── detection.py                 # OmniParser (YOLOv8) detector wrapper
│   ├── omniparser.py                # OmniParser model loading + inference
│   ├── florence.py                  # Florence-2 captioning
│   ├── embeddings.py                # CLIP embeddings + FAISS index
│   ├── vision.py                    # VLM analysis via LiteLLM
│   ├── executor.py                  # Mouse/keyboard action execution
│   ├── gpu.py                       # Centralized GPU memory management
│   ├── model_cache.py               # HuggingFace model weight management
│   ├── watcher.py                   # Background screen change monitor
│   └── accessibility.py             # Windows UI Automation bridge
│
├── recorder/                        # Recording UI + session logic
│   ├── record_controller.py         # Recording session orchestration
│   ├── overlay.py                   # OverlayController (mode, callbacks)
│   ├── overlay_view.py              # PyQt6 transparent overlay window
│   ├── overlay_items.py             # Interactive bounding boxes + handles
│   ├── hotkeys.py                   # Global hotkey listeners (Win32/pynput)
│   ├── smart_detect.py              # OmniParser + Florence-2 auto-detection
│   ├── dialog.py                    # Element tagging dialog
│   └── element_types.py             # ElementType enum
│
├── mapper/                          # Graph engine + replay
│   ├── execute_controller.py        # Skill execution orchestration
│   ├── runner.py                    # Step-by-step replay engine
│   ├── orchestrator.py              # Preflight checks + recovery LLM
│   ├── graph.py                     # OCSDGraph (NetworkX wrapper)
│   ├── pathfinder.py                # Weighted shortest path
│   ├── validator.py                 # Post-action verification
│   ├── export.py                    # Skill JSON serialization
│   └── layers.py                    # UI layer classification
│
├── hub/                             # Skill management
│   ├── manifest.py                  # Skill versioning + metadata
│   ├── scanner.py                   # Malicious pattern detection
│   └── schema.py                    # Skill file JSON schema validation
│
├── api/                             # REST API
│   └── server.py                    # FastAPI MCP-compatible endpoints
│
└── tests/                           # Test suite
    ├── conftest.py                  # Pre-mocks heavy deps
    ├── test_cascade.py              # Locate cascade tests
    ├── test_wave2.py                # Graph, watcher, vision tests
    ├── test_export.py               # Skill serialization tests
    ├── test_pathfinder.py           # Pathfinding tests
    └── test_milestone.py            # Integration milestone tests
```

---

## Configuration

All settings live in `config.yaml`:

| Section | What it controls |
|---------|-----------------|
| `models` | VLM model, CLIP model, OmniParser weights, Whisper |
| `detection` | Confidence thresholds, crop buffer, search radius |
| `execution` | Mouse speed, typing speed, human-like delay |
| `litellm` | LiteLLM proxy URL and API key |
| `paths` | Skills dir, snippets dir, FAISS index, replay logs |
| `recovery` | Recovery LLM for self-healing replay |

### Tesseract OCR

OCSD auto-detects Tesseract in standard locations. Custom path:

```bash
# In .env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### LiteLLM (optional)

```bash
# In .env
LITELLM_BASE_URL=http://your-litellm-host:4000/v1
LITELLM_API_KEY=sk-your-key
```

Without LiteLLM, OCSD uses OmniParser + CLIP + OCR + position fallback for element location.

---

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests mock all heavy dependencies (torch, transformers, faiss, ultralytics) — runs fast without GPU.

---

## License

TBD
