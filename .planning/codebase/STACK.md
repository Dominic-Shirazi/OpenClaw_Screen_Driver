# Stack

## Language & Runtime

- **Python 3.11+** (3.12.10 currently installed)
- **Package Manager:** pip via setuptools (`pyproject.toml`)
- **Entry Point:** `main.py` with `ocsd` CLI script

## Core Dependencies

### Screen Capture & Input Control
- `mss 9.0+` — Fast multi-monitor screenshot capture
- `pyautogui 0.9.54+` — Cross-platform mouse/keyboard automation
- `pynput 1.7+` — Global input monitoring (hotkeys)
- `opencv-python 4.8+` — Image processing (BGR format)
- `Pillow 10.0+` — Image manipulation
- `numpy 1.24+` — Numerical array operations

### Text & OCR
- `pytesseract 0.3.10+` — OCR engine wrapper
- **Tesseract** (system-level binary, platform-specific paths)

### Graph & Data
- `networkx 3.0+` — Graph operations (skill graphs)
- `pyyaml 6.0+` — YAML config parsing
- `python-dotenv 1.0.0+` — `.env` file loading

### GUI
- `PyQt6 6.5+` — Cross-platform GUI (recording overlay, debugger)

### API
- `fastapi 0.100+` — REST API framework
- `uvicorn[standard] 0.23+` — ASGI server
- `pydantic` — Data validation

### TUI
- `rich 13.0+` — Terminal UI (interactive menu)

## Optional Dependency Groups

### embeddings (CLIP + FAISS)
- `transformers 4.36+` — HuggingFace model loading
- `torch 2.1+` — Deep learning framework
- `faiss-cpu 1.7+` — Vector similarity search (`IndexFlatIP`, 512-dim)
- Model: `openai/clip-vit-base-patch32` (~400MB)

### vlm (Vision Language Model)
- `openai 1.0.0+` — OpenAI-compatible API client
- Routes through **LiteLLM proxy** (OpenAI-compatible)
- Model groups: `vision`, `quick`, `planning`, `coding`

### detection (OmniParser + Florence-2)
- `ultralytics 8.3+` — YOLOv8 icon detection
- Model: `microsoft/OmniParser-v2.0` (auto-downloaded)
- Optional: `microsoft/Florence-2-large` for captioning

### model-cache (HuggingFace Hub)
- `huggingface-hub 0.20+` — Model download & caching
- Cache: `~/.cache/ocsd/models/`

### windows (Accessibility)
- `pywinauto 0.6.8+` — Win32 accessibility bridge

### dev
- `pytest 7.0+`, `pytest-mock 3.0+`, `ruff 0.1+`

## Installation Variants

```bash
pip install "openclaw-screen-driver"              # Base
pip install "openclaw-screen-driver[embeddings]"   # CLIP + FAISS
pip install "openclaw-screen-driver[vlm]"          # LiteLLM VLM
pip install "openclaw-screen-driver[detection]"    # OmniParser
pip install "openclaw-screen-driver[api]"          # FastAPI server
pip install "openclaw-screen-driver[tui]"          # Rich TUI
pip install "openclaw-screen-driver[windows]"      # pywinauto
pip install "openclaw-screen-driver[all]"          # Everything
pip install "openclaw-screen-driver[dev]"          # Development
```

## Configuration System

**Priority order:** Environment variables > config.yaml > code defaults

### Key Environment Variables
- `LITELLM_BASE_URL` — LiteLLM proxy endpoint (default: `http://localhost:4000/v1`)
- `LITELLM_API_KEY` — API key
- `OCSD_VLM_MODEL`, `OCSD_QUICK_MODEL`, `OCSD_PLANNING_MODEL`, `OCSD_CODING_MODEL`
- `TESSERACT_CMD` — Tesseract binary path
- `OCSD_FAISS_INDEX` — FAISS index path override
- `OCSD_LOCAL_ONLY` — Force local-only mode
- `OCSD_CONFIG_PATH` — Alternative config.yaml path

### config.yaml Structure
```yaml
hardware:
  gpu_vlm: 0
  gpu_embeddings: 1
models:
  vlm: "vision"
  clip: "openai/clip-vit-base-patch32"
  detector: "omniparser"
detection:
  confidence_threshold: 0.3
  match_threshold: 0.7
execution:
  locate_timeout_s: 5.0
  human_delay: 1.0
api:
  host: "0.0.0.0"
  port: 8742
  mcp_compatible: true
```

## Platform-Specific Notes

### Windows
- DPI awareness auto-detected via ctypes
- Tesseract: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### macOS
- Metal GPU via PyTorch
- Tesseract: `/opt/homebrew/bin/tesseract` (Apple Silicon) or `/usr/local/bin/tesseract` (Intel)

### Linux
- Tesseract: `/usr/bin/tesseract`
