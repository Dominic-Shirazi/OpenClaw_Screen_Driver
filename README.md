# OCSD — OpenClaw Screen Driver

AI-powered screen automation framework. Show it how to do something once (record), it learns the path, and replays it autonomously — finding elements even if they moved, self-verifying each step, and recovering from failures.

**Stage 1 complete.** Record → Save → Execute → Verify loop works end-to-end.

---

## Quick Start

### Prerequisites

| Dependency | Install |
|------------|---------|
| **Python 3.11+** | [python.org](https://www.python.org/downloads/) |
| **Tesseract OCR** | `winget install UB-Mannheim.TesseractOCR` (Win) / `brew install tesseract` (Mac) / `sudo apt install tesseract-ocr` (Linux) |
| **LiteLLM proxy** (optional) | For VLM features — see [LiteLLM docs](https://docs.litellm.ai/) |

### Install

```bash
# Clone the repo
git clone https://github.com/openclaw/ocsd.git
cd ocsd

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Or install as editable package with all features:
pip install -e ".[all,dev]"
```

### Optional Feature Groups

Install only what you need:

```bash
pip install -e ".[embeddings]"   # CLIP + FAISS visual similarity search
pip install -e ".[yoloe]"        # YOLOE one-shot visual grounding (~50MB model)
pip install -e ".[vlm]"          # VLM analysis via LiteLLM proxy
pip install -e ".[api]"          # FastAPI REST server
pip install -e ".[windows]"      # Windows accessibility bridge (pywinauto)
pip install -e ".[dev]"          # pytest, pytest-mock, ruff
pip install -e ".[all]"          # Everything above
```

### Configure

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your LiteLLM proxy URL and API key
# (optional — OCSD works without VLM, just fewer locate strategies)
```

---

## Usage

### Record a workflow

```bash
python main.py --record
```

1. A transparent overlay appears on your screen
2. Click on UI elements to tag them (a dialog pops up for each)
3. Set element type, label, and layer for each element
4. Press **Ctrl+Shift+S** to save the skill
5. Press **Escape** to cancel

### Record a diagram (unconnected nodes)

```bash
python main.py --diagram
```

Same as `--record` but creates nodes without edges. Useful for mapping a page layout that you'll connect later.

### Execute a saved skill

```bash
python main.py --execute path/to/skill.json
```

Replays the recorded path using the 5-stage locate cascade:

1. **YOLOE** (~20ms) — saved snippet visual grounding
2. **CLIP** (~50ms) — embedding similarity search
3. **OCR** (~200ms) — scoped text match around expected position
4. **VLM** (1-3s) — full-screen LiteLLM analysis
5. **Position fallback** — blind click at recorded coordinates

### Run the API server

```bash
pip install -e ".[api]"
uvicorn api.server:app --port 8420 --reload
```

Endpoints:
- `GET /health` — health check
- `GET /skills` — list all skills
- `GET /skills/{id}` — get skill graph data
- `GET /skills/{id}/nodes` — list nodes
- `GET /skills/{id}/plan` — preview execution plan
- `POST /skills/{id}/execute` — run a skill
- `DELETE /skills/{id}` — delete a skill

---

## Project Structure

```
ocsd/
├── main.py                  # CLI entry point (--record, --diagram, --execute)
├── config.yaml              # All configuration (models, thresholds, paths)
├── .env.example             # Environment variables template
├── pyproject.toml           # Package config + optional dependency groups
├── requirements.txt         # Core deps for quick install
│
├── core/                    # Shared infrastructure
│   ├── types.py             # Data classes (Point, Rect, LocateResult, etc.)
│   ├── config.py            # YAML + .env config loader
│   ├── capture.py           # Screenshot capture, pixel diff, snippet loading
│   ├── ocr.py               # Tesseract OCR with scoped region search
│   ├── yoloe.py             # YOLOE targeted visual finder (one-shot grounding)
│   ├── embeddings.py        # CLIP embeddings + FAISS index
│   ├── vision.py            # VLM analysis via LiteLLM (element ID, action confirm)
│   ├── executor.py          # Mouse/keyboard action execution
│   ├── watcher.py           # Background screen change monitor
│   └── accessibility.py     # Windows UI Automation bridge
│
├── recorder/                # Recording UI
│   ├── overlay.py           # PyQt6 transparent overlay (click-to-tag, draw boxes)
│   ├── dialog.py            # Element tagging dialog (type, label, layer)
│   ├── element_types.py     # ElementType enum + display metadata
│   └── session.py           # Recording session state manager
│
├── mapper/                  # Graph engine + replay
│   ├── graph.py             # OCSDGraph (NetworkX wrapper, CRUD, stats)
│   ├── runner.py            # Replay orchestrator + 5-stage locate cascade
│   ├── pathfinder.py        # Weighted shortest path, waypoints, branch decisions
│   ├── validator.py         # Post-action verification (pixel diff + VLM)
│   ├── export.py            # Skill JSON serialization + SHA256 integrity
│   ├── layers.py            # UI layer classification (OS / App / Page)
│   └── diff.py              # Graph version diffing
│
├── hub/                     # Skill management
│   ├── manifest.py          # Skill versioning + metadata
│   ├── scanner.py           # Malicious pattern detection
│   └── schema.py            # Skill file JSON schema validation
│
├── api/                     # REST API
│   └── server.py            # FastAPI MCP-compatible endpoints
│
├── executor/                # Legacy execution engine (being reconciled)
│   ├── runner.py
│   ├── pathfinder.py
│   └── validator.py
│
└── tests/                   # Test suite (76 tests)
    ├── conftest.py           # Pre-mocks heavy deps for lightweight testing
    ├── test_cascade.py       # Locate cascade tests (19 tests)
    ├── test_wave2.py         # Graph, watcher, vision tests
    ├── test_export.py        # Skill serialization tests
    ├── test_pathfinder.py    # Pathfinding tests
    └── test_milestone.py     # Integration milestone tests
```

---

## Configuration

All settings live in `config.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `models` | VLM model name, CLIP model, YOLOE weights, Whisper |
| `detection` | Confidence thresholds, crop buffer, search radius |
| `execution` | Mouse speed, typing speed, typo simulation, human delay |
| `litellm` | LiteLLM proxy URL and API key |
| `paths` | Skills dir, snippets dir, FAISS index, replay logs |
| `recovery` | Optional recovery LLM (Claude Opus) for self-healing |
| `fingerprinting` | App identity detection timeout and retries |

Environment variables in `.env` override config.yaml values where noted.

---

## System Dependencies

### Tesseract OCR (required)

OCSD auto-detects Tesseract in standard install locations. If it's installed somewhere custom, set `TESSERACT_CMD` in your `.env`:

```bash
# Windows
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Mac (Homebrew)
TESSERACT_CMD=/opt/homebrew/bin/tesseract

# Linux
TESSERACT_CMD=/usr/bin/tesseract
```

### LiteLLM Proxy (optional, for VLM features)

VLM features (element identification, action confirmation) require a running LiteLLM proxy with vision model routes configured:

```bash
# In .env
LITELLM_BASE_URL=http://your-litellm-host:4000/v1
LITELLM_API_KEY=sk-your-key
```

Without LiteLLM, OCSD falls back to OCR + CLIP + YOLOE + position for element location. VLM is the most accurate but slowest strategy.

---

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests mock all heavy dependencies (torch, transformers, faiss, ultralytics) so they run fast without GPU or model weights.

---

## What's Next (Planned)

- **Smart Recording (Wave 4):** YOLOE auto-detects all elements on screen, VLM identifies each one, user reviews instead of clicking individually
- **Recovery LLM:** When replay fails, send screenshot to Claude Opus for corrective action
- **`--compose` mode:** Draw edges between existing diagram nodes
- **Docker:** Containerized deployment for portability
- **Voice input:** faster-whisper for hands-free element tagging
- **Skill Hub:** Share and download community skills
- **Multi-monitor support**
- **Mac/Linux overlay ports**

---

## License

TBD
