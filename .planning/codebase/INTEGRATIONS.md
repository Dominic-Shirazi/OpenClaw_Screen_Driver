# Integrations

## External Services

### 1. LiteLLM Proxy (Primary VLM Gateway)
- **Type:** HTTP, OpenAI-compatible API
- **Purpose:** Routes VLM requests through fallback chains
- **Connection:** `LITELLM_BASE_URL` (default: `http://localhost:4000/v1`)
- **Authentication:** `LITELLM_API_KEY`
- **Model Groups:**
  - `vision` → Qwen3 VL → Gemini → local fallback
  - `quick` → Gemini 3.1 Flash Lite
  - `planning` → DeepSeek → Cerebras → local
  - `coding` → Qwen3 → Kimi K2 → Gemini → local
- **Methods:** `chat/completions` (POST) with multimodal (text + base64 images)
- **Used in:** `core/vision.py` — `analyze_crop()`, `confirm_action()`, `first_pass_map()`

### 2. HuggingFace Hub
- **Type:** HTTP, model artifact download
- **Purpose:** Auto-download model weights on first use
- **Models:**
  - `microsoft/OmniParser-v2.0` → YOLOv8 icon detection weights
  - `microsoft/Florence-2-large` → Vision-language captioning
  - `openai/clip-vit-base-patch32` → Embeddings encoder
- **Cache:** `~/.cache/ocsd/models/`
- **Library:** `huggingface_hub.hf_hub_download()`, `snapshot_download()`
- **Used in:** `core/detection.py`, `core/omniparser.py`, `core/embeddings.py`

### 3. Tesseract OCR (System Binary)
- **Type:** System-level executable (spawned process)
- **Purpose:** Text detection & OCR
- **Discovery:** ENV var → PATH → platform-specific hints
- **Fallback Paths:**
  - Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - macOS: `/usr/local/bin/tesseract` or `/opt/homebrew/bin/tesseract`
  - Linux: `/usr/bin/tesseract`
- **Methods:** `image_to_string()`, `image_to_data()` (boxes + confidence)
- **Used in:** `core/ocr.py`

### 4. Ollama (Optional Fallback)
- **Type:** Local LLM server
- **Purpose:** Runs when LiteLLM proxy has no cloud models available
- **Default port:** 11434
- **Not directly integrated** — reached via LiteLLM router config
- **Used in:** Referenced in `config.yaml` model fallback chains

## Local Data Stores

### FAISS Vector Index
- **Type:** In-process vector database
- **Purpose:** CLIP embedding similarity matching for element relocation
- **Index type:** `IndexFlatIP` (inner product / cosine similarity)
- **Dimension:** 512 (CLIP vision output)
- **Storage:**
  - Index: `./assets/faiss.index`
  - Metadata: `./assets/faiss.index.meta.json` (element_id → position mapping)
- **Operations:** `generate_embedding()`, `save_to_index()`, `search_index()`
- **Used in:** `core/embeddings.py`

### Skill Graph Storage
- **Type:** JSON files on disk
- **Purpose:** Persist recorded skills as directed graphs
- **Location:** `./skills/` directory (configurable via `paths.skills_dir`)
- **Format:** NetworkX JSON graph format
- **Used in:** `mapper/graph.py`, `mapper/export.py`

## API Endpoints (FastAPI)

**Server:** `api/server.py` — Default port 8742, MCP-compatible

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| GET | `/skills` | List all recorded skills |
| GET | `/skills/{id}` | Get skill graph JSON |
| POST | `/skills/{id}/execute` | Execute a skill |
| GET | `/skills/{id}/plan` | Get execution plan |
| DELETE | `/skills/{id}` | Delete a skill |

## Integration Patterns

### VLM Analysis Pipeline
1. Screenshot captured (BGR numpy array) via `core/capture.py`
2. Region cropped and encoded to base64 data URI
3. Sent to LiteLLM proxy as OpenAI chat completion
4. Response parsed (JSON extraction with fuzzy fallback)
5. Element types validated against `ElementType` enum

### Element Matching (Replay)
1. Saved element crop from recording time
2. CLIP embedding generated (`normalize L2`)
3. Current screen analyzed via OmniParser detection
4. Candidate crops embedded and compared
5. FAISS `search_index()` finds nearest neighbors
6. Spatial filter applied (radius-based from expected position)
7. Top match returned if above `match_threshold`

### Model Caching
1. `huggingface_hub.hf_hub_download()` on first use
2. Cache: `~/.cache/ocsd/models/models--org--repo/snapshots/hash/`
3. Model loaded once, reused globally (lazy-loaded, thread-safe)

### Configuration Merging
1. Defaults loaded from code (`core/config.py`)
2. YAML overlays user config (`config.yaml`)
3. Environment variables override all (via `python-dotenv`)
4. `_deep_merge()` recursively applies overrides
