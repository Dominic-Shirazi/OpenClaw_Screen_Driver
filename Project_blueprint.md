# OCSD — OpenClaw Screen Driver

## Machine-Readable Build Blueprint v0.1

> Target: Claude Code + Gemini co-build. Human-readable hints marked `# NOTE`.
> Stack: Python 3.11+, Windows-first, modular, offline-capable.

---

## REPOSITORY STRUCTURE

```
OpenClaw_Screen_Driver/
├── core/
│   ├── capture.py           # screenshot, crop, pixel-diff
│   ├── accessibility.py     # UIA/DOM query layer (hint only, never trust labels)
│   ├── vision.py            # Qwen2-VL local inference wrapper
│   ├── ocr.py               # pytesseract wrapper
│   ├── embeddings.py        # CLIP embedding generation + FAISS index
│   ├── watcher.py           # window title + URL change monitor
│   └── executor.py          # mouse/keyboard actions via PyAutoGUI
├── recorder/
│   ├── overlay.py           # transparent fullscreen overlay (tkinter or PyQt6)
│   ├── dialog.py            # element tagging dialog box
│   ├── element_types.py     # enum: ELEMENT_TYPE library
│   └── session.py           # recording session state manager
├── mapper/
│   ├── graph.py             # NetworkX directed graph: nodes=elements, edges=actions
│   ├── layers.py            # OS_UI / APP_PERSISTENT / PAGE_SPECIFIC layer logic
│   ├── diff.py              # intelligent change detection between map versions
│   └── export.py            # serialize graph → JSON skill file
├── executor/
│   ├── pathfinder.py        # A* or BFS over graph to reach destination node
│   ├── runner.py            # execute node sequence, emit events
│   └── validator.py         # post-action pixel-diff + VLM confirm
├── hub/
│   ├── schema.py            # JSON schema validator for skill files
│   ├── scanner.py           # automated malicious pattern detection
│   └── manifest.py          # skill versioning + metadata
├── api/
│   └── server.py            # FastAPI: MCP-compatible tool call endpoints
├── skills/                  # local skill library (git-tracked)
│   └── .gitkeep
├── assets/
│   └── snippets/            # reference image crops per element
├── tests/
├── config.yaml
└── main.py
```

---

## MODULE SPECS

### `core/capture.py`

```python
# DEPS: mss, Pillow, numpy
# FUNCTIONS:
screenshot_full() -> np.ndarray                  # full screen, all monitors
screenshot_region(x, y, w, h) -> np.ndarray      # bounding box crop
pixel_diff(img_a, img_b, threshold=0.05) -> float  # % changed pixels
save_snippet(img, skill_id, element_id) -> str   # path: assets/snippets/{skill_id}/{element_id}.png
get_window_title() -> str                        # active window title
get_active_url() -> str | None                   # browser URL via UIA (best-effort, not trusted)
```

### `core/accessibility.py`

```python
# DEPS: pywinauto (Windows UIA wrapper)
# NOTE: UIA/DOM labels are HINTS ONLY. Obfuscated, unreliable on most real-world sites.
# USE FOR: element boundary detection, tab-walk structure, window title, URL sniffing ONLY.
# NEVER USE FOR: element labeling, type detection, trust decisions
# FUNCTIONS:
get_element_tree(hwnd) -> list[dict]             # raw UIA dump: {rect, control_type, name_raw}
tab_walk(hwnd) -> list[dict]                     # tab-order traversal, returns rects only
get_focused_element() -> dict                    # currently focused element rect + raw name
```

### `core/vision.py`

```python
# DEPS: ollama Python client
# MODEL: qwen2-vl (via local Ollama, GPU0)
# NOTE: Primary element type detector. Workhorse of the stack.
# FUNCTIONS:
analyze_crop(img_path: str, context_prompt: str) -> dict
# returns: {
#   "element_type": str,       # from ELEMENT_TYPE enum
#   "label_guess": str,        # human-readable guess
#   "confidence": float,       # 0.0-1.0
#   "ocr_text": str | None     # any visible text
# }

confirm_action(before_img: str, after_img: str, intended_action: str) -> dict
# returns: {
#   "success": bool,
#   "confidence": float,
#   "notes": str
# }

first_pass_map(screenshot_path: str) -> list[dict]
# returns list of candidate elements:
# [{rect: {x,y,w,h}, type_guess: str, label_guess: str, confidence: float}]
```

### `core/embeddings.py`

```python
# DEPS: transformers (CLIP: openai/clip-vit-base-patch32), faiss-cpu, numpy
# NOTE: Generates visual fingerprint of element appearance. Survives font changes, color shifts.
# GPU1 preferred, CPU fallback acceptable (FAISS is lightweight)
# FUNCTIONS:
generate_embedding(img: np.ndarray) -> np.ndarray     # 512-dim CLIP vector
save_to_index(element_id: str, embedding: np.ndarray) # FAISS index upsert
search_index(query_embedding: np.ndarray, top_k=5, radius_pct=0.20) -> list[dict]
# radius_pct: only search within 20% of last known screen position
load_index() -> faiss.Index
```

### `core/watcher.py`

```python
# DEPS: pywinauto, psutil, threading
# NOTE: Runs as background thread. Emits events on state change.
# EVENTS:
#   window_changed(old_title, new_title)
#   url_changed(old_url, new_url)           # browser only, best-effort
#   pixel_diff_exceeded(diff_pct, region)   # triggers branch point candidate
# FUNCTIONS:
start_watching(callback: callable, diff_threshold=0.08, poll_ms=200)
stop_watching()
```

### `core/executor.py`

```python
# DEPS: pyautogui, pynput
# NOTE: All mouse movement uses easing curves (pyautogui.easeInOutQuad). Never linear.
# FUNCTIONS:
click(x, y, button='left', easing='easeInOutQuad', duration=0.15)
right_click(x, y)
double_click(x, y)
drag(x1, y1, x2, y2, duration=0.3)
type_text(text: str, interval=0.03)              # human-like typing interval
scroll(x, y, direction: str, amount: int)
hotkey(*keys)                                    # e.g. hotkey('ctrl', 'c')
```

---

## ELEMENT TYPE LIBRARY

```python
# recorder/element_types.py
from enum import Enum

class ElementType(Enum):
    TEXTBOX        = "textbox"         # input field, user types into
    BUTTON         = "button"          # click, stays on page
    BUTTON_NAV     = "button_nav"      # click, navigates to new page/state
    TOGGLE         = "toggle"          # on/off, checkbox, radio
    TAB            = "tab"             # tab navigation within page
    DROPDOWN       = "dropdown"        # expands list of options
    SCROLLBAR      = "scrollbar"       # scroll target, vertical or horizontal
    READ_HERE      = "read_here"       # data destination: extract this value
    DRAG_SOURCE    = "drag_source"     # drag starts here
    DRAG_TARGET    = "drag_target"     # drag ends here
    IMAGE          = "image"           # non-interactive image (may contain text)
    MODAL          = "modal"           # popup/overlay container
    NOTIFICATION   = "notification"    # toast, alert, status message
    UNKNOWN        = "unknown"         # fallback, requires human tag
```

---

## UI LAYER SYSTEM

```python
# mapper/layers.py
# NOTE: 3-tier layer model. Determines how aggressively to flag missing elements.

class LayerType(Enum):
    OS_UI          = "os_ui"           # taskbar, window chrome, system tray
                                       # ASSUME ALWAYS PRESENT. Flag if deleted.
                                       # User must explicitly delete to remove from map.
    APP_PERSISTENT = "app_persistent"  # nav bars, headers, sidebars per app/site
                                       # ASSUME PRESENT per session. Flag if disappears mid-session.
    PAGE_SPECIFIC  = "page_specific"   # unique elements per view/state
                                       # Map fresh per page. Expected to change.

# LAYER ASSIGNMENT HEURISTICS:
# - Element position never changes across 3+ screenshots → candidate OS_UI
# - Element appears on >80% of pages in same app → candidate APP_PERSISTENT
# - Element appears on 1 specific page → PAGE_SPECIFIC
# - User can override any assignment manually
```

---

## MAP GRAPH SCHEMA

```python
# mapper/graph.py
# DEPS: networkx

# NODE schema (element):
{
    "node_id": str,              # uuid4
    "skill_id": str,             # parent skill file
    "element_type": ElementType,
    "layer": LayerType,
    "label": str,                # human-provided via voice/text in dialog
    "label_ai_guess": str,       # Qwen2-VL first-pass guess
    "ocr_text": str | None,
    "embedding_id": str,         # FAISS index key
    "snippet_path": str,         # assets/snippets/{skill_id}/{node_id}.png
    "relative_position": {
        "x_pct": float,          # 0.0-1.0 of screen width
        "y_pct": float,          # 0.0-1.0 of screen height
        "region_hint": str       # "top_right", "center", etc.
    },
    "last_seen_resolution": [int, int],   # [width, height]
    "confidence_threshold": float,        # min match confidence to use this node
    "verified_count": int,                # times successfully used in execution
    "fail_count": int,                    # times failed to locate
    "created_at": str,           # ISO8601
    "updated_at": str
}

# EDGE schema (action/transition):
{
    "edge_id": str,
    "source_node_id": str,
    "target_node_id": str,
    "action_type": ElementType,  # what action on source leads to target
    "action_payload": str | None, # text to type, key to press, etc.
    "is_branch": bool,           # does this edge represent a conditional path?
    "branch_condition": str | None,  # human-described condition for this branch
    "execution_count": int,
    "success_count": int
}
```

---

## RECORDER FLOW

```
# NOTE: This is the human-facing recording session. Runs as overlay on top of any screen.

INIT:
  1. Launch transparent fullscreen overlay (recorder/overlay.py)
  2. Start watcher.py background thread
  3. Take baseline screenshot → send to vision.first_pass_map() → store candidate list
  4. Render candidate highlights on overlay (bounding boxes, color-coded by type_guess)

HOVER LOOP (runs continuously):
  1. Track mouse position
  2. On hover >300ms over candidate region:
     a. Query accessibility.get_focused_element() → append raw UIA hint
     b. Find matching candidate from first_pass_map by proximity
     c. Pre-fill dialog with: type_guess, label_guess, ocr_text, UIA hint
  3. User LEFT-CLICKS element → open TagDialog

TAG DIALOG (recorder/dialog.py):
  Fields:
    - Element Type: dropdown (ElementType enum), pre-filled by Qwen3vl-8b
    - Label: text input, pre-filled by vision label_guess
    - Layer: dropdown (OS_UI / APP_PERSISTENT / PAGE_SPECIFIC), auto-guessed
    - Notes: free text
    - Voice Input: button → triggers faster-whisper → fills Label field
  Actions:
    - [Confirm] → save node, capture snippet, generate CLIP embedding
    - [Skip] → dismiss, mark as UNKNOWN
    - [Branch Point] → flag edge as is_branch=True, prompt for condition description
    - [This is a destination] → set element_type = READ_HERE

POST-TAG:
  1. executor.py executes the action (click/type/etc.)
  2. watcher.py detects state change (pixel_diff or window/URL change)
  3. IF new state detected:
     a. Take new screenshot
     b. Run vision.first_pass_map() on new state
     c. Prompt user: "New page/state detected. Continue mapping here? [Yes/No/Branch]"
  4. IF branch: create new edge with is_branch=True, fork map path
  5. Repeat hover loop on new state
```

---

## EXECUTION ENGINE

```python
# executor/pathfinder.py
# DEPS: networkx

def find_path(graph, start_node_id: str, destination_node_id: str) -> list[str]:
    # BFS on directed graph
    # Returns ordered list of node_ids to visit
    # Prefers edges with highest success_count / execution_count ratio
    # Avoids edges with success_rate < 0.5 unless no alternative

def find_by_label(graph, label: str) -> str | None:
    # fuzzy match label across all nodes
    # returns node_id of best match

def find_read_here_nodes(graph, page_hint: str | None = None) -> list[str]:
    # returns all READ_HERE nodes, optionally filtered by page context
```

```python
# executor/runner.py
# NOTE: Executes a path returned by pathfinder. Emits events. Writes replay log.

def run_path(path: list[str], graph, dry_run=False) -> ReplayLog:
    for node_id in path:
        node = graph.nodes[node_id]
        # 1. LOCATE element on current screen:
        located = locate_element(node)
        # 2. VALIDATE with Qwen2-VL before acting:
        confirmed = vision.confirm_action(screenshot, node['snippet_path'], node['label'])
        if confirmed.confidence < node['confidence_threshold']:
            emit_event('LOW_CONFIDENCE', node_id, confirmed)
            # fallback cascade (see below)
        # 3. ACT:
        executor.click(located.x, located.y)  # or type, drag, etc.
        # 4. POST-ACTION DIFF:
        after = capture.screenshot_full()
        diff = capture.pixel_diff(before, after)
        # 5. LOG:
        replay_log.append(node_id, located, confirmed, diff, success=True)
    return replay_log
```

```python
# executor/runner.py — LOCATE ELEMENT FALLBACK CASCADE
def locate_element(node: dict) -> Point:
    # STEP 1: OCR search — does node['ocr_text'] exist on screen?
    if node['ocr_text']:
        match = ocr.find_text_on_screen(node['ocr_text'])
        if match and match.confidence > 0.8:
            return match.center

    # STEP 2: CLIP embedding search within position radius
    current_screenshot = capture.screenshot_full()
    query_emb = embeddings.generate_embedding(current_screenshot)
    candidates = embeddings.search_index(query_emb, radius_pct=0.20)
    if candidates:
        best = candidates[0]
        if best['score'] > node['confidence_threshold']:
            return best['center']

    # STEP 3: VLM confirmation of candidate
    for candidate in candidates:
        crop = capture.screenshot_region(*candidate['rect'])
        result = vision.analyze_crop(crop, f"Is this a {node['element_type']} labeled '{node['label']}'?")
        if result['confidence'] > 0.75:
            return candidate['center']

    # STEP 4: FAIL — emit event, do not act
    emit_event('ELEMENT_NOT_FOUND', node['node_id'])
    raise ElementNotFoundError(node['node_id'])
```

---

## REPLAY LOG SCHEMA

```json
{
  "replay_id": "uuid4",
  "skill_id": "string",
  "executed_at": "ISO8601",
  "steps": [
    {
      "node_id": "string",
      "located_at": { "x": 0, "y": 0 },
      "locate_method": "ocr|embedding|vlm|failed",
      "vlm_confidence": 0.0,
      "pixel_diff_pct": 0.0,
      "success": true,
      "error": null
    }
  ],
  "overall_success": true,
  "duration_ms": 0
}
```

> NOTE: Replay logs are the self-improving dataset. Over time, fail patterns reveal which elements need re-recording or updated embeddings.

---

## SKILL FILE SCHEMA (FINAL)

```json
{
  "$schema": "ocsd-skill-v1",
  "skill_id": "string (uuid4)",
  "name": "string",
  "description": "string",
  "author": "string",
  "version": "semver string",
  "target_app": "string",
  "target_url": "string | null",
  "os": ["windows", "mac", "linux"],
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "checksum": "sha256 of nodes+edges arrays",
  "nodes": ["...node schema array..."],
  "edges": ["...edge schema array..."],
  "entry_node_id": "string",
  "exit_nodes": ["string"],
  "read_here_nodes": ["string"],
  "tags": ["string"]
}
```

---

## OCSD-HUB AUDIT PIPELINE

```
SUBMISSION FLOW:
  1. Author pushes skill JSON to ocsd-hub GitHub repo via PR
  2. CI runs hub/scanner.py (automated):
     - Validate JSON against schema (hub/schema.py)
     - Check action_payload fields for: shell commands, URLs, encoded strings, base64
     - Check fallback_rules for: eval(), exec(), subprocess patterns
     - Check snippet images for: steganography flags (basic LSB check)
     - Score: CLEAN / FLAGGED / REJECTED
  3. IF CLEAN → auto-merge to pending/
  4. IF FLAGGED → human moderator queue
  5. IF REJECTED → auto-close PR with reason
  6. Human moderator reviews FLAGGED:
     - Approve → move to verified/
     - Reject → close with reason
     - Escalate → security review label
  7. Verified skills available to all OCSD installs via hub/manifest.py

VERSIONING:
  - Skills follow semver (major.minor.patch)
  - Breaking changes (node_id changes) = major bump
  - New nodes added = minor bump
  - Label/threshold tweaks = patch bump
  - Git blame on every skill file
  - Deprecated skills flagged, not deleted (replay logs may reference them)
```

---

## MCP API ENDPOINTS

```python
# api/server.py
# DEPS: FastAPI, uvicorn
# NOTE: Exposes OCSD as MCP-compatible tool calls for Claw Stack integration

POST /run_skill
  body: { skill_id: str, destination_label: str | None, dry_run: bool }
  returns: ReplayLog

POST /run_path
  body: { skill_id: str, start_node_id: str, end_node_id: str }
  returns: ReplayLog

GET /skills
  returns: list[skill manifests]

GET /skill/{skill_id}/map
  returns: full graph JSON

POST /record/start
  body: { session_name: str }
  returns: { session_id: str }

POST /record/stop
  body: { session_id: str }
  returns: { skill_id: str, node_count: int, edge_count: int }

GET /replay_logs
  query: skill_id, limit, success_only
  returns: list[ReplayLog summaries]
```

---

## BUILD STAGES

### STAGE 1 — Core Infrastructure

> Single linear path recording + execution. No branches. No hub.

```
DELIVERABLES:
  - core/capture.py          ✓ screenshot, crop, pixel_diff
  - core/vision.py           ✓ Qwen2-VL wrapper (first_pass_map + analyze_crop)
  - core/embeddings.py       ✓ CLIP + FAISS
  - core/executor.py         ✓ mouse/keyboard
  - core/watcher.py          ✓ window/URL/diff events
  - recorder/overlay.py      ✓ transparent click-through overlay
  - recorder/dialog.py       ✓ tag dialog, voice input button
  - recorder/element_types.py ✓ ElementType enum
  - mapper/graph.py          ✓ node/edge CRUD
  - mapper/export.py         ✓ graph → skill JSON
  - executor/pathfinder.py   ✓ BFS path find
  - executor/runner.py       ✓ linear path execution + fallback cascade
  - executor/validator.py    ✓ post-action diff check
  MILESTONE: Record "login to X" → execute replay → verify success
```

### STAGE 2 — Layer System + Change Detection

> Teach the system about persistent vs page-specific elements.

```
DELIVERABLES:
  - mapper/layers.py         ✓ 3-tier layer logic
  - mapper/diff.py           ✓ compare map versions, flag missing elements
  - recorder/session.py      ✓ multi-page session state
  - watcher.py UPDATE        ✓ emit layer-aware change events
  MILESTONE: Map a 3-page flow. System flags when nav bar disappears.
```

### STAGE 3 — Branch Support

> One action can lead to multiple possible next states.

```
DELIVERABLES:
  - graph.py UPDATE          ✓ is_branch edges, branch_condition field
  - recorder/dialog.py UPDATE ✓ [Branch Point] button + condition text input
  - pathfinder.py UPDATE     ✓ handle branching paths, condition evaluation
  - runner.py UPDATE         ✓ detect which branch taken post-action
  MILESTONE: Map a form where dropdown choice changes next page.
```

### STAGE 4 — MCP API + Claw Stack Integration

> OCSD becomes a tool call.

```
DELIVERABLES:
  - api/server.py            ✓ FastAPI MCP endpoints
  - config.yaml              ✓ port, model paths, GPU assignment
  MILESTONE: Claude Code calls /run_skill via MCP. Gets ReplayLog back.
```

### STAGE 5 — OCSD-Hub

> Open source skill sharing.

```
DELIVERABLES:
  - hub/schema.py            ✓ JSON schema validator
  - hub/scanner.py           ✓ automated audit pipeline
  - hub/manifest.py          ✓ versioning + install from hub
  - GitHub repo structure    ✓ pending/ verified/ deprecated/
  - CI workflow (.github/workflows/audit.yml)
  MILESTONE: Submit a skill to hub. Auto-scan passes. Human approves. Other install pulls it.
```

---

## DEPENDENCIES (pip install)

```txt
# Core
mss                    # fast cross-platform screenshot
Pillow                 # image processing
numpy                  # array ops
opencv-python          # pixel diff, image ops
pytesseract            # OCR (requires tesseract binary)

# Vision / AI
ollama                 # local Qwen2-VL inference client
transformers           # CLIP embedding model
faiss-cpu              # vector similarity search
torch                  # CLIP dependency (CPU build acceptable for embeddings)

# Accessibility
pywinauto              # Windows UIA tree access

# Automation
pyautogui              # mouse + keyboard control
pynput                 # low-level input monitoring

# Graph
networkx               # directed graph for map

# API
fastapi                # MCP server
uvicorn                # ASGI server

# Audio (Stage 1 optional, voice button in dialog)
faster-whisper         # local STT, model:small

# Hub
jsonschema             # skill file validation
```

---

## CONFIG SCHEMA (config.yaml)

```yaml
ocsd:
  version: "0.1.0"

hardware:
  gpu_vlm: 0 # GPU index for Qwen2-VL
  gpu_embeddings: 1 # GPU index for CLIP (cpu fallback ok)

models:
  vlm: "qwen2-vl" # Ollama model name
  vlm_host: "http://localhost:11434"
  clip: "openai/clip-vit-base-patch32"
  whisper_model: "small" # faster-whisper model size

paths:
  skills_dir: "./skills"
  snippets_dir: "./assets/snippets"
  faiss_index: "./assets/faiss.index"
  replay_logs: "./logs/replays"

execution:
  mouse_duration: 0.15 # seconds per mouse move
  type_interval: 0.03 # seconds between keystrokes
  pixel_diff_threshold: 0.08 # % changed to trigger state change event
  hover_delay_ms: 300 # ms hover before auto-analyze
  default_confidence: 0.75 # min VLM confidence to act

api:
  host: "0.0.0.0"
  port: 8742
  mcp_compatible: true

hub:
  registry_url: "https://github.com/openclaw/ocsd-hub"
  auto_update_manifest: true
```

---

## KNOWN HARD PROBLEMS (flag for Claude Code + Gemini)

```
1. OVERLAY INPUT PASSTHROUGH
   Problem: Transparent fullscreen overlay must receive click events for tagging
            but also pass clicks through to the underlying app for recording.
   Approach: Two-mode overlay (RECORD_MODE captures clicks, PASSTHROUGH_MODE forwards them)
   Windows API: WS_EX_TRANSPARENT + WS_EX_LAYERED window flags

2. RESOLUTION SCALING
   Problem: element recorded at 1920x1080 must locate correctly at 2560x1440
   Approach: Always store x_pct/y_pct not absolute pixels. Scale search radius accordingly.

3. ANTI-BOT DETECTION
   Problem: PyAutoGUI mouse moves may trigger bot detection on hostile sites
   Approach: Randomize easing curve parameters slightly per action.
             Add jitter to timing intervals (±15% random).
             DO NOT solve Captchas. Flag and pause for human.

4. DYNAMIC DOM / REACT RE-RENDERS
   Problem: Element rects change on every render even if visually identical
   Approach: Never trust UIA rect as primary locator. Always use CLIP embedding + OCR combo.

5. MULTI-MONITOR
   Problem: mss screenshot coordinates differ from pyautogui coordinates on multi-monitor
   Approach: Normalize all coordinates to primary monitor origin at session start.
             Store monitor config in skill metadata.

6. FAISS INDEX GROWTH
   Problem: Index grows unbounded as skills are added
   Approach: Namespace index by skill_id. Allow per-skill index load/unload.
             Prune embeddings for nodes with fail_count > 10 (stale elements).
```

---

_OCSD Blueprint v0.1 — OpenClaw Screen Driver_
_Feed this file to Claude Code + Gemini CLI as primary build context._
_Start with STAGE 1 deliverables only. Do not scope-creep into Stage 2 until Stage 1 milestone is verified._
