# Architecture

**Analysis Date:** 2025-03-16

## Pattern Overview

**Overall:** Layered pipeline with event-driven recording and cascading element location during replay.

**Key Characteristics:**
- Separation of concerns across 5 layers: CLI/Entry → Recorder/Mapper → Graph Engine → Core Services → PyAutoGUI/System
- Dual operational modes: record (capture UI interactions) and execute (replay with resilience)
- Multi-stage location cascade (5 strategies ordered by speed/cost)
- Extensible detection system supporting OmniParser, CLIP, OCR, VLM, and position fallback
- Event-driven callbacks throughout recording and execution pipelines
- Thread-safe control of mouse/keyboard via mutex serialization

## Layers

**CLI/Entry:**
- Purpose: Command routing, argument parsing, setup (DPI awareness, logging, signal handling)
- Location: `main.py`
- Contains: `main()`, `cmd_record()`, `cmd_execute()`, `cmd_compose()`
- Depends on: `core.config`, `recorder.record_controller`, `mapper.execute_controller`
- Used by: System entry point

**Recording Layer (UI Capture):**
- Purpose: Capture user interactions on screen, detect UI elements, build graph from sequential clicks/drags
- Location: `recorder/`
- Contains: `record_controller.py`, `overlay.py`, `overlay_view.py`, `overlay_items.py`, `smart_detect.py`, `dialog.py`, `hotkeys.py`
- Depends on: `core.capture`, `core.config`, `core.detection`, `core.vision`, `core.types`, PyQt6
- Used by: `main.py` (--record, --diagram modes)

**Graph/Mapping Layer (Skill Structure):**
- Purpose: Build, manipulate, serialize, and query the UI automation graph; pathfinding for execution
- Location: `mapper/`
- Contains: `graph.py`, `export.py`, `pathfinder.py`, `validator.py`, `orchestrator.py`, `runner.py`, `execute_controller.py`
- Depends on: `core.types`, `core.locate`, `core.executor`, networkx, `hub.manifest`
- Used by: Recording layer (exports graph to JSON), execution layer (loads and traverses graph)

**Execution/Replay Layer:**
- Purpose: Execute saved skills with element location, action execution, validation, and recovery
- Location: `mapper/runner.py`, `mapper/orchestrator.py`, `mapper/execute_controller.py`
- Contains: Step-by-step replay, pre-flight checks, recovery LLM, event callbacks
- Depends on: `core.locate`, `core.executor`, `mapper.graph`, `mapper.validator`
- Used by: `main.py` (--execute mode)

**Core Services Layer (Vision + Control):**
- Purpose: Low-level screen sensing, element detection, action execution, configuration
- Location: `core/`
- Contains: `locate.py`, `capture.py`, `detection.py`, `ocr.py`, `embeddings.py`, `vision.py`, `executor.py`, `types.py`, `config.py`
- Depends on: mss, PIL, pytesseract, opencv-python, pyautogui, torch/transformers (optional), faiss-cpu (optional)
- Used by: Recording layer (for smart detection), execution layer (for element location and action execution)

**Skill Management Layer:**
- Purpose: Skill versioning, malicious pattern scanning, schema validation
- Location: `hub/`
- Contains: `manifest.py`, `scanner.py`, `schema.py`
- Depends on: `core.types`
- Used by: Recording layer (export), execution layer (import)

**API Layer (Optional):**
- Purpose: REST endpoints for skill execution and management
- Location: `api/server.py`
- Contains: FastAPI server with MCP-compatible endpoints
- Depends on: `mapper.execute_controller`, `core.config`
- Used by: External HTTP clients

## Data Flow

**Recording Session Flow:**

1. User launches with `--record` or `--diagram`
2. `record_controller.py:cmd_record()` initializes recording session
3. Hotkey listener (Ctrl+R) enters RECORD mode on overlay
4. For each UI element the user interacts with:
   - Screenshot captured (`core.capture.screenshot_full()`)
   - OmniParser detects candidates (`core.detection.get_detector()`)
   - Florence-2 generates captions (`core.florence.caption()`)
   - User confirms label in dialog (`recorder.dialog`)
   - Snippet PNG saved (`core.capture.save_snippet()`)
   - CLIP embedding computed (`core.embeddings.embed_image()`)
   - Node added to graph with position/embeddings stored
5. After each interaction, canvas edge drawn to next node
6. User presses Ctrl+Q to save
7. Graph exported to JSON skill file (`mapper.export.export_skill()`)
8. Snippets and embeddings saved to disk

**Execution Flow:**

1. User launches with `--execute skill.json`
2. `execute_controller.py:cmd_execute()` loads skill
3. `mapper.export.import_skill()` reconstructs OCSDGraph from JSON
4. Entry node determined (first node or --to label target)
5. `mapper.orchestrator.preflight_check()` validates starting screen state
6. For each edge in execution path:
   - `mapper.pathfinder.get_execution_plan()` computes path to target
   - `mapper.runner.execute_node()` executes single step:
     - `core.locate.locate_element()` finds element using cascade
     - `core.executor.click/type_text/scroll()` executes action
     - `mapper.validator.validate_action()` confirms change occurred
   - Event callbacks fired: STEP_START, ELEMENT_LOCATED, ACTION_EXECUTED, VALIDATION_PASSED
7. If element not found or validation fails, recovery LLM invoked
8. Replay log saved to disk

**State Management:**

- **Graph State:** OCSDGraph (NetworkX DiGraph) kept in memory during recording/execution
- **Screen State:** Captured on-demand as numpy arrays (full screenshots)
- **Configuration:** Loaded once from YAML at startup, cached in get_config()
- **Execution State:** Tracked via ReplayLog/ReplayStep dataclasses, saved to disk after execution
- **Thread State:** PyAutoGUI calls serialized via threading.Lock in `core.executor`

## Key Abstractions

**OCSDGraph:**
- Purpose: NetworkX-backed directed graph where nodes = UI elements, edges = transitions
- Examples: `mapper/graph.py`
- Pattern: Wraps nx.DiGraph with schema enforcement (node/edge attributes), CRUD methods, execution tracking
- Serializable to/from JSON (via `mapper.export`)

**LocateResult:**
- Purpose: Encapsulates successful element location result
- Examples: `core/types.py`
- Pattern: Immutable dataclass with point, method (e.g., "ocr", "yoloe", "vlm"), confidence, optional rect
- Used by: Runner to log which cascade stage succeeded

**CandidateElement:**
- Purpose: Detected UI element from OmniParser with bounding box and label guess
- Examples: `core/types.py`
- Pattern: Dataclass with rect, type_guess, label_guess, confidence
- Used by: Smart detection dialog, overlay rendering

**ReplayStep:**
- Purpose: Record of a single executed action: timing, location method, confidence, validation result
- Examples: `core/types.py`
- Pattern: Dataclass with node_id, action_type, locate_result, before/after screenshots, validation status
- Used by: Execution logging, debugging, recovery decision-making

**ElementNotFoundError:**
- Purpose: Raised when all 5 stages of locate cascade fail
- Examples: `core/types.py`
- Pattern: Custom exception with context (node_id, last_result, method_attempts)

## Entry Points

**main.py:main():**
- Location: `main.py`
- Triggers: CLI invocation (`ocsd` command or `python main.py`)
- Responsibilities: Parse args, setup logging/DPI/signal handlers, route to subcommand

**cmd_record():**
- Location: `recorder/record_controller.py`
- Triggers: `--record` or `--diagram` flag
- Responsibilities: Launch overlay, capture interactions, build graph, save skill

**cmd_execute():**
- Location: `mapper/execute_controller.py`
- Triggers: `--execute skill.json` flag
- Responsibilities: Load skill, run orchestrator, save replay log

**cmd_compose():**
- Location: `main.py`
- Triggers: `--compose diagram.json` flag
- Responsibilities: Interactive edge-drawing on pre-built diagram

## Error Handling

**Strategy:** Catch specific exceptions at layer boundaries; recover with fallback strategies in locate cascade.

**Patterns:**

1. **Locate Cascade Fallback:** If stage N fails, move to stage N+1. Only raise ElementNotFoundError after all 5 stages exhausted.
   - Example: OmniParser fails → try CLIP → try OCR → try VLM → try position fallback

2. **Action Validation:** If action produces no pixel change, flag for recovery. If pixel change detected but low confidence, ask VLM.
   - Example: Click button → pixel diff < threshold → recovery LLM diagnoses

3. **Pre-flight Check:** Verify screen state before replay begins. Abort early if expected element not found.
   - Example: User skipped to middle of workflow → preflight detects wrong screen → abort with explanation

4. **VLM Recovery:** When step fails, capture screen and ask LLM for recovery action (scroll, wait, navigate back).
   - Example: Element missing → VLM sees modal dialog → suggests clicking "OK" button

5. **Config Validation:** YAML loaded with defaults; missing keys don't crash.
   - Example: `get_config().get("detection", {}).get("match_threshold", 0.7)` supplies default

6. **Cross-Platform Guards:** Platform checks in code that uses Windows-only APIs.
   - Example: DPI awareness set only on `sys.platform == "win32"`

## Cross-Cutting Concerns

**Logging:**
- Every module imports `logging.getLogger(__name__)`
- Configured in `main.py:_setup_logging()` with format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- Levels: DEBUG for verbose, INFO for normal, WARNING for recoverable issues

**Validation:**
- Graph schema enforced by OCSDGraph (checks layer, element_type, required fields)
- Skill JSON validated by `hub.schema.validate_skill_json()`
- Configuration defaults supplied by `core.config.get_config()` accessor

**Authentication:**
- Tesseract path detected in `core.ocr` (from `TESSERACT_CMD` env var or standard locations)
- LiteLLM credentials from environment variables (LITELLM_BASE_URL, LITELLM_API_KEY)
- No session state — each replay is independent

**Thread Safety:**
- PyAutoGUI calls serialized via `_lock` in `core.executor` (click, type_text, scroll all acquire lock)
- Detection callbacks from smart_detect run on background thread, signal to main thread via Qt signal
- Watcher background thread emits events but doesn't modify shared state

---

*Architecture analysis: 2025-03-16*
