# Codebase Structure

**Analysis Date:** 2025-03-16

## Directory Layout

```
ocsd/
├── main.py                          # CLI entry point (record/diagram/execute/compose)
├── config.yaml                      # Application config (paths, detection, execution, models)
├── pyproject.toml                   # Package metadata + optional dependency groups
│
├── core/                            # Shared infrastructure — sensing, vision, control
│   ├── __init__.py
│   ├── types.py                     # Core data classes (Point, Rect, LocateResult, etc.)
│   ├── config.py                    # YAML config loader with defaults
│   ├── capture.py                   # Screenshots, pixel diff, snippet I/O
│   ├── locate.py                    # 5-stage element location cascade
│   ├── ocr.py                       # Tesseract OCR, scoped text search
│   ├── detection.py                 # OmniParser (YOLOv8) detector wrapper
│   ├── omniparser.py                # OmniParser model init + inference
│   ├── florence.py                  # Florence-2 image captioning
│   ├── embeddings.py                # CLIP embedding computation + FAISS index
│   ├── vision.py                    # VLM analysis via LiteLLM (confirm_action, analyze_crop)
│   ├── executor.py                  # Mouse/keyboard actions (click, type, scroll)
│   ├── gpu.py                       # Centralized GPU memory management
│   ├── model_cache.py               # HuggingFace model weight caching
│   ├── watcher.py                   # Background screen change monitor
│   └── accessibility.py             # Windows UI Automation bridge (pywinauto)
│
├── recorder/                        # Recording UI + session logic
│   ├── __init__.py
│   ├── record_controller.py         # Recording orchestration (cmd_record entry point)
│   ├── overlay.py                   # OverlayController (mode state, callbacks)
│   ├── overlay_view.py              # PyQt6 transparent window (rendering, hit detection)
│   ├── overlay_items.py             # Interactive bounding boxes + resize handles
│   ├── hotkeys.py                   # Global hotkey listeners (Win32/pynput)
│   ├── smart_detect.py              # OmniParser + Florence auto-detection thread
│   ├── dialog.py                    # Element tagging dialog (label, type confirmation)
│   ├── element_types.py             # ElementType enum (button, textbox, icon, etc.)
│   ├── session.py                   # Recording session state holder
│   ├── refine_dialog.py             # Side-by-side bbox refinement UI
│   ├── step_ui.py                   # Step-through UI for execution
│   ├── tui.py                       # Rich TUI menu (rich-based launcher)
│   └── hotkeys.py                   # Global hotkey listeners
│
├── mapper/                          # Graph engine + replay
│   ├── __init__.py
│   ├── graph.py                     # OCSDGraph (NetworkX wrapper + schema)
│   ├── export.py                    # Skill JSON serialization (export_skill, import_skill)
│   ├── execute_controller.py        # Execution orchestration (cmd_execute entry point)
│   ├── runner.py                    # Step-by-step replay (execute_node, run_skill)
│   ├── orchestrator.py              # Pre-flight checks + recovery LLM
│   ├── pathfinder.py                # Weighted shortest path, fingerprint checkpoints
│   ├── validator.py                 # Post-action verification (pixel diff + VLM)
│   ├── layers.py                    # UI layer classification
│   └── diff.py                      # Graph diffing utilities
│
├── hub/                             # Skill management + validation
│   ├── __init__.py
│   ├── manifest.py                  # SkillManifest dataclass + I/O
│   ├── scanner.py                   # Malicious pattern detection
│   └── schema.py                    # JSON schema validation
│
├── api/                             # REST API (optional)
│   └── server.py                    # FastAPI server (execute endpoint, health)
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures + mocks
│   ├── test_cascade.py              # Locate cascade tests
│   ├── test_detection_provider.py   # Detection module tests
│   ├── test_e2e_login.py            # Integration test (HTML login form)
│   ├── test_export.py               # Skill serialization tests
│   ├── test_florence.py             # Florence-2 captioning tests
│   ├── test_pathfinder.py           # Pathfinding tests
│   └── test_milestone.py            # Integration milestone tests
│
└── skills/                          # Recorded skill storage (created at runtime)
    ├── login.json
    ├── checkout.json
    └── <skill_name>/
        ├── snippets/                # Saved UI element screenshots
        └── embeddings/              # FAISS index + metadata
```

## Directory Purposes

**`core/`:**
- Purpose: Reusable services for screen sensing, vision, and control
- Contains: Detection models, OCR, embeddings, mouse/keyboard control, config management
- Key files: `types.py` (data contracts), `config.py` (settings), `locate.py` (element finding)
- Pattern: Stateless functions + lazy model loading (models loaded on first use, cached)

**`recorder/`:**
- Purpose: UI for recording workflows and diagrams
- Contains: PyQt6 overlay, hotkey listeners, element detection UI, dialog flows
- Key files: `record_controller.py` (orchestration), `overlay_view.py` (rendering), `smart_detect.py` (detection thread)
- Pattern: Event-driven (callbacks for element selection, mode changes, save)

**`mapper/`:**
- Purpose: Graph construction, pathfinding, and execution
- Contains: NetworkX graph wrapper, JSON import/export, execution engine, recovery logic
- Key files: `graph.py` (data structure), `runner.py` (replay), `orchestrator.py` (pre-flight + recovery)
- Pattern: Transactional (graph built in memory, exported atomically to JSON)

**`hub/`:**
- Purpose: Skill distribution and security
- Contains: Manifest versioning, malicious pattern scanning, JSON schema validation
- Key files: `manifest.py` (metadata), `schema.py` (validation rules)
- Pattern: Immutable manifests (dataclass), one-file skills (all metadata + graph in JSON)

**`api/`:**
- Purpose: HTTP interface for remote skill execution
- Contains: FastAPI endpoints, request/response marshaling
- Key files: `server.py` (routes)
- Pattern: Thin wrapper around execute_controller

**`tests/`:**
- Purpose: Unit + integration tests
- Contains: Mocked heavy deps (torch, faiss, ultralytics), real snapshot tests, HTML test pages
- Key files: `conftest.py` (fixtures), `test_cascade.py` (locate logic), `test_e2e_login.py` (integration)
- Pattern: Pre-mocked in conftest.py, fast execution without GPU

## Key File Locations

**Entry Points:**
- `main.py`: CLI argument parsing, subcommand routing, setup (DPI awareness, logging, signal handlers)
- `recorder/record_controller.py:cmd_record()`: Launch recording overlay, capture interactions
- `mapper/execute_controller.py:cmd_execute()`: Load skill, run replay, save logs
- `api/server.py`: FastAPI app with execute endpoint

**Configuration:**
- `config.yaml`: YAML file with sections for detection, execution, models, paths, litellm
- `core/config.py`: YAML loader with defaults (called via get_config() everywhere)
- `pyproject.toml`: Package metadata, optional dependency groups ([detection], [vlm], [all])

**Core Logic:**
- `core/locate.py`: Element location cascade (5-stage strategy)
- `core/executor.py`: Mouse/keyboard control with thread safety
- `core/capture.py`: Screenshot capture, pixel diffing, snippet I/O
- `mapper/graph.py`: OCSDGraph (NetworkX wrapper + schema enforcement)
- `mapper/runner.py`: Step-by-step replay with locate + validate loop
- `mapper/validator.py`: Post-action verification (pixel diff + VLM)

**Testing:**
- `tests/conftest.py`: Pre-mocked torch, faiss, ultralytics, transformers
- `tests/test_cascade.py`: Locate cascade unit tests
- `tests/test_e2e_login.py`: Integration test with local HTML form
- `.venv/`: Python virtual environment (excluded from git)

**Generated/Runtime:**
- `skills/`: Skill JSON files created by user during recording
- `.env`: Environment variables (secrets, paths — never committed)
- `__pycache__/`: Python bytecode (excluded from git)

## Naming Conventions

**Files:**
- Module files: `snake_case.py` (e.g., `record_controller.py`, `overlay_view.py`)
- Entry points: `main.py`, `cmd_*.py` functions
- Test files: `test_*.py` (pytest discovery pattern)

**Directories:**
- Package names: `lowercase` (e.g., `core`, `mapper`, `recorder`, `hub`)
- Feature subdirectories: `lowercase` (e.g., `skills/`, `snippets/`)

**Functions:**
- Public functions: `snake_case` (e.g., `locate_element()`, `execute_node()`)
- Private/internal: `_snake_case` (e.g., `_resolve_position_hint()`, `_hsleep()`)
- Event handlers: `on_*` (e.g., `on_element_clicked()`, `on_mode_changed()`)
- Command functions: `cmd_*` (e.g., `cmd_record()`, `cmd_execute()`, `cmd_compose()`)

**Classes:**
- Public classes: `PascalCase` (e.g., `OCSDGraph`, `OverlayController`, `LocateResult`)
- Data classes: `PascalCase` (e.g., `SkillManifest`, `CandidateElement`)
- Enums: `PascalCase` (e.g., `ElementType`, `RunnerEventType`)

**Variables/Constants:**
- Module-level constants: `UPPERCASE_WITH_UNDERSCORES` (e.g., `_VALID_LAYERS`, `_ADJACENT_KEYS`)
- Instance variables: `snake_case` (e.g., `self.graph`, `self.candidates`)

## Where to Add New Code

**New Vision Detection Strategy (e.g., new cascade stage):**
- Primary code: `core/locate.py` — add stage as new `if` block in cascade
- Config: Add thresholds to `core/config.py` defaults and `config.yaml` template
- Tests: `tests/test_cascade.py` — add test case for new stage
- Example: New "semantic search" stage would go after CLIP, before OCR

**New UI Element Type:**
- Enum definition: `recorder/element_types.py` (add to ElementType)
- Detection mapping: `mapper/runner.py:_action_type_for_node()` (add mapping)
- Dialog: `recorder/dialog.py` (add type to choices if needed)
- Tests: `tests/test_export.py` (verify serialization)

**New Recording Feature (e.g., voice input for labels):**
- Orchestration: `recorder/record_controller.py` (integrate into main flow)
- UI: `recorder/overlay_view.py` or new `recorder/voice_dialog.py`
- Config: Add config section to `config.yaml`
- Tests: `tests/test_*.py` (new test file for feature)

**New Recovery Strategy (e.g., retry with different cascade):**
- Main logic: `mapper/orchestrator.py` (add to recovery LLM path)
- Validation: `mapper/validator.py` (update decision tree)
- Tests: `tests/test_milestone.py` (integration test)

**New API Endpoint:**
- Route: `api/server.py` (add @app.post("/new-endpoint"))
- Controller: Extract logic to `mapper/execute_controller.py` if possible
- Tests: `tests/` (add integration test)

## Special Directories

**`skills/` (Skill Storage):**
- Purpose: Contains recorded skill JSON files and associated artifacts
- Generated: Yes (user creates via `--record`)
- Committed: No (user-specific, large)
- Structure: One JSON per skill, optional `skill_name/snippets/` and `skill_name/embeddings/` subdirs

**`tests/` (Test Suite):**
- Purpose: Unit + integration tests for all modules
- Generated: No (committed to repo)
- Committed: Yes
- Pattern: Mocked heavy dependencies (torch, faiss, ultralytics) in conftest.py for fast CI/CD

**`.venv/` (Virtual Environment):**
- Purpose: Isolated Python environment with all dependencies
- Generated: Yes (`python -m venv .venv`)
- Committed: No (created per-machine)
- Usage: Activate via `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)

**`.env` (Environment Variables):**
- Purpose: Secrets and system-specific paths
- Generated: Yes (user creates)
- Committed: No (secrets — listed in .gitignore)
- Contents: TESSERACT_CMD, LITELLM_BASE_URL, LITELLM_API_KEY (never shown in logs)

**`__pycache__/` (Python Bytecode):**
- Purpose: Cached compiled Python code
- Generated: Yes (Python runtime)
- Committed: No (regenerated every run)

---

*Structure analysis: 2025-03-16*
