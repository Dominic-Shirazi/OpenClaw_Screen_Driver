# Project: OCSD (OpenClaw Screen Driver)

## STATUS 2026-09-08: PROJECT LIKELY DEAD — READ THIS FIRST

Verdict after a repo audit on head_pc (Sep 8) and a market check the same day:

- **The token-cost pitch is gone.** OpenAI Codex Record & Replay (Jun 18 2026, macOS) and Anthropic's record-a-skill feature deliver "show it once, it repeats" to consumers on a $20/mo plan. Under the hood theirs keep the model in the loop every step; OCSD's zero-token deterministic replay is technically different but the buyer cannot tell.
- **The determinism pitch was never proven.** All 6 replay logs in logs/replays failed; every locate fell through to OCR. The CLIP/FAISS/Florence cascade never re-found an element end to end.
- **Cost math for a single user is trivial.** One 20-step process, 22 runs/month, costs ~$20 on ChatGPT Plus or ~$15-45 on the raw API. Nobody builds this to save that.
- **Remaining niche:** local-only models that cannot drive a GUI, screens that cannot leave the building, or fleet-scale volume. The founder knows nobody in those groups. UiPath already owns the enterprise version.
- **Original motivation was dated:** built when computer use was new and expensive, for a Mac Mini fleet sold to OpenClaw users that may no longer exist.

If someone revives this: the only design worth building is in the Sep 8 audit — window/accessibility anchor, multi-scale OpenCV template match on a context crop with expanding search rings and early exit, scoped OCR, one small local grounding model, then halt and ask. Never blind-click. Drop CLIP/FAISS/Florence from replay; OmniParser proposes boxes at record time only. Prove it on tests/fixtures/login.html (5 steps, resize browser, replay 3x) before touching anything else.

Code state: `main` has the Apr 8 bug-fix marathon and QWidget overlay; `feat/remaining-v1-tasks` adds the Apr 9 context-box, AI-text, and locate-timeout work and was never merged.

## Current Plan Summary
Screen automation tool that records user interactions as "routines" and replays them using computer vision. PyQt6 overlay for recording HUD, Ollama VLM for element detection, human-like mouse/keyboard input via human_mouse_moves + human_typing libraries. 6 of 11 UI/UX issues fixed (2026-04-07). Remaining: tag dialog draggable (#6A), VLM prompt rewrite (#6C), loading feedback (#7), API mark_waiting (#8), CLI dedup (#9), shimmer polish (#10), startup animation (#11).

## File Tree + Agent Registry

### Root (3 files)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| main.py | root-main | Backward-compat entry point redirecting to cli.app:main | main() → pyproject.toml scripts | cli.app.main ← cli/app.py |
| config.yaml | root-config | Default YAML config: hardware, models, detection, execution, overlay, API, hub | Config values → core/config.py | (none) |
| pyproject.toml | root-pyproject | Package metadata, deps, optional groups, scripts, tooling | `ocsd` console_scripts, dep groups → pip | (none) |

### cli/ (6 files) — Group: `cli`
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| cli/app.py | cli_app | Typer CLI with subcommands: record, run, list, inspect, update, fork, delete, serve | app, main() → pyproject.toml entry point | cli/output, cli/tui, cli/_minimize, recorder/*, routine/*, core/*, api/* |
| cli/tui.py | cli_tui | Rich TUI: loading screen, arrow-key menu, routine browser, run dispatch | run_tui(), show_loading_screen() → cli/app.py | cli/_keys, cli/_minimize, cli/output, recorder/*, routine/*, core/* |
| cli/output.py | cli_output | Shared CLI helpers: error panels, routine resolution, param parsing, variable collection | show_error(), resolve_routine_path(), prepare_run() → cli/app.py, cli/tui.py | routine/discovery, routine/format |
| cli/_keys.py | cli_keys | Cross-platform single-keypress reader for TUI navigation | read_key() → cli/tui.py | msvcrt (Win), termios (Unix) |
| cli/_minimize.py | cli_minimize | Win32 terminal minimize/restore via ctypes | minimize_terminal(), restore_terminal() → cli/app.py, cli/tui.py | ctypes (Win only) |
| cli/__init__.py | cli_init | Empty package init | (none) | (none) |

### core/ (17 files) — Groups: `core-exec`, `core-detection`, `core-ai`

#### core-exec (execution infrastructure)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| core/executor.py | executor | Human-like mouse/keyboard/scroll via human_mouse_moves + human_typing | click, type_text, drag, scroll, hotkey, prompt_user_blocking → routine/runner.py, recorder/* | core/config, human_mouse_moves (ext), human_typing (bundled), pyautogui |
| core/capture.py | capture | Screenshot capture (full/region), pixel diff, snippet save/load, window title | screenshot_full, screenshot_region, get_window_title → core/*, recorder/*, routine/* | core/config, mss, cv2, pywinauto (Win) |
| core/types.py | types | Shared dataclasses (Point, Rect, LocateResult, etc.) and domain exceptions | Point, Rect, LocateResult, ElementNotFoundError → all modules | (stdlib only) |
| core/config.py | config | YAML config loader with deep-merge defaults + .env auto-loading | get_config, reload_config → nearly all modules | pyyaml, python-dotenv |
| core/gpu.py | gpu | GPU detection and device assignment for torch models | get_device → core/vision.py, model loaders | core/config, torch (optional) |
| core/__init__.py | core-init | Package marker (docstring only) | (none) | (none) |

#### core-detection (vision/locate pipeline)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| core/vision.py | vision | VLM inference via LiteLLM proxy for element analysis and action confirmation | analyze_crop_array, confirm_action, first_pass_map_array → core/locate, routine/runner, recorder/record_session | core/config, core/types, openai, cv2 |
| core/locate.py | locate | 5-stage element location cascade (OmniParser, CLIP, OCR, VLM, position fallback) | locate_element, locate_element_from_step → routine/runner.py | core/capture, core/config, core/ocr, core/detection, core/vision, core/embeddings |
| core/detection.py | detection | Detection provider protocol + singleton factory (currently OmniParser) | DetectionProvider, get_detector → core/locate.py | core/config, core/omniparser |
| core/ocr.py | ocr | Tesseract OCR: text extraction, bbox detection, text search | find_text_on_screen → core/locate.py (Stage 3) | pytesseract, core/capture, core/types |

#### core-ai (AI models + monitoring)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| core/embeddings.py | embeddings | CLIP embedding generation + FAISS vector index for similarity search | generate_embedding, search_index → core/omniparser, core/locate | core/config, transformers, faiss-cpu, torch |
| core/model_cache.py | model-cache | HuggingFace model weight download + caching | ensure_model → core/florence, core/omniparser | huggingface_hub, core/config |
| core/florence.py | florence | Florence-2 model for UI element captioning | caption_crop, caption_batch, describe_element → recorder/smart_detect | core/model_cache, core/config, transformers, torch |
| core/omniparser.py | omniparser | OmniParser YOLOv8 for UI element detection + CLIP matching | OmniParserProvider.detect, detect_and_match → core/detection, core/locate | core/model_cache, core/embeddings, core/config, ultralytics, torch |
| core/conditions.py | conditions | Condition polling (timer, element, screen change, VLM, text, iteration) for wait/loop | ConditionChecker → routine/runner.py | core/locate, core/vision, core/ocr, core/capture |
| core/watcher.py | watcher | Daemon thread polling window title, URL, pixel-diff changes | start_watching, WatcherCallback → recorder/record_session | core/capture, core/config, core/types |
| core/accessibility.py | accessibility | Windows UIA tree via pywinauto for element discovery | get_element_tree, tab_walk → core/locate (fallback) | pywinauto (Win only), core/types |

### recorder/ (44 files) — Groups: `record-session`, `record-controller`, `overlay-core`, `overlay-visual-layers`, `overlay-panels`, `overlay-widgets`, `overlay-infra`, `overlay-tag-dialog`, `recorder-dialogs`, `recorder-utils`, `recorder-legacy`

#### Solo: record-session
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/record_session.py | record-session | State-machine orchestrating full recording pipeline: click→detect→VLM→tag→save | RecordSession → recorder/record_controller.py | PipelineBridge, RecordPhase, OverlayState, core/config, core/capture, core/detection, core/vision, core/executor, mapper/graph, routine/format, core/embeddings |

#### Solo: record-controller
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/record_controller.py | record-controller | Bridges CLI/TUI with overlay + recording session for element recording flow | cmd_record(args) → cli/app.py | core/config, core/capture, core/detection, core/vision, core/embeddings, core/florence, recorder/smart_detect, recorder/dialog, recorder/refine_dialog, recorder/overlay/controller, mapper/graph, mapper/export |

#### overlay-core (view/controller pair)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/overlay/view.py | overlay-view | QGraphicsView fullscreen transparent overlay managing all visual layers | OverlayView → recorder/overlay/controller.py | All overlay layers (shimmer, scan, bbox, panels, etc.), state.py, platform_*.py |
| recorder/overlay/controller.py | overlay-controller | Public API facade + state machine for overlay subsystem | OverlayController → recorder/record_session.py, record_controller.py, routine/runner.py | overlay/view.py, overlay/state.py, recorder/hotkeys.py |
| recorder/overlay/state.py | overlay-state | OverlayState enum (READY/RECORDING/PAUSED/REPLAYING), transition table, color map | OverlayState, STATE_COLORS, transition() → view.py, controller.py, shimmer_layer.py | (stdlib enum only) |
| recorder/overlay/__init__.py | overlay-init | Package init re-exporting key overlay types | AnimationClock, OverlayController, OverlayState, etc. → recorder/record_session.py, external | All overlay submodules |

#### Solo: overlay-tag-dialog
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/overlay/tag_dialog_panel.py | overlay-tag-dialog | Frosted-glass tag dialog with form fields, typewriter VLM fill, card glow | TagDialogPanel, signals confirmed/dismissed → overlay/controller.py | animation_clock, card_glow, typewriter_engine, hud_common |

#### overlay-visual-layers
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/overlay/scan_layer.py | overlay-scan | Multi-phase cinematic scan animation during AI element detection | ScanLayer → overlay/controller.py | PyQt6 only (no internal imports) |
| recorder/overlay/bbox_layer.py | overlay-bbox | Bounding box with 8 resize handles, label, morph animation | BboxLayer → overlay/controller.py | PyQt6 only |
| recorder/overlay/shimmer_layer.py | overlay-shimmer | Ocean-wave edge glow using OpenSimplex noise + mouse retreat | ShimmerLayer → overlay/controller.py, view.py | opensimplex, overlay/state.py, ctypes (Win32 cursor) |
| recorder/overlay/card_glow.py | overlay-card-glow | Continuous ambient underglow around card perimeters | paint_card_glow() → toolbar_panel.py, tag_dialog_panel.py | hud_common.py |

#### overlay-panels
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/overlay/toolbar_panel.py | toolbar-panel | Draggable pill toolbar with context-sensitive button sets across 7 modes | ToolbarMode, ToolbarPanel, button_clicked signal → overlay/controller.py | animation_clock, card_glow, hud_common |
| recorder/overlay/abort_panel.py | abort-panel | Centered abort confirmation dialog (Discard/Keep Recording) | AbortPanel, discard_clicked/keep_clicked → overlay/controller.py | card_glow, hud_common |
| recorder/overlay/mini_dialogs.py | mini-dialogs | Wait/Prompt/Loop dialogs for configuring special recording steps | WaitDialog, PromptDialog, LoopDialog → overlay/controller.py | animation_clock, card_glow, hud_common |

#### overlay-widgets
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/overlay/countdown_widget.py | countdown-widget | Cursor-following 3-2-1 countdown spinner | CountdownWidget, countdown_finished signal → controller.py | animation_clock, card_glow, hud_common |
| recorder/overlay/donut_cloud_layer.py | donut-cloud-layer | Gaussian heat-map blob with raindrop ripple for click confidence | DonutCloudLayer → overlay controller | PyQt6 only |
| recorder/overlay/typewriter_engine.py | typewriter-engine | Per-field character insertion engine for form filling | TypewriterEngine → tag_dialog_panel.py | animation_clock, hud_common |
| recorder/overlay/hud_common.py | hud-common | Shared HUD design tokens: colors, spacing, fonts, z-values, timing | All constants → all overlay panels/widgets | PyQt6 (QColor) |
| recorder/overlay/status_badge.py | status-badge | Top-center pill badge showing replay step progress | StatusBadge → routine/runner.py / replay controller | animation_clock |

#### overlay-infra (12 utility files)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/overlay/animation_clock.py | animation-clock | Shared 16ms QTimer delivering frame-independent dt to all animated layers | AnimationClock → all animated layers | PyQt6 |
| recorder/overlay/border_layer.py | border-layer | Four colored rectangles around screen perimeter for state indication | BorderLayer → view.py | PyQt6 |
| recorder/overlay/camera_flash.py | camera-flash | White 15px border flash (200ms fade) for screenshot capture feedback | CameraFlash → view.py | animation_clock |
| recorder/overlay/capture_guard.py | capture-guard | Context manager: hide overlay → flush compositor → screenshot → restore | capture_guard() → record_session.py, executor | platform_win32 (dwm_flush), core/config |
| recorder/overlay/click_catcher_layer.py | click-catcher | Full-screen invisible rect intercepting mouse in recording mode | ClickCatcherLayer → view.py | PyQt6 |
| recorder/overlay/dpi.py | dpi-utils | Qt logical ↔ mss physical pixel coordinate conversion | logical_to_physical(), physical_to_logical() → coordinate exchange boundary | PyQt6 |
| recorder/overlay/mode_indicator_layer.py | mode-indicator | HUD label showing current state (READY/RECORDING/PAUSED) + hotkey hints | ModeIndicatorLayer → view.py | overlay/state.py |
| recorder/overlay/pipeline_bridge.py | pipeline-bridge | QObject with pyqtSignals for thread-safe background→main-thread delivery | PipelineBridge → record_session.py, record_controller.py | PyQt6 |
| recorder/overlay/platform_linux.py | platform-linux | Linux: force XCB, toggle click-through via Qt attributes | ensure_xcb_platform(), set_click_through_linux() → view.py | PyQt6 |
| recorder/overlay/platform_win32.py | platform-win32 | Win32: layered window, click-through via WS_EX_TRANSPARENT, DwmFlush | setup_win32_layered(), set_click_through_win32(), dwm_flush() → view.py, capture_guard | ctypes |
| recorder/overlay/record_phase.py | record-phase | Enum: 15 recording sub-states (AWAITING_CLICK → LOOP_DEFINING) | RecordPhase → record_session.py, record_controller.py | (stdlib enum) |
| recorder/overlay/target_highlight.py | target-highlight | Purple glow border around located element during replay (300ms fade) | TargetHighlight → view.py | animation_clock |

#### recorder-dialogs
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/dialog.py | tag-dialog | Modal PyQt6 dialog for tagging UI elements (type, label, layer, notes) | TagDialog → record_session.py, record_controller.py | recorder/element_types, PyQt6 |
| recorder/refine_dialog.py | refine-dialog | Side-by-side bbox comparison: user crop vs YOLOE-refined suggestion | RefineDialog → record_session.py | core/types, PyQt6 |
| recorder/step_ui.py | step-ui | TUI step-through prompt for --step replay mode | step_through_prompt() → routine/runner.py | rich (optional) |

#### recorder-utils (8 supporting files)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/record_flow.py | record-flow | Entry point: TUI name prompt → QApp setup → minimize → overlay+session → event loop | cmd_record() → cli/app.py | overlay/controller, record_session, platform_utils, core/detection |
| recorder/hotkeys.py | hotkeys | Platform-aware global hotkey listeners (Win32 polling / pynput) | _create_hotkey_listener() → overlay/controller.py | PyQt6 (QTimer), ctypes (Win), pynput (macOS/Linux) |
| recorder/smart_detect.py | smart-detect | 3-stage UI detection: OmniParser → Florence-2 caption → OCR fallback | detect_ui_elements_async() → record_session.py | core/detection, core/florence, pytesseract |
| recorder/tui.py | tui | Rich interactive menu for mode selection (record/diagram/execute/compose) | launch_menu() → cli/app.py | rich |
| recorder/session.py | session | Recording session state: element list, undo history, mode tracking | RecordingSession → record_flow.py, record_session.py | (stdlib only) |
| recorder/element_types.py | element-types | ElementType enum (34 members: Interactive/Structural/Static/Meta) | ElementType → recorder/*, core/vision.py | (stdlib enum) |
| recorder/platform_utils.py | platform-utils | Win32 minimize_all_windows via keybd_event simulation | minimize_all_windows() → record_flow.py | ctypes (Win) |
| recorder/__init__.py | recorder-init | Empty package init | (none) | (none) |

#### recorder-legacy (pre-refactor overlay)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| recorder/_overlay_legacy.py | overlay-legacy-controller | Legacy overlay controller: mode switching, candidate rendering, click capture | OverlayMode, OverlayController → overlay_view.py | recorder/hotkeys |
| recorder/overlay_items.py | overlay-items | Legacy Qt graphics: interactive bboxes with draggable handles, donut viz | _ElementBoxGroup → overlay_view.py | PyQt6 only |
| recorder/overlay_view.py | overlay-view-legacy | Legacy fullscreen overlay: mouse events, rubber-band, candidate rendering | _OverlayView → _overlay_legacy.py | overlay_items, recorder/overlay (rewired) |

### routine/ (11 files) — Groups: `routine-runner`, `routine-support`

#### Solo: routine-runner
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| routine/runner.py | routine-runner | Step-sequential replay engine with 5-stage recovery cascade | run_routine(), RunResult, RunEvent, preflight_check() → cli/app.py, api/server | core/capture, core/config, core/executor, core/locate, core/types, routine/format, routine/run_log |

#### routine-support (10 files)
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| routine/format.py | routine-format | Routine dataclass with ordered JSON serialization, save/load, v1 step builder | Routine, build_v1_step → recorder/record_session, routine/*, cli/* | mapper/graph, routine/checksum |
| routine/discovery.py | routine-discovery | Scans ~/.ocsd/routines/ for routine.json directories | RoutineInfo, list_routines → cli/app.py, routine/management | routine/migration |
| routine/migration.py | routine-migration | Schema version detection + v0→v1 upgrade | detect_schema_version, upgrade_v0_to_v1 → routine/discovery, cli | routine/checksum, routine/format (lazy) |
| routine/version.py | routine-version | Semantic version bump helpers | bump_minor, bump_patch → routine/update_session | (standalone) |
| routine/checksum.py | routine-checksum | SHA256 checksum over steps+graph for integrity | calculate_routine_checksum → routine/format, routine/migration | (standalone) |
| routine/update_session.py | routine-update-session | Guided-replay walkthrough: keep/edit/delete/fork per step | UpdateSession → cli/app.py, TUI | mapper/graph, routine/format, routine/management, routine/version |
| routine/run_log.py | routine-run-log | Per-run dirs, annotated screenshots, result JSON, log pruning | create_run_dir, save_run_result → routine/runner.py | core/config, cv2 |
| routine/management.py | routine-management | Routine CRUD: fork (with truncation), delete, inspect | fork_routine, delete_routine, inspect_routine → cli/app.py, routine/update_session | routine/format, routine/discovery, mapper/graph |
| routine/replay_overlay.py | routine-replay-overlay | Thread-safe adapter: RunEvent callbacks → OverlayController Qt signals | ReplayOverlayAdapter → cli/app.py, TUI | routine/runner (RunEvent), core/config, PyQt6 |
| routine/__init__.py | routine-init | Re-exports Routine, VALID_CATEGORIES, calculate_routine_checksum | public API → importers | routine/checksum, routine/format |

### mapper/ (10 files) — Groups: `mapper-core`, `mapper-support`

#### mapper-core
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| mapper/graph.py | graph | NetworkX directed graph: node/edge CRUD, execution tracking, serialization | OCSDGraph → mapper/*, core/locate, recorder/record_session, routine/format | networkx, recorder/element_types |
| mapper/runner.py | runner | Replay executor: locate → act → validate per node with event callbacks | execute_node, run_skill, run_path → mapper/orchestrator, cli/app.py | mapper/graph, mapper/pathfinder, mapper/validator, core/locate, core/executor, core/capture |
| mapper/orchestrator.py | orchestrator | Top-level replay with preflight fingerprint validation + VLM recovery loops | orchestrate_skill, preflight_check, diagnose_failure → cli/app.py | mapper/graph, mapper/pathfinder, mapper/runner, mapper/validator, core/* |

#### mapper-support
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| mapper/pathfinder.py | pathfinder | Weighted shortest-path + execution plan generation over OCSDGraph | find_path, get_execution_plan → mapper/runner, mapper/orchestrator | mapper/graph, core/types |
| mapper/execute_controller.py | execute-controller | CLI entry: load skill JSON → resolve nodes → dispatch replay → save log | cmd_execute(args) → cli/app.py | mapper/export, mapper/runner, mapper/orchestrator |
| mapper/validator.py | validator | Post/pre-action verification via pixel-diff + optional VLM confirmation | validate_action, validate_destination → mapper/runner, mapper/orchestrator | core/capture, core/vision, core/config |
| mapper/diff.py | diff | Structured diff between two OCSDGraph versions | GraphDiff, diff_graphs → skill versioning | mapper/graph |
| mapper/export.py | export | Serialize OCSDGraph to/from JSON skill files with SHA256 checksums | export_skill, load_skill_from_file → mapper/execute_controller, cli | mapper/graph |
| mapper/layers.py | layers | Classify elements into OS_UI / APP_PERSISTENT / PAGE_SPECIFIC layers | Layer enum, classify_layer → mapper/graph | (stdlib only) |
| mapper/__init__.py | mapper-init | Package marker | (none) | (none) |

### api/ (4 files) — Group: `api`
| File | Agent Callsign | Purpose | Provides → To | Consumes ← From |
|---|---|---|---|---|
| api/server.py | api-server | FastAPI REST: health, routines, run, respond, screenshot, abort, MCP mount | app (FastAPI) → uvicorn/lifecycle.py | api/run_manager, core/config, routine/discovery, routine/format, routine/runner |
| api/run_manager.py | run-manager | Thread-safe run lifecycle state machine (single-active-run enforcement) | RunManager, RunStatus → api/server.py | (stdlib only) |
| api/lifecycle.py | api-lifecycle | Start FastAPI as daemon thread (signal-handler suppression for Qt coexistence) | start_api_daemon() → cli/app.py | uvicorn, core/config |
| api/__init__.py | api-init | Empty package init | (none) | (none) |

## Recent Decisions Log
| Date | Decision | Reason | Files Affected |
|---|---|---|---|
| 2026-04-03 | Use human_mouse_moves lib for mouse movement | User built proper kinematic sub-movement library with Fitts's Law | core/executor.py |
| 2026-04-03 | Use human_typing lib for keyboard input | Bundled library with burst cadence, QWERTY neighbor typos, fatigue | core/executor.py |
| 2026-04-03 | Shimmer uses OpenSimplex noise | Organic wave shapes, multiple octaves for realistic ocean swell | recorder/overlay/shimmer_layer.py |
| 2026-04-03 | Mouse retreat uses temporal smoothing (lerp) | Raw cursor polling caused waves to jump | recorder/overlay/shimmer_layer.py |
| 2026-04-03 | Full Bézier move under single lock | Two threads interleaving caused mouse bouncing between paths | core/executor.py |
| 2026-04-07 | All decorative overlay items setAcceptedMouseButtons(NoButton) | QGraphicsObject defaults to accepting all mouse buttons, swallowing clicks | 8 overlay layer files |
| 2026-04-07 | DwmFlush + processEvents after hide_for_capture | Compositor needs explicit flush to guarantee overlay is hidden before clicks | recorder/record_session.py |
| 2026-04-07 | Remap action_type → action in on_tag_confirmed | get_form_data() uses "action_type" key, executor reads "action" key | recorder/record_session.py |
| 2026-04-07 | Disconnect countdown_finished before reconnecting | Qt signals accumulate connections, causing N actions after N retries | recorder/record_session.py |

## Notable Concerns Surfaced During Onboarding

### Bugs (FIXED 2026-04-07)
- ~~**mapper/runner.py**: `execute_node()` references `node_data` not in scope~~ ✓
- ~~**mapper/runner.py**: `run_skill()` accepts `execution_params` but never passes to `_resolve_input_text()`~~ ✓
- ~~**core/conditions.py**: `_check_element_appears` VLM threshold never reached~~ ✓
- **mapper/runner.py**: `run_path()` doesn't forward `execution_params` — future gap for graph navigation
- **recorder/record_session.py**: `action_type` → `action` key remap needed (fixed, but indicates VLM/form field naming inconsistency)

### Cross-Platform Gaps
- **core/gpu.py**: No MPS (Apple Silicon) support despite macOS being primary audience
- **recorder/overlay/shimmer_layer.py**: Mouse retreat Win32-only — returns (-1000,-1000) on macOS/Linux
- **recorder/platform_utils.py**: minimize_all_windows() only Win32 — no macOS/Linux fallback
- **Several files**: Hardcoded "Segoe UI" font (Windows-specific)

### Architecture Debt
- **cli/app.py + cli/tui.py**: Qt app + overlay + thread wiring duplicated verbatim — needs shared helper
- **core/locate.py**: `locate_element` and `locate_element_from_step` duplicate ~200 lines of cascade logic
- **recorder/overlay/mini_dialogs.py**: WaitDialog/PromptDialog/LoopDialog have nearly identical paint()/boundingRect()/_tick() — needs base class
- **recorder-legacy**: 3 files (_overlay_legacy.py, overlay_items.py, overlay_view.py) are partially rewired to new overlay — may be dead code
- **api/run_manager.py**: `mark_waiting` defined but never called from server callback — /respond endpoint guard unreachable

### Thread Safety
- **core/config.py**: `_config_cache` has no lock — concurrent first-call race
- **core/embeddings.py**: Global mutable state with no thread lock
- **core/executor.py**: `human_mouse_moves` sys.path hardcoded relative — fragile in Docker/restructure
