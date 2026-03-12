# OCSD Build Progress — Stage 1

Last updated: 2026-03-11

## Wave 0: Scaffolding
| Module | File | Status | Author | Notes |
|--------|------|--------|--------|-------|
| Types | `core/types.py` | DONE | Claude | Point, Rect, ReplayLog, exceptions |
| Config | `core/config.py` | DONE | Claude | YAML + .env via python-dotenv |
| Element types | `recorder/element_types.py` | DONE | Claude | ElementType enum (14 types) |
| __init__.py files | all packages | DONE | Claude | Package scaffolding |
| pyproject.toml | `pyproject.toml` | DONE | Claude | Dependency groups |

## Wave 1: Leaf Modules (no internal deps)
| Module | File | Status | Author | Notes |
|--------|------|--------|--------|-------|
| OCR | `core/ocr.py` | DONE | Claude | Tesseract wrapper, find_text_on_screen |
| Capture | `core/capture.py` | DONE | Gemini | screenshot_full, pixel_diff, save_snippet |
| Executor | `core/executor.py` | DONE | Claude | PyAutoGUI with anti-bot jitter, thread-safe |
| Accessibility | `core/accessibility.py` | DONE | Claude | pywinauto UIA tree (Windows only) |
| Embeddings | `core/embeddings.py` | DONE | Gemini | CLIP + FAISS index |

## Wave 2: Core Intelligence
| Module | File | Status | Author | Notes |
|--------|------|--------|--------|-------|
| Vision (VLM) | `core/vision.py` | DONE | Claude | LiteLLM proxy → vision group (was Ollama) |
| Watcher | `core/watcher.py` | DONE | Claude | Daemon thread polling |
| Graph | `mapper/graph.py` | DONE | Claude | NetworkX DiGraph wrapper, CRUD, stats |

## Wave 3: Graph Consumers
| Module | File | Status | Author | Notes |
|--------|------|--------|--------|-------|
| Dialog | `recorder/dialog.py` | DONE | Gemini+Claude | PyQt6 tag dialog (Claude fixed indent) |
| Pathfinder | `executor/pathfinder.py` | DONE | Claude | Weighted BFS/Dijkstra, fuzzy label search |
| Validator | `executor/validator.py` | DONE | Claude | Post-action validation (pixel diff + VLM) |
| Export | `mapper/export.py` | DONE | Gemini | Skill JSON + SHA256 checksum |

## Wave 4: Overlay
| Module | File | Status | Author | Notes |
|--------|------|--------|--------|-------|
| Overlay | `recorder/overlay.py` | DONE | Claude | PyQt6 transparent overlay, Win32 flags |

## Wave 5: Orchestration
| Module | File | Status | Author | Notes |
|--------|------|--------|--------|-------|
| Runner | `executor/runner.py` | DONE | Claude | 4-step fallback cascade, event system |
| Main | `main.py` | DONE | Claude | argparse, DPI awareness, Ctrl+C |

## API Configuration
| Setting | Source | Value |
|---------|--------|-------|
| LiteLLM base URL | `.env` | `http://192.168.0.2:4000/v1` |
| VLM model group | `.env` | `vision` (Qwen3 VL → Gemini → local) |
| Quick model | `.env` | `quick` (Gemini Flash Lite) |
| Planning model | `.env` | `planning` (DeepSeek → Cerebras → local) |
| Coding model | `.env` | `coding` (Qwen3 → Kimi K2 → Gemini → local) |

## Tests
| Test File | Count | Status | Notes |
|-----------|-------|--------|-------|
| `tests/test_wave2.py` | 30 | PASS | JSON parsing, watcher, graph |
| `tests/test_pathfinder.py` | 16 | PASS | BFS, weighted, fuzzy search |
| `tests/test_export.py` | 4 | PASS | Round-trip, checksum, persistence |
| `tests/test_milestone.py` | 7 | PASS | Full pipeline E2E, dry-run, serialization |

**Total tests: 57/57 passing**

## Milestone: Record → Execute → Verify
| Step | Status | Notes |
|------|--------|-------|
| Local HTML login form | DONE | `tests/fixtures/login.html` |
| Build skill graph (programmatic) | DONE | 4 nodes, 3 edges |
| Export → save → load → import | DONE | Checksum verified |
| Pathfind entry → exit | DONE | 4-step linear path |
| Dry-run execute | DONE | `ReplayLog.overall_success == True` |
| `--record` overlay (live) | TODO | Requires manual testing |
| `--execute` replay (live) | TODO | Requires LiteLLM + Tesseract |

## Commits
| Hash | Message |
|------|---------|
| `c7a095b` | feat: Initialize project structure |
| `bded81a` | feat: Wave 0 scaffolding |
| `fb71e52` | feat: Wave 1 leaf modules + venv |
| `f289688` | docs: add Gemini CLI rules |
| `e98ebd3` | feat: Waves 2+3 — vision, watcher, graph, dialog, pathfinder, validator, export |
| `9cb606f` | feat: Waves 4+5 — overlay, runner, main |
| `016f67d` | test(milestone): Stage 1 E2E pipeline verification |
| `dad4253` | feat(api): migrate from Ollama to LiteLLM proxy + .env config |
