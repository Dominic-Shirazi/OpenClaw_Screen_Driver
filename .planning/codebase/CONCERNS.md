# Codebase Concerns

**Analysis Date:** 2026-03-16

## Tech Debt

**Exposed Pixel Diff Metric:**
- Issue: `mapper/runner.py` line 227 hardcodes `pixel_diff_pct=0.0` in `ReplayStep` with a TODO comment noting the validator's raw diff should be exposed
- Files: `mapper/runner.py:227`, `mapper/validator.py`
- Impact: Replay logs always report 0% pixel diff even when actions succeed via visual change; breaks debugging and analytics of which methods reliably detect changes
- Fix approach: Thread the `pixel_diff` value from `validate_action()` through `validator.py` into `ReplayStep.pixel_diff_pct`, removing the hardcoded zero

**Broad Exception Catching in Watcher:**
- Issue: `core/watcher.py` uses multiple bare `except Exception as e:` blocks (lines 69, 74, 97, 155, 157, 171, 189, 196) during polling and callback execution
- Files: `core/watcher.py:69-80`, `core/watcher.py:88-99`, `core/watcher.py:155-198`
- Impact: Subtle bugs in title/URL/diff detection swallowed silently; callback exceptions logged but not propagated, can hide state mutation errors
- Fix approach: Catch specific exception types (AttributeError, ctypes exceptions for Win32, TimeoutError); let unexpected errors propagate for visibility

**Global State in Core Modules:**
- Issue: `core/embeddings.py`, `core/detection.py`, and `core/watcher.py` use module-level globals (`_clip_model`, `_detector_instance`, `_watcher_thread`) with lazy initialization and locking
- Files: `core/embeddings.py:11-16`, `core/detection.py:20-21`, `core/config.py:21`
- Impact: Thread-safe but brittle; state persists across function calls making tests difficult to isolate; no way to reset state between test runs without restart
- Fix approach: Migrate to a singleton factory pattern or context manager; provide `reset()` function for test teardown

**VLM Unavailability Silent Fallback:**
- Issue: `mapper/validator.py:90-99` catches all exceptions from `confirm_action()` and silently falls back to pixel-diff-only validation
- Files: `mapper/validator.py:90-99`
- Impact: If LiteLLM proxy is down, execution continues without warning that VLM is unavailable; user unaware validation is degraded
- Fix approach: Log a warning with severity level ERROR; optionally abort execution if `vlm_confirm=true` in config but VLM unavailable

**Get Active URL Best-Effort Returns None Silently:**
- Issue: `core/capture.py:145-179` and `core/watcher.py:34-45` silently catch all exceptions from Windows UIA calls and return None, making failures undiagnosable
- Files: `core/capture.py:176-178`, `core/watcher.py:44`
- Impact: URL detection fails silently on browser/OS incompatibilities; no logged reason why URL is None
- Fix approach: Log at debug level which exception occurred (pywinauto version, browser, language mismatch); document known limitations

## Known Bugs

**OCR Tesseract Binary Not Found Silent Degradation:**
- Symptoms: Element location falls through to position fallback without user awareness that OCR stage failed; accuracy drops silently
- Files: `core/ocr.py:57-79`, `core/locate.py:158-177`
- Trigger: User hasn't installed Tesseract via `winget install UB-Mannheim.TesseractOCR` on Windows, or binary not on PATH on macOS/Linux
- Workaround: `core/ocr.py` has helpful error message but only raised if OCR functions explicitly called; Stage 3 in `locate.py` catches ImportError and silently skips

**Accessibility Module pywinauto Import Failure Returns Empty List:**
- Symptoms: Tab walk and element tree extraction return [] when pywinauto not installed, silently failing any accessibility-based workflows
- Files: `core/accessibility.py:5-13`, `core/accessibility.py:48-64`
- Trigger: Running on non-Windows, or Windows with pywinauto not installed
- Workaround: Module logs a warning at import time but continues; callers expect list output and don't detect failure

**Vision Module JSON Extraction Fragility:**
- Symptoms: VLM responses that are malformed or wrapped differently than expected raise ValueError with no guidance on what went wrong
- Files: `core/vision.py:125-170`, `core/vision.py:173-219`
- Trigger: Model returns non-JSON, JSON with different structure, or commentary outside fences
- Workaround: Logs error with first 200 chars of response; user must read logs and manually inspect response

## Security Considerations

**Hardcoded DPI Awareness Magic Numbers:**
- Risk: `main.py:45-46` hardcodes magic value `-4` for `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2` using ctypes without version guard
- Files: `main.py:40-65`
- Current mitigation: Wrapped in try/except that falls back to v1 then v0; but magic numbers not validated against Windows SDK
- Recommendations: Define named constants or check Windows version before calling; document which Windows versions support which constants

**Base64 Encoding of Full Screenshots in VLM Requests:**
- Risk: Images are base64-encoded and sent to LiteLLM proxy; if proxy is over HTTP (not HTTPS), images are transmitted in cleartext
- Files: `core/vision.py:53-58`, `core/vision.py:76-122`
- Current mitigation: Config allows setting custom proxy URL but no TLS validation; default is `http://localhost:4000/v1`
- Recommendations: Enforce HTTPS in production; validate SSL certs; sanitize screenshots before sending (crop to relevant region only)

**Temp Files Not Deleted on Exception:**
- Risk: `mapper/validator.py:76-82` creates temporary PNG files via `tempfile.NamedTemporaryFile(delete=False)` but cleanup only happens if no exception
- Files: `mapper/validator.py:73-99`
- Current mitigation: Try/except around VLM call but temp files leaked if exception occurs after creation
- Recommendations: Use context manager `with tempfile.NamedTemporaryFile(...delete=False) as tmp:` and register cleanup in finally block, or use `tempfile.TemporaryDirectory`

**No Validation of Action Payloads:**
- Risk: `mapper/runner.py:114-120` retrieves action_payload from edges and passes to executor without type checking or sanitization
- Files: `mapper/runner.py:114-120`, `mapper/graph.py`
- Current mitigation: Graph loading from JSON includes some validation but edges can be malformed
- Recommendations: Validate `action_payload` is a string; check length bounds (prevent DoS via massive paste); escape special characters if needed

## Performance Bottlenecks

**VLM Full-Screen Scan is Expensive Fallback:**
- Problem: If OmniParser, CLIP, and OCR all fail, `core/locate.py:182-210` calls `first_pass_map_array()` which invokes VLM on entire screen (~500ms-2s depending on model)
- Files: `core/locate.py:182-210`, `core/vision.py:220+`
- Cause: No other reliable way to find elements if visual features don't match; VLM is most reliable but slowest
- Improvement path: Cache VLM results for 5-10 seconds; reuse candidate list across multiple locate attempts in same step; precompute candidate list once per screenshot

**Pixel Diff Calculation is Slow on Large Images:**
- Problem: `core/capture.py:45-73` computes full-screen diff via numpy operations every validation; on 4K displays this is ~8M pixels × 3 channels
- Files: `core/capture.py:45-73`, `mapper/validator.py:54`
- Cause: Uses `cv2.absdiff` which is not optimized for sparse changes
- Improvement path: Subsample image or divide into grid; compute diff on lower-resolution version first; early exit if threshold exceeded

**Embeddings FAISS Index Loaded on Every Locate:**
- Problem: `core/locate.py:116-155` calls `get_embedding_by_id()` which loads FAISS index from disk every time during cascade
- Files: `core/locate.py:120`, `core/embeddings.py:75-96`
- Cause: `load_index()` checks if `_faiss_index is not None` so subsequent calls are cached, but first call does I/O
- Improvement path: Pre-warm FAISS index at start of replay; add metrics for cache hits/misses

**OCR Tesseract Subprocess Launch Per Call:**
- Problem: `core/ocr.py:90-103` and `ocr_with_boxes()` launch Tesseract process for every OCR call; no pooling
- Files: `core/ocr.py:90-103`, `core/ocr.py:105-148`
- Cause: pytesseract wraps subprocess calls; no persistent daemon available
- Improvement path: Consider faster OCR alternatives (EasyOCR, PaddleOCR, or local tiny ONNX model); pre-process image to remove low-contrast regions

## Fragile Areas

**Cascading Locate Strategy Depends on All Stage Implementations:**
- Files: `core/locate.py:49-239` (main function), `core/detection.py`, `core/embeddings.py`, `core/ocr.py`, `core/vision.py`
- Why fragile: Each stage can fail independently (missing dependency, service unavailable, model crash). If OmniParser crashes, detection fails silently; CLIP missing raises ImportError; OCR binary missing raises RuntimeError. Inconsistent error handling makes cascade unpredictable.
- Safe modification: Never skip a stage silently; log stage entry/exit; fail fast if a required stage crashes; provide debug flags to enable/disable stages independently
- Test coverage: Each stage tested separately but cascade integration test (`tests/test_cascade.py`) only covers happy path; no tests for partial availability

**Watcher Thread Lifecycle Management:**
- Files: `core/watcher.py:48-170`, `core/watcher.py:210-280`
- Why fragile: `start_watching()` creates thread but doesn't validate callback signature; thread state checked via `_running_lock` but `_watcher_thread` reference can be stale if stop/start called rapidly
- Safe modification: Use threading.Event more explicitly; prevent concurrent start/stop; validate callback is callable at start time
- Test coverage: No unit tests for watcher; only integration tests that depend on real screen polling

**VLM Response JSON Parsing Regex Fragility:**
- Files: `core/vision.py:125-170`, `core/vision.py:173-219` (regex patterns line 141, 153, 161, 186, 201)
- Why fragile: Regex `\{.*\}` (greedy) fails on nested objects with }, `\[.*\]` (greedy) fails if response has multiple arrays. If model adds newlines inside fences the `re.DOTALL` may over-match.
- Safe modification: Use JSON parser with lenient mode; add unit tests for 20+ response formats; fail with clear error message showing what was extracted vs expected
- Test coverage: `tests/test_smart_detect.py` mocks VLM but doesn't test malformed responses; real responses not in test fixtures

**Graph Edge Addition Validation Missing:**
- Files: `mapper/graph.py:`, `main.py:177` (compose mode)
- Why fragile: `cmd_compose()` allows user to draw edges via overlay click but graph doesn't validate that edges form a DAG or don't create cycles; invalid graphs silently load
- Safe modification: After each edge addition, run DAG check; warn user if cycle detected; reject invalid graphs at load time
- Test coverage: No test for graph validity; `tests/test_export.py` only tests serialization format

**Recorder Overlay Window Management on Non-Windows:**
- Files: `recorder/overlay.py`, `recorder/overlay_view.py`, `main.py:208-216`
- Why fragile: PyQt6 code assumes X11 on Linux, Cocoa on macOS; no platform-specific testing; overlay behavior undefined on Wayland, non-native window managers
- Safe modification: Document supported configurations; test on target platforms; gracefully degrade to CLI mode if GUI unavailable
- Test coverage: No tests for overlay rendering; only end-to-end test uses mocked dialog

## Scaling Limits

**FAISS Index In-Memory Scaling:**
- Current capacity: `core/embeddings.py` uses IndexFlatIP which is O(N) search; metadata map loaded entirely into RAM
- Limit: ~10,000 elements per GPU before memory exhaustion; no index persistence strategy for large skill libraries
- Scaling path: Use hierarchical clustering (IndexIVFFlat); implement on-disk persistence; shard index by skill_id

**Screenshot Memory Accumulation in Replay Logs:**
- Current capacity: `mapper/runner.py` and `mapper/orchestrator.py` store `before_screenshot` and `after_screenshot` in `ReplayStep`; logs not pruned
- Limit: Full-resolution 4K screenshots (~25MB each); 100-step replay = 5GB of logs; no rotation/cleanup
- Scaling path: Store only hash of screenshot + delta; save full images to separate cache with TTL; implement log rotation in `get_config()["paths"]["replay_logs"]`

**Watcher Poll Interval Fixed:**
- Current capacity: `core/watcher.py` polls at fixed interval (default 1000ms); no adaptive rate limiting
- Limit: High CPU usage if poll_ms < 100; misses quick state changes if poll_ms > 500
- Scaling path: Add exponential backoff; measure change detection lag; implement hysteresis

**Detection Candidate List Not Pruned:**
- Current capacity: `core/vision.py:220+` (first_pass_map_array) returns all candidates; `core/locate.py:191-206` iterates all to find best match
- Limit: 1000+ candidates on complex pages slow down matching; no filtering before VLM
- Scaling path: Pre-filter candidates by element type (skip invisible, hidden, min size); grid-based spatial indexing

## Dependencies at Risk

**Qwen2-VL Model Via Ollama (External Service):**
- Risk: Vision analysis depends on external Ollama service being online; no local fallback model configured
- Impact: If Ollama crashes or network fails, VLM validation disabled; execution continues with lower confidence
- Migration plan: Add fallback to smaller local model (e.g., PaliGemma via transformers) for non-critical validation; cache VLM results for common patterns

**Tesseract Binary Platform-Specific Installation:**
- Risk: OCR stage fails if binary not installed; installation paths differ per OS/distro
- Impact: Users on macOS must `brew install tesseract` or pass `TESSERACT_CMD`; Linux users vary by distro package manager
- Migration plan: Consider faster OCR library (EasyOCR, paddleOCR) that's pure-Python; or bundle Tesseract binary via PyInstaller

**PyAutoGUI Cross-Platform Mouse/Keyboard:**
- Risk: Behavior varies across platforms (Wayland vs X11, different window managers); failsafe (mouse to 0,0) may not work on all setups
- Impact: Mouse movement may stutter; clicks may miss on high-DPI displays even with DPI awareness; drag/scroll unreliable on some Linux DMs
- Migration plan: Consider human_mouse library (recommended in MEMORY.md); test on Wayland explicitly; add platform-specific timing offsets

**PyWinAuto Windows-Only Accessibility:**
- Risk: Accessibility tree walking only works on Windows with UIA; Mac/Linux have no equivalent
- Impact: Tab walk and accessibility features unavailable on non-Windows; graceful degradation documented but not tested
- Migration plan: Add native Cocoa/GTK accessibility APIs for macOS/Linux (non-trivial); document as Windows-only feature in README

## Missing Critical Features

**No Skill Versioning or Rollback:**
- Problem: Skills loaded from JSON have no version field; if a skill is updated, old recordings can't be downgraded or compared
- Blocks: Version management, A/B testing of skill changes, recovery from bad edits
- Fix approach: Add `version` field to skill metadata (SEMVER); implement skill export/import with version preservation; log version mismatch warnings

**No Skill Dependency Management:**
- Problem: Complex skills may reference sub-skills (e.g., "login_gmail" calls "open_browser" as a subtask) but no way to declare or validate dependencies
- Blocks: Composability, reuse, safe modification of base skills
- Fix approach: Add `dependencies: [{"skill_id": "...", "version": ">=1.0"}]` to skill metadata; validate before execution

**No Audit Trail for Execution:**
- Problem: Replay logs stored but no persistent record of which user ran what skill when; no access control
- Blocks: Compliance, debugging shared environments, detecting abuse
- Fix approach: Add user context to replay logs; store in database with timestamp, user, skill, success/failure; add query interface

**No Recovery Mode for Failed Steps:**
- Problem: Execution aborts on first locate/action failure; no way to pause, manually fix the state, and resume
- Blocks: Automated recovery, interactive debugging, resilience to UI changes
- Fix approach: Add `--pause-on-failure` flag; show dialog with options: skip/retry/abort; let user manually correct and continue from next step

**No Element Change Detection Between Recordings:**
- Problem: If a UI element moves or changes appearance between recording and playback, locate may fail silently; no warning of drift
- Blocks: Detecting when skills need re-recording, alerting on layout changes
- Fix approach: Add schema to snapshot UI element positions; compare at playback time; warn if drift > threshold

## Test Coverage Gaps

**Cascade Locate Missing Partial Availability Tests:**
- What's not tested: OmniParser unavailable but CLIP available; OCR available but not Tesseract binary; VLM unavailable
- Files: `tests/test_cascade.py:464` only covers happy path; no parameterized tests for stage combinations
- Risk: If dependency missing, locate fails silently or with confusing error; user doesn't know which stage failed
- Priority: High — locate is critical path; partial failures common in production

**Recorder Overlay Not Tested on Non-Windows:**
- What's not tested: Overlay behavior on macOS/Linux; geometry calculations on Wayland; HiDPI scaling on non-Windows
- Files: `recorder/overlay.py` and `recorder/overlay_view.py` have no platform-specific tests
- Risk: Overlay fails silently on macOS/Linux; user unaware feature not supported
- Priority: High — UI framework (PyQt6) depends on platform; visual correctness critical

**Watcher Thread Safety Under Rapid Stop/Start:**
- What's not tested: `start_watching()` called twice without `stop_watching()`; callback raises exception; thread crashes
- Files: `tests/test_cascade.py` uses watcher but only in normal flow; no concurrent access tests
- Risk: Leaking threads; deadlock if callback tries to stop watcher; state corruption
- Priority: Medium — race condition unlikely but consequences severe

**VLM Response Handling Edge Cases:**
- What's not tested: Model returns empty response; returns invalid JSON; returns JSON array instead of object; includes unicode emoji/special chars
- Files: `core/vision.py:125-170` tested only with mocked responses in `tests/test_smart_detect.py:268+`
- Risk: Actual VLM response crashes parser; user unaware of format issue
- Priority: Medium — would catch with golden dataset of real responses

**Graph Edge Validation Missing:**
- What's not tested: Adding cycles to graph; creating disconnected nodes; edge with missing src/dst; edge with invalid action_type
- Files: `mapper/graph.py` has no validation tests; `tests/test_export.py` only tests I/O format
- Risk: Invalid graphs load and fail during execution with unclear error
- Priority: Low — graph editing is compose mode (less common); validation would catch at load time

**Temp File Cleanup in Validator:**
- What's not tested: `mapper/validator.py:76-82` leaks temp files if exception occurs; no test for exception path
- Files: `tests/test_orchestrator.py:331` mocks VLM but doesn't force exception
- Risk: Disk space exhaustion if many validations fail
- Priority: Low — but easy to fix and verify with tempdir inspection

---

*Concerns audit: 2026-03-16*
