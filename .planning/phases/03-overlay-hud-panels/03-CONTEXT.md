# Phase 3: Overlay HUD Panels - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Two interactive overlay panels — a frosted-glass tag dialog and a floating toolbar — both rendered as QGraphicsObject items inside the existing QGraphicsScene. The tag dialog captures element metadata with VLM-assisted typewriter fill. The toolbar provides context-sensitive buttons for the recording workflow. Both panels register as shimmer avoidance rects and hide fully before screenshot capture. No record flow wiring (Phase 4), no routine file format (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Tag dialog appearance (HUD-01)
- **Positioning**: Fades in near the captured element but always fully on-screen with breathing room from edges. Smart positioning using common UI conventions (prefer below/right, flip if near edge)
- **Frosted glass**: Medium frost — semi-transparent dark panel, vaguely see what's behind. Not opaque, not see-through
- **Size**: Medium panel (~400x280px), resizes smoothly when conditional fields appear/disappear
- **Shape**: Rounded corners (14-16px radius), card-like feel
- **Fonts**: Thin white fonts on dark frost
- **Accents**: Subtle green glow — consistent with "ready" state color language
- **Card border glow**: Same "lights shining from behind" design as screen border but at card scale — tiny green light sources around the card perimeter shining outward parallel to the screen (not down, not toward screen). Like a backlit bezel
- **Card glow animation**: Slow shimmer sweep at idle. During VLM typewriter fill, brightness pulses/flashes synchronized with keystrokes — like you can "see" the AI typing by watching the glow. Quick flashes, varying brightness, like a fast typist
- **Field styling**: Subtle dark recessed input wells for each field. Clearly defined but low contrast against frost
- **Dismiss**: Smooth opacity fade out
- **Confirm/Cancel**: Green "Confirm" and dim "Cancel" buttons at bottom, plus Enter to confirm and Esc to cancel
- **Shimmer interaction**: Tag dialog registers as avoidance rect — screen border shimmer retreats from it

### Typewriter fill behavior (HUD-02, HUD-03)
- **First capture (HUD-02)**: All VLM-filled fields typewrite simultaneously at different speeds — label, caption, type all fill at once but at slightly varied rates. Chaotic-but-purposeful energy
- **Edit mode (HUD-03)**: Pre-filled fields appear instantly with no typewriter animation
- **Speed**: Fast typist (~30-50ms per character). Fields fill in 1-2 seconds for typical labels
- **Cursors**: Per-field blinking green cursors at insertion point during typewriter fill. Reinforces "someone is typing" feel
- **VLM waiting/failure**: Card border glow pulses/breathes while waiting for VLM (matching existing visual language — glow IS the loading indicator, no spinners). Cursor keeps blinking at current position. After timeout, fields become editable with what's filled so far
- **Card glow during typing**: Quick brightness pulses synchronized with character insertion — the card border "flickers" with the typing rhythm

### Action dropdown + conditional fields (HUD-04)
- **Layout**: Two separate dropdowns — action type first (what to DO), element type below (what it IS)
- **Auto-detection**: VLM auto-fills both dropdowns but they remain editable by user
- **Conditional fields**: Action type selection reveals/hides additional fields with smooth slide-and-fade transitions. Panel resizes smoothly
- **Action-specific fields**:
  - `type` → "Text to type" field + "Press Enter after?" toggle
  - `scroll` → direction/amount
  - `wait` → condition/timeout
  - `click_drag` → "Drag to where?" (second bbox capture)
  - `read` / `snip_and_search` → prompt field for VLM query ("look here" action)
  - `prompt_user` → question field
  - `click` / `double_click` / `right_click` → no extra fields
- **Helper tips**: Inline dimmed text below relevant fields, contextual to action type. E.g., for textbox+type: "I'll click here before typing — no need to set up a separate click step"
- **Send Enter option**: For `type` action, toggle to automatically press Enter after typing (common for search boxes, chat inputs)

### Floating toolbar (HUD-05, HUD-06)
- **Shape**: Horizontal pill bar (~250x40px)
- **Position**: Starts top-right corner, user can drag anywhere
- **Visual**: Same frosted glass + backlit green glow as tag dialog. Consistent HUD family
- **Context-sensitive buttons** that swap with fade transitions:
  - **Recording mode**: [Pause] [Undo Last]
  - **Tag dialog open**: [Confirm] [Cancel] [Skip]
  - **Dry-run mode**: [Run Step] [Skip Step] [Finish]
- **Draggable**: Full drag support, remembers position during session
- **Hide behavior (HUD-06)**: Fully hides before any screenshot capture (same as all overlay elements)
- **Shimmer interaction**: Toolbar registers as avoidance rect for screen border shimmer

### Claude's Discretion
- Exact frosted glass implementation approach (QGraphicsBlurEffect, pre-blurred snapshot, or semi-transparent dark fill)
- Smart positioning algorithm for tag dialog (prefer below-right, flip logic)
- Typewriter timing variance algorithm (how to vary per-field speeds)
- Toolbar icon design (text labels, icons, or both)
- How conditional fields animate (exact easing, duration)
- Keyboard navigation within the tag dialog

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### HUD requirements
- `.planning/REQUIREMENTS.md` — HUD-01 through HUD-06 define all HUD panel requirements
- `.planning/ROADMAP.md` — Phase 3 success criteria (5 criteria that must be TRUE)

### Phase 2 animation infrastructure (build on top of this)
- `recorder/overlay/animation_clock.py` — AnimationClock for typewriter timing and glow pulses
- `recorder/overlay/shimmer_layer.py` — Shimmer layer with set_avoidance_rects() API. HUD panels must register here
- `recorder/overlay/view.py` — OverlayView manages scene, apply_state(), hide_for_capture()
- `recorder/overlay/controller.py` — OverlayController public API

### Existing tag dialog (reference only — being rebuilt as overlay panel)
- `recorder/dialog.py` — Current TagDialog (QDialog) with type dropdown, explainer, form fields
- `recorder/element_types.py` — ElementType enum (~25 types in 4 categories)

### Prior phase context
- `.planning/phases/01-overlay-foundation/01-CONTEXT.md` — Layer-based composition, one file per layer, fault isolation
- `.planning/phases/02-overlay-animations/02-CONTEXT.md` — Animation patterns, color language (red=processing, green=ready)

### Project constraints
- `.planning/PROJECT.md` — Overlay screenshot cleanliness non-negotiable, VLM with manual fallback
- `CLAUDE.md` — Code quality rules, type hints, logging, thread safety

### Codebase patterns
- `.planning/codebase/CONVENTIONS.md` — Naming, imports, error handling
- `.planning/codebase/ARCHITECTURE.md` — Layer architecture, event-driven patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AnimationClock` (recorder/overlay/animation_clock.py): Register tick callbacks for typewriter timing and glow pulse animation
- `ShimmerLayer.set_avoidance_rects()`: HUD panels register their bounding rects here so shimmer retreats from them
- `ElementType` enum (recorder/element_types.py): ~25 types in 4 categories — reuse for type dropdown
- `TagDialog` (recorder/dialog.py): Reference for field layout, type dropdown grouping, explainer pattern — rebuild as QGraphicsObject
- `STATE_COLORS` (recorder/overlay/state.py): Green (50,200,50) and red (255,50,50) for consistent color language

### Established Patterns
- Each layer = independent QGraphicsObject subclass in its own file (Phase 1/2 pattern)
- QGraphicsObject base for animated items (avoids MRO issues)
- AnimationClock.register(callback) for all timed animations
- Additive blending (CompositionMode_Plus) for glow effects — reuse for card border glow
- apply_state() hook for state-driven visual changes

### Integration Points
- `OverlayView` scene — Tag dialog and toolbar are QGraphicsObject items added/removed from scene
- `OverlayController` — Needs new API: show_tag_dialog(), dismiss_tag_dialog(), show_toolbar(), etc.
- `hide_for_capture()` — Both HUD panels must hide during capture cycle
- `recorder/overlay/__init__.py` — New modules register here

</code_context>

<specifics>
## Specific Ideas

- Card border glow uses same "lights shining from behind" visual as screen shimmer but at card scale — smaller, more of them, shining outward parallel to screen
- During VLM typing, card glow flashes synchronized with keystrokes — "you can see the AI typing by watching the glow"
- Helper tips are contextual and helpful: "I'll click here before typing" for textbox actions, "Press Enter to send?" toggle for search/chat inputs
- Red = processing, Green = ready color language carries forward — green accents on the HUD panels
- No spinners anywhere — glow breathing/pulsing IS the loading indicator

</specifics>

<deferred>
## Deferred Ideas

- "Look here" action (send crop + full screen to VLM with context prompt) — maps to existing `read`/`snip_and_search` action types, full implementation in Phase 6 (Action Types)
- VLM prompting best practices for element analysis — research during Phase 4 (Record Flow) or Phase 6

</deferred>

---

*Phase: 03-overlay-hud-panels*
*Context gathered: 2026-03-16*
