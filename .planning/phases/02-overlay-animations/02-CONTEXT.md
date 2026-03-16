# Phase 2: Overlay Animations - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Cinematic animation items that run at 60fps without CPU spike: border shimmer glow, element scan animation, bbox morph, and donut cloud probability visualizer. All animations use QTimer (16ms interval), never trigger repaint loops from within paintEvent. Nothing "appears" — everything slides, fades, scales, or eases. This phase adds animation infrastructure and four distinct animation items on top of the Phase 1 overlay foundation.

</domain>

<decisions>
## Implementation Decisions

### Border shimmer glow (ANIM-01, ANIM-06)
- **Style**: Gradient sweep — a brighter-to-dimmer gradient rotates around the perimeter continuously
- **Width**: Between medium and thick (12-20px range)
- **Mouse reactivity**: The shimmer wave naturally retreats/shies away from the mouse cursor — not a simple bubble of clarity, but the wave "phobically" avoids the pointer with organic motion
- **UI element reactivity**: Shimmer also retreats from any active overlay UI elements (floating cards, bbox overlays, text panels). The border glow defers to content
- **Speed (ready/green)**: Slow ambient, 4-6 second full loop around the perimeter
- **Speed (recording/red)**: Faster (~2s loop), turns red, AND physically retreats/thins — "puts its head down and goes to work." Less prominent, not distracting
- **State transitions**: Green=ready/paused (wider, slower), Red=recording (thinner, faster, retreated). The border itself communicates the shift in intensity

### Element scan animation (ANIM-02, ANIM-06)
- **Trigger**: After click capture, starting from the rough snip boundary
- **Sequence**:
  1. Four corners glow red at the rough snip boundary
  2. Lines draw counter-clockwise from each corner toward the next
  3. As each line reaches the next corner, an internal red glow begins filling inward from that completed edge
  4. Once all 4 edges connect, interior has full red glow (30-40% opacity) — element still visible beneath
  5. Laser scan line sweeps top-to-bottom (~1s per pass), white core with red edge glow
  6. Then sweeps left-to-right (~1s per pass)
  7. Scan repeats until AI bbox resize result returns (minimum 1 full sweep each direction)
  8. When AI result arrives, corners snap to fitted bbox and red glow retreats completely
- **Scan line appearance**: White/bright core line with red glow halo — like an actual laser/scanner beam
- **Internal glow**: Medium opacity (30-40%), element remains visible underneath

### Bbox morph animation (ANIM-03, ANIM-06)
- **Style**: Smooth corner slide — each corner independently slides to its new position
- **Duration**: 400-600ms
- **Easing**: Ease-in-out (slow-fast-slow) — polished, cinematic feel
- **Final state**: Same red as scan phase but dimmed — thin red outline, glow retreats. Consistent with scan visual language
- **Transition**: Happens after scan animation completes, morphing from rough to AI-fitted bbox

### Donut cloud / probability visualizer (ANIM-04, ANIM-06)
- **Style**: Heat map blob — Gaussian distribution rendered as colored probability density
- **Color**: Red/translucent heat map while editing, turns green (same "ready" green from state machine) when user accepts
- **Appear animation**: Fades in from center outward
- **Editable**: Corners are draggable until user accepts the click zone
- **Idle animation**: Raindrop effect — slow simulated "click" ripples appear within the heat map, concentrated in high-probability center areas with occasional drops near edges. 3-5 raindrops visible at a time. Literally visualizes where randomized clicks would land
- **Acceptance**: When user confirms, heat map transitions from red to green = "ready"

### Animation infrastructure (ANIM-05)
- All animations driven by QTimer at 16ms interval (60fps target)
- No animation drives `self.update()` from within `paintEvent` — CPU stays under 10% during idle animation
- Each animation item is an independent QGraphicsItem subclass in its own file (consistent with Phase 1 layer pattern)

### Claude's Discretion
- Exact easing functions and animation curve implementations
- QGraphicsItem vs QGraphicsEffect vs QPainter approach for each animation type
- How to efficiently implement the mouse-reactive shimmer (distance field, physics sim, or simpler approach)
- Raindrop ripple rendering technique (QGraphicsEllipseItem pool, custom paint, etc.)
- Animation state machine design (how animations sequence/chain)
- Whether to use QPropertyAnimation or manual QTimer-driven interpolation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Overlay requirements
- `.planning/REQUIREMENTS.md` — ANIM-01 through ANIM-06 define all animation requirements
- `.planning/ROADMAP.md` — Phase 2 success criteria (5 criteria that must be TRUE)

### Phase 1 foundation (build on top of this)
- `recorder/overlay/border_layer.py` — Current static BorderLayer (4 QGraphicsRectItems, set_color API). Must evolve into shimmer glow
- `recorder/overlay/bbox_layer.py` — Current static BboxLayer with corner handles. Must gain morph animation
- `recorder/overlay/view.py` — OverlayView shell, manages scene and layers, apply_state() hook
- `recorder/overlay/controller.py` — OverlayController, state machine integration, public API
- `recorder/overlay/state.py` — OverlayState enum, STATE_COLORS map, transition table

### Phase 1 context (decisions that carry forward)
- `.planning/phases/01-overlay-foundation/01-CONTEXT.md` — Layer-based composition, one file per layer, fault isolation principles

### Project constraints
- `.planning/PROJECT.md` — Overlay screenshot cleanliness non-negotiable, all elements hide before capture
- `.planning/STATE.md` — Blocker: QObject+QGraphicsItem MRO ordering (QObject must come first)
- `CLAUDE.md` — Code quality rules, type hints, logging, thread safety requirements

### Codebase patterns
- `.planning/codebase/CONVENTIONS.md` — Naming, imports, error handling, logging patterns
- `.planning/codebase/ARCHITECTURE.md` — Layer architecture, event-driven patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BorderLayer` (recorder/overlay/border_layer.py): QGraphicsItemGroup with 4 rects. set_color() API. Must be evolved or replaced for shimmer glow
- `BboxLayer` (recorder/overlay/bbox_layer.py): QGraphicsItemGroup with rect + corner handles + label. highlight()/get_rect() API. Needs morph capability
- `STATE_COLORS` (recorder/overlay/state.py): RGBA map per state — green (50,200,50,150) and red (255,50,50,200). Reuse these base colors for animation
- `OverlayView.apply_state()` (recorder/overlay/view.py): Hook where state changes flow to layers — animation state changes should flow through here

### Established Patterns
- Each layer = independent QGraphicsItem subclass in its own file (Phase 1 decision)
- View manages scene, layers add/remove themselves
- Platform-specific code guarded with sys.platform checks
- setRenderHint(Antialiasing, True) already enabled on the view
- QTimer.singleShot(0, ...) for deferred setup

### Integration Points
- `OverlayView.apply_state()` — New animations must hook into state transitions here
- `OverlayView.hide_for_capture()` / `show_after_capture()` — All animation items must stop/hide cleanly for capture
- `OverlayController.set_bboxes()` — Bbox morph connects here (rough bbox → scan → AI fit → morph)
- `recorder/overlay/__init__.py` — Package init, new animation modules register here

</code_context>

<specifics>
## Specific Ideas

- Border shimmer should feel "phobic" of the mouse and UI elements — organic retreat, not mechanical. "The waves naturally shy away"
- Recording mode border: "puts its head down and went to work" — retreats, thins, speeds up. Not distracting
- Scan animation sequence is explicitly cinematic: corner glow → counter-clockwise line draw → fill inward → laser scan → bbox snap
- Donut cloud raindrop effect literally visualizes randomized human-like click distribution — shows the user what the executor will do
- Red = processing/editing, Green = accepted/ready — consistent color language across all animations

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-overlay-animations*
*Context gathered: 2026-03-16*
