# Issues to Fix

Functional first, polish later. Work through sequentially.

## 1. Replay clicks don't pass through
Clicks during replay should reach the underlying application. Currently the overlay intercepts them.

## 2. Scan highlight persists
After locating/sizing/saving a node, the red scan rectangle stays on screen. Should clear automatically so user can start fresh with the next action.

## 3. Tag dialog popup issues
The popup that appears after element detection has multiple problems:
- **A)** Can't drag/move it — should be movable
- **B)** Text fields aren't editable — should be editable
- **C)** VLM prompt needs updating — AI needs to understand what we're looking for, what each dropdown option indicates, and how to choose the right one

## 4. Loading/scanning feedback missing
After highlighting an element (F2), there's no visible feedback. Need:
- "Loading UI Detector" indicator until scan/resize is done
- "Processing Element" indicator while VLM analyzes the element

## 5. Shimmer still ugly (COSMETIC)
Ocean waves with OpenSimplex are much better than the old radial blobs, but still not polished enough visually. Needs refinement.

## 6. Startup animation missing (COSMETIC)
Top-left and top-right menus don't catch your eye when the program starts. Need a proper entrance sequence:
- Screen fades to dark
- Menus slide in
- Shimmer starts
- Screen fades back in with UI on top of clean, minimized desktop
