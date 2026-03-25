# Project: Mac Over Speak - Task Breakdown

## Goal
Remove the floating recording indicator and move all status feedback (recording state and language) to the system menu bar icon.

## Current State
- A floating `tkinter` window shows a colored dot and language text.
- The menu bar icon only shows a colored dot (generated as a dynamic image).
- The floating indicator is often obscured by other windows.

## Task Breakdown

### Phase 1: Planning and Analysis
- [x] Research best way to render text into the menu bar icon using `PIL` (Pillow).
- [x] Determine the layout for the enhanced menu bar icon (Circle with text inside).

### Phase 2: Refactoring `qwen_bridge.py`
- [x] **Cleanup `setup_ui`**: Remove the code that creates and configures the `self.indicator` floating window.
- [x] **Enhance `update_rumps_icon`**:
    - Modify the function signature to accept `state` (recording status) and `lang` (current language).
    - Update drawing logic to render a colored circle with the language character (e.g., "中", "A", "J") centered inside.
- [x] **Update `_set_lang_text`**: Change it to trigger a menu bar icon update instead of modifying the canvas.
- [x] **Update `_update_ui_internal`**:
    - Remove all operations on `self.indicator`.
    - Ensure it calls `update_tray_status` with the latest state.
- [x] **Update `update_tray_status`**: Ensure it passes both `state` and `self.current_language_ui` to `update_rumps_icon`.

### Phase 3: Testing and Polish
- [x] Verify that the floating indicator no longer appears.
- [x] Verify that the menu bar icon correctly reflects:
    - Recording state (Red: Recording, Yellow: Processing, Green: Typing, Gray: Idle).
    - Language status (Correct character for Chinese, English, and Japanese).
- [x] Ensure high-resolution (Retina) support for the icons.
- [x] Final code cleanup and removal of unused `tkinter` code (except for the main root used for the loop).

## Workflow Notes
- Keep the `tkinter` main loop (`tick_tk`) as it's necessary for the current architecture.
- Ensure `PIL` fonts are handled correctly across different macOS versions if possible (or use a default system font).
