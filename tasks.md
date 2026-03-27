# Project: Mac Over Speak - Task Breakdown

## Goal
Successfully removed the floating recording indicator and moved all status feedback (recording state and language) to the system menu bar icon.

## Current State
- ✅ All status indicators (Recording, Processing, Typing) are now integrated into the system tray icon via dynamic image generation.
- ✅ Floating `tkinter` window has been removed.
- ✅ Added "Hard Restart Service" option in the menu for improved stability.

# Tasks
- [x] check if loading qwen asr model process is downlaoding model from network or from local. If load from network should make sure that it download it to local and load from local data after first download
- [x] make sure that this program use as small memory as possible

## Workflow Notes
- Keep the `tkinter` main loop (`tick_tk`) as it's necessary for the current architecture (event loop and background worker sync).
- Ensure `PIL` fonts are handled correctly across different macOS versions using system font paths.
