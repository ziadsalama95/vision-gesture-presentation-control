"""Sends keyboard shortcuts to PowerPoint via PyAutoGUI.

Optionally focuses the PowerPoint window first via pygetwindow (Windows).
Falls into "dry-run" mode if `actions.dry_run` is true so you can test
gestures without actually pressing keys.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List


class PresentationActionController:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config["actions"]["enabled"])
        self.pyautogui = None
        self.pygetwindow = None
        self.last_focus_status = ""

        if not self.enabled:
            return

        try:
            import pyautogui

            pyautogui.PAUSE = 0.03
            self.pyautogui = pyautogui
        except Exception as exc:
            self.enabled = False
            print(f"External controls disabled. Install pyautogui. Details: {exc}")
            return

        # Window focusing is optional. Without it actions still work, they
        # just go to whatever window is currently focused.
        try:
            import pygetwindow

            self.pygetwindow = pygetwindow
        except Exception:
            self.pygetwindow = None

    def valid_actions(self) -> List[str]:
        return sorted(self.config["actions"]["key_profile"].keys())

    def run_action(self, action: str) -> str:
        if not self.enabled or self.pyautogui is None:
            return "External controls disabled"

        keys = self.config["actions"]["key_profile"].get(action)
        if not keys:
            return f"No key binding for {action}"

        if bool(self.config["actions"].get("dry_run", False)):
            return f"DRY RUN: {action} -> {'+'.join(keys)}"

        try:
            focus_status = self._focus_target_window()
            if focus_status.startswith("Target not found") and bool(
                self.config["actions"].get("require_target_window", False)
            ):
                return focus_status

            if len(keys) == 1:
                self.pyautogui.press(keys[0])
            else:
                self.pyautogui.hotkey(*keys)

            suffix = f" | {focus_status}" if focus_status else ""
            return f"{action}{suffix}"
        except Exception as exc:
            return f"Action failed: {exc}"

    # -----------------------------------------------------------------

    def _focus_target_window(self) -> str:
        controls = self.config["actions"]
        if not bool(controls.get("focus_before_action", True)):
            self.last_focus_status = "Focus disabled"
            return ""
        if self.pygetwindow is None:
            self.last_focus_status = "Window focus unavailable"
            return self.last_focus_status

        titles = controls.get("target_window_titles", [])
        if not titles:
            self.last_focus_status = "No target title configured"
            return self.last_focus_status

        for title in titles:
            try:
                windows = self.pygetwindow.getWindowsWithTitle(title)
            except Exception:
                windows = []
            for window in windows:
                try:
                    if getattr(window, "isMinimized", False):
                        window.restore()
                    window.activate()
                    time.sleep(0.06)
                    self.last_focus_status = f"Focused: {title}"
                    return self.last_focus_status
                except Exception:
                    continue

        self.last_focus_status = "Target not found: PowerPoint"
        return self.last_focus_status
