"""Configuration constants, defaults, and JSON load/save helpers.

Everything that used to live in many `DEFAULT_CONFIG` sections has been
collapsed down to what the simplified presentation-control app actually needs:
camera, face recognition, gestures, key bindings, PowerPoint keys, and UI.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))

CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

DB_PATH = os.path.join(DATA_DIR, "db")
CONFIG_PATH = os.path.join(CONFIG_DIR, "gesture_config.json")


# Pretty labels used in the on-screen legend.
ACTION_LABELS: Dict[str, str] = {
    "next_slide": "Next",
    "previous_slide": "Previous",
    "start_slideshow": "Start show",
    "exit_slideshow": "Exit",
}

GESTURE_LABELS: Dict[str, str] = {
    "Closed_Fist": "Fist",
    "Open_Palm": "Palm",
    "Pointing_Up": "Point",
    "Thumb_Down": "Thumb down",
    "Thumb_Up": "Thumb up",
    "Victory": "Victory",
    "ILoveYou": "I love you",
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "primary_index": 0,
        "fallback_index": 1,
        "width": 960,
        "height": 540,
        "fps": 30,
        "mirror": True,
    },
    "face": {
        # YuNet detector
        "score_threshold": 0.6,
        "nms_threshold": 0.3,
        "top_k": 5000,
        # Tracking + active-controller logic
        "track_iou_threshold": 0.25,
        "stable_seconds": 0.25,
        "unknown_retry_seconds": 2.0,
        "active_grace_seconds": 1.25,
        # SFace recognition acceptance threshold (cosine similarity)
        "sface_cosine_threshold": 0.363,
        # Lock if two authorized people are within this rank gap
        "ambiguous_rank_gap": 0.15,
        # Lock if an unknown face is this much larger than the authorized one
        "foreground_override_ratio": 1.12,
    },
    "gestures": {
        "enabled": True,
        "confidence_threshold": 0.55,
        "stable_seconds": 0.35,
        "action_cooldown_seconds": 1.0,
    },
    # Gesture -> action mapping. Add/remove entries here to change behaviour
    # without touching code.
    "bindings": {
        "Thumb_Up": "next_slide",
        "Thumb_Down": "previous_slide",
        "Victory": "start_slideshow",
        "Closed_Fist": "exit_slideshow",
    },
    "actions": {
        "enabled": True,
        # When true, the camera shows what *would* happen but no key is pressed.
        "dry_run": False,
        "focus_before_action": True,
        # When true, refuses to send keys unless a PowerPoint window is found.
        "require_target_window": False,
        "target_window_titles": ["PowerPoint", "Slide Show"],
        # action name -> list of keys to press (single key = press, multiple = hotkey)
        "key_profile": {
            "next_slide": ["right"],
            "previous_slide": ["left"],
            "start_slideshow": ["f5"],
            "exit_slideshow": ["esc"],
        },
    },
    "ui": {
        "show_legend": True,
        "show_performance": True,
        "overlay_alpha": 0.58,
    },
}


def deep_merge(defaults: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load config from disk, falling back to defaults if missing/broken.

    Defaults are merged in so newly-added keys appear automatically when the
    user upgrades.
    """

    if not os.path.exists(path):
        save_json(path, DEFAULT_CONFIG)
        return copy.deepcopy(DEFAULT_CONFIG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            current = json.load(f)
    except (OSError, json.JSONDecodeError):
        # Move a broken config aside so the user can recover it, then rewrite.
        backup = f"{path}.broken"
        try:
            os.replace(path, backup)
            print(f"Config was invalid. Moved it to {backup}")
        except OSError:
            pass
        save_json(path, DEFAULT_CONFIG)
        return copy.deepcopy(DEFAULT_CONFIG)

    merged = deep_merge(DEFAULT_CONFIG, current)
    if merged != current:
        save_json(path, merged)
    return merged


def ensure_dirs() -> None:
    for path in (CONFIG_DIR, DATA_DIR, MODEL_DIR, DB_PATH):
        os.makedirs(path, exist_ok=True)
