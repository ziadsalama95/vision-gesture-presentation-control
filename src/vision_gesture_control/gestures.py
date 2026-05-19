"""MediaPipe hand-gesture recognition + a stability/cooldown action resolver.

This file is intentionally short. The full app had custom-gesture templates,
swipe motion detection, and mode switching — none of those are needed for
simple presentation control, so they're gone.
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .models import GESTURE_MODEL


@dataclass
class GestureFrame:
    """One snapshot of MediaPipe's gesture output."""

    name: str = ""
    score: float = 0.0
    handedness: str = ""
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    timestamp_ms: int = 0


class MediaPipeGestureEngine:
    """Wraps the MediaPipe LIVE_STREAM gesture recognizer.

    `submit(frame, ts)` is fire-and-forget; results land asynchronously and
    `latest()` reads the most recent one. This keeps the camera loop smooth.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config["gestures"]["enabled"])
        self.last_result = GestureFrame()
        self.lock = threading.Lock()
        self.recognizer = None
        self.mp = None

        if not self.enabled:
            return
        try:
            import mediapipe as mp

            self.mp = mp
            BaseOptions = mp.tasks.BaseOptions
            GestureRecognizer = mp.tasks.vision.GestureRecognizer
            GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._callback,
            )
            self.recognizer = GestureRecognizer.create_from_options(options)
        except Exception as exc:
            self.enabled = False
            print(
                "Gesture control disabled. Install mediapipe and verify model file. "
                f"Details: {exc}"
            )

    def close(self) -> None:
        if self.recognizer is not None:
            self.recognizer.close()

    def submit(self, frame, timestamp_ms: int) -> None:
        if not self.enabled or self.recognizer is None or self.mp is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            self.recognizer.recognize_async(mp_image, timestamp_ms)
        except Exception:
            pass

    def latest(self) -> GestureFrame:
        with self.lock:
            return copy.deepcopy(self.last_result)

    def _callback(self, result: Any, _output_image: Any, timestamp_ms: int) -> None:
        frame = GestureFrame(timestamp_ms=timestamp_ms)
        try:
            if result.gestures and result.gestures[0]:
                category = result.gestures[0][0]
                frame.name = category.category_name
                frame.score = float(category.score)
            if result.handedness and result.handedness[0]:
                frame.handedness = result.handedness[0][0].category_name
            if result.hand_landmarks and result.hand_landmarks[0]:
                frame.landmarks = [
                    (float(p.x), float(p.y), float(p.z))
                    for p in result.hand_landmarks[0]
                ]
        except Exception:
            frame = GestureFrame(timestamp_ms=timestamp_ms)

        with self.lock:
            self.last_result = frame


class GestureCommandResolver:
    """Turns a stream of GestureFrames into PowerPoint actions.

    A gesture must:
      1. Have a confidence >= `gestures.confidence_threshold`.
      2. Stay the same for at least `gestures.stable_seconds`.
      3. Not fire the same action again within `gestures.action_cooldown_seconds`.
    """

    def __init__(self, config: Dict[str, Any], actions: "Any") -> None:
        self.config = config
        self.actions = actions
        # Stability tracking: which gesture+action is currently being "held"
        self.candidate = ""             # "name:action" composite key
        self.candidate_name = ""        # gesture name only (e.g. "Thumb_Up")
        self.candidate_action = ""      # mapped action name (e.g. "next_slide")
        self.candidate_since = 0.0      # when the current candidate started
        self.last_action_at: Dict[str, float] = {}
        self.last_status = ""
        # Last action that actually fired, used by the UI for the action toast.
        self.last_fired_action = ""
        self.last_fired_at = 0.0

    def process(
        self,
        gesture: GestureFrame,
        authorized: bool,
        hand_ok: bool,
        now: float,
    ) -> str:
        if not authorized or not hand_ok:
            self._reset()
            return ""

        name = gesture.name
        if name in ("", "None", "Unknown"):
            self._reset()
            return ""
        if gesture.score < float(self.config["gestures"]["confidence_threshold"]):
            self._reset()
            return ""

        action = str(self.config["bindings"].get(name, ""))
        if not action:
            self._reset()
            return ""

        key = f"{name}:{action}"
        if key != self.candidate:
            self.candidate = key
            self.candidate_name = name
            self.candidate_action = action
            self.candidate_since = now
            return f"Gesture: {name} ({gesture.score:.2f})"

        stable_seconds = float(self.config["gestures"]["stable_seconds"])
        cooldown = float(self.config["gestures"]["action_cooldown_seconds"])
        last_at = self.last_action_at.get(action, 0.0)

        if now - self.candidate_since >= stable_seconds and now - last_at >= cooldown:
            self.last_action_at[action] = now
            self.candidate_since = now
            self.last_fired_action = action
            self.last_fired_at = now
            self.last_status = self.actions.run_action(action)
            return self.last_status

        return f"Gesture: {name} ({gesture.score:.2f})"

    def _reset(self) -> None:
        self.candidate = ""
        self.candidate_name = ""
        self.candidate_action = ""
        self.candidate_since = 0.0
