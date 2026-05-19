"""Hand-landmark detection (MediaPipe) + a custom geometric gesture classifier.

Design notes
------------
MediaPipe's GestureRecognizer ships with a built-in classifier head trained on
~7 canonical gestures. This project deliberately ignores that classifier and
runs its own geometry-based rules on the 21 hand landmarks. Why?

  1. Easier to explain and extend (no black-box ML head).
  2. Lets us add gestures the bundled classifier doesn't know (e.g. OK sign,
     a "Three" finger count).
  3. Per-finger logic gives the UI richer hints (which fingers are extended).

We still keep the GestureRecognizer model because we already need its landmark
output; we just discard `result.gestures` in the async callback.
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .models import GESTURE_MODEL


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class GestureFrame:
    """One snapshot of the hand pipeline output."""

    name: str = ""
    score: float = 0.0
    handedness: str = ""
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    fingers_extended: Tuple[bool, bool, bool, bool, bool] = (False,) * 5
    timestamp_ms: int = 0


# ---------------------------------------------------------------------------
# MediaPipe wrapper (landmarks only, classifier output ignored)
# ---------------------------------------------------------------------------

class MediaPipeHandEngine:
    """Async wrapper around MediaPipe that exposes hand landmarks only.

    `submit(frame, ts)` is fire-and-forget. Results land on a callback thread
    and `latest()` reads the most recent one. Keeps the camera loop smooth.
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
                "Hand detection disabled. Install mediapipe and verify model file. "
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
        # We intentionally ignore `result.gestures` - this project runs its
        # own geometric classifier in GeometricGestureClassifier below.
        frame = GestureFrame(timestamp_ms=timestamp_ms)
        try:
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


# ---------------------------------------------------------------------------
# Geometric gesture classifier
# ---------------------------------------------------------------------------

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance in (x, y) only - z is a relative estimate from MP."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class GeometricGestureClassifier:
    """Classifies a hand pose from 21 MediaPipe landmarks using finger geometry.

    Rules:

    * A non-thumb finger is **extended** when its TIP is significantly farther
      from the wrist than its PIP joint (ratio > `finger_extended_ratio`).
      Robust under hand rotation.

    * The thumb is **extended** when its TIP is significantly farther from the
      pinky MCP than the thumb MCP is. Independent of hand orientation.

    * The "OK" gesture is checked first - it requires the thumb tip and index
      tip to be close (pinch ratio < `ok_pinch_ratio`) AND the other three
      fingers extended.

    Recognized gesture names:
        Closed_Fist, Open_Palm, Thumb_Up, Thumb_Down, Pointing_Up,
        Victory, Three, OK, ILoveYou
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        gcfg = config.get("gestures", {})
        self.finger_ratio = float(gcfg.get("finger_extended_ratio", 1.10))
        self.thumb_ratio = float(gcfg.get("thumb_extended_ratio", 1.20))
        self.ok_pinch_ratio = float(gcfg.get("ok_pinch_ratio", 0.40))
        self.thumb_up_offset = float(gcfg.get("thumb_up_y_offset", 0.05))

    # ----- per-finger rules ---------------------------------------------------

    def _finger_extended(
        self,
        landmarks: List[Tuple[float, float, float]],
        pip: int,
        tip: int,
    ) -> bool:
        wrist = landmarks[WRIST]
        d_pip = _dist(landmarks[pip], wrist)
        d_tip = _dist(landmarks[tip], wrist)
        return d_tip > d_pip * self.finger_ratio

    def _thumb_extended(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        ref = landmarks[PINKY_MCP]
        return (
            _dist(landmarks[THUMB_TIP], ref)
            > _dist(landmarks[THUMB_MCP], ref) * self.thumb_ratio
        )

    def _thumb_pointing_up(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        # OpenCV y increases downward, so "above" the wrist => smaller y.
        return landmarks[THUMB_TIP][1] < landmarks[WRIST][1] - self.thumb_up_offset

    def _hand_scale(self, landmarks: List[Tuple[float, float, float]]) -> float:
        return max(1e-6, _dist(landmarks[WRIST], landmarks[MIDDLE_MCP]))

    def _is_ok_sign(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        pinch = _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / self._hand_scale(landmarks)
        if pinch > self.ok_pinch_ratio:
            return False
        return (
            self._finger_extended(landmarks, MIDDLE_PIP, MIDDLE_TIP)
            and self._finger_extended(landmarks, RING_PIP, RING_TIP)
            and self._finger_extended(landmarks, PINKY_PIP, PINKY_TIP)
        )

    # ----- top-level classify --------------------------------------------------

    def fingers_extended(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> Tuple[bool, bool, bool, bool, bool]:
        """Returns (thumb, index, middle, ring, pinky) as booleans."""
        if len(landmarks) < 21:
            return (False, False, False, False, False)
        return (
            self._thumb_extended(landmarks),
            self._finger_extended(landmarks, INDEX_PIP, INDEX_TIP),
            self._finger_extended(landmarks, MIDDLE_PIP, MIDDLE_TIP),
            self._finger_extended(landmarks, RING_PIP, RING_TIP),
            self._finger_extended(landmarks, PINKY_PIP, PINKY_TIP),
        )

    def classify(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> Tuple[str, float, Tuple[bool, bool, bool, bool, bool]]:
        """Returns (gesture_name, score, finger_state). Empty name = unrecognized."""
        if len(landmarks) < 21:
            return "", 0.0, (False,) * 5

        # OK sign overlaps with several finger-state lookups, so check first.
        if self._is_ok_sign(landmarks):
            # Reconstruct fingers_extended for the UI hint.
            state = (
                False,
                False,
                True, True, True,
            )
            return "OK", 0.95, state

        state = self.fingers_extended(landmarks)
        thumb, index, middle, ring, pinky = state

        # Lookup table of finger-state -> gesture name.
        if state == (False, False, False, False, False):
            return "Closed_Fist", 0.95, state
        if state == (True, True, True, True, True):
            return "Open_Palm", 0.95, state
        if state == (True, False, False, False, False):
            if self._thumb_pointing_up(landmarks):
                return "Thumb_Up", 0.95, state
            return "Thumb_Down", 0.95, state
        if state == (False, True, False, False, False):
            return "Pointing_Up", 0.90, state
        if state == (False, True, True, False, False):
            return "Victory", 0.95, state
        if state == (False, True, True, True, False):
            return "Three", 0.90, state
        if state == (True, True, False, False, True):
            return "ILoveYou", 0.90, state

        return "", 0.0, state


# ---------------------------------------------------------------------------
# Stability + cooldown resolver
# ---------------------------------------------------------------------------

class GestureCommandResolver:
    """Turns a stream of GestureFrames into discrete actions.

    A gesture must:
      1. Pass `gestures.confidence_threshold` (geometric classifier confidence).
      2. Stay the same for at least `gestures.stable_seconds`.
      3. Wait at least `gestures.action_cooldown_seconds` before firing again.
    """

    def __init__(self, config: Dict[str, Any], actions: Any) -> None:
        self.config = config
        self.actions = actions
        self.classifier = GeometricGestureClassifier(config)

        self.candidate = ""              # "name:action" composite key
        self.candidate_name = ""
        self.candidate_action = ""
        self.candidate_since = 0.0
        self.last_action_at: Dict[str, float] = {}
        self.last_status = ""
        # Used by the UI to render the action toast.
        self.last_fired_action = ""
        self.last_fired_at = 0.0
        # Optional CSV logger (set externally by app.py)
        self.session_logger: Optional[Any] = None
        self.actor_provider: Optional[Any] = None

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

        # Run our geometric classifier on the landmarks and patch the
        # gesture frame so the UI sees consistent naming.
        name, score, fingers = self.classifier.classify(gesture.landmarks)
        gesture.name = name
        gesture.score = score
        gesture.fingers_extended = fingers

        if not name:
            self._reset()
            return ""
        if score < float(self.config["gestures"]["confidence_threshold"]):
            self._reset()
            return ""

        action = str(self.config["bindings"].get(name, ""))
        if not action:
            self._reset()
            return f"Gesture: {name} (no binding)"

        key = f"{name}:{action}"
        if key != self.candidate:
            self.candidate = key
            self.candidate_name = name
            self.candidate_action = action
            self.candidate_since = now
            return f"Gesture: {name} ({score:.2f})"

        stable_seconds = float(self.config["gestures"]["stable_seconds"])
        cooldown = float(self.config["gestures"]["action_cooldown_seconds"])
        last_at = self.last_action_at.get(action, 0.0)

        if now - self.candidate_since >= stable_seconds and now - last_at >= cooldown:
            self.last_action_at[action] = now
            self.candidate_since = now
            self.last_fired_action = action
            self.last_fired_at = now
            self.last_status = self.actions.run_action(action)
            # Optional session log
            if self.session_logger is not None:
                actor = ""
                if self.actor_provider is not None:
                    try:
                        actor = str(self.actor_provider() or "")
                    except Exception:
                        actor = ""
                try:
                    self.session_logger.log(
                        actor=actor,
                        gesture=name,
                        action=action,
                        status=self.last_status,
                    )
                except Exception:
                    pass
            return self.last_status

        return f"Gesture: {name} ({score:.2f})"

    def _reset(self) -> None:
        self.candidate = ""
        self.candidate_name = ""
        self.candidate_action = ""
        self.candidate_since = 0.0
