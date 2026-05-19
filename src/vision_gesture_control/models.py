"""Auto-downloads the three model files used by the app.

The simplified app no longer needs Vosk (voice) or DeepFace (emotion), so this
module only handles YuNet, SFace, and the MediaPipe gesture recognizer.
"""

from __future__ import annotations

import os
import urllib.request

from .config import MODEL_DIR


YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
SFACE_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
GESTURE_MODEL = os.path.join(MODEL_DIR, "gesture_recognizer.task")

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
GESTURE_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)


def _ensure_file(path: str, url: str, label: str) -> bool:
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {label}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {label}: {path}")
        return True
    except Exception as exc:
        print(f"Could not download {label}: {exc}")
        return False


def ensure_face_models() -> bool:
    """Returns True only if both YuNet and SFace are available."""
    ok_yunet = _ensure_file(YUNET_MODEL, YUNET_URL, "YuNet face detector")
    ok_sface = _ensure_file(SFACE_MODEL, SFACE_URL, "SFace recognizer")
    return ok_yunet and ok_sface


def ensure_gesture_model() -> bool:
    return _ensure_file(GESTURE_MODEL, GESTURE_URL, "MediaPipe gesture recognizer")
