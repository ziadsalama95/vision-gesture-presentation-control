"""Face detection, recognition, tracking, and active-controller selection.

This is the heart of the "only me can control it" guarantee:

1. YuNet detects every face in the frame.
2. SFace produces a 128-D embedding for each registered sample.
3. FaceIndex compares a live face against the cached embeddings.
4. FaceTracker keeps stable IDs across frames and picks one "active controller":
   - It must be authorized (matches a person in `data/db/`).
   - It must be the most prominent (largest + most central) authorized face.
   - If two authorized people are roughly equal, controls LOCK.
   - If an unknown face is significantly more foreground, controls LOCK.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .models import SFACE_MODEL, YUNET_MODEL


@dataclass
class FaceEntry:
    person: str
    path: str
    feature: np.ndarray


@dataclass
class FaceTrack:
    track_id: int
    face: np.ndarray
    bbox: Tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    identity: str = "Unknown"
    similarity: float = 0.0
    identified_at: float = 0.0
    last_attempt: float = 0.0
    seen_count: int = 1
    pending_identification: bool = True


# ---------------------------------------------------------------------------
# Geometry helpers (kept module-level so they can be reused by the UI module)
# ---------------------------------------------------------------------------

def face_bbox(face: np.ndarray, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x, y, box_w, box_h = map(int, face[:4])
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + box_w)
    y2 = min(h, y + box_h)
    return x1, y1, x2, y2


def bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


def crop_face(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return None
    return crop.copy()


# ---------------------------------------------------------------------------
# Detector + recognizer factories
# ---------------------------------------------------------------------------

def create_detector(config: Dict[str, Any]) -> Optional[Any]:
    face_cfg = config["face"]
    try:
        return cv2.FaceDetectorYN.create(
            model=YUNET_MODEL,
            config="",
            input_size=(320, 320),
            score_threshold=float(face_cfg["score_threshold"]),
            nms_threshold=float(face_cfg["nms_threshold"]),
            top_k=int(face_cfg["top_k"]),
        )
    except Exception as exc:
        print(f"Error loading YuNet model: {exc}")
        return None


def create_recognizer() -> Optional[Any]:
    try:
        return cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    except Exception as exc:
        print(f"Error loading SFace model: {exc}")
        return None


def detect_faces(detector: Any, frame: np.ndarray) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    # Sort biggest first so downstream "largest face" logic is deterministic.
    return sorted([face for face in faces], key=lambda f: f[2] * f[3], reverse=True)


# ---------------------------------------------------------------------------
# Face database (one embedding per sample image)
# ---------------------------------------------------------------------------

def _image_paths(db_path: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(db_path):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


class FaceIndex:
    """Caches an SFace embedding for every image in `data/db/<person>/`."""

    def __init__(self, db_path: str, detector: Any, recognizer: Any, threshold: float) -> None:
        self.db_path = db_path
        self.detector = detector
        self.recognizer = recognizer
        self.threshold = threshold
        self.entries: List[FaceEntry] = []

    def reload(self) -> None:
        self.entries.clear()
        os.makedirs(self.db_path, exist_ok=True)
        for path in _image_paths(self.db_path):
            person = os.path.basename(os.path.dirname(path))
            img = cv2.imread(path)
            if img is None:
                continue
            feature = self._embed_database_image(img)
            if feature is not None:
                self.entries.append(FaceEntry(person=person, path=path, feature=feature))

        people = sorted({entry.person for entry in self.entries})
        print(f"SFace identity index: {len(self.entries)} samples, {len(people)} people.")

    def _embed_database_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        try:
            faces = detect_faces(self.detector, img)
            aligned = (
                self.recognizer.alignCrop(img, faces[0])
                if faces
                else cv2.resize(img, (112, 112))
            )
            feature = self.recognizer.feature(aligned)
            return np.asarray(feature).copy()
        except Exception:
            return None

    def identify(self, frame: np.ndarray, face: np.ndarray) -> Tuple[str, float]:
        """Return (person_name, similarity). 'Unknown' if no entry passes threshold."""
        if not self.entries:
            return "Unknown", 0.0
        try:
            aligned = self.recognizer.alignCrop(frame, face)
            feature = self.recognizer.feature(aligned)
        except Exception:
            return "Unknown", 0.0

        best_person, best_score = "Unknown", -1.0
        for entry in self.entries:
            try:
                score = float(
                    self.recognizer.match(
                        feature, entry.feature, cv2.FaceRecognizerSF_FR_COSINE
                    )
                )
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_person = entry.person

        if best_score >= self.threshold:
            return best_person, best_score
        return "Unknown", max(0.0, best_score)


# ---------------------------------------------------------------------------
# Tracker + active-controller selection
# ---------------------------------------------------------------------------

class FaceTracker:
    """Tracks faces by IoU and picks the single 'active controller'.

    `lock_reason` is a human-readable string that the UI shows so the user
    always knows *why* controls are locked.
    """

    def __init__(self, config: Dict[str, Any], face_index: FaceIndex) -> None:
        self.config = config
        self.face_index = face_index
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_id = 1
        self.active_track_id: Optional[int] = None
        self.lock_reason = "Waiting for authorized user"

    def reset_identities(self) -> None:
        for track in self.tracks.values():
            track.identity = "Unknown"
            track.similarity = 0.0
            track.pending_identification = True
            track.last_attempt = 0.0
        self.active_track_id = None
        self.lock_reason = "Identity database changed"

    def update(self, frame: np.ndarray, faces: List[np.ndarray], now: float) -> List[FaceTrack]:
        assigned_tracks: set[int] = set()
        assigned_faces: set[int] = set()
        iou_threshold = float(self.config["face"]["track_iou_threshold"])

        visible_bboxes = [face_bbox(face, frame.shape) for face in faces]

        # Greedy assignment of new detections to existing tracks via best IoU.
        for face_idx, bbox in enumerate(visible_bboxes):
            best_track_id, best_iou = None, 0.0
            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                score = iou(track.bbox, bbox)
                if score > best_iou:
                    best_iou, best_track_id = score, track_id

            if best_track_id is not None and best_iou >= iou_threshold:
                track = self.tracks[best_track_id]
                track.face = faces[face_idx]
                track.bbox = bbox
                track.last_seen = now
                track.seen_count += 1
                assigned_tracks.add(best_track_id)
                assigned_faces.add(face_idx)

        # Anything not assigned becomes a brand-new track.
        for face_idx, face in enumerate(faces):
            if face_idx in assigned_faces:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = FaceTrack(
                track_id=track_id,
                face=face,
                bbox=visible_bboxes[face_idx],
                first_seen=now,
                last_seen=now,
            )

        self._drop_stale(now)
        self._identify_ready_tracks(frame, now)
        self._select_active(frame.shape, now)
        return self.visible_tracks(now)

    def visible_tracks(self, now: float) -> List[FaceTrack]:
        return [t for t in self.tracks.values() if now - t.last_seen <= 0.35]

    def active_track(self, now: float) -> Optional[FaceTrack]:
        if self.active_track_id is None:
            return None
        track = self.tracks.get(self.active_track_id)
        if track is None:
            return None
        if now - track.last_seen <= float(self.config["face"]["active_grace_seconds"]):
            return track
        return None

    def is_unlocked(self, now: float) -> bool:
        track = self.active_track(now)
        return track is not None and track.identity != "Unknown"

    def hand_belongs_to_active(
        self,
        landmarks: List[Tuple[float, float, float]],
        frame_shape: Tuple[int, int, int],
        now: float,
    ) -> bool:
        """Returns True if the recognized hand is closer to the active face
        than to any other visible face. Stops a bystander from gesturing for you.
        """
        active = self.active_track(now)
        if active is None or not landmarks:
            return False

        visible = self.visible_tracks(now)
        if len(visible) <= 1:
            return True

        h, w = frame_shape[:2]
        hand_x = float(np.mean([p[0] for p in landmarks])) * w
        hand_y = float(np.mean([p[1] for p in landmarks])) * h

        def distance(track: FaceTrack) -> float:
            cx, cy = bbox_center(track.bbox)
            width = max(1.0, track.bbox[2] - track.bbox[0])
            height = max(1.0, track.bbox[3] - track.bbox[1])
            return math.sqrt(((hand_x - cx) / width) ** 2 + ((hand_y - cy) / height) ** 2)

        return min(visible, key=distance).track_id == active.track_id

    # --- internals ---------------------------------------------------------

    def _drop_stale(self, now: float) -> None:
        grace = float(self.config["face"]["active_grace_seconds"])
        stale = [tid for tid, t in self.tracks.items() if now - t.last_seen > max(3.0, grace + 1.0)]
        for tid in stale:
            self.tracks.pop(tid, None)
            if self.active_track_id == tid:
                self.active_track_id = None

    def _identify_ready_tracks(self, frame: np.ndarray, now: float) -> None:
        face_cfg = self.config["face"]
        stable_seconds = float(face_cfg["stable_seconds"])
        unknown_retry = float(face_cfg["unknown_retry_seconds"])

        for track in self.visible_tracks(now):
            if now - track.first_seen < stable_seconds:
                continue
            needs_first_id = track.pending_identification
            retry_unknown = (
                track.identity == "Unknown"
                and now - track.last_attempt >= unknown_retry
            )
            if needs_first_id or retry_unknown:
                identity, score = self.face_index.identify(frame, track.face)
                track.identity = identity
                track.similarity = score
                track.identified_at = now
                track.last_attempt = now
                track.pending_identification = False

    def _select_active(self, frame_shape: Tuple[int, int, int], now: float) -> None:
        visible = self.visible_tracks(now)
        authorized = [t for t in visible if t.identity != "Unknown"]

        if not visible:
            if self.active_track(now) is None:
                self.active_track_id = None
                self.lock_reason = "No visible face"
            return

        if not authorized:
            self.active_track_id = None
            self.lock_reason = "No authorized face"
            return

        ranked = sorted(
            authorized,
            key=lambda t: self._controller_rank(t, frame_shape),
            reverse=True,
        )
        best = ranked[0]
        best_rank = self._controller_rank(best, frame_shape)

        if len(ranked) > 1:
            second_rank = self._controller_rank(ranked[1], frame_shape)
            if best_rank - second_rank < float(self.config["face"]["ambiguous_rank_gap"]):
                self.active_track_id = None
                self.lock_reason = "Ambiguous authorized controller"
                return

        # If an unknown face is dramatically larger than the best authorized one,
        # treat the scene as untrusted (e.g. a stranger leaned over your shoulder).
        largest = max(visible, key=lambda t: bbox_area(t.bbox))
        if largest.identity == "Unknown":
            ratio = float(self.config["face"]["foreground_override_ratio"])
            if bbox_area(largest.bbox) > bbox_area(best.bbox) * ratio:
                self.active_track_id = None
                self.lock_reason = "Foreground face is not authorized"
                return

        self.active_track_id = best.track_id
        self.lock_reason = f"Unlocked: {best.identity}"

    def _controller_rank(self, track: FaceTrack, frame_shape: Tuple[int, int, int]) -> float:
        h, w = frame_shape[:2]
        area_score = bbox_area(track.bbox) / max(1.0, float(w * h))
        cx, cy = bbox_center(track.bbox)
        dx = abs(cx - (w / 2.0)) / max(1.0, w / 2.0)
        dy = abs(cy - (h / 2.0)) / max(1.0, h / 2.0)
        center_score = 1.0 - min(1.0, math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0))
        return area_score * 4.0 + center_score
