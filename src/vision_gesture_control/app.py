"""Simplified Vision Gesture Presentation Control.

Pipeline:
    Webcam -> YuNet face detection -> SFace identity -> active controller
           -> MediaPipe hand gesture -> stability + cooldown
           -> PowerPoint keystroke (next / prev / start / exit)

Keyboard shortcuts inside the camera window:
    q  Quit
    a  Register a new authorized person (captures 5 samples)
    d  Delete a registered person
    r  Reload the face database
    t  Toggle dry-run mode (shows what would happen, no key press)
"""

from __future__ import annotations

import os
import shutil
import time
import tkinter as tk
from tkinter import simpledialog
from typing import List, Optional

import cv2

from .actions import PresentationActionController
from .config import CONFIG_PATH, DB_PATH, ensure_dirs, load_config, save_json
from .face import (
    FaceIndex,
    FaceTracker,
    bbox_area,
    create_detector,
    create_recognizer,
    crop_face,
    detect_faces,
)
from .gestures import GestureCommandResolver, MediaPipeGestureEngine
from .models import ensure_face_models, ensure_gesture_model
from .ui import (
    PerformanceStats,
    draw_action_toast,
    draw_active_crown,
    draw_capture_progress,
    draw_faces,
    draw_gesture_badge,
    draw_gesture_progress_ring,
    draw_hand_skeleton,
    draw_help_hint,
    draw_legend,
    draw_locked_indicator,
    draw_performance,
    draw_status_bar,
)


CAPTURE_SAMPLES = 5
CAPTURE_INTERVAL_SECONDS = 0.6


# ---------------------------------------------------------------------------
# Tk dialogs for the keyboard shortcuts
# ---------------------------------------------------------------------------

def _make_modal_root() -> tk.Tk:
    """Tk root that pops above the OpenCV camera window on Windows."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def _ask_person_name() -> Optional[str]:
    root = _make_modal_root()
    name = simpledialog.askstring(
        "Add Person", "Enter the name of the new person:", parent=root
    )
    root.destroy()
    if name and name.strip():
        return name.strip()
    return None


def _db_people(db_path: str) -> List[str]:
    if not os.path.exists(db_path):
        return []
    return sorted(
        name
        for name in os.listdir(db_path)
        if os.path.isdir(os.path.join(db_path, name))
    )


def _ask_person_to_delete(db_path: str) -> Optional[str]:
    people = _db_people(db_path)
    if not people:
        print("No people are registered.")
        return None

    root = _make_modal_root()
    name = simpledialog.askstring(
        "Delete Person",
        "Enter person name to delete:\n" + ", ".join(people),
        parent=root,
    )
    root.destroy()
    if not name:
        return None

    normalized = name.strip().casefold()
    matches = [p for p in people if p.casefold() == normalized]
    if not matches:
        print(f"Person '{name}' was not found.")
        return None
    return matches[0]


def _delete_person_folder(db_path: str, person: str) -> bool:
    target = os.path.abspath(os.path.join(db_path, person))
    db_root = os.path.abspath(db_path)
    if not target.startswith(db_root + os.sep):
        print(f"Refusing to delete unsafe path: {target}")
        return False
    if not os.path.isdir(target):
        return False
    shutil.rmtree(target)
    return True


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _configure_camera(cap: cv2.VideoCapture, config: dict) -> None:
    cam_cfg = config["camera"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg["height"]))
    cap.set(cv2.CAP_PROP_FPS, int(cam_cfg["fps"]))


def _try_open(index: int, config: dict) -> Optional[cv2.VideoCapture]:
    # On Windows, DSHOW is more forgiving with consumer webcams than the
    # default MSMF backend. Fall back to the default if DSHOW fails.
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        _configure_camera(cap, config)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
        cap.release()
    return None


def _open_camera(config: dict) -> Optional[cv2.VideoCapture]:
    primary = int(config["camera"]["primary_index"])
    fallback = int(config["camera"]["fallback_index"])

    print(f"Attempting to open primary webcam (index {primary})...")
    cap = _try_open(primary, config)
    if cap is not None:
        return cap

    print(f"Primary webcam failed. Trying fallback (index {fallback})...")
    return _try_open(fallback, config)


def _toggle_dry_run(config: dict) -> bool:
    controls = config["actions"]
    controls["dry_run"] = not bool(controls.get("dry_run", False))
    save_json(CONFIG_PATH, config)
    return bool(controls["dry_run"])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    config = load_config()

    print("Initializing models...")
    if not ensure_face_models():
        return
    if bool(config["gestures"]["enabled"]):
        ensure_gesture_model()

    detector = create_detector(config)
    recognizer = create_recognizer()
    if detector is None or recognizer is None:
        return

    face_index = FaceIndex(
        DB_PATH, detector, recognizer,
        float(config["face"]["sface_cosine_threshold"]),
    )
    face_index.reload()

    tracker = FaceTracker(config, face_index)
    gesture_engine = MediaPipeGestureEngine(config)
    actions = PresentationActionController(config)
    resolver = GestureCommandResolver(config, actions)
    stats = PerformanceStats()

    cap = _open_camera(config)
    if cap is None:
        print("Error: Could not open any webcam stream.")
        gesture_engine.close()
        return

    print("Stream successfully opened.")
    print("==========================")
    print("KEYS:")
    print("  q -> Quit")
    print("  a -> Add a new authorized person (5 samples)")
    print("  d -> Delete an authorized person")
    print("  r -> Reload face database")
    print("  t -> Toggle dry-run mode")
    print("==========================")

    capture_mode = False
    capture_name = ""
    capture_count = 0
    last_capture_at = 0.0
    gesture_status = ""

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            if bool(config["camera"]["mirror"]):
                frame = cv2.flip(frame, 1)

            now = time.time()
            stats.tick(now)
            # Monotonic ms for MediaPipe's strictly-increasing-timestamp
            # requirement (`time.time()` can jump if the system clock is set).
            timestamp_ms = time.monotonic_ns() // 1_000_000

            detect_start = time.perf_counter()
            faces = detect_faces(detector, frame)
            stats.observe("detect_ms", (time.perf_counter() - detect_start) * 1000.0)

            tracks = tracker.update(frame, faces, now)

            # ------ keyboard input ----------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                face_index.reload()
                tracker.reset_identities()
                gesture_status = "Identity database reloaded"
            elif key == ord("t"):
                dry = _toggle_dry_run(config)
                gesture_status = f"Dry-run {'enabled' if dry else 'disabled'}"
            elif key == ord("d") and not capture_mode:
                person = _ask_person_to_delete(DB_PATH)
                if person and _delete_person_folder(DB_PATH, person):
                    face_index.reload()
                    tracker.reset_identities()
                    gesture_status = f"Deleted {person}"
                    print(gesture_status)
            elif key == ord("a") and not capture_mode:
                name = _ask_person_name()
                if name:
                    capture_name = name
                    os.makedirs(os.path.join(DB_PATH, capture_name), exist_ok=True)
                    capture_mode = True
                    capture_count = 0
                    last_capture_at = now
                    print(
                        f"Starting capture for {capture_name}. "
                        "Move your head slightly between samples."
                    )

            # ------ registration capture ----------------------------------
            if capture_mode and tracks:
                largest = max(tracks, key=lambda t: bbox_area(t.bbox))
                if now - last_capture_at > CAPTURE_INTERVAL_SECONDS:
                    crop = crop_face(frame, largest.bbox)
                    if crop is not None:
                        img_path = os.path.join(
                            DB_PATH, capture_name, f"sample_{capture_count}.jpg"
                        )
                        cv2.imwrite(img_path, crop)
                        capture_count += 1
                        last_capture_at = now
                    if capture_count >= CAPTURE_SAMPLES:
                        capture_mode = False
                        face_index.reload()
                        tracker.reset_identities()
                        print(f"Finished capturing {CAPTURE_SAMPLES} samples for {capture_name}.")

            # ------ gesture pipeline --------------------------------------
            recognized = tracker.is_unlocked(now) and not capture_mode
            if recognized:
                gesture_engine.submit(frame, timestamp_ms)

            gesture = gesture_engine.latest()
            if gesture.timestamp_ms:
                stats.observe(
                    "gesture_latency_ms",
                    max(0.0, float(timestamp_ms - gesture.timestamp_ms)),
                )

            hand_ok = tracker.hand_belongs_to_active(gesture.landmarks, frame.shape, now)

            if recognized and not capture_mode:
                status = resolver.process(gesture, recognized, hand_ok, now)
                if status:
                    gesture_status = status
            else:
                resolver.process(gesture, False, False, now)

            # ------ drawing -----------------------------------------------
            active_track = tracker.active_track(now)
            dry_run = bool(config["actions"].get("dry_run", False))
            stable_seconds = float(config["gestures"]["stable_seconds"])

            draw_faces(frame, tracks, tracker.active_track_id)
            draw_active_crown(frame, active_track, now)
            draw_hand_skeleton(frame, gesture.landmarks, recognized)
            draw_gesture_progress_ring(
                frame,
                gesture.landmarks,
                resolver.candidate_action,
                resolver.candidate_since,
                stable_seconds,
                resolver.last_fired_at,
                now,
            )
            draw_gesture_badge(
                frame,
                gesture.landmarks,
                gesture.name,
                gesture.score,
                bound=bool(config["bindings"].get(gesture.name)),
                recognized=recognized,
            )
            draw_status_bar(
                frame,
                tracker,
                dry_run,
                gesture_status,
                gesture_engine.enabled,
                now,
            )
            draw_legend(frame, config, resolver.candidate_name if recognized else "")
            draw_action_toast(
                frame, resolver.last_fired_action, resolver.last_fired_at, now,
            )
            draw_capture_progress(
                frame, capture_mode, capture_name, capture_count,
                CAPTURE_SAMPLES, last_capture_at, now,
            )
            draw_locked_indicator(frame, tracker, now, suppress=capture_mode)
            draw_performance(frame, config, stats)
            draw_help_hint(frame)

            cv2.imshow("Vision Gesture Presentation Control", frame)
    finally:
        cap.release()
        gesture_engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
