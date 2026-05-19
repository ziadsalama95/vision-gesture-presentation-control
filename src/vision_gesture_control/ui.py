"""All OpenCV overlay drawing for the gesture-controlled presentation app.

Modern dark HUD with viewfinder-style face boxes, color-coded hand skeleton,
gesture progress ring, animated action toasts, and a live-highlighted legend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import ACTION_LABELS, GESTURE_LABELS
from .face import FaceTrack, FaceTracker


# =============================================================================
# Color palette (BGR)
# =============================================================================

COLOR_PANEL_BG = (24, 24, 30)
COLOR_PANEL_BORDER = (90, 95, 110)
COLOR_TEXT_PRIMARY = (240, 240, 245)
COLOR_TEXT_SECONDARY = (165, 170, 185)
COLOR_TEXT_MUTED = (110, 115, 130)

COLOR_ACCENT = (255, 195, 70)        # warm cyan/azure (BGR)
COLOR_ACTIVE = (96, 255, 124)        # lime
COLOR_ACTIVE_DIM = (54, 168, 78)
COLOR_AUTH = (78, 200, 110)
COLOR_UNKNOWN = (60, 175, 240)       # amber
COLOR_LOCKED = (80, 90, 240)         # warm red
COLOR_DRYRUN_BG = (40, 130, 220)

COLOR_PROG_PENDING = (255, 195, 70)
COLOR_PROG_FIRING = (96, 255, 124)


# =============================================================================
# Hand topology (21-landmark connections, grouped by finger for coloring)
# =============================================================================

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky + palm base
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# Map each landmark to a finger group: 0=palm, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
FINGER_OF_LANDMARK: Dict[int, int] = {
    0: 0,
    1: 1, 2: 1, 3: 1, 4: 1,
    5: 2, 6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3, 12: 3,
    13: 4, 14: 4, 15: 4, 16: 4,
    17: 5, 18: 5, 19: 5, 20: 5,
}

FINGER_COLORS: List[Tuple[int, int, int]] = [
    (200, 200, 220),   # 0 palm - light grey-blue
    (180, 220, 100),   # 1 thumb - teal
    (255, 180, 255),   # 2 index - pink
    (60, 230, 255),    # 3 middle - yellow
    (140, 255, 140),   # 4 ring - light green
    (255, 200, 100),   # 5 pinky - sky blue
]

FINGERTIPS = {4, 8, 12, 16, 20}


# =============================================================================
# Stats container
# =============================================================================

@dataclass
class PerformanceStats:
    fps: float = 0.0
    frame_count: int = 0
    last_fps_at: float = 0.0
    detect_ms: float = 0.0
    gesture_latency_ms: float = 0.0

    def tick(self, now: float) -> None:
        self.frame_count += 1
        if self.last_fps_at == 0.0:
            self.last_fps_at = now
            return
        elapsed = now - self.last_fps_at
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_at = now

    def observe(self, field: str, value: float) -> None:
        previous = getattr(self, field)
        if previous == 0.0:
            setattr(self, field, value)
        else:
            setattr(self, field, previous * 0.85 + value * 0.15)


# =============================================================================
# Label helpers
# =============================================================================

def _action_label(action: str) -> str:
    return ACTION_LABELS.get(action, action.replace("_", " ").title())


def _gesture_label(gesture: str) -> str:
    return GESTURE_LABELS.get(gesture, gesture.replace("_", " "))


# =============================================================================
# Low-level drawing primitives
# =============================================================================

def _draw_alpha_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_viewfinder(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    corner: int = 18,
) -> None:
    """Camera-viewfinder L-corners instead of a full rectangle."""
    corner = min(corner, (x2 - x1) // 3, (y2 - y1) // 3)
    corner = max(6, corner)
    # top-left
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness, cv2.LINE_AA)
    # top-right
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness, cv2.LINE_AA)
    # bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness, cv2.LINE_AA)
    # bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness, cv2.LINE_AA)


def _draw_lock_icon(
    frame: np.ndarray,
    cx: int, cy: int,
    size: int,
    color: Tuple[int, int, int],
    unlocked: bool = False,
) -> None:
    """Tiny padlock icon centered at (cx, cy)."""
    body_w = size
    body_h = int(size * 0.7)
    shackle_r = max(3, size // 3)
    body_top = cy - body_h // 2 + shackle_r // 2
    body_bot = body_top + body_h
    body_left = cx - body_w // 2
    body_right = body_left + body_w

    cv2.rectangle(frame, (body_left, body_top), (body_right, body_bot), color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, (body_top + body_bot) // 2 + 1), 2, COLOR_PANEL_BG, -1, cv2.LINE_AA)

    shackle_cx = cx + (shackle_r if unlocked else 0)
    cv2.ellipse(
        frame, (shackle_cx, body_top), (shackle_r, shackle_r),
        0, 180, 360, color, 2, cv2.LINE_AA,
    )


def _draw_action_icon(
    frame: np.ndarray,
    action: str,
    cx: int, cy: int,
    size: int,
    color: Tuple[int, int, int],
) -> None:
    s = size // 2
    if action == "next_slide":
        pts = np.array([[cx - s, cy - s], [cx + s, cy], [cx - s, cy + s]], np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
    elif action == "previous_slide":
        pts = np.array([[cx + s, cy - s], [cx - s, cy], [cx + s, cy + s]], np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
    elif action == "start_slideshow":
        pts = np.array([[cx - s + 2, cy - s], [cx + s - 2, cy], [cx - s + 2, cy + s]], np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
    elif action == "exit_slideshow":
        cv2.line(frame, (cx - s, cy - s), (cx + s, cy + s), color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx - s, cy + s), (cx + s, cy - s), color, 3, cv2.LINE_AA)
    elif action == "blank_screen":
        cv2.rectangle(frame, (cx - s, cy - s), (cx + s, cy + s), color, -1, cv2.LINE_AA)
    elif action == "first_slide":
        bar = max(2, size // 6)
        cv2.rectangle(frame, (cx - s, cy - s), (cx - s + bar, cy + s), color, -1, cv2.LINE_AA)
        pts = np.array([[cx + s, cy - s], [cx - s + bar + 3, cy], [cx + s, cy + s]], np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
    elif action == "last_slide":
        bar = max(2, size // 6)
        cv2.rectangle(frame, (cx + s - bar, cy - s), (cx + s, cy + s), color, -1, cv2.LINE_AA)
        pts = np.array([[cx - s, cy - s], [cx + s - bar - 3, cy], [cx - s, cy + s]], np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
    else:
        cv2.circle(frame, (cx, cy), s, color, -1, cv2.LINE_AA)


# =============================================================================
# Public draw functions used by app.py
# =============================================================================

def draw_faces(
    frame: np.ndarray,
    tracks: List[FaceTrack],
    active_id: Optional[int],
) -> None:
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        if track.track_id == active_id:
            color = COLOR_ACTIVE
            thickness = 3
        elif track.identity != "Unknown":
            color = COLOR_AUTH
            thickness = 2
        else:
            color = COLOR_UNKNOWN
            thickness = 2

        _draw_viewfinder(frame, x1, y1, x2, y2, color, thickness=thickness, corner=20)

        # Identity pill below the face (or above if no room below)
        font = cv2.FONT_HERSHEY_DUPLEX
        label = track.identity
        sim_text = f"{track.similarity:.2f}"
        sharp_text = f"q{int(min(999, track.sharpness)):d}" if track.sharpness > 0 else ""
        (lw, lh), _ = cv2.getTextSize(label, font, 0.5, 1)
        (sw, _), _ = cv2.getTextSize(sim_text, font, 0.4, 1)
        (qw, _), _ = cv2.getTextSize(sharp_text, font, 0.38, 1)
        pad = 6
        gap = pad
        panel_w = lw + sw + (qw + gap if sharp_text else 0) + pad * 3
        panel_h = lh + pad * 2
        ly = y2 + 6
        if ly + panel_h > frame.shape[0] - 70:
            ly = y1 - panel_h - 6
        lx = max(8, min(x1, frame.shape[1] - panel_w - 8))

        _draw_alpha_rect(frame, lx, ly, lx + panel_w, ly + panel_h, COLOR_PANEL_BG, 0.82)
        cv2.rectangle(frame, (lx, ly), (lx + panel_w, ly + panel_h), color, 1, cv2.LINE_AA)
        cv2.putText(
            frame, label, (lx + pad, ly + pad + lh - 4),
            font, 0.5, color, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, sim_text, (lx + pad + lw + pad, ly + pad + lh - 4),
            font, 0.4, COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA,
        )
        if sharp_text:
            # Higher Laplacian variance = sharper. Color codes the value:
            # green (>=180), amber (>=80), red (<80).
            if track.sharpness >= 180:
                qc = COLOR_ACTIVE
            elif track.sharpness >= 80:
                qc = COLOR_UNKNOWN
            else:
                qc = COLOR_LOCKED
            cv2.putText(
                frame, sharp_text,
                (lx + pad + lw + pad + sw + gap, ly + pad + lh - 4),
                font, 0.38, qc, 1, cv2.LINE_AA,
            )


def draw_active_crown(
    frame: np.ndarray,
    active_track: Optional[FaceTrack],
    now: float,
) -> None:
    """Pulsing 'ACTIVE' indicator above the controller's face box."""
    if active_track is None:
        return
    x1, y1, x2, _ = active_track.bbox
    cx = (x1 + x2) // 2

    pulse = 0.75 + 0.25 * math.sin(now * 5.0)
    color = tuple(min(255, int(c * pulse)) for c in COLOR_ACTIVE)

    arrow_size = 10
    y_tip = max(arrow_size + 12, y1 - 8)
    pts = np.array([
        [cx - arrow_size, y_tip - arrow_size],
        [cx + arrow_size, y_tip - arrow_size],
        [cx, y_tip],
    ], np.int32)
    cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)

    text = "ACTIVE"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
    label_y = y_tip - arrow_size - 4
    if label_y - th < 60:
        return
    cv2.putText(
        frame, text, (cx - tw // 2, label_y),
        font, 0.4, color, 1, cv2.LINE_AA,
    )


def draw_hand_skeleton(
    frame: np.ndarray,
    landmarks: List[Tuple[float, float, float]],
    recognized: bool,
) -> None:
    if not landmarks or not recognized or len(landmarks) < 21:
        return
    h, w = frame.shape[:2]
    points = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]

    for a, b in HAND_CONNECTIONS:
        finger = FINGER_OF_LANDMARK.get(a, 0)
        color = FINGER_COLORS[finger]
        cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)

    for idx, pt in enumerate(points):
        finger = FINGER_OF_LANDMARK.get(idx, 0)
        color = FINGER_COLORS[finger]
        r = 5 if idx in FINGERTIPS else 3
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        if idx in FINGERTIPS:
            cv2.circle(frame, pt, r + 2, (255, 255, 255), 1, cv2.LINE_AA)


def draw_gesture_progress_ring(
    frame: np.ndarray,
    landmarks: List[Tuple[float, float, float]],
    candidate_action: str,
    candidate_since: float,
    stable_seconds: float,
    last_fired_at: float,
    now: float,
) -> None:
    """Circular progress arc around the wrist, fills as the gesture is held."""
    if not landmarks or len(landmarks) < 21:
        return
    h, w = frame.shape[:2]
    wrist = (int(landmarks[0][0] * w), int(landmarks[0][1] * h))
    mid = (int(landmarks[9][0] * w), int(landmarks[9][1] * h))
    hand_size = max(40.0, math.hypot(mid[0] - wrist[0], mid[1] - wrist[1]))
    radius = int(hand_size * 1.6)

    progress = 0.0
    if candidate_action and candidate_since > 0:
        progress = min(1.0, (now - candidate_since) / max(0.05, stable_seconds))

    fired_recently = last_fired_at > 0 and (now - last_fired_at) < 0.45

    if progress <= 0.02 and not fired_recently:
        return

    cv2.circle(frame, wrist, radius, (55, 60, 72), 2, cv2.LINE_AA)

    if fired_recently:
        ring_color = COLOR_PROG_FIRING
        end_angle = 270  # full ring
    else:
        ring_color = COLOR_PROG_FIRING if progress >= 1.0 else COLOR_PROG_PENDING
        end_angle = -90 + int(progress * 360)

    cv2.ellipse(
        frame, wrist, (radius, radius),
        0, -90, end_angle, ring_color, 3, cv2.LINE_AA,
    )


def draw_gesture_badge(
    frame: np.ndarray,
    landmarks: List[Tuple[float, float, float]],
    gesture_name: str,
    score: float,
    bound: bool,
    recognized: bool,
) -> None:
    """Small floating badge near the hand showing gesture name + confidence bar."""
    if not landmarks or not recognized or len(landmarks) < 21 or not gesture_name:
        return
    h, w = frame.shape[:2]
    wrist = (int(landmarks[0][0] * w), int(landmarks[0][1] * h))

    label = _gesture_label(gesture_name).upper()
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.45
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    pad_x, pad_y = 10, 6
    panel_w = max(tw + pad_x * 2, 80)
    panel_h = th + pad_y * 2 + 6  # extra for confidence bar

    bx = wrist[0] + 36
    by = max(64, wrist[1] - panel_h - 16)
    if bx + panel_w > w - 12:
        bx = wrist[0] - 36 - panel_w
    bx = max(10, bx)

    border = COLOR_ACCENT if bound else COLOR_TEXT_MUTED
    _draw_alpha_rect(frame, bx, by, bx + panel_w, by + panel_h, COLOR_PANEL_BG, 0.85)
    cv2.rectangle(frame, (bx, by), (bx + panel_w, by + panel_h), border, 1, cv2.LINE_AA)
    cv2.putText(
        frame, label, (bx + pad_x, by + pad_y + th - 2),
        font, scale, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA,
    )

    bar_y = by + panel_h - 4
    bar_x = bx + pad_x
    bar_w = panel_w - pad_x * 2
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 2), (60, 60, 72), -1)
    fill = int(bar_w * max(0.0, min(1.0, score)))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 2), border, -1)


def draw_action_toast(
    frame: np.ndarray,
    last_fired_action: str,
    last_fired_at: float,
    now: float,
) -> None:
    """Large centered pill that slides in when an action fires, then fades out."""
    if not last_fired_action or last_fired_at <= 0:
        return
    elapsed = now - last_fired_at
    total = 1.3
    if elapsed > total:
        return

    if elapsed < 0.08:
        alpha = elapsed / 0.08
        slide = int((1.0 - alpha) * 24)
    elif elapsed < 0.95:
        alpha = 1.0
        slide = 0
    else:
        alpha = max(0.0, 1.0 - (elapsed - 0.95) / 0.35)
        slide = 0

    if alpha <= 0:
        return

    label = _action_label(last_fired_action).upper()
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.85
    (tw, th), _ = cv2.getTextSize(label, font, scale, 2)
    icon_size = 22
    gap = 14
    pad_x, pad_y = 22, 14
    panel_w = icon_size + gap + tw + pad_x * 2
    panel_h = max(icon_size, th) + pad_y * 2
    x = (frame.shape[1] - panel_w) // 2
    y = 70 - slide

    # Background + border drawn into a temporary overlay so we can fade
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), COLOR_ACCENT, 2, cv2.LINE_AA)
    icon_cx = x + pad_x + icon_size // 2
    icon_cy = y + panel_h // 2
    _draw_action_icon(overlay, last_fired_action, icon_cx, icon_cy, icon_size, COLOR_ACCENT)
    text_y = y + panel_h // 2 + th // 2 - 2
    cv2.putText(
        overlay, label, (icon_cx + icon_size // 2 + gap, text_y),
        font, scale, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_status_bar(
    frame: np.ndarray,
    tracker: FaceTracker,
    dry_run: bool,
    gesture_status: str,
    gesture_engine_enabled: bool,
    now: float,
) -> None:
    w = frame.shape[1]
    bar_h = 58
    _draw_alpha_rect(frame, 0, 0, w, bar_h, COLOR_PANEL_BG, 0.82)

    unlocked = tracker.is_unlocked(now)
    accent = COLOR_ACTIVE if unlocked else COLOR_LOCKED
    cv2.rectangle(frame, (0, bar_h - 2), (w, bar_h), accent, -1)

    # --- Left: lock icon + identity title + lock reason -------------------
    _draw_lock_icon(frame, 24, 22, 14, accent, unlocked=unlocked)
    active = tracker.active_track(now)
    if unlocked and active is not None:
        title = active.identity.upper()
        subtitle = f"sim {active.similarity:.2f} - {tracker.lock_reason}"
    else:
        title = "LOCKED"
        subtitle = tracker.lock_reason
    cv2.putText(
        frame, title, (44, 24),
        cv2.FONT_HERSHEY_DUPLEX, 0.6, accent, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, subtitle[:70], (44, 44),
        cv2.FONT_HERSHEY_DUPLEX, 0.42, COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA,
    )

    # --- Right: mode badge + (optional) dry-run badge --------------------
    rx = w - 14

    def _draw_right_pill(text: str, fill: Optional[Tuple[int, int, int]],
                        border: Optional[Tuple[int, int, int]],
                        text_color: Tuple[int, int, int]) -> int:
        nonlocal rx
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)
        bx2 = rx
        bx1 = bx2 - tw - 18
        by1 = 10
        by2 = by1 + th + 12
        if fill is not None:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), fill, -1, cv2.LINE_AA)
        if border is not None:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), border, 1, cv2.LINE_AA)
        cv2.putText(
            frame, text, (bx1 + 9, by2 - 6),
            font, 0.45, text_color, 1, cv2.LINE_AA,
        )
        rx = bx1 - 10
        return bx1

    if dry_run:
        _draw_right_pill("DRY RUN", COLOR_DRYRUN_BG, None, COLOR_TEXT_PRIMARY)
    _draw_right_pill("PRESENTATION", None, COLOR_PANEL_BORDER, COLOR_TEXT_PRIMARY)

    # --- Sub-line: rolling gesture status / disabled notice ---------------
    if not gesture_engine_enabled:
        cv2.putText(
            frame, "Gestures disabled - install mediapipe and rerun",
            (w - 380, 44), cv2.FONT_HERSHEY_DUPLEX, 0.4, COLOR_LOCKED, 1, cv2.LINE_AA,
        )
    elif gesture_status:
        text = gesture_status[:54]
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, _), _ = cv2.getTextSize(text, font, 0.4, 1)
        cv2.putText(
            frame, text, (w - tw - 14, 44),
            font, 0.4, COLOR_TEXT_MUTED, 1, cv2.LINE_AA,
        )


def draw_legend(
    frame: np.ndarray,
    config: Dict[str, Any],
    current_gesture: str,
) -> None:
    """Top-right panel listing every binding. Current gesture row is highlighted."""
    if not bool(config.get("ui", {}).get("show_legend", True)):
        return
    bindings = config.get("bindings", {})
    if not bindings:
        return

    font = cv2.FONT_HERSHEY_DUPLEX
    title = "GESTURE CONTROLS"
    row_height = 22
    padding = 12

    rows = []
    max_left = 0
    max_right = 0
    for gesture, action in bindings.items():
        left = _gesture_label(gesture)
        right = _action_label(action)
        rows.append((gesture, left, right))
        (lw, _), _ = cv2.getTextSize(left, font, 0.44, 1)
        (rw, _), _ = cv2.getTextSize(right, font, 0.44, 1)
        max_left = max(max_left, lw)
        max_right = max(max_right, rw)

    (title_w, title_h), _ = cv2.getTextSize(title, font, 0.42, 1)
    panel_w = max(title_w, max_left + max_right + 56) + padding * 2 + 12
    panel_h = title_h + 10 + len(rows) * row_height + padding * 2

    x = frame.shape[1] - panel_w - 14
    y = 74

    _draw_alpha_rect(frame, x, y, x + panel_w, y + panel_h, COLOR_PANEL_BG, 0.82)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), COLOR_PANEL_BORDER, 1, cv2.LINE_AA)

    cv2.putText(
        frame, title, (x + padding, y + padding + title_h - 2),
        font, 0.42, COLOR_ACCENT, 1, cv2.LINE_AA,
    )
    sep_y = y + padding + title_h + 4
    cv2.line(
        frame, (x + padding, sep_y), (x + panel_w - padding, sep_y),
        COLOR_PANEL_BORDER, 1, cv2.LINE_AA,
    )

    ry = sep_y + 18
    arrow_x = x + padding + 18 + max_left + 10
    for gesture, left, right in rows:
        is_current = gesture == current_gesture
        if is_current:
            _draw_alpha_rect(
                frame,
                x + 4, ry - row_height + 6,
                x + panel_w - 4, ry + 4,
                COLOR_ACTIVE_DIM, 0.35,
            )
        bullet_color = COLOR_ACTIVE if is_current else COLOR_TEXT_MUTED
        cv2.circle(frame, (x + padding + 4, ry - 5), 3, bullet_color, -1, cv2.LINE_AA)
        text_color = COLOR_TEXT_PRIMARY if is_current else COLOR_TEXT_SECONDARY
        cv2.putText(
            frame, left, (x + padding + 16, ry),
            font, 0.44, text_color, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "->", (arrow_x, ry),
            font, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, right, (arrow_x + 24, ry),
            font, 0.44, text_color, 1, cv2.LINE_AA,
        )
        ry += row_height


def draw_performance(
    frame: np.ndarray,
    config: Dict[str, Any],
    stats: PerformanceStats,
) -> None:
    if not bool(config.get("ui", {}).get("show_performance", True)):
        return
    items = [
        ("FPS", f"{stats.fps:.0f}"),
        ("DETECT", f"{stats.detect_ms:.0f}ms"),
        ("LAG", f"{stats.gesture_latency_ms:.0f}ms"),
    ]
    font = cv2.FONT_HERSHEY_DUPLEX
    pad = 12
    item_w = 74
    panel_w = item_w * len(items) + pad
    panel_h = 42
    x = 14
    y = frame.shape[0] - panel_h - 14

    _draw_alpha_rect(frame, x, y, x + panel_w, y + panel_h, COLOR_PANEL_BG, 0.78)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), COLOR_PANEL_BORDER, 1, cv2.LINE_AA)

    cx = x + pad
    for label, value in items:
        cv2.putText(
            frame, label, (cx, y + 15),
            font, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, value, (cx, y + 33),
            font, 0.5, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA,
        )
        cx += item_w


def draw_capture_progress(
    frame: np.ndarray,
    capturing: bool,
    capture_name: str,
    capture_count: int,
    total: int,
    last_capture_at: float,
    now: float,
) -> None:
    if not capturing:
        return

    # Header pill with dot progress
    label = f"REGISTERING  {capture_name.upper()}"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    dot_step = 18
    pad = 16
    panel_w = tw + pad * 2 + dot_step * total + 18
    panel_h = 42
    x = (frame.shape[1] - panel_w) // 2
    y = 72

    _draw_alpha_rect(frame, x, y, x + panel_w, y + panel_h, COLOR_PANEL_BG, 0.92)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), COLOR_LOCKED, 2, cv2.LINE_AA)
    cv2.putText(
        frame, label, (x + pad, y + (panel_h + th) // 2 - 2),
        font, scale, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA,
    )

    dot_x0 = x + pad + tw + 18
    dot_y = y + panel_h // 2
    for i in range(total):
        cx = dot_x0 + i * dot_step
        if i < capture_count:
            cv2.circle(frame, (cx, dot_y), 6, COLOR_ACTIVE, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, dot_y), 6, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Hint line below the pill
    hint = "Hold steady and slowly tilt your head between samples"
    (hw, hh), _ = cv2.getTextSize(hint, font, 0.42, 1)
    hx = (frame.shape[1] - hw) // 2
    hy = y + panel_h + hh + 12
    cv2.putText(
        frame, hint, (hx, hy),
        font, 0.42, COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA,
    )

    # Brief green frame flash whenever a new sample lands
    if last_capture_at > 0 and (now - last_capture_at) < 0.22:
        a = 1.0 - (now - last_capture_at) / 0.22
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (frame.shape[1], frame.shape[0]),
            COLOR_ACTIVE, 10,
        )
        cv2.addWeighted(overlay, 0.35 * a, frame, 1 - 0.35 * a, 0, frame)


def draw_locked_indicator(
    frame: np.ndarray,
    tracker: FaceTracker,
    now: float,
    suppress: bool = False,
) -> None:
    """Subtle red border below the status bar when controls are locked."""
    if suppress or tracker.is_unlocked(now):
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 58), (w, h), COLOR_LOCKED, 3)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)


def draw_session_log(
    frame: np.ndarray,
    rows: List[List[str]],
) -> None:
    """Tiny bottom-right panel showing the last 3 logged actions."""
    if not rows:
        return
    font = cv2.FONT_HERSHEY_DUPLEX
    title = "RECENT ACTIONS"
    items: List[str] = []
    for row in rows[-3:]:
        if len(row) < 4:
            continue
        ts = row[0].split("T")[-1] if "T" in row[0] else row[0]
        gesture = row[2]
        action = row[3]
        items.append(f"{ts}  {_gesture_label(gesture)} -> {_action_label(action)}")
    if not items:
        return

    line_h = 16
    padding = 10
    (title_w, title_h), _ = cv2.getTextSize(title, font, 0.38, 1)
    widths = [cv2.getTextSize(s, font, 0.36, 1)[0][0] for s in items]
    panel_w = max(title_w, max(widths)) + padding * 2
    panel_h = title_h + 6 + len(items) * line_h + padding * 2
    x = frame.shape[1] - panel_w - 14
    # Lift above the centered help hint at the very bottom.
    y = frame.shape[0] - panel_h - 56

    _draw_alpha_rect(frame, x, y, x + panel_w, y + panel_h, COLOR_PANEL_BG, 0.78)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), COLOR_PANEL_BORDER, 1, cv2.LINE_AA)
    cv2.putText(
        frame, title, (x + padding, y + padding + title_h - 2),
        font, 0.38, COLOR_ACCENT, 1, cv2.LINE_AA,
    )
    ry = y + padding + title_h + line_h - 4
    for line in items:
        cv2.putText(
            frame, line, (x + padding, ry),
            font, 0.36, COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA,
        )
        ry += line_h


def draw_help_hint(frame: np.ndarray) -> None:
    """Small key-shortcut hint along the bottom-center."""
    text = "[A] add   [D] delete   [R] reload   [T] dry-run   [Q] quit"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.4
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    pad = 10
    x = (frame.shape[1] - tw - pad * 2) // 2
    y = frame.shape[0] - th - pad * 2 - 14
    _draw_alpha_rect(frame, x, y, x + tw + pad * 2, y + th + pad * 2, COLOR_PANEL_BG, 0.7)
    cv2.rectangle(
        frame, (x, y), (x + tw + pad * 2, y + th + pad * 2),
        COLOR_PANEL_BORDER, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (x + pad, y + pad + th - 2),
        font, scale, COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA,
    )
