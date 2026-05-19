# Vision Gesture Presentation Control

A computer-vision app that lets **only you** control PowerPoint slides with
hand gestures from your webcam. Any other person in the frame is ignored.

Recognized hand poses are classified by a **custom geometric rule set**
(finger-state analysis on the 21 MediaPipe hand landmarks) instead of relying
on MediaPipe's built-in gesture classifier. This makes new gestures easy to
add and gives you full control over the recognition rules.

The on-screen HUD shows your identity, face-quality (Laplacian-variance
sharpness), a color-coded hand skeleton, a circular progress ring around the
wrist while a gesture is being held, an animated toast whenever an action
fires, and a live tail of the recent-actions CSV log.

## What it does

1. Opens your webcam.
2. Detects every face with **OpenCV YuNet**.
3. Recognises identity with **OpenCV SFace** against the images in
   `data/db/<your_name>/`.
4. Picks one "active controller" - the largest, most central, **authorized**
   face. Locks if it's ambiguous or if an unknown face is more foreground.
5. Reads **MediaPipe** hand landmarks from the active controller's hand and
   classifies the pose with a **custom geometric classifier** based on
   per-finger extension/curl tests.
6. After the same gesture stays stable for ~0.35 s, sends a key press to
   PowerPoint via **PyAutoGUI** and appends a row to `data/session_log.csv`.

## Recognized gestures

| Gesture       | How it is detected (finger states T,I,M,R,P)        |
| ------------- | --------------------------------------------------- |
| `Closed_Fist` | All fingers curled                                  |
| `Open_Palm`   | All fingers extended                                |
| `Thumb_Up`    | Only thumb extended, thumb tip above wrist          |
| `Thumb_Down`  | Only thumb extended, thumb tip below wrist          |
| `Pointing_Up` | Only index extended                                 |
| `Victory`     | Index + middle extended                             |
| `Three`       | Index + middle + ring extended (**new**)            |
| `OK`          | Thumb tip touches index tip, other three extended (**new**) |
| `ILoveYou`    | Thumb + index + pinky extended                      |

## Default gesture-to-action mapping

| Gesture        | Action          | Key sent      |
| -------------- | --------------- | ------------- |
| Thumb Up       | Next slide      | `Right`       |
| Thumb Down     | Previous slide  | `Left`        |
| Victory        | Start slideshow | `F5`          |
| Closed Fist    | Exit slideshow  | `Esc`         |
| Open Palm      | Blank screen    | `B`           |
| Three          | First slide     | `Home`        |
| OK             | Last slide      | `End`         |

Change them in `config/gesture_config.json` under `bindings` and
`actions.key_profile`. No code changes needed.

## Requirements

* Python 3.11 recommended
* A webcam
* Windows is the primary tested platform (window-focus uses `pygetwindow`)

Install:

```powershell
python -m pip install -r requirements.txt
```

Or in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

1. `python main.py`
2. First run downloads three small model files into `models/`
   (YuNet ~230 KB, SFace ~36 MB, MediaPipe gesture ~9 MB).
3. Press **`a`**, type your name, hold still while it captures 5 face samples.
4. Open PowerPoint, load your deck, click the slide once to focus the window.
5. Show **Victory** to start the slideshow, then **Thumb Up / Thumb Down**
   to navigate. **Closed Fist** exits.

## Keyboard shortcuts (inside the camera window)

| Key | Action                                         |
| --- | ---------------------------------------------- |
| `q` | Quit                                           |
| `a` | Register a new authorized person (5 samples)   |
| `d` | Delete a registered person                     |
| `r` | Reload the face database                       |
| `t` | Toggle dry-run mode (no real key presses)      |

## Project layout

```
main.py                                # Thin launcher
requirements.txt                       # 5 deps, no PyTorch / TF / Vosk
config/
  gesture_config.json                  # Auto-created on first run
data/
  db/<person>/sample_X.jpg             # Your 5 face samples (gitignored)
  session_log.csv                      # Action log (gitignored)
models/                                # Auto-downloaded ONNX + MediaPipe (gitignored)
src/vision_gesture_control/
  __init__.py
  config.py        # Defaults + JSON load/save
  models.py        # Auto-download YuNet / SFace / MediaPipe
  face.py          # Detector, recognizer, FaceIndex, FaceTracker
  gestures.py      # MediaPipe wrapper + custom geometric classifier
  actions.py       # PowerPoint keystroke sender (PyAutoGUI)
  session_log.py   # CSV logger for fired actions
  ui.py            # OpenCV overlay drawing (HUD)
  app.py           # Main loop wiring it all together
```

## Why does it only work for "me"?

The `FaceTracker._select_active` method in `src/vision_gesture_control/face.py`
makes that guarantee:

* The active controller is the most prominent **authorized** face
  (largest + most central in frame).
* If two authorized people are roughly equally prominent
  (within `face.ambiguous_rank_gap`), the system **locks**.
* If an unknown face is significantly larger than the best authorized one
  (`face.foreground_override_ratio`, default 1.12 x), the system **locks**.
* The hand has to be closer to the active face than to anyone else, or the
  gesture is ignored (`hand_belongs_to_active`).

The on-screen status bar always tells you why controls are locked.

## How the geometric classifier works

`GeometricGestureClassifier` in `src/vision_gesture_control/gestures.py`
analyses the 21 hand landmarks returned by MediaPipe and computes 5 booleans:
which fingers are extended (thumb, index, middle, ring, pinky).

* A non-thumb finger is "extended" when its TIP is farther from the wrist
  than its PIP joint by `finger_extended_ratio` (default 1.10). This works
  regardless of hand rotation.
* The thumb is "extended" when its TIP is farther from the pinky MCP than
  the thumb MCP is, scaled by `thumb_extended_ratio` (default 1.20).
* The OK sign is checked first as a special case: thumb tip and index tip
  must be within `ok_pinch_ratio` (default 0.40) of the hand size of each
  other, and the other three fingers must be extended.

The resulting finger-state tuple is mapped to a canonical gesture name by a
short lookup table; the action resolver then enforces stability and cooldown
before firing.

## Session log

Every fired action is appended to `data/session_log.csv`:

```
timestamp_iso,actor,gesture,action,status
2026-05-19T18:28:31,Ziad,Thumb_Up,next_slide,DRY RUN: next_slide -> right
2026-05-19T18:28:32,Ziad,Thumb_Down,previous_slide,DRY RUN: previous_slide -> left
...
```

The HUD shows the last 3 entries in a `RECENT ACTIONS` panel.

Disable logging by setting `logging.enabled` to `false` in
`config/gesture_config.json`. The file is gitignored.

## Configuration

`config/gesture_config.json` is generated on first run. Key knobs:

| Setting                              | Purpose                                            |
| ------------------------------------ | -------------------------------------------------- |
| `camera.primary_index`               | Main webcam index                                  |
| `camera.mirror`                      | Mirror the camera feed                             |
| `face.sface_cosine_threshold`        | Identity acceptance threshold (lower = strict)     |
| `face.ambiguous_rank_gap`            | When to lock between two authorized people         |
| `face.foreground_override_ratio`     | When an unknown foreground face locks controls     |
| `face.quality_refresh_seconds`       | How often to recompute the Laplacian sharpness     |
| `gestures.confidence_threshold`      | Minimum classifier confidence                      |
| `gestures.stable_seconds`            | How long the gesture must stay before firing       |
| `gestures.action_cooldown_seconds`   | Minimum time between repeated actions              |
| `gestures.finger_extended_ratio`     | Threshold for "non-thumb finger extended" rule     |
| `gestures.thumb_extended_ratio`      | Threshold for "thumb extended" rule                |
| `gestures.ok_pinch_ratio`            | How close thumb/index tips must be for the OK sign |
| `gestures.thumb_up_y_offset`         | Thumb tip offset above wrist for Thumb_Up vs Down  |
| `bindings`                           | Gesture name -> action name                        |
| `actions.key_profile`                | Action name -> list of keys to press               |
| `actions.dry_run`                    | Show what would happen without pressing keys       |
| `actions.target_window_titles`       | Window titles considered "PowerPoint"              |
| `logging.enabled`                    | Append fired actions to `data/session_log.csv`     |

## Troubleshooting

* **Webcam doesn't open** - close other apps that may be using it; try
  changing `camera.primary_index` to 1 or 2.
* **It never recognises me** - re-register with `a` in good light, capture
  samples from slightly different angles. Try raising
  `face.sface_cosine_threshold` slightly (e.g. 0.40).
* **A gesture isn't being detected** - the per-finger thresholds are tunable.
  Lower `gestures.finger_extended_ratio` (e.g. 1.05) to be more lenient,
  raise it (e.g. 1.20) to be stricter.
* **Gestures detected but no slide change** - click on the PowerPoint window
  first so it's focused. Use `t` (dry-run) to confirm the gesture is mapping
  correctly before sending keys.
* **First start is slow** - it's downloading three model files; subsequent
  starts are instant.

## Stack

* **OpenCV YuNet** - lightweight ONNX face detector (~230 KB) returning 5 landmarks
* **OpenCV SFace** - 128-D face recognition embedding compared with cosine similarity
* **MediaPipe Hand Landmarker** - 21 3D landmarks per hand (the bundled gesture
  classifier head is NOT used; a custom geometric classifier replaces it)
* **PyAutoGUI** + **pygetwindow** - OS-level keystroke injection and PowerPoint
  window focusing
