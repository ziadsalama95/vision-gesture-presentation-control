# Vision Gesture Presentation Control

A small computer-vision app that lets **only you** control PowerPoint slides
with hand gestures from your webcam. Any other person in the frame is ignored.

The on-screen HUD shows your identity, a color-coded hand skeleton, a circular
progress ring around the wrist while a gesture is being held, and an animated
toast whenever an action fires.

## What it does

1. Opens your webcam.
2. Detects every face with **OpenCV YuNet**.
3. Recognises identity with **OpenCV SFace** against the images in
   `data/db/<your_name>/`.
4. Picks one "active controller" — the largest, most central, **authorized**
   face. Locks if it's ambiguous or if an unknown face is more foreground.
5. Reads **MediaPipe** hand gestures from the active controller's hand.
6. After the same gesture stays stable for ~0.35 s, sends a key press to
   PowerPoint via **PyAutoGUI** (`right`, `left`, `F5`, `Esc`).

## Default gesture mapping

| Gesture        | Action          | Key sent  |
| -------------- | --------------- | --------- |
| Thumb Up       | Next slide      | `Right`   |
| Thumb Down     | Previous slide  | `Left`    |
| Victory        | Start slideshow | `F5`      |
| Closed Fist    | Exit slideshow  | `Esc`     |

Change them anytime in `config/gesture_config.json` under `bindings` and
`actions.key_profile`.

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
models/                                # Auto-downloaded ONNX + MediaPipe (gitignored)
src/vision_gesture_control/
  __init__.py
  config.py     # Defaults + JSON load/save
  models.py     # Auto-download YuNet / SFace / MediaPipe gesture
  face.py       # Detector, recognizer, FaceIndex, FaceTracker
  gestures.py   # MediaPipe engine + stability/cooldown resolver
  actions.py    # PowerPoint keystroke sender (PyAutoGUI)
  ui.py         # OpenCV overlay drawing
  app.py        # Main loop wiring it all together
```

## Why does it only work for "me"?

The `FaceTracker._select_active` method in `src/vision_gesture_control/face.py`
makes that guarantee:

* The active controller is the most prominent **authorized** face
  (largest + most central in frame).
* If two authorized people are roughly equally prominent
  (within `face.ambiguous_rank_gap`), the system **locks**.
* If an unknown face is significantly larger than the best authorized one
  (`face.foreground_override_ratio`, default 1.12 ×), the system **locks**.
* The hand has to be closer to the active face than to anyone else, or the
  gesture is ignored (`hand_belongs_to_active`).

The on-screen status bar always tells you why controls are locked.

## Configuration

`config/gesture_config.json` is generated on first run. You can edit it freely.
Everything is hot-reloadable on restart. Key knobs:

| Setting                              | Purpose                                            |
| ------------------------------------ | -------------------------------------------------- |
| `camera.primary_index`               | Main webcam index                                  |
| `camera.mirror`                      | Mirror the camera feed                             |
| `face.sface_cosine_threshold`        | Identity acceptance threshold (lower = strict)     |
| `face.ambiguous_rank_gap`            | When to lock between two authorized people         |
| `face.foreground_override_ratio`     | When an unknown foreground face locks controls     |
| `gestures.confidence_threshold`      | Minimum MediaPipe gesture confidence               |
| `gestures.stable_seconds`            | How long the gesture must stay before firing       |
| `gestures.action_cooldown_seconds`   | Minimum time between repeated actions              |
| `bindings`                           | Gesture name -> action name                        |
| `actions.key_profile`                | Action name -> list of keys to press               |
| `actions.dry_run`                    | Show what would happen without pressing keys       |
| `actions.target_window_titles`       | Window titles considered "PowerPoint"              |

## Troubleshooting

* **Webcam doesn't open** - close other apps that may be using it; try
  changing `camera.primary_index` to 1 or 2.
* **It never recognises me** - re-register with `a` in good light, capture
  samples from slightly different angles. Try raising
  `face.sface_cosine_threshold` slightly (e.g. 0.40).
* **Gestures detected but no slide change** - click on the PowerPoint window
  first so it's focused. Use `t` (dry-run) to confirm the gesture is mapping
  correctly before sending keys.
* **First start is slow** - it's downloading three model files; subsequent
  starts are instant.

## Stack

* **OpenCV YuNet** - lightweight ONNX face detector (~230 KB) returning 5 landmarks
* **OpenCV SFace** - 128-D face recognition embedding compared with cosine similarity
* **MediaPipe Gesture Recognizer** - hand-landmark + classifier head, 8 built-in classes
* **PyAutoGUI** + **pygetwindow** - OS-level keystroke injection and PowerPoint window focusing
