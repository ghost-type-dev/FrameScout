# FrameScout

FrameScout is a small desktop video frame browser. It opens a video file, shows
one frame at a time, and lazily generates thumbnail previews for sampled frames.

## Requirements

- Python 3.12 or newer
- OpenCV
- NumPy
- PySide6

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python framescout.py
```

Use **Open Video** to choose a video file, then select thumbnails to jump between
frames. The **Step** dropdown controls how densely thumbnails are sampled.
