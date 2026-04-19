# FrameScout

A small desktop tool for scrubbing through a video frame-by-frame. Open a
video, sample it as a thumbnail strip, jump to specific frames, and export
the ones you want as image files.

## Features

- Lazy thumbnail strip — only frames near the viewport are decoded.
- Adjustable sampling step (every Nth frame), in powers of two.
- Range filter to restrict navigation to a `[start, end]` frame window.
- Multi-frame selection via drag-rectangle, shift+click, or shift+arrow.
- Export selected frames as PNG or JPG to a chosen folder.
- Adaptive seek strategy that benchmarks each video's decode cost on load.

## Requirements

- Python 3.12 or newer
- OpenCV (`opencv-python`)
- NumPy
- PySide6

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python framescout.py
```

Supported containers: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`.

## Usage

Click **Open Video** to load a file. The main view shows one frame; the
thumbnail strip below samples the video at the current step.

### Stepping

The strip shows every Nth frame, where N is the **Step** value.

- **+** doubles the step (sparser thumbnails); **−** halves it.
- Keyboard: `=` doubles, `-` halves.

### Range filter

Restrict both the thumbnails and main-view navigation to a frame window:

- Type a **start** and **end** frame number; press Enter to apply.
- **Reset** restores the full video range.

### Selecting frames

The thumbnail strip uses extended selection:

- **Click** a thumbnail to view that frame (single-select).
- **Shift+click** another thumbnail to select the inclusive range.
- **Drag** across thumbnails for a rubber-band selection.
- **Shift+arrow** extends the selection one frame at a time.
- **Ctrl+click** toggles an individual thumbnail.

### Exporting

Click **Export Selected** to save the chosen frames at full resolution:

1. Pick a destination folder.
2. Pick a format (**png** default, or **jpg**).
3. A progress dialog shows write progress; **Cancel** stops it.

Files are named `{videoname}_frame_{index:06d}.{ext}`.

## How it works

On load, FrameScout times a short burst of sequential `grab()` decodes
against a handful of full seeks to estimate the per-video grab-vs-seek
breakeven. That threshold drives both the main view's frame fetcher and
the thumbnail worker, so navigation stays fast on long-GOP files (where a
forward walk would otherwise decode hundreds of frames per jump) without
penalising short-GOP files (where seeks are needlessly expensive).

Thumbnails are generated in a background thread with a replaceable
request queue: scrolling cancels in-flight requests and replaces them
with the new viewport's needs, so the strip stays responsive even on
large videos.
