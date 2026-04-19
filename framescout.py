import bisect
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QEvent, QTimer
from PySide6.QtGui import QImage, QPixmap, QIcon, QIntValidator, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


THUMB_WIDTH = 160
THUMB_HEIGHT = THUMB_WIDTH * 9 // 16  # default 16:9; aspect preserved during render


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    # Format_BGR888 lets us skip cv2.cvtColor — Qt reads the BGR buffer directly.
    h, w, _ = frame_bgr.shape
    return QImage(frame_bgr.data, w, h, 3 * w, QImage.Format_BGR888).copy()


def format_timestamp(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    if h:
        return f"{h:d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


class AspectLabel(QLabel):
    """QLabel that scales its pixmap to fit, preserving aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background-color: #111;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setFramePixmap(self, pm: QPixmap | None) -> None:
        self._pixmap = pm
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self.clear()
            return
        super().setPixmap(
            self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )


class ThumbnailWorker(QThread):
    """Thumbnail generator with a replaceable request queue.

    The queue holds frame indices the main window currently wants filled
    (typically: visible items that don't yet have an icon). `set_queue` can
    be called as often as the user scrolls — each call drops any pending
    work and waits for the new request list.
    """

    thumb_ready = Signal(int, QImage)

    def __init__(
        self, path: str, thumb_w: int, max_forward_grab: int, parent=None
    ):
        super().__init__(parent)
        self._path = path
        self._thumb_w = thumb_w
        # Past this many frames forward, seeking to the nearest keyframe is
        # cheaper than walking via grab(). Set per-video by the caller.
        self._max_forward_grab = max_forward_grab
        self._cond = threading.Condition()
        self._queue: list[int] = []
        self._stop = False

    def set_queue(self, indices: list[int]) -> None:
        with self._cond:
            self._queue = list(indices)
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def run(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            return
        try:
            while True:
                with self._cond:
                    while not self._stop and not self._queue:
                        self._cond.wait()
                    if self._stop:
                        return
                    batch = sorted(set(self._queue))
                    self._queue = []
                if not batch:
                    continue
                # Decode each target. For small gaps we walk forward with
                # grab(); for big gaps we re-seek so OpenCV jumps to the
                # nearest prior keyframe instead of decoding everything.
                cap.set(cv2.CAP_PROP_POS_FRAMES, batch[0])
                pos = batch[0]
                for target in batch:
                    # Preempt if stopped or a fresh queue arrived.
                    with self._cond:
                        if self._stop:
                            return
                        if self._queue:
                            break
                    forward = target - pos
                    if forward < 0 or forward > self._max_forward_grab:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        pos = target
                    else:
                        while pos < target:
                            if not cap.grab():
                                pos = target  # stream ended / corrupt
                                break
                            pos += 1
                    ok, frame = cap.read()
                    pos += 1
                    if not ok or frame is None:
                        continue
                    h, w = frame.shape[:2]
                    new_w = self._thumb_w
                    new_h = max(1, int(h * new_w / w))
                    thumb = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                    self.thumb_ready.emit(target, bgr_to_qimage(thumb))
        finally:
            cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FrameScout")
        self.resize(1200, 820)

        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.current_frame: int = 0
        self.step: int = 32
        self.range_start: int = 0
        self.range_end: int = 0  # inclusive; equals total_frames-1 when full
        self.max_forward_grab: int = 60  # benchmarked at video load
        self.worker: ThumbnailWorker | None = None
        self._item_by_index: dict[int, QListWidgetItem] = {}
        self._thumb_indices: list[int] = []

        self._visible_timer = QTimer(self)
        self._visible_timer.setSingleShot(True)
        self._visible_timer.setInterval(30)
        self._visible_timer.timeout.connect(self._update_visible_thumbs)

        # Coalesce rapid navigation (held arrow keys) so the main view only
        # decodes the latest target instead of backing up a queue of decodes.
        self._pending_frame: int | None = None
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(0)
        self._frame_timer.timeout.connect(self._render_pending_frame)
        # Tracks where the capture will read next; lets us grab() forward for
        # small jumps instead of doing an expensive keyframe-aligned seek.
        self._cap_pos: int = 0

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.open_btn = QPushButton("Open Video")
        self.open_btn.clicked.connect(self.open_video)
        top.addWidget(self.open_btn)

        top.addWidget(QLabel("Step (every N frames):"))
        self.step_minus_btn = QPushButton("−")
        self.step_minus_btn.setFixedWidth(32)
        self.step_minus_btn.setAutoRepeat(False)
        self.step_minus_btn.clicked.connect(lambda: self._nudge_step(-1))
        top.addWidget(self.step_minus_btn)

        self.step_label = QLabel(str(self.step))
        self.step_label.setAlignment(Qt.AlignCenter)
        self.step_label.setMinimumWidth(48)
        top.addWidget(self.step_label)

        self.step_plus_btn = QPushButton("+")
        self.step_plus_btn.setFixedWidth(32)
        self.step_plus_btn.setAutoRepeat(False)
        self.step_plus_btn.clicked.connect(lambda: self._nudge_step(+1))
        top.addWidget(self.step_plus_btn)

        top.addSpacing(16)
        top.addWidget(QLabel("Range:"))
        self.range_start_edit = QLineEdit()
        self.range_start_edit.setFixedWidth(80)
        self.range_start_edit.setValidator(QIntValidator(0, 10**9, self))
        self.range_start_edit.setPlaceholderText("start")
        self.range_start_edit.editingFinished.connect(self._commit_range)
        top.addWidget(self.range_start_edit)
        top.addWidget(QLabel("–"))
        self.range_end_edit = QLineEdit()
        self.range_end_edit.setFixedWidth(80)
        self.range_end_edit.setValidator(QIntValidator(0, 10**9, self))
        self.range_end_edit.setPlaceholderText("end")
        self.range_end_edit.editingFinished.connect(self._commit_range)
        top.addWidget(self.range_end_edit)
        self.range_reset_btn = QPushButton("Reset")
        self.range_reset_btn.clicked.connect(self._reset_range)
        top.addWidget(self.range_reset_btn)

        top.addStretch(1)

        self.info_label = QLabel("No video loaded")
        top.addWidget(self.info_label)
        root.addLayout(top)

        self.frame_view = AspectLabel()

        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        self.thumb_list.setFlow(QListWidget.LeftToRight)
        self.thumb_list.setWrapping(True)
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setMovement(QListWidget.Static)
        self.thumb_list.setIconSize(QSize(THUMB_WIDTH, THUMB_HEIGHT))
        self.thumb_list.setSpacing(6)
        self.thumb_list.setUniformItemSizes(True)
        self.thumb_list.setMinimumHeight(THUMB_HEIGHT + 70)
        # Arrow keys (Left/Right/Up/Down) move the list's current item and
        # currentItemChanged drives the main view update.
        self.thumb_list.setFocusPolicy(Qt.StrongFocus)
        self.thumb_list.currentItemChanged.connect(self._on_current_thumb_changed)

        # "-" halves the step, "=" doubles it (matches the on-screen buttons).
        for seq, delta in (("-", -1), ("=", +1)):
            sc = QShortcut(QKeySequence(seq), self.thumb_list)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.setAutoRepeat(False)
            sc.activated.connect(lambda d=delta: self._nudge_step(d))
        self.thumb_list.verticalScrollBar().valueChanged.connect(
            self._schedule_visible_update
        )
        self.thumb_list.horizontalScrollBar().valueChanged.connect(
            self._schedule_visible_update
        )
        self.thumb_list.viewport().installEventFilter(self)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.frame_view)
        splitter.addWidget(self.thumb_list)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStretchFactor(0, 1)  # main view absorbs window resizes
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([600, THUMB_HEIGHT + 70])
        root.addWidget(splitter, 1)

        self.setStatusBar(QStatusBar())
        self.setFocusPolicy(Qt.StrongFocus)

    # -------- video loading --------

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All files (*)",
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.statusBar().showMessage(f"Failed to open: {path}", 5000)
            return

        self._stop_worker()
        if self.cap is not None:
            self.cap.release()

        self.cap = cap
        self.video_path = path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.current_frame = 0
        self.range_start = 0
        self.range_end = max(self.total_frames - 1, 0)
        self.range_start_edit.setText(str(self.range_start))
        self.range_end_edit.setText(str(self.range_end))
        self._cap_pos = 0
        self._pending_frame = None
        self._frame_timer.stop()

        self.setWindowTitle(f"FrameScout — {Path(path).name}")
        if self.total_frames <= 0:
            self.statusBar().showMessage("Video reports zero frames", 5000)
            self.info_label.setText("Frame 0 / 0   --:--")
            self.frame_view.setFramePixmap(None)
            self.thumb_list.clear()
            return

        self.max_forward_grab = self._benchmark_seek_threshold()
        # Benchmark moved the cap; reset to a known position before display.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._cap_pos = 0
        self.statusBar().showMessage(
            f"Seek/grab threshold: {self.max_forward_grab} frames", 3000
        )

        self._show_frame(0)
        self._regen_thumbnails()
        self.thumb_list.setFocus()

    def _benchmark_seek_threshold(self) -> int:
        """Estimate forward-grab vs seek breakeven for this video.

        Times a burst of grab() calls (decode-only) and a handful of seeks
        to spread-out positions (seek + decode-from-keyframe). The ratio is
        roughly the GOP-driven breakeven: short-GOP videos get a small
        threshold, long-GOP / I-frame-only videos get a large one. Clamped
        to a sane band so weird timings can't break navigation.
        """
        if self.cap is None or self.total_frames < 100:
            return 60
        cap = self.cap
        n = self.total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 4)
        cap.read()  # prime decoder so the first grab isn't artificially slow
        grab_n = 30
        t0 = time.perf_counter()
        grabbed = 0
        for _ in range(grab_n):
            if not cap.grab():
                break
            grabbed += 1
        grab_avg = (time.perf_counter() - t0) / max(grabbed, 1)

        targets = [n // 6, n // 3, n // 2, (2 * n) // 3, (5 * n) // 6]
        seeked = 0
        t0 = time.perf_counter()
        for tgt in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
            ok, _ = cap.read()
            if ok:
                seeked += 1
        if seeked == 0 or grab_avg <= 0:
            return 60
        seek_avg = (time.perf_counter() - t0) / seeked

        return max(10, min(600, int(round(seek_avg / grab_avg))))

    # -------- frame display --------

    def _read_frame(self, index: int) -> np.ndarray | None:
        if self.cap is None:
            return None
        forward = index - self._cap_pos
        if 0 <= forward <= self.max_forward_grab:
            # Walk forward from current position: cheaper than re-seeking
            # to a prior keyframe and decoding back up to `index`.
            for _ in range(forward):
                if not self.cap.grab():
                    self._cap_pos = index
                    return None
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        self._cap_pos = index + 1 if ok else index
        return frame if ok else None

    def _show_frame(self, index: int) -> None:
        """Request a frame be displayed; coalesced so rapid calls don't back up.

        When arrow keys repeat, we may get events faster than we can decode.
        Instead of queuing a decode per event, we just record the most recent
        target and let a 0 ms timer render it on the next event-loop tick.
        Any further calls that arrive before the timer fires just overwrite
        the target — intermediate frames are skipped.
        """
        if self.cap is None or self.total_frames <= 0:
            return
        index = max(self.range_start, min(index, self.range_end))
        self._pending_frame = index
        if not self._frame_timer.isActive():
            self._frame_timer.start()

    def _render_pending_frame(self) -> None:
        if self.cap is None or self._pending_frame is None:
            return
        index = self._pending_frame
        self._pending_frame = None
        frame = self._read_frame(index)
        if frame is None:
            return
        self.frame_view.setFramePixmap(QPixmap.fromImage(bgr_to_qimage(frame)))
        self.current_frame = index
        self._update_info()
        self._sync_thumb_selection()
        # If more navigation piled up during this decode, render again.
        if self._pending_frame is not None and self._pending_frame != index:
            self._frame_timer.start()

    def _sync_thumb_selection(self) -> None:
        """Highlight the thumbnail whose sampled frame is nearest (≤) current."""
        if not self._thumb_indices:
            return
        pos = bisect.bisect_right(self._thumb_indices, self.current_frame) - 1
        if pos < 0:
            pos = 0
        idx = self._thumb_indices[pos]
        item = self._item_by_index.get(idx)
        if item is None:
            return
        if self.thumb_list.currentItem() is not item:
            self.thumb_list.setCurrentItem(item)
        self.thumb_list.scrollToItem(item, QListWidget.EnsureVisible)

    def _update_info(self) -> None:
        ts = format_timestamp(self.current_frame / self.fps) if self.fps > 0 else "--:--"
        self.info_label.setText(
            f"Frame {self.current_frame} / {max(self.total_frames - 1, 0)}   {ts}"
        )

    # -------- thumbnails --------

    def _commit_range(self) -> None:
        if self.cap is None or self.total_frames <= 0:
            return
        last = self.total_frames - 1
        try:
            s = int(self.range_start_edit.text())
            e = int(self.range_end_edit.text())
        except ValueError:
            self.range_start_edit.setText(str(self.range_start))
            self.range_end_edit.setText(str(self.range_end))
            return
        s = max(0, min(s, last))
        e = max(0, min(e, last))
        if s > e:
            s, e = e, s
        self.range_start_edit.setText(str(s))
        self.range_end_edit.setText(str(e))
        if s == self.range_start and e == self.range_end:
            return
        self.range_start = s
        self.range_end = e
        if self.current_frame < s or self.current_frame > e:
            self._show_frame(s)
        self._regen_thumbnails()
        self.thumb_list.setFocus()

    def _reset_range(self) -> None:
        if self.cap is None or self.total_frames <= 0:
            return
        last = self.total_frames - 1
        if self.range_start == 0 and self.range_end == last:
            self.thumb_list.setFocus()
            return
        self.range_start = 0
        self.range_end = last
        self.range_start_edit.setText(str(self.range_start))
        self.range_end_edit.setText(str(self.range_end))
        self._regen_thumbnails()
        self.thumb_list.setFocus()

    def _nudge_step(self, delta: int) -> None:
        # "-" halves the step, "+" doubles it. Floor at 1.
        if delta < 0:
            n = max(1, self.step // 2)
        else:
            n = self.step * 2
        if n == self.step:
            return
        self.step = n
        self.step_label.setText(str(n))
        if self.cap is not None:
            self._regen_thumbnails()
        self.thumb_list.setFocus()

    def _regen_thumbnails(self) -> None:
        self._stop_worker()
        self.thumb_list.clear()
        self._item_by_index = {}
        self._thumb_indices = []
        if self.cap is None or self.total_frames <= 0 or not self.video_path:
            return

        indices = list(range(self.range_start, self.range_end + 1, self.step))
        self._thumb_indices = indices
        # Taller caption area for the second line (timestamp).
        item_size = QSize(THUMB_WIDTH + 16, THUMB_HEIGHT + 52)
        for idx in indices:
            ts = format_timestamp(idx / self.fps) if self.fps > 0 else ""
            label = f"{idx}\n{ts}" if ts else str(idx)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, idx)
            item.setSizeHint(item_size)
            item.setTextAlignment(Qt.AlignCenter)
            self.thumb_list.addItem(item)
            self._item_by_index[idx] = item

        self.worker = ThumbnailWorker(
            self.video_path, THUMB_WIDTH, self.max_forward_grab, self
        )
        self.worker.thumb_ready.connect(self._on_thumb_ready)
        self.worker.start()
        self._sync_thumb_selection()
        # Defer so QListWidget has laid out its items before we query geometry.
        QTimer.singleShot(0, self._update_visible_thumbs)

    def _on_thumb_ready(self, idx: int, qimg: QImage) -> None:
        item = self._item_by_index.get(idx)
        if item is None:
            return
        item.setIcon(QIcon(QPixmap.fromImage(qimg)))

    def _schedule_visible_update(self) -> None:
        self._visible_timer.start()

    def _update_visible_thumbs(self) -> None:
        """Queue thumbnail generation for items that are (near-)visible and empty."""
        if self.worker is None or not self._thumb_indices:
            return
        viewport = self.thumb_list.viewport()
        v_rect = viewport.rect()
        if v_rect.isEmpty():
            return
        # Prefetch one extra row above and below the viewport so scrolling
        # a little doesn't momentarily reveal a blank thumbnail.
        margin = THUMB_HEIGHT + 40
        probe_rect = v_rect.adjusted(0, -margin, 0, margin)
        center_y = v_rect.center().y()

        needed: list[tuple[int, int]] = []  # (distance_from_center, frame_idx)
        for frame_idx in self._thumb_indices:
            item = self._item_by_index.get(frame_idx)
            if item is None or not item.icon().isNull():
                continue
            rect = self.thumb_list.visualItemRect(item)
            if not rect.intersects(probe_rect):
                continue
            needed.append((abs(rect.center().y() - center_y), frame_idx))

        needed.sort()
        self.worker.set_queue([idx for _, idx in needed])

    def eventFilter(self, obj, event):
        if (
            event.type() == QEvent.Resize
            and obj is self.thumb_list.viewport()
        ):
            self._schedule_visible_update()
        return super().eventFilter(obj, event)

    def _on_current_thumb_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        idx = current.data(Qt.UserRole)
        if idx is None:
            return
        idx = int(idx)
        if idx == self.current_frame:
            return
        self._show_frame(idx)

    def _stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

    # -------- cleanup --------

    def closeEvent(self, event) -> None:
        self._stop_worker()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
