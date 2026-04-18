import bisect
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QEvent, QTimer
from PySide6.QtGui import QImage, QPixmap, QIcon, QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


THUMB_WIDTH = 160
THUMB_HEIGHT = THUMB_WIDTH * 9 // 16  # default 16:9; aspect preserved during render


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


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

    def __init__(self, path: str, thumb_w: int, parent=None):
        super().__init__(parent)
        self._path = path
        self._thumb_w = thumb_w
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
                    idx = self._queue.pop(0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                new_w = self._thumb_w
                new_h = max(1, int(h * new_w / w))
                thumb = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.thumb_ready.emit(idx, bgr_to_qimage(thumb))
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
        self.step: int = 10
        self.worker: ThumbnailWorker | None = None
        self._item_by_index: dict[int, QListWidgetItem] = {}
        self._thumb_indices: list[int] = []

        self._visible_timer = QTimer(self)
        self._visible_timer.setSingleShot(True)
        self._visible_timer.setInterval(30)
        self._visible_timer.timeout.connect(self._update_visible_thumbs)

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
        self.step_combo = QComboBox()
        self.step_combo.setEditable(True)
        # Don't auto-insert typed values as new dropdown entries.
        self.step_combo.setInsertPolicy(QComboBox.NoInsert)
        for n in (1, 3, 5, 10):
            self.step_combo.addItem(str(n), n)
        self.step_combo.setCurrentIndex(3)  # default: 10
        self.step_combo.lineEdit().setValidator(QIntValidator(1, 10**9, self))
        # activated fires on dropdown pick; editingFinished fires on Enter /
        # focus loss after typing. Both funnel into the same commit path.
        self.step_combo.activated.connect(lambda _: self._commit_step())
        self.step_combo.lineEdit().editingFinished.connect(self._commit_step)
        self.step_combo.setFixedWidth(80)
        top.addWidget(self.step_combo)

        top.addStretch(1)

        self.info_label = QLabel("No video loaded")
        top.addWidget(self.info_label)
        root.addLayout(top)

        self.frame_view = AspectLabel()
        root.addWidget(self.frame_view, 1)

        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        self.thumb_list.setFlow(QListWidget.LeftToRight)
        self.thumb_list.setWrapping(True)
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setMovement(QListWidget.Static)
        self.thumb_list.setIconSize(QSize(THUMB_WIDTH, THUMB_HEIGHT))
        self.thumb_list.setSpacing(6)
        self.thumb_list.setUniformItemSizes(True)
        self.thumb_list.setFixedHeight(THUMB_HEIGHT + 70)
        # Arrow keys (Left/Right/Up/Down) move the list's current item and
        # currentItemChanged drives the main view update.
        self.thumb_list.setFocusPolicy(Qt.StrongFocus)
        self.thumb_list.currentItemChanged.connect(self._on_current_thumb_changed)
        self.thumb_list.verticalScrollBar().valueChanged.connect(
            self._schedule_visible_update
        )
        self.thumb_list.horizontalScrollBar().valueChanged.connect(
            self._schedule_visible_update
        )
        self.thumb_list.viewport().installEventFilter(self)
        root.addWidget(self.thumb_list)

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

        self.setWindowTitle(f"FrameScout — {Path(path).name}")
        if self.total_frames <= 0:
            self.statusBar().showMessage("Video reports zero frames", 5000)
            self.info_label.setText("Frame 0 / 0   --:--")
            self.frame_view.setFramePixmap(None)
            self.thumb_list.clear()
            return

        self._show_frame(0)
        self._regen_thumbnails()
        self.thumb_list.setFocus()

    # -------- frame display --------

    def _read_frame(self, index: int) -> np.ndarray | None:
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        return frame if ok else None

    def _show_frame(self, index: int) -> None:
        if self.cap is None or self.total_frames <= 0:
            return
        index = max(0, min(index, self.total_frames - 1))
        frame = self._read_frame(index)
        if frame is None:
            return
        self.frame_view.setFramePixmap(QPixmap.fromImage(bgr_to_qimage(frame)))
        self.current_frame = index
        self._update_info()
        self._sync_thumb_selection()

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

    def _commit_step(self) -> None:
        text = self.step_combo.currentText().strip()
        try:
            n = int(text)
        except ValueError:
            n = self.step
        if n < 1:
            n = self.step
        # Reflect the accepted value in the editor (in case we rejected input).
        if text != str(n):
            self.step_combo.blockSignals(True)
            self.step_combo.setCurrentText(str(n))
            self.step_combo.blockSignals(False)
        if n == self.step:
            return
        self.step = n
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

        indices = list(range(0, self.total_frames, self.step))
        self._thumb_indices = indices
        item_size = QSize(THUMB_WIDTH + 16, THUMB_HEIGHT + 36)
        for idx in indices:
            item = QListWidgetItem(str(idx))
            item.setData(Qt.UserRole, idx)
            item.setSizeHint(item_size)
            item.setTextAlignment(Qt.AlignCenter)
            self.thumb_list.addItem(item)
            self._item_by_index[idx] = item

        self.worker = ThumbnailWorker(self.video_path, THUMB_WIDTH, self)
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
