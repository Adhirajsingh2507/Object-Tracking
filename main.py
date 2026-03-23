"""
YOLOv8 + DeepSORT Real-Time Object Tracker — PySide6 GUI
---------------------------------------------------------
Threaded video capture, ONNX-ready inference, frame skipping,
resolution control, and live FPS overlay.

Usage:
    python main.py
"""

import sys
from queue import Queue, Empty

import cv2
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QComboBox,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from core import TrackerCore


# ──────────────────────────────────────────────
# Capture Thread — reads frames in background
# ──────────────────────────────────────────────
class CaptureThread(QThread):
    """Reads frames from a cv2.VideoCapture in a background thread."""

    frame_ready = Signal(object)  # emits numpy frame

    def __init__(self, source, queue: Queue, parent=None):
        super().__init__(parent)
        self.source = source
        self.queue = queue
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.source}")
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                # End of video — stop (no infinite loop)
                break
            # Drop old frames if queue is full (keep latest)
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
            self.queue.put(frame)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()


# ──────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # -------- WINDOW --------
        self.setWindowTitle("YOLOv8 + DeepSORT Tracker")
        self.setGeometry(100, 100, 960, 600)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QPushButton {
                padding: 8px 14px;
                font-size: 13px;
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #45475a; }
            QPushButton:pressed { background-color: #585b70; }
            QComboBox {
                padding: 6px;
                font-size: 13px;
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
            }
        """)

        # -------- VIDEO DISPLAY --------
        self.video_label = QLabel("No Video — Upload a file or use your camera")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #11111b; color: #6c7086; font-size: 16px;"
        )
        self.video_label.setMinimumHeight(400)

        # -------- BUTTONS --------
        self.upload_btn = QPushButton("📂 Upload Video")
        self.camera_btn = QPushButton("📷 Camera")
        self.start_btn = QPushButton("▶ Start Tracking")
        self.stop_tracking_btn = QPushButton("⏸ Stop Tracking")
        self.stop_btn = QPushButton("⏹ Stop Video")

        # -------- RESOLUTION SELECTOR --------
        self.res_combo = QComboBox()
        self.res_combo.blockSignals(True)  # prevent signal during init
        self.res_combo.addItems(["320px (fastest)", "480px (balanced)", "640px (quality)"])
        self.res_combo.setCurrentIndex(1)  # default: 480
        self.res_combo.blockSignals(False)
        self.res_combo.currentIndexChanged.connect(self._on_res_change)

        # -------- LAYOUT --------
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.camera_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_tracking_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.res_combo)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # -------- TRACKER CORE --------
        self._init_tracker(process_width=480)
        self.tracking_enabled = False

        # -------- CAPTURE THREAD --------
        self.frame_queue: Queue = Queue(maxsize=2)
        self.capture_thread = None

        # -------- DISPLAY TIMER --------
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        # -------- BUTTON CONNECTIONS --------
        self.upload_btn.clicked.connect(self._open_video)
        self.camera_btn.clicked.connect(self._use_camera)
        self.start_btn.clicked.connect(self._start_tracking)
        self.stop_tracking_btn.clicked.connect(self._stop_tracking)
        self.stop_btn.clicked.connect(self._stop_video)

    # ------------------------------------------------------------------
    # Tracker init
    # ------------------------------------------------------------------
    def _init_tracker(self, process_width: int = 480) -> None:
        """(Re)initialise the TrackerCore. Tries ONNX first, falls back to .pt."""
        from pathlib import Path

        onnx_path = Path("yolov8n.onnx")
        model = str(onnx_path) if onnx_path.exists() else "yolov8n.pt"
        self.core = TrackerCore(
            model_path=model,
            conf_threshold=0.4,
            process_width=process_width,
            frame_skip=2,
        )

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------
    def _open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self._start_source(file_path)

    def _use_camera(self):
        self._start_source(0)

    def _start_source(self, source):
        """Stop any existing capture, then start a new one."""
        self._stop_video()
        self.frame_queue = Queue(maxsize=2)
        self.capture_thread = CaptureThread(source, self.frame_queue)
        self.capture_thread.start()
        self.tracking_enabled = False
        self.timer.start(15)  # ~66 Hz poll rate (actual display is frame-rate limited)

    def _start_tracking(self):
        self.tracking_enabled = True

    def _stop_tracking(self):
        self.tracking_enabled = False

    def _stop_video(self):
        self.timer.stop()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None
        # Drain queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        self.video_label.setText("Video Stopped")

    def _on_res_change(self, index: int):
        widths = [320, 480, 640]
        self._init_tracker(process_width=widths[index])

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------
    def _update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
        except Empty:
            return

        if self.tracking_enabled:
            frame = self.core.process(frame)
            self.core.draw_fps(frame)

        # BGR → RGB for Qt
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._stop_video()
        event.accept()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
