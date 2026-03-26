"""
final.py — Maximum-FPS Object Tracker (PySide6 GUI)
-----------------------------------------------------
Self-contained, optimised for 10+ FPS on a minimal CPU-only laptop.

Key optimisations vs main.py:
  1. ONNX Runtime at 320×320 (~36ms vs 210ms PyTorch)  → 6× faster inference
  2. Lightweight IoU tracker (no CNN re-ID)              → <1ms vs ~30ms DeepSORT
  3. Frame skip of 3                                     → 3× effective throughput
  4. Threaded capture                                    → I/O never blocks inference

Usage:
    python3.10 final.py
"""

import sys
import time
from collections import OrderedDict
from pathlib import Path
from queue import Queue, Empty
from threading import Thread

import cv2
import numpy as np
import onnxruntime as ort

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
from PySide6.QtCore import Qt, QTimer, QThread
from PySide6.QtGui import QImage, QPixmap


# ═══════════════════════════════════════════════════════════════
# IoU Tracker — lightweight replacement for DeepSORT
# ═══════════════════════════════════════════════════════════════
class IoUTracker:
    """
    Simple multi-object tracker using only Intersection-over-Union matching.
    No CNN features, no Kalman filter — pure geometry.
    Cost: <0.1ms per frame vs ~30ms for DeepSORT.
    """

    def __init__(self, iou_threshold=0.3, max_age=15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = OrderedDict()
        self._next_id = 1

    def update(self, boxes):
        """
        Match new detections to existing tracks via IoU.
        Args:  boxes — list of [x1, y1, x2, y2]
        Returns:  list of (track_id, [x1, y1, x2, y2])
        """
        if len(boxes) == 0:
            to_delete = []
            for tid in self.tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return [(tid, t["bbox"]) for tid, t in self.tracks.items()]

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["bbox"] for tid in track_ids]
        matched_tracks = set()
        matched_dets = set()
        results = []

        if track_boxes:
            iou_matrix = self._compute_iou_matrix(track_boxes, boxes)
            while True:
                if iou_matrix.size == 0:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                if iou_matrix[i, j] < self.iou_threshold:
                    break
                tid = track_ids[i]
                self.tracks[tid]["bbox"] = boxes[j]
                self.tracks[tid]["age"] = 0
                self.tracks[tid]["hits"] += 1
                matched_tracks.add(i)
                matched_dets.add(j)
                results.append((tid, boxes[j]))
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

        to_delete = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_delete.append(tid)
                else:
                    results.append((tid, self.tracks[tid]["bbox"]))
        for tid in to_delete:
            del self.tracks[tid]

        for j, box in enumerate(boxes):
            if j not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {"bbox": box, "age": 0, "hits": 1}
                results.append((tid, box))

        return results

    @staticmethod
    def _compute_iou_matrix(boxes_a, boxes_b):
        a = np.array(boxes_a, dtype=np.float32)
        b = np.array(boxes_b, dtype=np.float32)
        x1 = np.maximum(a[:, 0:1], b[:, 0].T)
        y1 = np.maximum(a[:, 1:2], b[:, 1].T)
        x2 = np.minimum(a[:, 2:3], b[:, 2].T)
        y2 = np.minimum(a[:, 3:4], b[:, 3].T)
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-6)


# ═══════════════════════════════════════════════════════════════
# ONNX Detector — direct ONNX Runtime, no ultralytics overhead
# ═══════════════════════════════════════════════════════════════
class ONNXDetector:
    """Runs YOLOv8n ONNX inference directly via onnxruntime."""

    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.45):
        self.conf = conf_threshold
        self.nms = nms_threshold
        print(f"[INFO] Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_h = inp.shape[2]
        self.input_w = inp.shape[3]
        print(f"[INFO] Input size: {self.input_w}×{self.input_h}")

    def detect(self, frame):
        """Run detection on a BGR frame. Returns list of [x1,y1,x2,y2]."""
        orig_h, orig_w = frame.shape[:2]

        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...])

        outputs = self.session.run(None, {self.input_name: img})
        preds = outputs[0][0].T

        class_scores = preds[:, 4:]
        max_scores = class_scores.max(axis=1)
        mask = max_scores >= self.conf
        preds = preds[mask]
        max_scores = max_scores[mask]

        if len(preds) == 0:
            return []

        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h

        cx = preds[:, 0] * scale_x
        cy = preds[:, 1] * scale_y
        w = preds[:, 2] * scale_x
        h = preds[:, 3] * scale_y

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes_xywh = list(zip(x1.tolist(), y1.tolist(), w.tolist(), h.tolist()))
        indices = cv2.dnn.NMSBoxes(boxes_xywh, max_scores.tolist(), self.conf, self.nms)

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        return [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in indices]


# ═══════════════════════════════════════════════════════════════
# Capture Thread
# ═══════════════════════════════════════════════════════════════
class CaptureThread(QThread):
    """Reads frames from cv2.VideoCapture in a background QThread."""

    def __init__(self, source, queue, parent=None):
        super().__init__(parent)
        self.source = source
        self.queue = queue
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {self.source}")
            return
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
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


# ═══════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # -------- WINDOW --------
        self.setWindowTitle("⚡ Final Tracker — ONNX + IoU (Max FPS)")
        self.setGeometry(80, 80, 1000, 640)
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f1a; }
            QLabel { color: #e0e0e0; }
            QPushButton {
                padding: 10px 16px;
                font-size: 13px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d44, stop:1 #1e1e30);
                color: #e0e0ff;
                border: 1px solid #3a3a5c;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a5c, stop:1 #2d2d44);
                border-color: #6c6cff;
            }
            QPushButton:pressed { background-color: #4a4a7a; }
            QComboBox {
                padding: 8px 12px;
                font-size: 13px;
                font-weight: bold;
                background-color: #1e1e30;
                color: #a0ffa0;
                border: 1px solid #3a5c3a;
                border-radius: 8px;
            }
            QComboBox:hover { border-color: #6cff6c; }
            QComboBox QAbstractItemView {
                background-color: #1e1e30;
                color: #a0ffa0;
                selection-background-color: #3a5c3a;
            }
        """)

        # -------- VIDEO DISPLAY --------
        self.video_label = QLabel("No Video — Upload a file or use your camera")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #08080f; color: #4a4a6a; font-size: 15px;"
            "border: 1px solid #1a1a2e; border-radius: 8px;"
        )
        self.video_label.setMinimumHeight(430)

        # -------- BUTTONS --------
        self.upload_btn = QPushButton("📂 Upload Video")
        self.camera_btn = QPushButton("📷 Camera")
        self.start_btn = QPushButton("▶ Start Tracking")
        self.stop_tracking_btn = QPushButton("⏸ Stop Tracking")
        self.stop_btn = QPushButton("⏹ Stop Video")

        # -------- QUALITY SELECTOR --------
        self.quality_combo = QComboBox()
        self.quality_combo.blockSignals(True)
        self.quality_combo.addItems([
            "⚡ Speed (skip 4)",
            "⚖️ Balanced (skip 3)",
            "🎯 Quality (skip 2)",
            "🔬 Max (skip 1)",
        ])
        self.quality_combo.setCurrentIndex(1)  # default: balanced
        self.quality_combo.blockSignals(False)
        self.quality_combo.currentIndexChanged.connect(self._on_quality_change)

        # -------- LAYOUT --------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.camera_btn)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_tracking_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.quality_combo)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(btn_row)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # -------- DETECTION + TRACKING --------
        model_path = "yolov8n.onnx"
        if not Path(model_path).exists():
            print("[ERROR] yolov8n.onnx not found. Run: python3.10 export_onnx.py --imgsz 320")
            sys.exit(1)

        self.detector = ONNXDetector(model_path, conf_threshold=0.5)
        self.tracker = IoUTracker(iou_threshold=0.3, max_age=15)
        self.tracking_enabled = False
        self.frame_skip = 3
        self._frame_count = 0
        self._last_boxes = []
        self._fps_times = []

        # -------- CAPTURE --------
        self.frame_queue = Queue(maxsize=2)
        self.capture_thread = None

        # -------- TIMER --------
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        # -------- CONNECTIONS --------
        self.upload_btn.clicked.connect(self._open_video)
        self.camera_btn.clicked.connect(self._use_camera)
        self.start_btn.clicked.connect(self._start_tracking)
        self.stop_tracking_btn.clicked.connect(self._stop_tracking)
        self.stop_btn.clicked.connect(self._stop_video)

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------
    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if path:
            self._start_source(path)

    def _use_camera(self):
        self._start_source(0)

    def _start_source(self, source):
        self._stop_video()
        self.frame_queue = Queue(maxsize=2)
        self.capture_thread = CaptureThread(source, self.frame_queue)
        self.capture_thread.start()
        self.tracking_enabled = False
        self._frame_count = 0
        self._last_boxes = []
        self.tracker = IoUTracker(iou_threshold=0.3, max_age=15)
        self.timer.start(10)

    def _start_tracking(self):
        self.tracking_enabled = True

    def _stop_tracking(self):
        self.tracking_enabled = False

    def _stop_video(self):
        self.timer.stop()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        self.video_label.setText("Video Stopped")

    def _on_quality_change(self, index):
        skip_map = [4, 3, 2, 1]
        self.frame_skip = skip_map[index]
        print(f"[INFO] Frame skip set to: {self.frame_skip}")

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def _update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
        except Empty:
            return

        t_start = time.perf_counter()

        if self.tracking_enabled:
            self._frame_count += 1

            # Run detection every Nth frame
            if self._frame_count % self.frame_skip == 0:
                self._last_boxes = self.detector.detect(frame)

            # Update tracker
            active = self.tracker.update(self._last_boxes)

            # Draw
            for track_id, bbox in active:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS
            elapsed = time.perf_counter() - t_start
            self._fps_times.append(elapsed)
            if len(self._fps_times) > 30:
                self._fps_times.pop(0)
            fps = len(self._fps_times) / max(sum(self._fps_times), 1e-9)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._stop_video()
        event.accept()


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
