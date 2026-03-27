"""
pipeline.py — Frame Processing Pipeline
------------------------------------------
Replaces QTimer._update_frame() from final.py.
Runs as a continuous thread: pull frame → detect → track → draw → encode JPEG.
"""

import time
from threading import Thread, Lock

import cv2
import numpy as np

from backend.tracker import ONNXDetector, IoUTracker
from backend.capture import CaptureThread


class Pipeline:
    """
    Per-user tracking pipeline. Preserves final.py architecture:
      CaptureThread → Queue(2) → frame_skip → ONNXDetector → NMS → IoUTracker → Draw → JPEG
    """

    def __init__(self, detector: ONNXDetector):
        self.detector = detector
        self.tracker = IoUTracker(iou_threshold=0.3, max_age=15)

        # State
        self.tracking_enabled = False
        self.frame_skip = 3
        self._frame_count = 0
        self._last_boxes = []
        self._fps_times = []
        self._fps = 0.0

        # Capture
        self._capture: CaptureThread | None = None

        # Output frame (latest JPEG bytes, thread-safe)
        self._output_lock = Lock()
        self._output_frame: bytes | None = None

        # Pipeline thread
        self._running = False
        self._thread: Thread | None = None

    @property
    def fps(self):
        return self._fps

    @property
    def state(self):
        if self._capture is None or not self._capture.is_alive():
            return "idle"
        if self.tracking_enabled:
            return "tracking"
        return "playing"

    def start_source(self, source):
        """Start capturing from a video file path or webcam index."""
        self.stop()
        self._capture = CaptureThread(source)
        self.tracker = IoUTracker(iou_threshold=0.3, max_age=15)
        self._frame_count = 0
        self._last_boxes = []
        self._fps_times = []
        self._fps = 0.0
        self.tracking_enabled = False

        self._running = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def start_tracking(self):
        self.tracking_enabled = True

    def stop_tracking(self):
        self.tracking_enabled = False

    def set_quality(self, frame_skip: int):
        self.frame_skip = max(1, min(frame_skip, 10))

    def stop(self):
        """Stop the pipeline and capture thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._capture:
            self._capture.stop()
            self._capture = None
        with self._output_lock:
            self._output_frame = None

    def get_frame(self) -> bytes | None:
        """Get the latest JPEG-encoded frame (thread-safe)."""
        with self._output_lock:
            return self._output_frame

    # ──────────────────────────────────────────
    # Main processing loop (replaces QTimer)
    # ──────────────────────────────────────────
    def _loop(self):
        while self._running and self._capture and self._capture.is_alive():
            frame = self._capture.read()
            if frame is None:
                time.sleep(0.005)  # brief sleep to avoid busy-wait
                continue

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
                self._fps = len(self._fps_times) / max(sum(self._fps_times), 1e-9)
                cv2.putText(frame, f"FPS: {self._fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Encode to JPEG
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._output_lock:
                self._output_frame = jpeg.tobytes()

        # Pipeline ended (video finished or stopped)
        self._running = False
