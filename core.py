"""
Shared Tracking Core
--------------------
Provides TrackerCore: a single class that handles YOLOv8 / ONNX inference
and DeepSORT multi-object tracking. Used by both main.py (GUI) and track.py (CLI).

Optimised for minimal-laptop CPU-only execution.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Check onnxruntime availability at import time
_ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except (ImportError, OSError):
    pass


class TrackerCore:
    """
    Encapsulates detection (YOLOv8 / ONNX) + tracking (DeepSORT).

    Args:
        model_path:      Path to .pt or .onnx weights.
        conf_threshold:  Minimum detection confidence (0–1).
        process_width:   Resize frames to this width before inference (smaller = faster).
        frame_skip:      Process every N-th frame (1 = every frame, 2 = every other, …).
        max_age:         DeepSORT: frames to keep a lost track alive.
        n_init:          DeepSORT: consecutive hits before a track is confirmed.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        process_width: int = 640,
        frame_skip: int = 2,
        max_age: int = 20,
        n_init: int = 3,
    ):
        self.conf_threshold = conf_threshold
        self.process_width = process_width
        self.frame_skip = max(1, frame_skip)
        self._frame_count = 0
        self._last_tracks = []

        # --- FPS tracking ---
        self._fps = 0.0
        self._frame_times = []  # list of float

        # --- Load model ---
        self._use_onnx = model_path.endswith(".onnx") and _ONNX_AVAILABLE
        if model_path.endswith(".onnx") and not _ONNX_AVAILABLE:
            print("[WARN] onnxruntime not available — falling back to yolov8n.pt")
            model_path = model_path.replace(".onnx", ".pt")
        if self._use_onnx:
            self._load_onnx(model_path)
        else:
            self._load_yolo(model_path)

        # --- DeepSORT ---
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    # ------------------------------------------------------------------
    # Model loaders
    # ------------------------------------------------------------------

    def _load_yolo(self, path: str) -> None:
        from ultralytics import YOLO

        print(f"[INFO] Loading YOLO model: {path}")
        self.model = YOLO(path)
        self.model.to("cpu")

    def _load_onnx(self, path: str) -> None:
        print(f"[INFO] Loading ONNX model: {path}")
        self.ort_session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        self._onnx_input_name = self.ort_session.get_inputs()[0].name
        self._onnx_input_shape = self.ort_session.get_inputs()[0].shape  # [1,3,H,W]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_yolo(self, frame: np.ndarray) -> list:
        """Run YOLOv8 .pt inference, return list of (xywh, conf, cls)."""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        return detections

    def _infer_onnx(self, frame: np.ndarray) -> list:
        """Run ONNX Runtime inference, return list of (xywh, conf, cls)."""
        h, w = frame.shape[:2]
        input_h, input_w = self._onnx_input_shape[2], self._onnx_input_shape[3]

        # Preprocess: resize, BGR→RGB, HWC→CHW, normalise, add batch dim
        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

        outputs = self.ort_session.run(None, {self._onnx_input_name: img})
        # YOLOv8 ONNX output shape: [1, 84, 8400] — transpose to [8400, 84]
        preds = outputs[0][0].T

        detections = []
        for pred in preds:
            # pred: [cx, cy, w, h, cls0_conf, cls1_conf, ...]
            class_scores = pred[4:]
            max_cls = int(np.argmax(class_scores))
            conf = float(class_scores[max_cls])
            if conf < self.conf_threshold:
                continue

            # Convert centre-xywh to top-left-xywh, scale back to original frame
            cx, cy, bw, bh = pred[:4]
            x1 = (cx - bw / 2) * w / input_w
            y1 = (cy - bh / 2) * h / input_h
            bw_scaled = bw * w / input_w
            bh_scaled = bh * h / input_h
            detections.append(([x1, y1, bw_scaled, bh_scaled], conf, max_cls))

        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection + tracking on a frame.
        Respects frame_skip: only runs inference every N-th frame,
        reuses the last known tracks for skipped frames.

        Returns the annotated frame (drawn in-place).
        """
        t_start = time.perf_counter()
        self._frame_count += 1

        if self._frame_count % self.frame_skip == 0:
            # Resize for inference (faster), but draw on original
            if self.process_width and frame.shape[1] > self.process_width:
                scale = self.process_width / frame.shape[1]
                small = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                small = frame
                scale = 1.0

            # Run inference
            if self._use_onnx:
                detections = self._infer_onnx(small)
            else:
                detections = self._infer_yolo(small)

            # Scale detections back to original resolution
            if scale != 1.0:
                inv = 1.0 / scale
                detections = [
                    ([d[0][0] * inv, d[0][1] * inv, d[0][2] * inv, d[0][3] * inv], d[1], d[2])
                    for d in detections
                ]

            # Update tracker
            self._last_tracks = self.tracker.update_tracks(detections, frame=frame)
        else:
            # Skipped frame — do NOT call update_tracks with empty detections,
            # as that would kill unconfirmed tracks before they reach n_init.
            # Just reuse the last known tracks.
            pass

        # Draw confirmed tracks
        self._draw_tracks(frame, self._last_tracks)

        # FPS calculation (rolling window of last 30 frames)
        elapsed = time.perf_counter() - t_start
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        self._fps = len(self._frame_times) / max(sum(self._frame_times), 1e-9)

        return frame

    @property
    def fps(self) -> float:
        """Current smoothed FPS."""
        return self._fps

    def draw_fps(self, frame: np.ndarray) -> None:
        """Overlay FPS counter on the frame (top-left corner)."""
        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_tracks(frame: np.ndarray, tracks) -> None:
        """Draw green bounding boxes + ID labels for confirmed tracks."""
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            track_id = track.track_id

            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
