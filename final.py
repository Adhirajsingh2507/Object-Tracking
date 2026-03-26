"""
final.py — Maximum-FPS Object Tracker
---------------------------------------
Self-contained, optimised for 10+ FPS on a minimal CPU-only laptop.

Key optimisations vs main.py / track.py:
  1. ONNX Runtime at 320×320 (~36ms vs 210ms PyTorch)  → 6× faster inference
  2. Lightweight IoU tracker (no CNN re-ID)              → <1ms vs ~30ms DeepSORT
  3. Frame skip of 3                                     → 3× effective throughput
  4. Threaded capture                                    → I/O never blocks inference
  5. Minimal overhead: no Qt, just OpenCV highgui

Usage:
    python3.10 final.py                         # webcam
    python3.10 final.py --source video.mp4      # video file
    python3.10 final.py --frame-skip 4          # skip more for extra speed
"""

import argparse
import time
from collections import OrderedDict
from pathlib import Path
from queue import Queue, Empty
from threading import Thread

import cv2
import numpy as np
import onnxruntime as ort


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
        self.tracks = OrderedDict()   # track_id → {bbox, age, hits}
        self._next_id = 1

    def update(self, boxes):
        """
        Match new detections to existing tracks via IoU.

        Args:
            boxes: list of [x1, y1, x2, y2] in original frame coords.

        Returns:
            list of (track_id, [x1, y1, x2, y2]) for all active tracks.
        """
        if len(boxes) == 0:
            # Age all tracks
            to_delete = []
            for tid in self.tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return [(tid, t["bbox"]) for tid, t in self.tracks.items()]

        # Compute IoU matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        matched_tracks = set()
        matched_dets = set()
        results = []

        if track_boxes:
            iou_matrix = self._compute_iou_matrix(track_boxes, boxes)

            # Greedy matching — best IoU first
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

        # Unmatched tracks — age them
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

        # Unmatched detections — create new tracks
        for j, box in enumerate(boxes):
            if j not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {"bbox": box, "age": 0, "hits": 1}
                results.append((tid, box))

        return results

    @staticmethod
    def _compute_iou_matrix(boxes_a, boxes_b):
        """Vectorised IoU between two lists of [x1,y1,x2,y2] boxes."""
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
    """
    Runs YOLOv8n ONNX inference directly via onnxruntime.
    No ultralytics wrapper = faster startup + lower per-frame overhead.
    """

    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.45):
        self.conf = conf_threshold
        self.nms = nms_threshold

        print(f"[INFO] Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_h = inp.shape[2]  # 320
        self.input_w = inp.shape[3]  # 320
        print(f"[INFO] Input size: {self.input_w}×{self.input_h}")

    def detect(self, frame):
        """
        Run detection on a BGR frame.

        Returns:
            list of [x1, y1, x2, y2] boxes in original frame coordinates.
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess: resize, BGR→RGB, HWC→CHW, float32, /255, batch
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB + normalise
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...])

        # Inference
        outputs = self.session.run(None, {self.input_name: img})

        # Decode: YOLOv8 output = [1, 84, N] → [N, 84]
        preds = outputs[0][0].T

        # Filter by confidence
        class_scores = preds[:, 4:]
        max_scores = class_scores.max(axis=1)
        mask = max_scores >= self.conf
        preds = preds[mask]
        max_scores = max_scores[mask]

        if len(preds) == 0:
            return []

        # Convert centre-xywh → x1y1x2y2 in original frame coords
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

        # NMS
        boxes_xywh = list(zip(x1.tolist(), y1.tolist(), w.tolist(), h.tolist()))
        indices = cv2.dnn.NMSBoxes(boxes_xywh, max_scores.tolist(), self.conf, self.nms)

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        return [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in indices]


# ═══════════════════════════════════════════════════════════════
# Threaded Video Capture
# ═══════════════════════════════════════════════════════════════
class VideoCaptureThread:
    """Reads frames in a background thread — I/O never blocks processing."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.queue = Queue(maxsize=2)
        self._running = True
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put(None)  # signal end
                break
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
            self.queue.put(frame)

    def read(self):
        try:
            return self.queue.get(timeout=1)
        except Empty:
            return None

    def release(self):
        self._running = False
        self.cap.release()


# ═══════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="final.py — Max-FPS Object Tracker")
    p.add_argument("--source", default="0", help="'0' for webcam or path to video")
    p.add_argument("--model", default="yolov8n.onnx", help="ONNX model path")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--frame-skip", type=int, default=3, help="Process every Nth frame")
    return p.parse_args()


def run(args):
    model_path = args.model
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("[INFO]  Run: python3.10 export_onnx.py --imgsz 320")
        return

    detector = ONNXDetector(model_path, conf_threshold=args.conf)
    tracker = IoUTracker(iou_threshold=0.3, max_age=15)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = VideoCaptureThread(source)

    print(f"[INFO] Tracking started — press 'q' to quit")
    print(f"[INFO] Frame skip: {args.frame_skip} | Conf: {args.conf}")

    frame_count = 0
    fps_times = []
    last_boxes = []

    while True:
        t_start = time.perf_counter()
        frame = cap.read()
        if frame is None:
            print("[INFO] End of stream.")
            break

        frame_count += 1

        # Run detection only every Nth frame
        if frame_count % args.frame_skip == 0:
            last_boxes = detector.detect(frame)

        # Update tracker (even on skipped frames, uses last known boxes)
        active_tracks = tracker.update(last_boxes)

        # Draw
        for track_id, bbox in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS counter (rolling 30-frame window)
        elapsed = time.perf_counter() - t_start
        fps_times.append(elapsed)
        if len(fps_times) > 30:
            fps_times.pop(0)
        fps = len(fps_times) / max(sum(fps_times), 1e-9)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Final Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run(args)
