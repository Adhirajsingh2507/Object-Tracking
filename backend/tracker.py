"""
tracker.py — ONNXDetector + IoUTracker
---------------------------------------
Direct port from final.py. Zero logic changes.
"""

from collections import OrderedDict

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
