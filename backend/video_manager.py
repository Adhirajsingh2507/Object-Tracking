"""
video_manager.py — Per-User Session Manager
----------------------------------------------
Each authenticated user gets their own Pipeline instance.
Manages lifecycle, prevents interference between users.
"""

import os
from threading import Lock
from pathlib import Path

from backend.tracker import ONNXDetector
from backend.pipeline import Pipeline


# Singleton detector — loaded once, shared read-only across all pipelines
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolov8n.onnx")
_detector: ONNXDetector | None = None
_detector_lock = Lock()


def get_detector() -> ONNXDetector:
    """Get or create the singleton ONNX detector."""
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = ONNXDetector(_MODEL_PATH, conf_threshold=0.5)
        return _detector


# Per-user sessions
_sessions: dict[str, Pipeline] = {}
_sessions_lock = Lock()


def get_pipeline(user_id: str) -> Pipeline:
    """Get or create a Pipeline for a user."""
    with _sessions_lock:
        if user_id not in _sessions:
            _sessions[user_id] = Pipeline(get_detector())
        return _sessions[user_id]


def remove_pipeline(user_id: str):
    """Stop and remove a user's pipeline."""
    with _sessions_lock:
        pipeline = _sessions.pop(user_id, None)
        if pipeline:
            pipeline.stop()


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
