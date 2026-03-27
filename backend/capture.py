"""
capture.py — Threaded Video Capture
--------------------------------------
Adapted from QThread to threading.Thread.
Same queue + drop-old policy as final.py.
"""

from queue import Queue, Empty
from threading import Thread

import cv2


class CaptureThread:
    """Reads frames from cv2.VideoCapture in a background thread."""

    def __init__(self, source):
        self.source = source
        self.queue: Queue = Queue(maxsize=2)
        self._running = True
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {self.source}")
            self._running = False
            return
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            # Drop old frames if queue is full (keep latest)
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
            self.queue.put(frame)
        cap.release()
        self._running = False

    def read(self):
        """Get a frame from the queue (non-blocking, returns None if empty)."""
        try:
            return self.queue.get_nowait()
        except Empty:
            return None

    def is_alive(self):
        return self._running and self._thread.is_alive()

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=3)
        # Drain queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
