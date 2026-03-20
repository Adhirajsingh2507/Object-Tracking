import sys
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # -------- WINDOW --------
        self.setWindowTitle("YOLOv8 + DeepSORT Tracker")
        self.setGeometry(100, 100, 900, 600)

        self.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-size: 14px;
            }
        """)

        # -------- VIDEO DISPLAY --------
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setFixedHeight(400)

        # -------- BUTTONS --------
        self.upload_btn = QPushButton("Upload Video")
        self.camera_btn = QPushButton("Use Camera")
        self.start_btn = QPushButton("Start Tracking")
        self.stop_tracking_btn = QPushButton("Stop Tracking")
        self.stop_btn = QPushButton("Stop Video")

        # -------- LAYOUT --------
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.camera_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_tracking_btn)
        button_layout.addWidget(self.stop_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # -------- VIDEO --------
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # -------- MODEL --------
        print("[INFO] Loading YOLO...")
        self.model = YOLO("yolov8n.pt")
        self.model.to("cpu")

        print("[INFO] Initializing DeepSORT...")
        self.tracker = DeepSort(max_age=20, n_init=3)

        self.conf_threshold = 0.4
        self.tracking_enabled = False

        # -------- BUTTON CONNECTIONS --------
        self.upload_btn.clicked.connect(self.open_video)
        self.camera_btn.clicked.connect(self.use_camera)
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_tracking_btn.clicked.connect(self.stop_tracking)
        self.stop_btn.clicked.connect(self.stop_video)

    # -------- BUTTON ACTIONS --------
    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.tracking_enabled = False  # reset tracking
            self.timer.start(30)

    def use_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.tracking_enabled = False
        self.timer.start(30)

    def start_tracking(self):
        self.tracking_enabled = True

    def stop_tracking(self):
        self.tracking_enabled = False

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.setText("Video Stopped")

    # -------- PROCESS FRAME --------
    def process_frame(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = track.to_ltrb()
            track_id = track.track_id

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
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

        return frame

    # -------- UPDATE FRAME --------
    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if ret:
            if self.tracking_enabled:
                frame = self.process_frame(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = frame.shape
            bytes_per_line = ch * w

            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            self.video_label.setPixmap(pixmap)
        else:
            # -------- INFINITE LOOP --------
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -------- RUN APP --------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
