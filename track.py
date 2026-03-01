"""
Real-Time Object Tracking System
---------------------------------
Uses YOLOv8 for detection and DeepSORT for multi-object tracking.
Supports webcam input or any video file without requiring a GPU.
"""

import argparse

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Tracking with YOLOv8 + DeepSORT"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for webcam, or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to the YOLOv8 model weights file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum detection confidence threshold (default: 0.4)",
    )
    return parser.parse_args()


def run_tracker(source, model_path, conf_threshold):
    """
    Run the detection + tracking loop.

    Args:
        source (str|int): Webcam index (int) or path to a video file (str).
        model_path (str): Path to the YOLOv8 .pt weights file.
        conf_threshold (float): Detections below this confidence are ignored.
    """
    # Load YOLOv8 model. Force CPU so it runs on any machine without a GPU.
    model = YOLO(model_path)
    model.to("cpu")

    # DeepSORT parameters:
    #   max_age  – frames to keep a track alive without a new detection match
    #   n_init   – consecutive detections needed before a track is confirmed
    tracker = DeepSort(max_age=20, n_init=3)

    # Accept either an integer webcam index or a file path string
    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    print(f"[INFO] Tracking started — press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream or video.")
                break

            # Run YOLO inference; verbose=False suppresses per-frame console logs
            results = model(frame, verbose=False, conf=conf_threshold)[0]

            # Build the detection list expected by DeepSORT:
            # each entry is ([x, y, w, h], confidence, class_id)
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

            # Update tracker; returns active Track objects
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                # Skip tentative (unconfirmed) tracks to reduce noise
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

            cv2.imshow("Real-Time Object Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit signal received.")
                break
    finally:
        # Always release resources, even if an exception occurs
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    run_tracker(
        source=args.source,
        model_path=args.model,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
