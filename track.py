"""
Real-Time Object Tracking — CLI
---------------------------------
Uses the shared TrackerCore for YOLOv8 / ONNX + DeepSORT tracking.
Supports webcam input or any video file, runs fully on CPU.

Usage:
    python track.py
    python track.py --source video.mp4
    python track.py --source 0 --model yolov8n.onnx --process-width 480 --frame-skip 2
"""

import argparse

import cv2
from core import TrackerCore


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Tracking with YOLOv8 + DeepSORT (CLI)"
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
        default=None,
        help="Path to .pt or .onnx model (default: auto-detect yolov8n.onnx → yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum detection confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=480,
        help="Resize frame width for inference — smaller = faster (default: 480)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process every N-th frame; 1 = every frame (default: 2)",
    )
    return parser.parse_args()


def resolve_model(model_arg=None):
    # type: (str) -> str
    """Pick ONNX if available, else fall back to .pt."""
    if model_arg:
        return model_arg
    from pathlib import Path

    if Path("yolov8n.onnx").exists():
        return "yolov8n.onnx"
    return "yolov8n.pt"


def run_tracker(args):
    """Run the CLI detection + tracking loop."""
    model_path = resolve_model(args.model)

    core = TrackerCore(
        model_path=model_path,
        conf_threshold=args.conf,
        process_width=args.process_width,
        frame_skip=args.frame_skip,
    )

    video_source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    print(f"[INFO] Tracking started — press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream or video.")
                break

            frame = core.process(frame)
            core.draw_fps(frame)

            cv2.imshow("Real-Time Object Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit signal received.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    run_tracker(args)


if __name__ == "__main__":
    main()
