"""
Export YOLOv8n to ONNX format for faster CPU inference.

Usage:
    python export_onnx.py
    python export_onnx.py --model yolov8s.pt --imgsz 640

Produces yolov8n.onnx (or equivalent) in the same directory.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO .pt → .onnx")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Path to .pt weights"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size for export"
    )
    args = parser.parse_args()

    print(f"[INFO] Exporting {args.model} → ONNX (imgsz={args.imgsz})...")
    model = YOLO(args.model)
    path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    print(f"[INFO] ONNX model saved to: {path}")


if __name__ == "__main__":
    main()
