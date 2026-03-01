<div align="center">

# 🎯 Real-Time Object Tracking System

**Multi-object detection and tracking using YOLOv8 + DeepSORT — runs fully on CPU.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-orange)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv)](https://opencv.org/)

</div>

---

## 📌 Description

A lightweight, real-time object tracking system that detects and tracks multiple objects in a live webcam feed or video file. It combines **YOLOv8** for fast, accurate object detection with **DeepSORT** for persistent ID assignment across frames — all without requiring a GPU.

---

## ✨ Features

- 🟢 **Multi-object tracking** — assigns stable IDs to each detected object across frames
- 🖥️ **CPU-only** — no GPU or CUDA required; runs on any laptop
- 📷 **Flexible input** — plug in a webcam or point it at any video file
- ⚙️ **Configurable** — tune model, confidence threshold, and source via CLI flags
- 💡 **Minimal dependencies** — just 4 pip packages

---

## 🛠️ Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Object Detection | [Ultralytics YOLOv8](https://docs.ultralytics.com/) | Per-frame bounding box detection |
| Multi-Object Tracking | [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) | ReID + Kalman filter tracking |
| Camera / Video I/O | [OpenCV](https://opencv.org/) | Frame capture and rendering |
| Numerical Processing | [NumPy](https://numpy.org/) | Bounding box coordinate handling |

---

## 📁 Project Structure

```
track_project/
├── track.py           # Main entry point — run this to start tracking
├── requirements.txt   # Direct Python dependencies
├── .gitignore         # Ignores venv, weights, cache, output videos
├── LICENSE            # MIT License
└── README.md          # This file
```

> **Note:** `yolov8n.pt` model weights are **not** committed to the repo (they are gitignored). They are auto-downloaded on first run.

---

## 🚀 Installation

### Prerequisites

- Python **3.10** or higher
- A webcam **or** a `.mp4` / `.avi` video file

### 1. Clone the repository

```bash
git clone https://github.com/your-username/track_project.git
cd track_project
```

### 2. Create and activate a virtual environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `torch` and `torchvision` are installed automatically as dependencies of `ultralytics`.

---

## ▶️ Usage

### Webcam (default)

```bash
python track.py
```

### Video File

```bash
python track.py --source path/to/video.mp4
```

### All Options

```
python track.py [--source SOURCE] [--model MODEL] [--conf CONF]

Arguments:
  --source    Video source: '0' for webcam, or path to video file  (default: 0)
  --model     Path to YOLOv8 .pt weights file                      (default: yolov8n.pt)
  --conf      Minimum detection confidence threshold (0.0–1.0)     (default: 0.4)
```

**Example — custom model with a higher confidence threshold on a video:**
```bash
python track.py --source demo.mp4 --model yolov8s.pt --conf 0.55
```

Press **`q`** at any time to quit the tracking window.

---

## ⚙️ How It Works

```
Video Frame
    │
    ▼
┌─────────────┐
│  YOLOv8n   │  ← detects all objects, outputs (x,y,w,h) + confidence + class
└─────────────┘
    │
    ▼ bounding boxes
┌─────────────────┐
│   DeepSORT      │  ← matches current detections to existing tracks using
│   Tracker       │    Kalman Filter (motion) + cosine distance (appearance)
└─────────────────┘
    │
    ▼ confirmed tracks with stable IDs
┌─────────────────┐
│   OpenCV Draw   │  ← draws green bounding box + "ID: N" label per object
└─────────────────┘
    │
    ▼
 Display Window
```

**YOLOv8n** (nano) is chosen for best speed-vs-accuracy trade-off on CPU. **DeepSORT** keeps each object's identity stable even through brief occlusions by combining a Kalman filter for motion prediction with a deep appearance descriptor for re-identification.

---

## 🔮 Future Improvements

- [ ] Class-label display next to track ID (e.g., "person", "car")
- [ ] Video output saving (`--save` flag)
- [ ] FPS counter overlay
- [ ] REST API / WebSocket stream endpoint
- [ ] ONNX export for further CPU optimisation
- [ ] Docker containerisation for zero-setup deployment

---

## 👤 Author

**Adhiraj Singh**  
B.Tech — Computer Science Engineering (AI/ML)  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
