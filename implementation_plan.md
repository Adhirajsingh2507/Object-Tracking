# Object Tracking System — Complete Architecture & Implementation Plan

> **YOLOv8 + DeepSORT Real-Time Multi-Object Tracker**
> Threaded GUI (PySide6) + CLI — CPU-only, ONNX-ready

---

## 1. Full System Architecture

```mermaid
graph TD
    subgraph Entry["🚀 Entry Points"]
        GUI["python3.10 main.py<br/>(PySide6 GUI)"]
        CLI["python3.10 track.py<br/>(OpenCV CLI)"]
    end

    subgraph Core["⚙️ core.py — TrackerCore"]
        LOAD{"Model Loader"}
        LOAD -->|.pt file| YOLO["YOLOv8 PyTorch<br/>_infer_yolo()"]
        LOAD -->|.onnx file| ONNX["ONNX Runtime<br/>_infer_onnx()"]
        LOAD -->|.onnx but no onnxruntime| FALLBACK["Auto-fallback → .pt"]

        YOLO --> NMS1["NMS<br/>cv2.dnn.NMSBoxes<br/>IoU=0.45"]
        ONNX --> NMS2["NMS<br/>cv2.dnn.NMSBoxes<br/>IoU=0.45"]

        NMS1 --> DS["DeepSORT<br/>update_tracks()"]
        NMS2 --> DS

        DS --> DRAW["Draw Boxes + IDs"]
        DRAW --> FPS["FPS Overlay"]
    end

    subgraph Perf["🚀 Performance Layer"]
        RESIZE["Resolution Downscale<br/>320 / 480 / 640 px"]
        SKIP["Frame Skip<br/>process every Nth frame"]
        THREAD["CaptureThread<br/>background I/O"]
    end

    GUI --> Core
    CLI --> Core
    GUI --> THREAD
    RESIZE --> YOLO
    RESIZE --> ONNX
    SKIP --> DS

    style Entry fill:#238636,stroke:#fff,color:#fff
    style Core fill:#1a1a2e,stroke:#e94560,color:#fff
    style Perf fill:#16213e,stroke:#0f3460,color:#fff
```

---

## 2. Component Dependency Graph

```mermaid
graph LR
    subgraph Python["🐍 Python 3.10+"]
        direction TB
        STD_SYS["sys"]
        STD_Q["queue"]
        STD_TIME["time"]
        STD_PATH["pathlib"]
        STD_ARGS["argparse"]
    end

    subgraph External["📦 External Libraries"]
        QT["PySide6 (Qt6)"]
        CV["OpenCV 4.13"]
        UL["Ultralytics 8.4"]
        DS["deep-sort-realtime"]
        ORT["onnxruntime 1.23"]
        NP["NumPy"]
        PT["PyTorch (auto)"]
    end

    subgraph Project["📄 Project Files"]
        CORE["core.py<br/>TrackerCore"]
        MAIN["main.py<br/>GUI"]
        TRACK["track.py<br/>CLI"]
        EXPORT["export_onnx.py"]
    end

    MAIN --> QT
    MAIN --> CV
    MAIN --> CORE
    TRACK --> CV
    TRACK --> CORE
    CORE --> CV
    CORE --> NP
    CORE --> DS
    CORE --> UL
    CORE -.->|optional| ORT
    UL --> PT
    EXPORT --> UL

    style Python fill:#3572A5,stroke:#fff,color:#fff
    style External fill:#161b22,stroke:#30363d,color:#c9d1d9
    style Project fill:#0d1117,stroke:#58a6ff,color:#58a6ff
```

---

## 3. Class Diagram (UML)

```mermaid
classDiagram
    class TrackerCore {
        +float conf_threshold
        +int process_width
        +int frame_skip
        +float nms_threshold
        -int _frame_count
        -list _last_tracks
        -float _fps
        -list _frame_times
        -bool _use_onnx
        -YOLO model
        -InferenceSession ort_session
        +DeepSort tracker
        +__init__(model_path, conf, width, skip, max_age, n_init, nms)
        +process(frame) ndarray
        +draw_fps(frame)
        +fps : float
        -_load_yolo(path)
        -_load_onnx(path)
        -_infer_yolo(frame) list
        -_infer_onnx(frame) list
        -_draw_tracks(frame, tracks)$
    }

    class CaptureThread {
        +Signal frame_ready
        +source
        +Queue queue
        -bool _running
        +run()
        +stop()
    }

    class MainWindow {
        -QLabel video_label
        -QPushButton upload_btn
        -QPushButton camera_btn
        -QPushButton start_btn
        -QPushButton stop_tracking_btn
        -QPushButton stop_btn
        -QComboBox res_combo
        -Queue frame_queue
        -CaptureThread capture_thread
        -QTimer timer
        -TrackerCore core
        -bool tracking_enabled
        +_init_tracker(process_width)
        +_open_video()
        +_use_camera()
        +_start_source(source)
        +_start_tracking()
        +_stop_tracking()
        +_stop_video()
        +_on_res_change(index)
        +_update_frame()
        +closeEvent(event)
    }

    MainWindow --|> QMainWindow : inherits
    CaptureThread --|> QThread : inherits
    MainWindow --> TrackerCore : owns
    MainWindow --> CaptureThread : owns
    TrackerCore --> DeepSort : owns
    TrackerCore --> YOLO : uses
    TrackerCore ..> InferenceSession : optional
```

---

## 4. Detection Pipeline — NMS Detail

```mermaid
flowchart LR
    subgraph Input["📹 Frame In"]
        F["Raw Frame<br/>(e.g. 1920×1080)"]
    end

    subgraph Downscale["📐 Resize"]
        F --> R["cv2.resize<br/>→ process_width px"]
    end

    subgraph Detect["🔍 Detection"]
        R --> Y{"Model Type?"}
        Y -->|.pt| YP["YOLOv8 PyTorch<br/>model(frame, conf=0.5)"]
        Y -->|.onnx| YO["ONNX Runtime<br/>ort_session.run()"]
        YP --> B1["Raw Boxes<br/>[x,y,w,h] + conf + cls"]
        YO --> P["Preprocess<br/>RGB, CHW, float32, /255"]
        P --> I["Inference"]
        I --> D["Decode<br/>cx,cy,w,h → x,y,w,h"]
        D --> B2["Raw Boxes"]
    end

    subgraph NMS["🧹 NMS Dedup"]
        B1 --> N["cv2.dnn.NMSBoxes<br/>conf ≥ 0.5, IoU ≤ 0.45"]
        B2 --> N
        N --> CLEAN["Deduplicated<br/>Detections"]
    end

    subgraph Scale["📏 Rescale"]
        CLEAN --> SC["Scale coords<br/>back to original res"]
    end

    subgraph Track["🎯 DeepSORT"]
        SC --> DS["update_tracks()<br/>n_init=1, max_age=30"]
        DS --> OUT["Confirmed Tracks<br/>[ltrb, track_id]"]
    end

    style Input fill:#11111b,stroke:#45475a,color:#cdd6f4
    style Downscale fill:#1e1e2e,stroke:#45475a,color:#cdd6f4
    style Detect fill:#1a1a2e,stroke:#e94560,color:#fff
    style NMS fill:#238636,stroke:#fff,color:#fff
    style Scale fill:#1e1e2e,stroke:#45475a,color:#cdd6f4
    style Track fill:#1f6feb,stroke:#fff,color:#fff
```

---

## 5. Data-Flow Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant GUI as MainWindow
    participant CT as CaptureThread
    participant Q as frame_queue
    participant TC as TrackerCore
    participant YOLO as YOLOv8n
    participant NMS as NMSBoxes
    participant DS as DeepSORT
    participant CV as OpenCV Draw

    User->>GUI: Click "Camera" / "Upload"
    GUI->>CT: CaptureThread(source).start()

    User->>GUI: Click "Start Tracking"
    GUI->>GUI: tracking_enabled = True

    loop CaptureThread (background)
        CT->>CT: cap.read()
        CT->>Q: queue.put(frame)
    end

    loop QTimer every 15ms
        GUI->>Q: queue.get_nowait()
        Q-->>GUI: frame

        alt tracking_enabled AND frame_count % skip == 0
            GUI->>TC: process(frame)
            TC->>TC: resize to process_width
            TC->>YOLO: model(small_frame, conf=0.5)
            YOLO-->>TC: raw boxes [xyxy, conf, cls]
            TC->>TC: convert xyxy → xywh
            TC->>NMS: NMSBoxes(boxes, scores, 0.5, 0.45)
            NMS-->>TC: deduplicated indices
            TC->>TC: scale coords to original res
            TC->>DS: update_tracks(detections, frame)
            DS-->>TC: confirmed tracks [ltrb, id]
            TC->>CV: rectangle() + putText()
            TC->>TC: draw_fps()
        else frame skipped
            TC->>TC: reuse _last_tracks
            TC->>CV: draw cached boxes
        end

        GUI->>GUI: BGR→RGB → QImage → QPixmap → display
    end

    User->>GUI: Click "Stop Video"
    GUI->>CT: stop()
```

---

## 6. Application State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle : App launched, model loaded

    Idle --> Playing : _open_video() / _use_camera()
    Playing --> Tracking : _start_tracking()
    Tracking --> Playing : _stop_tracking()
    Playing --> Idle : _stop_video()
    Tracking --> Idle : _stop_video()

    state Playing {
        [*] --> PollQueue
        PollQueue --> DisplayRaw : frame available
        DisplayRaw --> PollQueue : next timer tick
        PollQueue --> PollQueue : queue empty
    }

    state Tracking {
        [*] --> PollQueue2
        PollQueue2 --> CheckSkip : frame available
        CheckSkip --> RunInference : frame_count % skip == 0
        CheckSkip --> ReuseTracks : skipped frame
        RunInference --> YOLO_NMS
        YOLO_NMS --> DeepSORT_Update
        DeepSORT_Update --> DrawAndDisplay
        ReuseTracks --> DrawAndDisplay
        DrawAndDisplay --> PollQueue2 : next timer tick
    }

    Idle --> ResChanged : _on_res_change()
    ResChanged --> Idle : TrackerCore reinitialised
```

---

## 7. Method Reference Tables

### [core.py](file:///home/adhiraj-singh/track_project/core.py) — [TrackerCore](file:///home/adhiraj-singh/track_project/core.py#26-272)

| Method | Lines | Purpose |
|---|---|---|
| [__init__](file:///home/adhiraj-singh/track_project/core.py#39-74) | 39–73 | Load model (.pt/.onnx with fallback), init DeepSORT, set thresholds |
| [_load_yolo](file:///home/adhiraj-singh/track_project/core.py#79-85) | 79–84 | Load YOLOv8 via ultralytics, force CPU |
| [_load_onnx](file:///home/adhiraj-singh/track_project/core.py#86-93) | 86–92 | Load ONNX model via onnxruntime |
| [_infer_yolo](file:///home/adhiraj-singh/track_project/core.py#98-124) | 98–123 | YOLO .pt inference → collect boxes → **NMS** → return detections |
| [_infer_onnx](file:///home/adhiraj-singh/track_project/core.py#125-171) | 125–170 | ONNX inference → decode predictions → **NMS** → return detections |
| [process](file:///home/adhiraj-singh/track_project/core.py#176-229) | 176–228 | Resize → infer (or skip) → DeepSORT → draw → FPS |
| [draw_fps](file:///home/adhiraj-singh/track_project/core.py#235-246) | 235–245 | Overlay FPS counter (top-left, cyan) |
| [_draw_tracks](file:///home/adhiraj-singh/track_project/core.py#251-272) | 251–271 | Draw green rectangles + "ID: N" labels |

### [main.py](file:///home/adhiraj-singh/track_project/main.py) — [MainWindow](file:///home/adhiraj-singh/track_project/main.py#75-254) + [CaptureThread](file:///home/adhiraj-singh/track_project/main.py#35-70)

| Method | Lines | Purpose |
|---|---|---|
| `CaptureThread.run` | 46–65 | Background thread: read frames, push to queue (drop old) |
| `CaptureThread.stop` | 67–69 | Signal thread to stop, wait for join |
| `MainWindow.__init__` | 76–162 | Build UI, init TrackerCore, wire signals |
| [_init_tracker](file:///home/adhiraj-singh/track_project/main.py#167-179) | 167–178 | Create TrackerCore (auto-detect ONNX → PT fallback) |
| [_open_video](file:///home/adhiraj-singh/track_project/main.py#183-189) | 183–188 | File dialog → start capture |
| [_use_camera](file:///home/adhiraj-singh/track_project/main.py#190-192) | 190–191 | Open webcam index 0 |
| [_start_source](file:///home/adhiraj-singh/track_project/main.py#193-201) | 193–200 | Stop old capture → start new CaptureThread + timer |
| [_update_frame](file:///home/adhiraj-singh/track_project/main.py#228-247) | 228–246 | Poll queue → process → BGR→RGB → display on QLabel |
| [closeEvent](file:///home/adhiraj-singh/track_project/main.py#251-254) | 251–253 | Clean shutdown: stop thread + timer |

### [track.py](file:///home/adhiraj-singh/track_project/track.py) — CLI

| Method | Lines | Purpose |
|---|---|---|
| [parse_args](file:///home/adhiraj-singh/track_project/track.py#19-55) | 19–54 | CLI flags: --source, --model, --conf, --process-width, --frame-skip |
| [resolve_model](file:///home/adhiraj-singh/track_project/track.py#57-67) | 57–66 | Auto-pick .onnx if exists, else .pt |
| [run_tracker](file:///home/adhiraj-singh/track_project/track.py#69-105) | 69–104 | Main loop: read → process → display → quit on 'q' |

---

## 8. Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `conf_threshold` | **0.5** | Higher = fewer false positives, fewer duplicate boxes |
| `nms_threshold` | **0.45** | IoU overlap limit — boxes above this are merged |
| `process_width` | **480** | Inference resolution — smaller = faster, less precise |
| `frame_skip` | **2** | Run YOLO every 2nd frame — doubles effective FPS |
| `n_init` | **1** | Track confirmed on first match (required for frame_skip > 1) |
| `max_age` | **30** | Frames before a lost track is deleted |

---

## 9. Project File Map

```mermaid
graph TB
    subgraph Project["track_project/"]
        CORE["core.py<br/>TrackerCore (272 lines)<br/>shared detection + tracking"]
        MAIN["main.py<br/>PySide6 GUI (264 lines)<br/>threaded capture + display"]
        TRACK["track.py<br/>CLI tracker (114 lines)<br/>headless OpenCV loop"]
        EXPORT["export_onnx.py<br/>PT → ONNX converter"]
        PT["yolov8n.pt<br/>PyTorch weights (6.5 MB)"]
        ONNX["yolov8n.onnx<br/>ONNX weights (12.8 MB)"]
        REQ["requirements.txt<br/>6 dependencies"]
        README["README.md"]
    end

    MAIN -->|imports| CORE
    TRACK -->|imports| CORE
    CORE -->|loads| ONNX
    CORE -->|loads| PT
    EXPORT -->|generates| ONNX

    style CORE fill:#238636,stroke:#fff,color:#fff
    style MAIN fill:#1f6feb,stroke:#fff,color:#fff
    style TRACK fill:#1f6feb,stroke:#fff,color:#fff
    style EXPORT fill:#f0883e,stroke:#fff,color:#fff
    style ONNX fill:#f0883e,stroke:#fff,color:#fff
    style PT fill:#8b949e,stroke:#fff,color:#fff
```

---

## 10. Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Detection | Ultralytics YOLOv8n | 8.4 |
| Tracking | deep-sort-realtime (Kalman + ReID) | 1.3+ |
| NMS | cv2.dnn.NMSBoxes | — |
| Inference (fast) | ONNX Runtime | 1.23 |
| Inference (fallback) | PyTorch (CPU) | auto |
| GUI | PySide6 (Qt6) | 6.11 |
| Video I/O | OpenCV | 4.13 |
| Numerics | NumPy | 1.24+ |
