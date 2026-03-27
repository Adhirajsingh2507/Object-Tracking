"""
main.py — FastAPI Application
-------------------------------
Routes for auth, video control, and MJPEG streaming.
Serves frontend static files.
"""

import os
import time

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.database import init_db
from backend.auth import register_user, login_user, get_current_user
from backend.video_manager import get_pipeline, UPLOAD_DIR

# ── App ──
app = FastAPI(title="Object Tracker")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.on_event("startup")
def startup():
    init_db()


# ── Static files ──
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Page routes ──
@app.get("/", response_class=HTMLResponse)
def serve_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/login", response_class=HTMLResponse)
def serve_login():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))


# ── Auth models ──
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class QualityRequest(BaseModel):
    frame_skip: int


# ── Auth routes ──
@app.post("/api/register")
def api_register(req: RegisterRequest):
    return register_user(req.username, req.email, req.password)


@app.post("/api/login")
def api_login(req: LoginRequest):
    return login_user(req.username, req.password)


# ── Video control routes ──
@app.post("/api/upload-video")
async def api_upload_video(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload a video file and start the capture pipeline."""
    # Save uploaded file
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    pipeline = get_pipeline(str(user["id"]))
    pipeline.start_source(save_path)
    return {"status": "playing", "source": file.filename}


@app.post("/api/start-webcam")
def api_start_webcam(user: dict = Depends(get_current_user)):
    pipeline = get_pipeline(str(user["id"]))
    pipeline.start_source(0)
    return {"status": "playing", "source": "webcam"}


@app.post("/api/start-tracking")
def api_start_tracking(user: dict = Depends(get_current_user)):
    pipeline = get_pipeline(str(user["id"]))
    pipeline.start_tracking()
    return {"status": "tracking"}


@app.post("/api/stop-tracking")
def api_stop_tracking(user: dict = Depends(get_current_user)):
    pipeline = get_pipeline(str(user["id"]))
    pipeline.stop_tracking()
    return {"status": "playing"}


@app.post("/api/stop-video")
def api_stop_video(user: dict = Depends(get_current_user)):
    pipeline = get_pipeline(str(user["id"]))
    pipeline.stop()
    return {"status": "idle"}


@app.post("/api/set-quality")
def api_set_quality(req: QualityRequest, user: dict = Depends(get_current_user)):
    pipeline = get_pipeline(str(user["id"]))
    pipeline.set_quality(req.frame_skip)
    return {"frame_skip": pipeline.frame_skip}


@app.get("/api/status")
def api_status(user: dict = Depends(get_current_user)):
    pipeline = get_pipeline(str(user["id"]))
    return {
        "state": pipeline.state,
        "fps": round(pipeline.fps, 1),
        "frame_skip": pipeline.frame_skip,
        "tracking_enabled": pipeline.tracking_enabled,
    }


# ── MJPEG Stream ──
@app.get("/stream")
def video_stream(user: dict = Depends(get_current_user)):
    """MJPEG video stream — use as <img src="/stream?token=...">>."""
    pipeline = get_pipeline(str(user["id"]))

    def generate():
        while True:
            frame = pipeline.get_frame()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            else:
                # No frame yet, send a small delay
                time.sleep(0.03)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
