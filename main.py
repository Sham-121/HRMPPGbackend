import shutil
import tempfile
from typing import List, Optional, Tuple


import cv2
import numpy as np
import heartpy as hp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="PPG Heart Rate API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BpmResponse(BaseModel):
    bpm: float
    num_samples: int
    fps: float


# ---------- Core PPG processing ----------

def extract_ppg_from_video(path: str) -> Tuple[List[float], float]:
    """Open video, extract per-frame brightness, return signal + fps."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not read video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    values: List[float] = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        # Take central 40% x 40% region (where finger should be)
        x1, x2 = int(w * 0.3), int(w * 0.7)
        y1, y2 = int(h * 0.3), int(h * 0.7)
        roi = frame_rgb[y1:y2, x1:x2]

        # Use RED channel (index 0 or 2 depending on ordering; here RGB so 0=R,1=G,2=B)
        red = roi[:, :, 0]
        avg_intensity = float(np.mean(red))
        values.append(avg_intensity)

    cap.release()

    if len(values) < fps * 4:  # need ~4+ seconds of data
        raise HTTPException(
            status_code=422,
            detail="Video too short or not enough usable frames for BPM calculation",
        )

    return values, float(fps)


def calculate_bpm(values: List[float], fs: float) -> Optional[float]:
    """Use HeartPy to get BPM from a PPG signal."""
    signal = np.array(values, dtype=float)

    # HeartPy likes at least a few seconds of data
    if len(signal) < fs * 4:
        return None

    try:
        working_data, measures = hp.process(signal, sample_rate=fs)
        bpm = measures.get("bpm", None)
        if bpm is None or not np.isfinite(bpm):
            return None
        return float(bpm)
    except Exception as e:
        print("HeartPy error:", e)
        return None


# ---------- API endpoints ----------


@app.get("/")
def root():
    return {"status": "ok", "message": "PPG Heart Rate API running"}


@app.post("/analyze_ppg_video", response_model=BpmResponse)
async def analyze_ppg_video(file: UploadFile = File(...)):
    # Save upload to a temp file
    try:
        suffix = "." + (file.filename.split(".")[-1] if "." in file.filename else "mp4")
    except Exception:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        values, fps = extract_ppg_from_video(temp_path)
        bpm = calculate_bpm(values, fps)
        if bpm is None:
            raise HTTPException(
                status_code=422,
                detail="Unable to compute BPM – signal too noisy or finger not stable",
            )

        return BpmResponse(
            bpm=round(bpm, 1),
            num_samples=len(values),
            fps=fps,
        )
    finally:
        pass
