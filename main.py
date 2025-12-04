import shutil
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import heartpy as hp
from scipy.signal import find_peaks

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="PPG Heart Rate API")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: allow all; restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BpmResponse(BaseModel):
    bpm: float
    num_samples: int
    fps: float


# ---------- Core PPG processing ----------

def extract_ppg_from_video(path: str):
    """
    Open video, extract per-frame brightness from a central crop,
    using the GREEN channel. Returns (values, fps).
    """
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

        # Central 40% x 40% region
        x1, x2 = int(w * 0.3), int(w * 0.7)
        y1, y2 = int(h * 0.3), int(h * 0.7)
        roi = frame_rgb[y1:y2, x1:x2]

        # Use GREEN channel: index 1 (RGB)
        green = roi[:, :, 1]
        avg_intensity = float(np.mean(green))
        values.append(avg_intensity)

    cap.release()

    # Need at least ~4 seconds of data
    if len(values) < fps * 4:
        raise HTTPException(
            status_code=422,
            detail="Video too short or not enough usable frames for BPM calculation",
        )

    return values, float(fps)


def calculate_bpm(values: List[float], fs: float) -> Optional[float]:
    """
    Try HeartPy first with filtering and bpm constraints.
    If that fails, fallback to a simple peak-based BPM.
    """
    signal = np.array(values, dtype=float)

    # Need at least ~4 seconds
    if len(signal) < fs * 4:
        return None

    # 1) Detrend & normalize
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std > 1e-8:
        signal = signal / std

    # 2) Bandpass filter (0.7–4 Hz ≈ 42–240 bpm)
    try:
        filtered = hp.filter_signal(
            signal,
            cutoff=[0.7, 4.0],
            sample_rate=fs,
            order=3,
            filtertype="bandpass",
        )
    except Exception as e:
        print("HeartPy filter error:", e)
        filtered = signal

    # 3) HeartPy processing with bpm range
    try:
        working_data, measures = hp.process(
            filtered,
            sample_rate=fs,
            bpmmin=40,
            bpmmax=180,
        )
        bpm = measures.get("bpm", None)
        if bpm is not None and np.isfinite(bpm):
            return float(bpm)
    except Exception as e:
        print("HeartPy error (process):", e)

    # 4) Fallback: simple peak detection
    try:
        # minimum distance between peaks: 0.3s => ~200 bpm max
        min_distance = int(0.3 * fs)
        peaks, _ = find_peaks(filtered, distance=max(1, min_distance))

        if len(peaks) < 2:
            return None

        intervals = np.diff(peaks) / fs  # seconds between peaks

        # Keep intervals corresponding to ~40–180 bpm => 0.33–1.5s
        mask = (intervals > 0.33) & (intervals < 1.5)
        intervals = intervals[mask]
        if intervals.size == 0:
            return None

        avg_interval = float(np.mean(intervals))
        if avg_interval <= 0:
            return None

        bpm_fb = 60.0 / avg_interval
        if np.isfinite(bpm_fb) and 40.0 <= bpm_fb <= 180.0:
            print("Using fallback BPM:", bpm_fb)
            return float(bpm_fb)

        return None
    except Exception as e:
        print("Fallback peak detection error:", e)
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
        # let OS clean up temp file; explicit delete optional
        pass
