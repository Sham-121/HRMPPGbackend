import shutil
import tempfile
from typing import List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
from scipy.signal import butter, filtfilt, find_peaks
from typing import List, Optional
from pydantic import BaseModel, Field


app = FastAPI(title="PPG Heart Rate API")

# --------- CORS (so your Expo app can call this) ---------
origins = ["*"]  # dev only; lock down in prod if you want

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Optional: raw PPG endpoint (values -> bpm) ---------

class PpgRequest(BaseModel):
    values: List[float] = Field(..., min_items=10)
    fs: Optional[float] = 30.0

class BpmResponse(BaseModel):
    bpm: float
    num_samples: int
    fs: float

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def calculate_bpm_from_signal(values: List[float], fs: float) -> Optional[float]:
    signal = np.array(values, dtype=float)

    # Need at least ~3 seconds of data
    if len(signal) < fs * 3:
        return None

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Band-pass 0.7–4 Hz (~42–240 BPM)
    b, a = butter_bandpass(0.7, 4.0, fs)
    filtered = filtfilt(b, a, signal)

    # Peaks
    peaks, _ = find_peaks(filtered, distance=int(0.25 * fs))  # min 0.25s apart
    if len(peaks) < 2:
        return None

    # BPM from avg RR interval
    intervals = np.diff(peaks) / fs  # seconds
    avg_interval = np.mean(intervals)
    bpm = 60.0 / avg_interval
    return float(bpm)

@app.post("/calculate_bpm", response_model=BpmResponse)
def calculate_bpm_endpoint(req: PpgRequest):
    bpm = calculate_bpm_from_signal(req.values, req.fs or 30.0)
    if bpm is None:
        raise HTTPException(
            status_code=422,
            detail="Unable to compute BPM – signal too short or too noisy",
        )
    return BpmResponse(bpm=round(bpm, 1), num_samples=len(req.values), fs=req.fs or 30.0)

# --------- Video endpoint: upload camera clip -> BPM ---------

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

    # Read video with OpenCV
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not read video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    values: List[float] = []

    # Extract average brightness of a small central region per frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        # Central crop (e.g. middle 40% x 40%)
        x1, x2 = int(w * 0.3), int(w * 0.7)
        y1, y2 = int(h * 0.3), int(h * 0.7)
        roi = frame_rgb[y1:y2, x1:x2]

        # Use green channel (index 1) which often gives good PPG
        green = roi[:, :, 1]
        avg_intensity = float(np.mean(green))
        values.append(avg_intensity)

    cap.release()

    if len(values) < fps * 3:
        raise HTTPException(
            status_code=422,
            detail="Video too short or not enough usable frames for BPM calculation",
        )

    bpm = calculate_bpm_from_signal(values, fps)
    if bpm is None:
        raise HTTPException(
            status_code=422,
            detail="Unable to compute BPM – signal too noisy",
        )

    return BpmResponse(bpm=round(bpm, 1), num_samples=len(values), fs=fps)

@app.get("/")
def root():
    return {"status": "ok", "message": "PPG Heart Rate API running"}
