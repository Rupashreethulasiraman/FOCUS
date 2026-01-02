from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import os
import sys

# ---------- PATH SETUP (WINDOWS SAFE) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "rec_model")
sys.path.insert(0, MODEL_DIR)

# ---------- APP ----------
app = FastAPI(title="Focus AI Backend")

# Global tracker
tracker = None

@app.on_event("startup")
def startup_event():
    global tracker
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model", "rec_model")
    import sys
    sys.path.insert(0, MODEL_DIR)
    from cctv_tracker3 import CCTVTracker
    MODEL_BASE_DIR = os.path.join(BASE_DIR, "model", "rec_model")
    tracker = CCTVTracker(model_dir=MODEL_BASE_DIR)


# ---------- HEALTH ----------
@app.get("/health")
def health():
    if tracker is None:
        return {"status": "Initializing"}
    return {
        "status": "Focus AI backend running",
        "suspects_loaded": tracker.suspects_count
    }

# ---------- PROCESS FRAME ----------
@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Accepts an image frame and returns detections
    """
    if tracker is None:
        return {"error": "Model not loaded"}

    # Read image bytes
    image_bytes = await file.read()

    # Convert to OpenCV frame
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    result = tracker.process_frame(frame)

    return result
