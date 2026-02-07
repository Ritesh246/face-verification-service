# app/main.py

from fastapi import FastAPI, HTTPException
from typing import List
import numpy as np

from app.schemas import VerifyFaceRequest, VerifyFaceResponse, StudentResult
from app.image_loader import (
    get_registered_faces_by_rolls,
    get_attendance_selfie
)
from app.attendance_writer import write_attendance_records

from insightface.app import FaceAnalysis
import threading

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Face Verification Service")

# -----------------------------
# Lazy-loaded FaceAnalysis
# -----------------------------
_face_app = None
_face_lock = threading.Lock()

def get_face_app():
    global _face_app
    if _face_app is None:
        with _face_lock:
            if _face_app is None:
                print("🔄 Loading InsightFace model (lazy)...")
                fa = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"]
                )
                fa.prepare(ctx_id=0, det_size=(640, 640))
                _face_app = fa
                print("✅ InsightFace model loaded")
    return _face_app

# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

SIMILARITY_THRESHOLD = 0.35

# -----------------------------
# Health check (VERY IMPORTANT)
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# -----------------------------
# Verify Face Endpoint
# -----------------------------
@app.post("/verify-face", response_model=VerifyFaceResponse)
def verify_face(payload: VerifyFaceRequest):

    face_app = get_face_app()

    roll_numbers = [s.roll for s in payload.students]
    if not roll_numbers:
        raise HTTPException(status_code=400, detail="No students provided")

    registered_faces = get_registered_faces_by_rolls(roll_numbers)

    selfie_img = get_attendance_selfie(payload.selfie_image_url)
    selfie_faces = face_app.get(selfie_img)
    selfie_embeddings = [face.embedding for face in selfie_faces]

    api_results: List[StudentResult] = []
    attendance_results = []

    for roll in roll_numbers:
        status = "absent"

        if roll in registered_faces:
            reg_img = registered_faces[roll]["image"]
            reg_faces = face_app.get(reg_img)

            if reg_faces:
                reg_embedding = reg_faces[0].embedding
                best_score = max(
                    cosine_similarity(reg_embedding, emb)
                    for emb in selfie_embeddings
                )
                if best_score >= SIMILARITY_THRESHOLD:
                    status = "present"

        api_results.append(StudentResult(roll=roll, status=status))
        attendance_results.append({"roll": roll, "status": status})

    write_attendance_records(
        session_id=payload.session_id,
        class_id=payload.class_id,
        attendance_results=attendance_results,
        registered_faces=registered_faces
    )

    return VerifyFaceResponse(results=api_results)
