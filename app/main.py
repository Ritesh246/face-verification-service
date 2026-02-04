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


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Face Verification Service")


# -----------------------------
# Initialize InsightFace ONCE
# -----------------------------
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))


# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


SIMILARITY_THRESHOLD = 0.35


# -----------------------------
# Verify Face Endpoint
# -----------------------------
@app.post("/verify-face", response_model=VerifyFaceResponse)
def verify_face(payload: VerifyFaceRequest):

    # 1️⃣ Extract roll numbers
    roll_numbers = [s.roll for s in payload.students]

    if not roll_numbers:
        raise HTTPException(status_code=400, detail="No students provided")

    # 2️⃣ Fetch registered faces
    registered_faces = get_registered_faces_by_rolls(roll_numbers)

    print("🧪 REGISTERED_FACES DEBUG:")
    for roll, data in registered_faces.items():
        print(roll, data.keys())  # must contain student_id & image

    # 3️⃣ Fetch attendance selfie
    selfie_img = get_attendance_selfie(payload.selfie_image_url)

    # 4️⃣ Detect faces in selfie
    selfie_faces = face_app.get(selfie_img)
    selfie_embeddings = [face.embedding for face in selfie_faces]

    # --------------------------------
    # Two separate containers
    # --------------------------------
    api_results: List[StudentResult] = []     # for API response
    attendance_results = []                   # for DB insert (PLAIN DICT)

    # 5️⃣ Match each student
    for roll in roll_numbers:

        status = "absent"

        if roll in registered_faces:
            reg_img = registered_faces[roll]["image"]
            reg_faces = face_app.get(reg_img)

            if reg_faces:
                reg_embedding = reg_faces[0].embedding
                best_score = 0.0

                for selfie_emb in selfie_embeddings:
                    score = cosine_similarity(reg_embedding, selfie_emb)
                    best_score = max(best_score, score)

                if best_score >= SIMILARITY_THRESHOLD:
                    status = "present"

        # API response (Pydantic)
        api_results.append(
            StudentResult(roll=roll, status=status)
        )

        # DB insert payload (DICT ONLY)
        attendance_results.append({
            "roll": roll,
            "status": status
        })

    print("🧪 FINAL ATTENDANCE RESULTS (DB):", attendance_results)

    # 6️⃣ Write attendance to DB
    write_attendance_records(
        session_id=payload.session_id,
        class_id=payload.class_id,
        attendance_results=attendance_results,   # ✅ dicts
        registered_faces=registered_faces
    )

    return VerifyFaceResponse(results=api_results)
