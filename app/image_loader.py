# app/image_loader.py

import cv2
import numpy as np
import requests

from app.supabase_client import get_supabase_client

# Create Supabase client once
supabase = get_supabase_client()


# -------------------------------
# Utils
# -------------------------------

def load_image_from_url(url: str) -> np.ndarray:
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    return img


# -------------------------------
# Attendance selfie (PUBLIC bucket)
# -------------------------------

def get_attendance_selfie(selfie_url: str) -> np.ndarray:
    """
    selfie_url example:
    https://<project>.supabase.co/storage/v1/object/public/attendance-selfies/...
    """

    print(f"🔥 SELFIE URL RECEIVED (PUBLIC): {selfie_url}")

    # ✅ Direct HTTP fetch (NO signed URL)
    return load_image_from_url(selfie_url)


# -------------------------------
# Registered student face (PRIVATE bucket)
# -------------------------------

def get_registered_face(face_image_path: str) -> np.ndarray:
    """
    face_image_path example:
    students/<uuid>.jpg
    """

    signed = (
        supabase
        .storage
        .from_("face-images")
        .create_signed_url(face_image_path, expires_in=60)
    )

    signed_url = signed["signedURL"]
    return load_image_from_url(signed_url)


# -------------------------------
# Fetch registered faces by roll numbers
# -------------------------------

def get_registered_faces_by_rolls(roll_numbers: list[int]) -> dict:
    """
    Returns:
    {
        roll_no: {
            "student_id": uuid,
            "image": np.ndarray
        }
    }
    """

    response = (
        supabase
        .table("profiles")
        .select("id, roll_no, face_image_path")
        .in_("roll_no", roll_numbers)
        .execute()
    )

    registered_faces = {}

    for row in response.data:
        roll = row["roll_no"]

        img = get_registered_face(row["face_image_path"])

        registered_faces[roll] = {
            "student_id": row["id"],
            "image": img
        }

    return registered_faces
