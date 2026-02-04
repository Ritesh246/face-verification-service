# app/attendance_writer.py

from datetime import datetime, timezone
from app.supabase_client import get_supabase_client

supabase = get_supabase_client()

def write_attendance_records(
    session_id: str,
    class_id: str,
    attendance_results: list,
    registered_faces: dict
):
    """
    attendance_results:
    [
        { "roll": 47, "status": "present" },
        { "roll": 48, "status": "absent" }
    ]
    """

    rows = []

    for result in attendance_results:
        roll = result["roll"]          # ✅ STRICT
        status = result["status"]

        if roll not in registered_faces:
            print(f"⚠️ Roll {roll} not found in registered_faces")
            continue

        student_id = registered_faces[roll]["student_id"]

        rows.append({
            "session_id": session_id,
            "class_id": class_id,
            "student_id": student_id,           # ✅ NOT NULL
            "roll_number": roll,                # ✅ NOT NULL
            "status": status,
            "marked_at": datetime.now(timezone.utc).isoformat()  # ✅ SERIALIZABLE
        })

    if not rows:
        raise Exception("No attendance rows to insert")

    supabase.table("attendance_records").insert(rows).execute()
