from datetime import date
from app.supabase_client import get_supabase_client
from postgrest.exceptions import APIError

supabase = get_supabase_client()

def write_attendance_records(session_id, class_id, attendance_results, registered_faces):
    today = date.today()

    inserted = []
    skipped = []

    for record in attendance_results:
        roll = record["roll"]
        status = record["status"]

        if status != "present":
            continue

        try:
            supabase.table("attendance_records").insert({
                "session_id": session_id,
                "class_id": class_id,
                "student_id": registered_faces[roll]["student_id"],
                "roll_number": str(roll),
                "status": "present",
                "attendance_date": today
            }).execute()

            inserted.append(roll)

        except APIError as e:
            # Unique constraint violation → already marked today
            if "unique_daily_attendance" in str(e):
                print(f"⚠️ Roll {roll} already marked today. Skipping.")
                skipped.append(roll)
            else:
                raise e

    return {
        "inserted": inserted,
        "skipped": skipped
    }
