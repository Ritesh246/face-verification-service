from pydantic import BaseModel
from typing import List


class StudentInput(BaseModel):
    roll: int


class VerifyFaceRequest(BaseModel):
    class_id: str
    session_id: str
    selfie_image_url: str
    students: List[StudentInput]


class StudentResult(BaseModel):
    roll: int
    status: str


class VerifyFaceResponse(BaseModel):
    results: List[StudentResult]
