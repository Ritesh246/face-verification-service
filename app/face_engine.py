import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis

THRESHOLD = 0.6

class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def load_image_from_url(self, url: str):
        resp = requests.get(url, timeout=10)
        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    def load_image_from_file(self, path: str):
        return cv2.imread(path)

    def cosine_similarity(self, a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return float(np.dot(a, b))

    def match(self, selfie_img, registered_images: dict):
        """
        registered_images: { roll: cv2_image }
        returns: { roll: "present" | "absent" }
        """

        results = {}
        selfie_faces = self.app.get(selfie_img)

        used_faces = set()

        for roll, reg_img in registered_images.items():
            reg_faces = self.app.get(reg_img)

            if len(reg_faces) != 1 or len(selfie_faces) == 0:
                results[roll] = "absent"
                continue

            reg_emb = reg_faces[0].embedding

            best_score = -1
            best_index = None

            for i, face in enumerate(selfie_faces):
                if i in used_faces:
                    continue

                score = self.cosine_similarity(reg_emb, face.embedding)
                if score > best_score:
                    best_score = score
                    best_index = i

            if best_score >= THRESHOLD and best_index is not None:
                results[roll] = "present"
                used_faces.add(best_index)
            else:
                results[roll] = "absent"

        return results
