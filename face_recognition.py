import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from Models import User

class FaceRecognizer:
    def __init__(self, users):
        self.model = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider'])

        self.model.prepare(ctx_id=0)
        self.users: list[User] = users
        self.load()


    def load(self):
        for user in self.users:
            for image in user.images:
                user.embeddings.append(self.get_embedding(image))

    def train(self, user_id: str, image: np.ndarray):
        faces = self.model.get(image)
        if not faces:
            return False
        emb = faces[0]['embedding']
        x, y, x2, y2 = map(int, faces[0]['bbox'])
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        result = [u for u in self.users if user_id == u.user_id]
        if len(result) > 0:
            result = result[0]
        else:
            result = User(user_id)
            self.users.append(result)
        result.add_image(image, save=True)
        result.embeddings.append(emb)
        return True

    def recognize(self, image: np.ndarray, threshold: float = 0.4):
        faces = self.model.get(image)
        results = []
        for face in faces:
            emb = face['embedding']
            best_match, best_score = "unknown", 1.0
            for user in self.users:
                for known_emb in user.embeddings:
                    score = self.get_similarity(emb, known_emb)
                    if score < best_score:
                        best_match, best_score = user.user_id, score

            x, y, x2, y2 = map(int, face['bbox'])
            center = ((x + x2) / 2, (y + y2) / 2)
            width = x2 - x
            height = y2 - y
            if best_score < threshold:
                results.append((best_match, best_score, (center, width, height)))
                cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{best_match} , {best_score: 0.01f}",
                            (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

            else:
                results.append(("unknown", best_score, (center, width, height)))
                cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, f"unknown , {best_score: 0.01f}",
                            (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
        return results

    def get_rectangle(self, image: np.ndarray):
        faces = self.model.get(image)
        return [face['bbox'] for face in faces]

    def get_embedding(self, image: np.ndarray):
        faces = self.model.get(image)
        return faces[0]['embedding'] if faces else None

    @staticmethod
    def get_similarity(emb1: np.ndarray, emb2: np.ndarray):
        return cosine(emb1, emb2)
