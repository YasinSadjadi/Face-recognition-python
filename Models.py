import os
import cv2

class User:
    def __init__(self, user_id: str, base_dir: str = 'faces'):
        self.user_id = user_id
        self.user_dir = os.path.join(base_dir, user_id)
        self.images = []

        self.embeddings = []

        os.makedirs(self.user_dir, exist_ok=True)
        self._load_images()

    def _load_images(self):
        self.images.clear()
        for filename in sorted(os.listdir(self.user_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.user_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    self.images.append(img)

    def add_image(self, image, save=True):
        if save:
            index = len(self.images) + 1
            filename = os.path.join(self.user_dir, f"{index:03d}.jpg")
            cv2.imwrite(filename, image)
        self.images.append(image)

    def get_images(self):
        return self.images

    @staticmethod
    def read_all_users():
        users = []
        user_names = os.listdir('faces')
        for user_name in user_names:
            users.append(User(user_name))

        return users
