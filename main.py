import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from Models import User
from face_recognition import FaceRecognizer
from machine import Main, States

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face & Hand Recognition System")

        self.main = Main(FaceRecognizer(users=User.read_all_users()))
        self.main.run = self.run_gui  # override

        # UI Elements
        self.image_label = QLabel()
        self.start_button = QPushButton("Start Recognition")
        self.train_button = QPushButton("Start Training")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.train_button)
        layout.addLayout(buttons)
        self.setLayout(layout)

        # Signals
        self.start_button.clicked.connect(lambda: self.main.transition(States.RECOGNIZING))
        self.train_button.clicked.connect(lambda: self.main.transition(States.TRAINING))

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def run_gui(self, default_state=States.TRAINING):
        self.main.state = default_state

    def update_frame(self):
        frame = self.main.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # جایگزینی بخش training و recognizing
        if self.main.state == States.TRAINING:
            # کدی مثل self.main.training() رو بازنویسی می‌کنیم
            cv2.putText(frame, "Training mode - press T in terminal", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        elif self.main.state == States.RECOGNIZING:
            res = self.main.model.recognize(frame)
            for result in res:
                if result[0] != "unknown":
                    x, y = result[2][0]
                    cv2.putText(frame, f"Hello {result[0]}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # نمایش فریم
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
