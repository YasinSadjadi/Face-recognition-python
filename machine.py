import os
import cv2 as cv
from TTS.api import TTS
import sounddevice as sd
from face_recognition import FaceRecognizer
from hand_detection import HandDetector
from Models import User
from enum import Enum
from capture import CaptureMulty as Capture

class States(Enum):
    TRAINING = 0
    RECOGNIZING = 1


class Main:
    SAMPLE_RATE = 20500
    def __init__(self, face_recognition_model):
        self.cap = Capture(0)
        self.model = face_recognition_model
        self.hand_detector = HandDetector()
        self.detected_users = []
        self.state = States.TRAINING
        self.tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=True)

        self.enter_your_name = self.tts.tts("if you want to train, enter your name, else press enter")
        self.press_button_train = self.tts.tts("press 't' to train, press 'Esc' to exit")
        self.press_button_recognize = self.tts.tts("if you want training press 't', and if not, press 'Esc' to exit")


    def run(self, default_state=States.TRAINING):
        self.state = default_state
        self.machine()

    def transition(self, next_state):
        Main.clear_console()
        print(f"transitioning from {self.state.name} to {next_state.name}")
        self.state = next_state

    def machine(self):
        while True:
            if self.state == States.TRAINING:
                self.training()
            elif self.state == States.RECOGNIZING:
                self.recognizing()

    def training(self):

        sd.play(self.enter_your_name, samplerate=Main.SAMPLE_RATE)
        name = input("if you want to train, enter your name, else press enter: ")

        if name == "":
            self.transition(States.RECOGNIZING)
            return


        sd.play(self.press_button_train, samplerate=Main.SAMPLE_RATE)
        print("press 't' to train, press 'Esc' to exit")

        read = -1
        while True:
            frame = self.cap.read()
            if read == 27:
                self.transition(States.RECOGNIZING)
                break

            elif read == ord('t') or read == ord('T'):
                self.model.train(user_id=name, image=frame)

            cv.imshow("frame", frame)
            read = cv.waitKey(1)


    def recognizing(self):

        sd.play(self.press_button_recognize, samplerate=Main.SAMPLE_RATE)
        print("if you want training press 't', and if not, press 'Esc' to exit")

        read = -1
        while True:
            frame = self.cap.read()
            if read == 27:
                quit()
            elif read == ord('t') or read == ord('T'):
                self.transition(States.TRAINING)
                break

            res = self.model.recognize(frame)
            not_detected = [user[0] for user in res if not user[0] in self.detected_users and user[0] == "unknown"]
            for user in not_detected:
                if user == "unknown":
                    continue
                audio_array = self.tts.tts(f"hello {user}")
                sd.play(audio_array, samplerate=Main.SAMPLE_RATE)

            for result in res:
                if result[0] == "unknown":
                    continue
                panel_center = []

                face_center, face_width, face_height = result[2]

                panel_center.append(int(face_center[0] + round(face_width * 1.5)))
                panel_center.append(int(face_center[1] + round(face_height * 1.5)))

                panel_first = (panel_center[0] - int(face_width * 1.4), panel_center[1] - face_height)
                panel_second = (panel_center[0] + face_width, panel_center[1] + face_height)

                cropped_panel = frame[panel_first[1]:panel_second[1], panel_first[0]:panel_second[0]]

                cropped_panel = self.hand_detector.detect_hand(cropped_panel)

                frame[panel_first[1]:panel_second[1], panel_first[0]:panel_second[0]] = cropped_panel
                cv.rectangle(frame, panel_first, panel_second, (0, 0, 255), 2)


            self.detected_users.extend(not_detected)

            cv.imshow("frame", frame)
            read = cv.waitKey(1)


    @staticmethod
    def clear_console():
        os.system('cls' if os.name == 'nt' else 'clear')


main = Main(FaceRecognizer(users=User.read_all_users()))
main.run()