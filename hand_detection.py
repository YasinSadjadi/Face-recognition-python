import cv2
import mediapipe as mp
import pyautogui
import time



class HandDetector:

    def __init__(self):

        self.screen_width, self.screen_height = pyautogui.size()


        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.dragging = False
        self.pinch_start_time = None
        self.pinch_active = False

    # >>>>>>>>>>>>>>>>> main loop >>>>>>>>>>>>>>>>>

    def detect_hand(self, frame):
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

                x1 = int(index_finger_tip.x * frame_width)
                y1 = int(index_finger_tip.y * frame_height)

                x2 = int(thumb_tip.x * frame_width)
                y2 = int(thumb_tip.y * frame_height)

                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)

                cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)

                distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5

                screen_x = int(index_finger_tip.x * self.screen_width)
                screen_y = int(index_finger_tip.y * self.screen_height)

                # if not dragging:
                #     pyautogui.moveTo(screen_x, screen_y)
                # else:
                #     pyautogui.dragTo(screen_x, screen_y)

                # if distance < 0.05:
                #     if not dragging:
                #         # pyautogui.mouseDown()
                #         dragging = True
                #         print("Drag started")
                # else:
                #     if dragging:
                #         # pyautogui.mouseUp()
                #         dragging = False
                #         print("Drag stopped")

                # # if distance < 0.05:
                # #     pyautogui.click()
                # #     print(distance)

                if not self.dragging:
                    pyautogui.moveTo(screen_x, screen_y)
                else:
                    pyautogui.dragTo(screen_x, screen_y)

                if distance < 0.05:
                    if not self.pinch_active:
                        pinch_start_time = time.time()
                        pinch_active = True
                    else:
                        pinch_duration = time.time() - self.pinch_start_time

                        if pinch_duration > 0.3 and not self.dragging:
                            dragging = True
                            # pyautogui.mouseDown()
                            print("Drag started (after 1 second)")
                else:
                    if self.pinch_active:
                        pinch_duration = time.time() - self.pinch_start_time
                        self.pinch_active = False

                        if self.dragging:
                            self.dragging = False
                            # pyautogui.mouseUp()
                            print("Drag stopped")
                        elif pinch_duration < 0.3:
                            pyautogui.click()
                            print(f"Click performed (pinch duration: {pinch_duration:.2f} seconds)")

        return frame
