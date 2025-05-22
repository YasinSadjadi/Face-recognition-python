import cv2 as cv
import numpy as np
import threading


class Capture:
    def __init__(self, camera_index: int or str):  # type: ignore
        self._capture = cv.VideoCapture(camera_index)
        if self._capture.isOpened() is False:
            raise Exception("Failed to load camera")

        res = self._capture.set(cv.CAP_PROP_AUTOFOCUS, 0)
        print("Auto Focus:", res)
        print("bufferSize: ", self._capture.get(cv.CAP_PROP_BUFFERSIZE))
        print("fps: ", self._capture.get(cv.CAP_PROP_FPS))

    def read(self) -> np.ndarray:
        image = self._capture.read()[1]
        image = cv.resize(image, (640, 480))
        return image


class CaptureMulty:
    def __init__(self, camera_index):
        self._capture = cv.VideoCapture(camera_index)
        if self._capture.isOpened() is False:
            raise Exception("Failed to load camera")

        # setting camera properties
        res = self._capture.set(cv.CAP_PROP_AUTOFOCUS, 0)
        print("Auto Focus:", res)
        self._capture.set(cv.CAP_PROP_FPS, 30)
        self._capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self._capture.set(cv.CAP_PROP_N_THREADS, 2)

        # our class variables
        self._frame = None
        self._running = True

        # start the thread to read frames from the video stream
        self._thread = threading.Thread(target=self._read_frame)
        self._thread.daemon = True
        self._thread.start()

    def _read_frame(self):
        # keep looping infinitely until the thread is stopped
        while self._running:
            ret, frame = self._capture.read()
            if not ret:
                continue
            self._frame = frame

    def read(self) -> np.ndarray:
        # return the frame most recently read
        return cv.flip(self._frame, 1)

    def release(self):
        # indicate that the thread should be stopped
        self._running = False
        self._thread.join()
        self._capture.release()