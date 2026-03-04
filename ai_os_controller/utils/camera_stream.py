"""
utils/camera_stream.py

Threaded webcam capture to prevent frame-blocking in the main loop.
Reading frames in a dedicated thread ensures the processing pipeline
always gets the latest frame without waiting for I/O.
"""

import cv2
import threading
import time


class CameraStream:
    """
    Non-blocking, threaded webcam wrapper.

    Usage:
        cam = CameraStream(src=0, width=1280, height=720).start()
        frame = cam.read()
        cam.stop()
    """

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # CAP_DSHOW is faster on Windows
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # keep buffer minimal for low latency

        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera source {src}")

        self.ret = False
        self.frame = None
        self.stopped = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraStream":
        """Spawn the background capture thread and return self for chaining."""
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        time.sleep(0.3)          # give the camera a moment to warm up
        return self

    def read(self):
        """Return the most-recently captured frame (thread-safe)."""
        with self._lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Signal the background thread to exit and release the device."""
        self.stopped = True
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)
        self.cap.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update(self):
        """Continuously grab frames until stop() is called."""
        while not self.stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret
                self.frame = frame
