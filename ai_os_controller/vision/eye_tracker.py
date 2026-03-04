"""
vision/eye_tracker.py

Iris-based gaze direction detection using MediaPipe FaceMesh.

MediaPipe FaceMesh (with refine_landmarks=True) exposes iris landmarks
468–477.  We use:

    468 – left iris centre
    473 – right iris centre

And the eye-corner landmarks for horizontal normalisation:

    33  – left eye outer corner
    133 – left eye inner corner
    362 – right eye inner corner
    263 – right eye outer corner

Gaze output:  "left" | "right" | "center"

Blink detection is performed using the Eye Aspect Ratio (EAR):

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

A blink is detected when EAR drops below EAR_THRESHOLD for several
consecutive frames.  A double-blink is two blinks within DOUBLE_BLINK_WINDOW
seconds.
"""

import time
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# ── gaze labels ──────────────────────────────────────────────────────────────
GAZE_LEFT   = "left"
GAZE_RIGHT  = "right"
GAZE_CENTER = "center"

# ── blink labels ─────────────────────────────────────────────────────────────
BLINK_NONE   = None
BLINK_SINGLE = "single"
BLINK_DOUBLE = "double"

# ── FaceMesh landmark indices ─────────────────────────────────────────────────
LEFT_IRIS   = 468
RIGHT_IRIS  = 473

# Left eye EAR landmarks (p1..p6 clockwise)
LEFT_EYE_EAR  = [33, 160, 158, 133, 153, 144]
# Right eye EAR landmarks
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Horizontal normalisation corners
LEFT_EYE_OUTER  = 33
LEFT_EYE_INNER  = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Gaze thresholds (normalised to eye width)
GAZE_RIGHT_THRESH = 0.55
GAZE_LEFT_THRESH  = 0.45

# Blink thresholds
EAR_THRESHOLD          = 0.20
EAR_CONSEC_FRAMES      = 2       # frames iris must stay closed
DOUBLE_BLINK_WINDOW    = 0.5     # seconds between two blinks = double blink


class EyeTracker:
    """
    Detects gaze direction and blinks from a single webcam frame.

    Args:
        ear_threshold:   EAR value below which an eye is considered closed.
        gaze_left_th:    Iris-ratio threshold for leftward gaze.
        gaze_right_th:   Iris-ratio threshold for rightward gaze.
    """

    def __init__(
        self,
        ear_threshold:  float = EAR_THRESHOLD,
        gaze_left_th:   float = GAZE_LEFT_THRESH,
        gaze_right_th:  float = GAZE_RIGHT_THRESH,
    ):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,           # enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._draw = mp.solutions.drawing_utils
        self._draw_spec = self._draw.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1
        )

        self.ear_threshold  = ear_threshold
        self.gaze_left_th   = gaze_left_th
        self.gaze_right_th  = gaze_right_th

        # Blink state
        self._blink_counter = 0
        self._blink_in_progress = False
        self._last_blink_time   = 0.0
        self._blink_count_window = 0    # blinks within the double-blink window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_gaze(self, frame: np.ndarray) -> Optional[str]:
        """
        Analyse frame and return gaze direction string, or None if no face.
        """
        lm = self._get_landmarks(frame)
        if lm is None:
            return None

        gaze_ratio = self._iris_ratio(lm)

        if gaze_ratio > self.gaze_right_th:
            return GAZE_RIGHT
        if gaze_ratio < self.gaze_left_th:
            return GAZE_LEFT
        return GAZE_CENTER

    def get_blink(self, frame: np.ndarray) -> Optional[str]:
        """
        Analyse frame and return blink event string, or None if no blink.

        Returns:
            BLINK_SINGLE – one complete blink detected
            BLINK_DOUBLE – second blink within the time window
            None         – no blink event this frame
        """
        lm = self._get_landmarks(frame)
        if lm is None:
            return None

        ear = self._calculate_ear(lm)

        if ear < self.ear_threshold:
            self._blink_counter += 1
            self._blink_in_progress = True
        else:
            if self._blink_in_progress and self._blink_counter >= EAR_CONSEC_FRAMES:
                return self._register_blink()
            self._blink_counter = 0
            self._blink_in_progress = False

        return None

    def process(self, frame: np.ndarray):
        """
        Convenience method: run both gaze and blink in one call.

        Returns:
            (gaze: str | None, blink: str | None)
        """
        lm = self._get_landmarks(frame)
        if lm is None:
            return None, None

        gaze_ratio = self._iris_ratio(lm)
        if gaze_ratio > self.gaze_right_th:
            gaze = GAZE_RIGHT
        elif gaze_ratio < self.gaze_left_th:
            gaze = GAZE_LEFT
        else:
            gaze = GAZE_CENTER

        ear = self._calculate_ear(lm)
        blink = None
        if ear < self.ear_threshold:
            self._blink_counter += 1
            self._blink_in_progress = True
        else:
            if self._blink_in_progress and self._blink_counter >= EAR_CONSEC_FRAMES:
                blink = self._register_blink()
            self._blink_counter = 0
            self._blink_in_progress = False

        return gaze, blink

    def close(self):
        self._mesh.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_landmarks(self, frame: np.ndarray):
        """Run FaceMesh and return the landmark list for the first face."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        rgb.flags.writeable = True

        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    def _iris_ratio(self, lm) -> float:
        """
        Return a value 0–1 representing horizontal iris position within the eye.

        0.0 = far left  |  0.5 = centre  |  1.0 = far right
        """
        left_iris  = lm[LEFT_IRIS]
        right_iris = lm[RIGHT_IRIS]
        avg_iris_x = (left_iris.x + right_iris.x) / 2.0

        # Normalise using eye width
        outer = lm[LEFT_EYE_OUTER]
        inner = lm[RIGHT_EYE_OUTER]
        eye_width = abs(inner.x - outer.x) + 1e-6

        return (avg_iris_x - outer.x) / eye_width

    @staticmethod
    def _calculate_ear(lm) -> float:
        """
        Eye Aspect Ratio – average of both eyes to reduce noise.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        def ear_one(indices):
            p = [lm[i] for i in indices]
            v1 = abs(p[1].y - p[5].y)
            v2 = abs(p[2].y - p[4].y)
            h  = abs(p[0].x - p[3].x) + 1e-6
            return (v1 + v2) / (2.0 * h)

        return (ear_one(LEFT_EYE_EAR) + ear_one(RIGHT_EYE_EAR)) / 2.0

    def _register_blink(self) -> str:
        """Record blink timestamp and decide single vs. double."""
        now = time.monotonic()
        self._blink_counter = 0
        self._blink_in_progress = False

        if now - self._last_blink_time < DOUBLE_BLINK_WINDOW:
            self._last_blink_time = 0.0   # reset so triple doesn't re-trigger
            return BLINK_DOUBLE

        self._last_blink_time = now
        return BLINK_SINGLE
