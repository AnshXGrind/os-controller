"""
vision/hand_tracker.py

MediaPipe-based hand gesture recogniser.

Detects 21 hand landmarks and classifies the hand shape into one of
the following gesture constants:

    G_INDEX_UP  – only index finger extended    → next reel
    G_PEACE     – index + middle extended        → previous reel
    G_THUMB_UP  – only thumb extended            → like
    G_FIST      – all fingers closed             → pause
    G_OPEN      – all five fingers extended      → cursor control mode
    G_UNKNOWN   – anything else
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

from ai_os_controller.utils.gesture_utils import get_finger_states
from ai_os_controller.utils.smoothing import LandmarkSmoother, GestureConfirmer

# ── gesture label constants ──────────────────────────────────────────────────
G_INDEX_UP  = "index_up"
G_PEACE     = "peace"
G_THUMB_UP  = "thumb_up"
G_FIST      = "fist"
G_OPEN      = "open"
G_UNKNOWN   = "unknown"


class HandTracker:
    """
    Wraps MediaPipe Hands and classifies hand gestures each frame.

    Args:
        max_hands:          Maximum number of hands to detect.
        min_detection_conf: Minimum detection confidence threshold.
        min_tracking_conf:  Minimum tracking confidence threshold.
        confirm_frames:     Number of consecutive frames before a gesture
                            is confirmed (reduces false positives).
        smooth_alpha:       EMA alpha for landmark smoothing (0–1).
    """

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_conf: float = 0.7,
        min_tracking_conf: float = 0.6,
        confirm_frames: int = 4,
        smooth_alpha: float = 0.5,
    ):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_draw_styles = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

        # Smoother for 21 landmarks (x, y, z) each
        self._smoother   = LandmarkSmoother(num_landmarks=21, alpha=smooth_alpha)
        self._confirmer  = GestureConfirmer(required_frames=confirm_frames)

        # Last stable gesture (persists until a new one is confirmed)
        self.current_gesture: str = G_UNKNOWN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> str:
        """
        Process a single BGR frame and return the confirmed gesture string.

        Side-effect: draws landmark overlay *onto* the frame in-place.

        Returns:
            One of the G_* gesture constants.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            self._smoother.reset()
            raw_gesture = G_UNKNOWN
        else:
            hand_lm = results.multi_hand_landmarks[0]

            # ── draw skeleton ─────────────────────────────────────────
            self._mp_draw.draw_landmarks(
                frame,
                hand_lm,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_draw_styles.get_default_hand_landmarks_style(),
                self._mp_draw_styles.get_default_hand_connections_style(),
            )

            # ── smooth landmarks ──────────────────────────────────────
            raw_pts = np.array([[lm.x, lm.y, lm.z]
                                 for lm in hand_lm.landmark])
            smoothed = self._smoother.smooth(raw_pts)

            # Rebuild a lightweight list of objects with .x/.y/.z
            lm_list = [_LM(p[0], p[1], p[2]) for p in smoothed]

            raw_gesture = self._classify(lm_list)

        # ── confirm gesture over N frames ────────────────────────────
        confirmed = self._confirmer.confirm(raw_gesture)
        if confirmed:
            self.current_gesture = confirmed

        return self.current_gesture

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(lm) -> str:
        """
        Classify the hand shape from a smoothed landmark list.

        Uses simple finger-state booleans (extended / bent).
        """
        fs = get_finger_states(lm)

        _t = fs["thumb"]
        _i = fs["index"]
        _m = fs["middle"]
        _r = fs["ring"]
        _p = fs["pinky"]

        # Open palm – all fingers extended
        if _t and _i and _m and _r and _p:
            return G_OPEN

        # Peace sign – index + middle up, rest down
        if _i and _m and not _r and not _p:
            return G_PEACE

        # Index up – only index extended
        if _i and not _m and not _r and not _p:
            return G_INDEX_UP

        # Thumb up – only thumb extended
        if _t and not _i and not _m and not _r and not _p:
            return G_THUMB_UP

        # Closed fist – nothing extended
        if not _t and not _i and not _m and not _r and not _p:
            return G_FIST

        return G_UNKNOWN


# ---------------------------------------------------------------------------
# Minimal landmark proxy
# ---------------------------------------------------------------------------

class _LM:
    """Lightweight stand-in for mediapipe NormalizedLandmark."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
