"""
vision/pose_tracker.py

Full-body pose tracking using MediaPipe Pose.

Detects raised hands and basic body postures to trigger OS actions:

    right hand raised  → next reel
    left hand raised   → previous reel
    both hands raised  → volume up
    lean forward       → (placeholder for future interaction)

Pose landmark reference (selected):
    11 – LEFT_SHOULDER
    12 – RIGHT_SHOULDER
    15 – LEFT_WRIST
    16 – RIGHT_WRIST
    23 – LEFT_HIP
    24 – RIGHT_HIP
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# ── pose action constants ─────────────────────────────────────────────────────
POSE_NONE          = "none"
POSE_RIGHT_HAND_UP = "right_hand_up"
POSE_LEFT_HAND_UP  = "left_hand_up"
POSE_BOTH_HANDS_UP = "both_hands_up"

# MediaPipe landmark indices
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_WRIST    = 15
_R_WRIST    = 16
_L_HIP      = 23
_R_HIP      = 24


class PoseTracker:
    """
    Classifies full-body pose gestures from a single BGR frame.

    Args:
        min_detection_conf: Minimum detection confidence.
        min_tracking_conf:  Minimum tracking confidence.
        wrist_shoulder_margin: How far (in normalised units) above the
                               shoulder the wrist must be to count as "raised".
    """

    def __init__(
        self,
        min_detection_conf: float = 0.5,
        min_tracking_conf:  float = 0.5,
        wrist_shoulder_margin: float = 0.05,
    ):
        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_draw_styles = mp.solutions.drawing_styles

        self.pose = self._mp_pose.Pose(
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self.margin = wrist_shoulder_margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> str:
        """
        Process a BGR frame and return a POSE_* action string.

        Side-effect: draws the pose skeleton onto the frame in-place.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not result.pose_landmarks:
            return POSE_NONE

        self._mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            self._mp_draw_styles.get_default_pose_landmarks_style(),
        )

        lm = result.pose_landmarks.landmark
        return self._classify(lm)

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify(self, lm) -> str:
        """
        Determine pose action from wrist / shoulder positions.

        A wrist is "raised" when its y-coordinate is above (lower number)
        the shoulder y-coordinate by more than `margin`.
        Memory aid: in image space y=0 is at the *top*.
        """
        l_wrist_up = (lm[_L_SHOULDER].y - lm[_L_WRIST].y) > self.margin
        r_wrist_up = (lm[_R_SHOULDER].y - lm[_R_WRIST].y) > self.margin

        if l_wrist_up and r_wrist_up:
            return POSE_BOTH_HANDS_UP
        if r_wrist_up:
            return POSE_RIGHT_HAND_UP
        if l_wrist_up:
            return POSE_LEFT_HAND_UP

        return POSE_NONE
