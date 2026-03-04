"""
vision/pose_tracker.py

Full-body pose tracking using MediaPipe PoseLandmarker
(Tasks API – mediapipe ≥ 0.10.30).

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

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mpv
import numpy as np

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

# ── model path ───────────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).parent.parent / "models"
_POSE_MODEL = _MODELS_DIR / "pose_landmarker.task"

# Pose connections for drawing (subset of 33 landmarks)
_POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),          # face outline
    (0,4),(4,5),(5,6),(6,8),
    (9,10),                            # mouth
    (11,12),(11,13),(13,15),(15,17),(15,19),(17,19),  # left arm
    (12,14),(14,16),(16,18),(16,20),(18,20),           # right arm
    (11,23),(12,24),(23,24),           # torso
    (23,25),(25,27),(27,29),(27,31),(29,31),           # left leg
    (24,26),(26,28),(28,30),(28,32),(30,32),           # right leg
]


class PoseTracker:
    """
    Classifies full-body pose gestures from a single BGR frame.

    Args:
        min_detection_conf:    Minimum detection confidence.
        min_tracking_conf:     Minimum tracking confidence.
        wrist_shoulder_margin: How far (in normalised units) above the
                               shoulder the wrist must be to count as "raised".
    """

    def __init__(
        self,
        min_detection_conf: float = 0.5,
        min_tracking_conf:  float = 0.5,
        wrist_shoulder_margin: float = 0.05,
    ):
        base_options = mp_python.BaseOptions(model_asset_path=str(_POSE_MODEL))
        options = mpv.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mpv.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_conf,
            min_pose_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._landmarker = mpv.PoseLandmarker.create_from_options(options)
        self._start_ms   = int(time.time() * 1000)
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return POSE_NONE

        lm = result.pose_landmarks[0]

        # ── draw skeleton ─────────────────────────────────────────────
        h, w = frame.shape[:2]
        for a, b in _POSE_CONNECTIONS:
            if a < len(lm) and b < len(lm):
                p1 = lm[a]
                p2 = lm[b]
                cv2.line(frame,
                         (int(p1.x * w), int(p1.y * h)),
                         (int(p2.x * w), int(p2.y * h)),
                         (0, 255, 255), 2)
        for landmark in lm:
            cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 3, (0, 0, 255), -1)

        return self._classify(lm)

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()

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
