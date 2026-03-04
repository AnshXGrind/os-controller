"""
utils/gesture_utils.py

Geometric helpers used by the hand gesture classifier.

All helpers work directly with MediaPipe NormalizedLandmark objects
(or any object with .x, .y, .z attributes).
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Distance / angle helpers
# ---------------------------------------------------------------------------

def euclidean_distance(p1, p2) -> float:
    """Return the Euclidean distance between two 2-D landmark points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def landmark_to_np(landmark) -> np.ndarray:
    """Convert a single MediaPipe landmark to a NumPy (x, y, z) array."""
    return np.array([landmark.x, landmark.y, getattr(landmark, "z", 0.0)])


def angle_between_points(a, b, c) -> float:
    """
    Compute the interior angle (degrees) at vertex *b* formed by the
    three points a → b → c.

    Used to determine whether a finger joint is extended or bent.
    """
    ba = landmark_to_np(a) - landmark_to_np(b)
    bc = landmark_to_np(c) - landmark_to_np(b)

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


# ---------------------------------------------------------------------------
# Finger state helpers (straight / bent)
# ---------------------------------------------------------------------------

# MediaPipe hand landmark indices
MP_WRIST        = 0
MP_THUMB_CMC    = 1
MP_THUMB_MCP    = 2
MP_THUMB_IP     = 3
MP_THUMB_TIP    = 4

MP_INDEX_MCP    = 5
MP_INDEX_PIP    = 6
MP_INDEX_DIP    = 7
MP_INDEX_TIP    = 8

MP_MIDDLE_MCP   = 9
MP_MIDDLE_PIP   = 10
MP_MIDDLE_DIP   = 11
MP_MIDDLE_TIP   = 12

MP_RING_MCP     = 13
MP_RING_PIP     = 14
MP_RING_DIP     = 15
MP_RING_TIP     = 16

MP_PINKY_MCP    = 17
MP_PINKY_PIP    = 18
MP_PINKY_DIP    = 19
MP_PINKY_TIP    = 20


def finger_is_extended(lm, tip_idx: int, mcp_idx: int, threshold: float = 0.04) -> bool:
    """
    A finger is 'extended' when its TIP landmark is farther from the wrist
    (in the y-axis) than its MCP knuckle.  Works in normalized image space
    where y=0 is the top of the frame.

    For thumbs, we use the x-axis comparison instead.
    """
    wrist = lm[MP_WRIST]
    tip = lm[tip_idx]
    mcp = lm[mcp_idx]

    # Vertical distance from wrist to tip vs. from wrist to mcp
    tip_dist  = euclidean_distance(tip, wrist)
    mcp_dist  = euclidean_distance(mcp, wrist)
    return (tip_dist - mcp_dist) > threshold


def get_finger_states(lm) -> dict:
    """
    Return a dict of booleans for each finger.

    Keys: 'thumb', 'index', 'middle', 'ring', 'pinky'
    """
    return {
        "thumb":  finger_is_extended(lm, MP_THUMB_TIP,  MP_THUMB_MCP,  threshold=0.03),
        "index":  finger_is_extended(lm, MP_INDEX_TIP,  MP_INDEX_MCP,  threshold=0.04),
        "middle": finger_is_extended(lm, MP_MIDDLE_TIP, MP_MIDDLE_MCP, threshold=0.04),
        "ring":   finger_is_extended(lm, MP_RING_TIP,   MP_RING_MCP,   threshold=0.04),
        "pinky":  finger_is_extended(lm, MP_PINKY_TIP,  MP_PINKY_MCP,  threshold=0.04),
    }


# ---------------------------------------------------------------------------
# Cooldown timer
# ---------------------------------------------------------------------------

import time


class CooldownTimer:
    """
    Prevents an action from firing too frequently.

    Usage:
        timer = CooldownTimer(seconds=1.0)
        if timer.ready():
            do_action()
    """

    def __init__(self, seconds: float = 1.0):
        self.interval = seconds
        self._last_trigger = 0.0

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._last_trigger >= self.interval:
            self._last_trigger = now
            return True
        return False

    def reset(self):
        self._last_trigger = 0.0
