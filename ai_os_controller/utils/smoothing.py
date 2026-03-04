"""
utils/smoothing.py

Landmark and coordinate smoothing utilities.

Applying smoothing to raw landmark coordinates removes jitter caused by
minor hand/face movements and makes gesture detection far more stable.
"""

from collections import deque
import numpy as np


class LandmarkSmoother:
    """
    Exponential moving average (EMA) smoother for a set of 2-D/3-D points.

    alpha close to 1.0 → fast response, more jitter.
    alpha close to 0.0 → very smooth, but laggy.
    """

    def __init__(self, num_landmarks: int, alpha: float = 0.5):
        """
        Args:
            num_landmarks: Number of (x, y) or (x, y, z) points.
            alpha:         EMA weight for the newest sample (0 < alpha ≤ 1).
        """
        self.alpha = alpha
        self.smoothed = None             # shape: (num_landmarks, dims)
        self.num_landmarks = num_landmarks

    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Update internal EMA state and return the smoothed landmark array.

        Args:
            landmarks: np.ndarray of shape (num_landmarks, dims).

        Returns:
            Smoothed landmark array of the same shape.
        """
        if self.smoothed is None:
            self.smoothed = landmarks.copy()
        else:
            self.smoothed = self.alpha * landmarks + (1 - self.alpha) * self.smoothed
        return self.smoothed

    def reset(self):
        """Clear state (e.g. when hand leaves the frame)."""
        self.smoothed = None


class CoordinateSmoother:
    """
    Simple moving-average smoother for a single (x, y) coordinate pair.
    Useful for cursor control to prevent jittery mouse movement.
    """

    def __init__(self, window: int = 5):
        self._x = deque(maxlen=window)
        self._y = deque(maxlen=window)

    def smooth(self, x: float, y: float):
        self._x.append(x)
        self._y.append(y)
        return float(np.mean(self._x)), float(np.mean(self._y))

    def reset(self):
        self._x.clear()
        self._y.clear()


class GestureConfirmer:
    """
    Require a gesture to appear consistently over N consecutive frames
    before it is considered 'confirmed'.

    This eliminates one-frame false positives.
    """

    def __init__(self, required_frames: int = 5):
        self.required_frames = required_frames
        self._last_gesture: str = ""
        self._count: int = 0

    def confirm(self, gesture: str) -> str | None:
        """
        Feed raw gesture string each frame.  Returns the confirmed gesture
        string only on the frame it first reaches the threshold, otherwise
        returns None.
        """
        if gesture == self._last_gesture:
            self._count += 1
            if self._count == self.required_frames:
                return gesture          # exactly on the confirmation frame
        else:
            self._last_gesture = gesture
            self._count = 1
        return None
