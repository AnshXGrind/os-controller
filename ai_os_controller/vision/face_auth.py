"""
vision/face_auth.py

Face authentication gate using the face_recognition library.

Workflow:
    1. At startup load a reference face image of the authorised user.
    2. Each frame: extract face encodings from the live webcam feed.
    3. Compare against the reference encoding.
    4. Return True only when the live face matches the reference within
       the configured tolerance.

IMPORTANT: face_recognition requires dlib which is compiled with C++.
On Windows, install via:
    pip install cmake
    pip install dlib
    pip install face_recognition
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FaceAuth:
    """
    Authenticates a user by comparing the webcam frame against a stored
    reference image.

    Args:
        image_path:   Path to a JPEG/PNG containing the authorised user's face.
        tolerance:    How much distance between face encodings to consider a
                      match. Lower values are stricter (default 0.6).
        enabled:      Set to False to skip authentication (dev/testing mode).
    """

    def __init__(
        self,
        image_path: str = "user.jpg",
        tolerance: float = 0.6,
        enabled: bool = True,
    ):
        self.enabled   = enabled
        self.tolerance = tolerance
        self._encoding: Optional[np.ndarray] = None

        if not self.enabled:
            logger.warning("FaceAuth is DISABLED – all users will be permitted.")
            return

        self._load_reference(image_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def authenticate(self, frame: np.ndarray) -> bool:
        """
        Check whether the authorised user's face is present in *frame*.

        Args:
            frame: BGR NumPy array from OpenCV.

        Returns:
            True if authenticated, False otherwise.
        """
        if not self.enabled:
            return True

        if self._encoding is None:
            # No reference loaded – fail safe
            return False

        # face_recognition expects RGB
        try:
            import face_recognition  # lazy import – graceful fallback
        except ImportError:
            logger.error(
                "face_recognition is not installed. "
                "Run: pip install cmake dlib face_recognition"
            )
            return True   # fallback: allow access when library is missing

        rgb = frame[:, :, ::-1].copy()   # BGR → RGB

        live_encodings = face_recognition.face_encodings(rgb)

        if not live_encodings:
            return False   # no face detected in this frame

        results = face_recognition.compare_faces(
            [self._encoding],
            live_encodings[0],
            tolerance=self.tolerance,
        )
        return bool(results[0])

    def is_ready(self) -> bool:
        """Return True when a valid reference encoding has been loaded."""
        return self._encoding is not None or not self.enabled

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_reference(self, image_path: str):
        """Load and encode the reference image."""
        path = Path(image_path)

        if not path.exists():
            logger.warning(
                f"Reference image '{image_path}' not found. "
                "FaceAuth will reject all frames until a valid image is provided. "
                "Place a photo named 'user.jpg' in the project root."
            )
            return

        try:
            import face_recognition
        except ImportError:
            logger.error(
                "face_recognition is not installed – skipping encoding load."
            )
            return

        image     = face_recognition.load_image_file(str(path))
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            logger.error(
                f"No face detected in reference image '{image_path}'. "
                "Please use a clear, front-facing portrait."
            )
            return

        self._encoding = encodings[0]
        logger.info(f"FaceAuth ready – reference loaded from '{image_path}'.")
