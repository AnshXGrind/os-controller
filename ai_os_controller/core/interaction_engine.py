"""
core/interaction_engine.py

The central brain of the AI OS Controller.

Combines signals from:
    • Hand gesture tracker
    • Eye gaze / blink tracker
    • Voice command listener
    • Face authentication gate
    • Pose tracker  (optional)

Priority order for actions:
    voice  >  gesture  >  gaze

Workflow each frame:
    1. Face authentication – skip all actions if user not recognised.
    2. Hand gesture detection.
    3. Eye gaze + blink detection.
    4. Poll latest voice command.
    5. Apply priority-ordered action dispatch.
    6. Draw UI feedback overlay on the frame.

Optional advanced stubs (future):
    • Emotion detection (deepface / fer)
    • Reinforcement-learning interaction
    • Multi-user face profiles
    • Gesture training system
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from ai_os_controller.vision.hand_tracker  import (
    HandTracker, G_INDEX_UP, G_PEACE, G_THUMB_UP, G_FIST, G_OPEN, G_UNKNOWN
)
from ai_os_controller.vision.eye_tracker   import (
    EyeTracker, GAZE_LEFT, GAZE_RIGHT, GAZE_CENTER,
    BLINK_SINGLE, BLINK_DOUBLE
)
from ai_os_controller.vision.face_auth     import FaceAuth
from ai_os_controller.vision.pose_tracker  import (
    PoseTracker, POSE_RIGHT_HAND_UP, POSE_LEFT_HAND_UP, POSE_BOTH_HANDS_UP
)
from ai_os_controller.voice.voice_commands  import VoiceController
from ai_os_controller.control.reels_controller import ReelsController
from ai_os_controller.control.os_controller    import OSController
from ai_os_controller.utils.smoothing          import CoordinateSmoother

logger = logging.getLogger(__name__)

# ── Overlay colours (BGR) ────────────────────────────────────────────────────
COL_GREEN  = (0,   220,  80)
COL_RED    = (0,    40, 220)
COL_YELLOW = (0,   200, 220)
COL_CYAN   = (220, 200,   0)
COL_WHITE  = (255, 255, 255)
COL_DARK   = (20,   20,  20)

# ── Cursor control deadzone ──────────────────────────────────────────────────
CURSOR_DEADZONE = 0.05   # fraction of frame width/height to ignore near centre


class InteractionEngine:
    """
    Orchestrates all sensing modalities and triggers OS actions.

    Args:
        hand:           HandTracker instance.
        eye:            EyeTracker instance.
        voice:          VoiceController instance.
        face:           FaceAuth instance.
        pose:           PoseTracker instance (optional).
        controller:     ReelsController / high-level action layer.
        os_ctrl:        OSController for cursor and low-level events.
        cursor_mode:    If True, open-palm gesture moves the cursor.
        screen_w:       Screen width in pixels (for cursor mapping).
        screen_h:       Screen height in pixels (for cursor mapping).
        pose_enabled:   Whether to run pose tracking (adds CPU cost).
    """

    def __init__(
        self,
        hand:        HandTracker,
        eye:         EyeTracker,
        voice:       VoiceController,
        face:        FaceAuth,
        controller:  ReelsController,
        os_ctrl:     OSController       = None,
        pose:        PoseTracker        = None,
        cursor_mode: bool               = False,
        screen_w:    int                = 1920,
        screen_h:    int                = 1080,
        pose_enabled: bool              = True,
    ):
        self.hand       = hand
        self.eye        = eye
        self.voice      = voice
        self.face       = face
        self.controller = controller
        self.os_ctrl    = os_ctrl or OSController()
        self.pose       = pose
        self.pose_enabled = pose_enabled and pose is not None

        self.cursor_mode = cursor_mode
        self.screen_w    = screen_w
        self.screen_h    = screen_h

        self._cursor_smoother = CoordinateSmoother(window=7)

        # State flags
        self._authenticated   = False
        self._last_auth_check = 0.0
        self._auth_interval   = 0.5    # re-check face every 0.5 s
        self._last_action     = ""
        self._last_action_time = 0.0

        # FPS tracking
        self._fps_counter = 0
        self._fps_display = 0.0
        self._fps_timer   = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray):
        """
        Main per-frame processing pipeline.

        Mutates *frame* in-place by drawing the UI overlay.

        Args:
            frame: BGR NumPy array from OpenCV.
        """
        self._update_fps()

        # ── 1. Face authentication ─────────────────────────────────────
        now = time.monotonic()
        if now - self._last_auth_check > self._auth_interval:
            self._authenticated    = self.face.authenticate(frame)
            self._last_auth_check  = now

        if not self._authenticated:
            self._draw_auth_failed(frame)
            return

        # ── 2. Hand gesture ───────────────────────────────────────────
        gesture = self.hand.detect(frame)

        # ── 3. Eye gaze + blink ───────────────────────────────────────
        gaze, blink = self.eye.process(frame)

        # ── 4. Pose (optional) ────────────────────────────────────────
        pose_action = None
        if self.pose_enabled:
            pose_action = self.pose.detect(frame)

        # ── 5. Voice command (non-blocking queue poll) ────────────────
        voice_cmd = self.voice.get_command()

        # ── 6. Action dispatch (voice > gesture > gaze > pose) ────────
        action_taken = self._dispatch(gesture, gaze, blink, voice_cmd, pose_action, frame)

        # ── 7. Cursor control (open palm) ─────────────────────────────
        if gesture == G_OPEN:
            self._handle_cursor(frame)

        # ── 8. UI overlay ─────────────────────────────────────────────
        self._draw_overlay(frame, gesture, gaze, blink, voice_cmd, action_taken)

    # ------------------------------------------------------------------
    # Dispatch logic
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        gesture:    str,
        gaze:       Optional[str],
        blink:      Optional[str],
        voice_cmd:  Optional[str],
        pose_action: Optional[str],
        frame:      np.ndarray,
    ) -> str:
        """
        Apply input events to the controller using voice > gesture > gaze priority.

        Returns:
            A short description of the action taken, or "" if none.
        """

        # ── Voice (highest priority) ───────────────────────────────────
        if voice_cmd:
            if self.controller.dispatch(voice_cmd):
                return f"Voice: {voice_cmd}"

        # ── Gesture ───────────────────────────────────────────────────
        if gesture == G_INDEX_UP:
            self.controller.next_reel()
            return "Gesture: next reel"

        if gesture == G_PEACE:
            self.controller.prev_reel()
            return "Gesture: prev reel"

        if gesture == G_THUMB_UP:
            self.controller.like()
            return "Gesture: like"

        if gesture == G_FIST:
            self.controller.pause()
            return "Gesture: pause"

        # ── Blink actions ─────────────────────────────────────────────
        if blink == BLINK_SINGLE:
            self.controller.like()
            return "Blink: like"

        if blink == BLINK_DOUBLE:
            self.controller.pause()
            return "Blink: pause"

        # ── Eye gaze (lowest vision priority) ─────────────────────────
        if gaze == GAZE_RIGHT:
            self.controller.next_reel()
            return "Gaze: next reel"

        if gaze == GAZE_LEFT:
            self.controller.prev_reel()
            return "Gaze: prev reel"

        # ── Pose ──────────────────────────────────────────────────────
        if pose_action == POSE_RIGHT_HAND_UP:
            self.controller.next_reel()
            return "Pose: next reel"

        if pose_action == POSE_LEFT_HAND_UP:
            self.controller.prev_reel()
            return "Pose: prev reel"

        if pose_action == POSE_BOTH_HANDS_UP:
            self.controller.volume_up()
            return "Pose: volume up"

        return ""

    # ------------------------------------------------------------------
    # Cursor mode
    # ------------------------------------------------------------------

    def _handle_cursor(self, frame: np.ndarray):
        """
        Map index fingertip position (from hand tracker) to screen coords
        and move the system cursor.

        Activated when gesture == G_OPEN (open palm).
        """
        # Retrieve index fingertip normalised coordinates from current hand landmarks
        if not hasattr(self.hand, "_smoother") or self.hand._smoother.smoothed is None:
            return

        # Landmark 8 = index fingertip
        tip = self.hand._smoother.smoothed[8]
        nx, ny = tip[0], tip[1]   # normalised 0–1

        # Apply deadzone around centre
        if abs(nx - 0.5) < CURSOR_DEADZONE and abs(ny - 0.5) < CURSOR_DEADZONE:
            return

        # Flip x (mirror) and map to screen
        sx = (1.0 - nx) * self.screen_w
        sy = ny * self.screen_h

        sx, sy = self._cursor_smoother.smooth(sx, sy)
        self.os_ctrl.move_cursor(sx, sy)

    # ------------------------------------------------------------------
    # Overlay / UI
    # ------------------------------------------------------------------

    def _draw_overlay(
        self,
        frame:    np.ndarray,
        gesture:  str,
        gaze:     Optional[str],
        blink:    Optional[str],
        voice:    Optional[str],
        action:   str,
    ):
        """Draw status labels and FPS counter on the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent status bar at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), COL_DARK, -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 1

        cv2.putText(frame, f"FPS: {self._fps_display:.1f}",
                    (w - 120, 28), font, scale, COL_CYAN, thick, cv2.LINE_AA)

        cv2.putText(frame, f"Gesture: {gesture}",
                    (10, 28), font, scale, COL_GREEN, thick, cv2.LINE_AA)
        cv2.putText(frame, f"Gaze: {gaze or 'n/a'}",
                    (10, 55), font, scale, COL_YELLOW, thick, cv2.LINE_AA)
        cv2.putText(frame, f"Blink: {blink or '-'}",
                    (10, 82), font, scale, COL_CYAN, thick, cv2.LINE_AA)
        if voice:
            cv2.putText(frame, f"Voice: {voice}",
                        (10, 109), font, scale, COL_WHITE, thick, cv2.LINE_AA)
        if action:
            cv2.putText(frame, f"Action: {action}",
                        (w // 2 - 120, h - 18), font, 0.7,
                        COL_GREEN, 2, cv2.LINE_AA)

    def _draw_auth_failed(self, frame: np.ndarray):
        """Display a blocked overlay when face authentication fails."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        cv2.putText(
            frame,
            "FACE AUTH REQUIRED",
            (w // 2 - 175, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Place user.jpg in project root",
            (w // 2 - 200, h // 2 + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 255),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------

    def _update_fps(self):
        self._fps_counter += 1
        now = time.monotonic()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps_display = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer   = now

    # ------------------------------------------------------------------
    # Stub hooks (future integrations)
    # ------------------------------------------------------------------

    def _emotion_hook(self, frame: np.ndarray) -> Optional[str]:
        """
        PLACEHOLDER – Emotion detection (deepface / fer).

        Example: happy → like video, surprised → replay.
        Return None until implemented.
        """
        return None

    def _rl_feedback_hook(self, action: str, outcome: int):
        """
        PLACEHOLDER – Reinforcement learning feedback loop.

        Call after each action with a reward signal to train a policy
        that learns user-specific gesture preferences.
        """
        pass

    def _gesture_training_hook(self, frame: np.ndarray, label: str):
        """
        PLACEHOLDER – Allow user to record custom gestures for training.
        """
        pass
