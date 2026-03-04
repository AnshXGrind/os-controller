"""
main.py

Entry point for the Multimodal AI OS Controller.

Usage:
    python main.py                         # default webcam, auth enabled
    python main.py --no-auth               # skip face auth (dev mode)
    python main.py --no-voice              # disable voice recognition
    python main.py --no-pose               # disable pose tracking
    python main.py --user-image path.jpg   # custom reference photo
    python main.py --src 1                 # use camera index 1

Press  Q  or  ESC  to quit.

──────────────────────────────────────────────────────────────────────────────
Module initialisation order
──────────────────────────────────────────────────────────────────────────────
 1. CameraStream   – threaded webcam capture
 2. HandTracker    – MediaPipe hand gestures
 3. EyeTracker     – MediaPipe FaceMesh gaze + blink
 4. FaceAuth       – face_recognition authentication gate
 5. PoseTracker    – MediaPipe full-body pose
 6. VoiceController– background speech recognition thread
 7. OSController   – PyAutoGUI wrapper
 8. ReelsController– high-level reel action layer
 9. InteractionEngine – combines all signals and triggers actions
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
import time

import cv2

# ── Module imports ────────────────────────────────────────────────────────────
from ai_os_controller.utils.camera_stream         import CameraStream
from ai_os_controller.vision.hand_tracker         import HandTracker
from ai_os_controller.vision.eye_tracker          import EyeTracker
from ai_os_controller.vision.face_auth            import FaceAuth
from ai_os_controller.vision.pose_tracker         import PoseTracker
from ai_os_controller.voice.voice_commands        import VoiceController
from ai_os_controller.control.os_controller       import OSController
from ai_os_controller.control.reels_controller    import ReelsController
from ai_os_controller.core.interaction_engine     import InteractionEngine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Multimodal AI OS Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src",        type=int,   default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--width",      type=int,   default=1280,
                   help="Capture width  (default: 1280)")
    p.add_argument("--height",     type=int,   default=720,
                   help="Capture height (default: 720)")
    p.add_argument("--user-image", type=str,   default="user.jpg",
                   help="Path to authorised user portrait (default: user.jpg)")
    p.add_argument("--no-auth",    action="store_true",
                   help="Disable face authentication (all users allowed)")
    p.add_argument("--no-voice",   action="store_true",
                   help="Disable voice control")
    p.add_argument("--no-pose",    action="store_true",
                   help="Disable pose tracking (saves CPU)")
    p.add_argument("--screen-w",   type=int,   default=1920,
                   help="Screen width  for cursor mapping (default: 1920)")
    p.add_argument("--screen-h",   type=int,   default=1080,
                   help="Screen height for cursor mapping (default: 1080)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    logger.info("═" * 60)
    logger.info("  Multimodal AI OS Controller – starting up")
    logger.info("═" * 60)

    # ── Camera ───────────────────────────────────────────────────────────────
    logger.info(f"Opening camera {args.src}  ({args.width}×{args.height})")
    try:
        cam = CameraStream(src=args.src, width=args.width, height=args.height).start()
    except IOError as exc:
        logger.critical(f"Camera error: {exc}")
        sys.exit(1)

    # ── Vision modules ────────────────────────────────────────────────────────
    logger.info("Initialising vision modules …")
    hand = HandTracker(max_hands=1, min_detection_conf=0.7, confirm_frames=4)
    eye  = EyeTracker()
    face = FaceAuth(image_path=args.user_image, enabled=not args.no_auth)
    pose = None if args.no_pose else PoseTracker()

    # ── Voice module ──────────────────────────────────────────────────────────
    voice = VoiceController()
    if not args.no_voice:
        logger.info("Starting voice recognition thread …")
        voice.start()
    else:
        logger.info("Voice recognition disabled by --no-voice flag.")

    # ── Control layer ─────────────────────────────────────────────────────────
    os_ctrl    = OSController(cooldown=0.8)
    controller = ReelsController(os_ctrl=os_ctrl)

    # ── Interaction engine ────────────────────────────────────────────────────
    engine = InteractionEngine(
        hand        = hand,
        eye         = eye,
        voice       = voice,
        face        = face,
        controller  = controller,
        os_ctrl     = os_ctrl,
        pose        = pose,
        screen_w    = args.screen_w,
        screen_h    = args.screen_h,
        pose_enabled = not args.no_pose,
    )

    logger.info("All modules ready.  Press Q or ESC to exit.")
    logger.info("─" * 60)

    # ── Main processing loop ──────────────────────────────────────────────────
    window_name = "AI OS Controller  (press Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                logger.warning("Empty frame – waiting …")
                time.sleep(0.05)
                continue

            # Run the full sensing + action pipeline
            engine.process(frame)

            # Show result
            cv2.imshow(window_name, frame)

            # Exit on Q or ESC
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                logger.info("Quit key pressed – shutting down.")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt – shutting down.")

    finally:
        _cleanup(cam, hand, eye, pose, voice)
        cv2.destroyAllWindows()
        logger.info("Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup(cam, hand, eye, pose, voice):
    logger.info("Releasing resources …")
    try:
        cam.stop()
    except Exception as exc:
        logger.debug(f"cam.stop(): {exc}")
    try:
        hand.close()
    except Exception as exc:
        logger.debug(f"hand.close(): {exc}")
    try:
        eye.close()
    except Exception as exc:
        logger.debug(f"eye.close(): {exc}")
    if pose:
        try:
            pose.close()
        except Exception as exc:
            logger.debug(f"pose.close(): {exc}")
    try:
        voice.stop()
    except Exception as exc:
        logger.debug(f"voice.stop(): {exc}")


if __name__ == "__main__":
    main()
