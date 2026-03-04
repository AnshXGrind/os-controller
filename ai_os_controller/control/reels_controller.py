"""
control/reels_controller.py

High-level reels / social-media action layer built on top of OSController.

Centralises all reel-specific action mappings so that the interaction
engine never needs to know which keys or scroll amounts to use.
"""

import logging

from ai_os_controller.control.os_controller import OSController

logger = logging.getLogger(__name__)

# Scroll delta for reel navigation (pixels)
SCROLL_AMOUNT = 500


class ReelsController:
    """
    Maps semantic reel actions to concrete OS events.

    Args:
        os_ctrl: An OSController instance to delegate to.
                 If None, a new one is created automatically.
    """

    def __init__(self, os_ctrl: OSController = None):
        self._os = os_ctrl or OSController()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def next_reel(self):
        """Advance to the next reel (scroll down)."""
        logger.debug("ReelsController: next_reel")
        self._os.scroll(-SCROLL_AMOUNT)

    def prev_reel(self):
        """Go back to the previous reel (scroll up)."""
        logger.debug("ReelsController: prev_reel")
        self._os.scroll(SCROLL_AMOUNT)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def like(self):
        """Like the current reel (press L)."""
        logger.debug("ReelsController: like")
        self._os.like()

    def pause(self):
        """Pause / play the current reel (press Space)."""
        logger.debug("ReelsController: pause")
        self._os.pause()

    def volume_up(self):
        """Increase playback volume."""
        logger.debug("ReelsController: volume_up")
        self._os.volume_up()

    def volume_down(self):
        """Decrease playback volume."""
        logger.debug("ReelsController: volume_down")
        self._os.volume_down()

    # ------------------------------------------------------------------
    # Action dispatch helper
    # ------------------------------------------------------------------

    def dispatch(self, command: str) -> bool:
        """
        Execute a reel action from a string label.

        Returns True if the command was recognised, False otherwise.

        Args:
            command: One of 'next', 'previous', 'like', 'pause',
                     'volume_up', 'volume_down'.
        """
        actions = {
            "next":        self.next_reel,
            "previous":    self.prev_reel,
            "like":        self.like,
            "pause":       self.pause,
            "volume_up":   self.volume_up,
            "volume_down": self.volume_down,
        }
        fn = actions.get(command)
        if fn:
            fn()
            return True
        logger.debug(f"ReelsController.dispatch: unknown command '{command}'")
        return False
