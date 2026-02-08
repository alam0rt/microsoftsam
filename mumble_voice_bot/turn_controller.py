"""Turn-taking and barge-in control."""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class TurnState(Enum):
    """States for turn-taking state machine."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class TurnController:
    """Manages turn-taking with barge-in support.

    This controller tracks the state of the conversation turn and enables
    barge-in detection, allowing users to interrupt the bot while it's speaking.

    State transitions:
        IDLE -> LISTENING: User starts speaking
        LISTENING -> PROCESSING: User stops speaking, processing begins
        PROCESSING -> SPEAKING: Bot starts speaking
        SPEAKING -> LISTENING: User interrupts (barge-in)
        SPEAKING -> IDLE: Bot finishes speaking
        Any -> IDLE: reset()

    Usage:
        controller = TurnController()

        # When user starts speaking
        controller.start_listening(user_id)

        # When processing user's speech
        controller.start_processing()

        # When bot starts speaking
        controller.start_speaking()

        # Check if user is interrupting
        if controller.state == TurnState.SPEAKING and user_is_speaking:
            if controller.request_barge_in():
                # Stop TTS, clear audio buffer
                pass

        # During TTS generation, check if cancelled
        for chunk in tts_stream:
            if controller.is_cancelled():
                break
            play(chunk)
    """

    state: TurnState = TurnState.IDLE
    _cancel_event: threading.Event = field(default_factory=threading.Event)
    _current_user: Optional[str] = None
    _state_lock: threading.Lock = field(default_factory=threading.Lock)
    _barge_in_callback: Optional[Callable[[], None]] = None
    _barge_in_count: int = 0
    _last_state_change: float = field(default_factory=time.time)

    # Configuration
    barge_in_delay_ms: int = 200  # Minimum ms before barge-in is allowed

    def start_listening(self, user_id: str):
        """User started speaking.

        Args:
            user_id: Identifier for the speaking user.
        """
        with self._state_lock:
            self._current_user = user_id
            self.state = TurnState.LISTENING
            self._cancel_event.clear()
            self._last_state_change = time.time()

    def start_processing(self):
        """Processing user's speech."""
        with self._state_lock:
            self.state = TurnState.PROCESSING
            self._last_state_change = time.time()

    def start_speaking(self):
        """Bot is now speaking."""
        with self._state_lock:
            self.state = TurnState.SPEAKING
            self._cancel_event.clear()
            self._last_state_change = time.time()

    def request_barge_in(self) -> bool:
        """Called when user speaks while bot is outputting.

        Returns:
            True if barge-in was triggered, False otherwise.
        """
        with self._state_lock:
            if self.state != TurnState.SPEAKING:
                return False

            # Prevent very quick barge-ins (accidental)
            elapsed_ms = (time.time() - self._last_state_change) * 1000
            if elapsed_ms < self.barge_in_delay_ms:
                return False

            self._cancel_event.set()
            self.state = TurnState.LISTENING
            self._barge_in_count += 1
            self._last_state_change = time.time()

            # Call barge-in callback if registered
            if self._barge_in_callback:
                try:
                    self._barge_in_callback()
                except Exception:
                    pass

            return True

    def is_cancelled(self) -> bool:
        """Check if current generation should be cancelled.

        Returns:
            True if cancelled (barge-in occurred), False otherwise.
        """
        return self._cancel_event.is_set()

    def reset(self):
        """Reset to idle state."""
        with self._state_lock:
            self._cancel_event.clear()
            self._current_user = None
            self.state = TurnState.IDLE
            self._last_state_change = time.time()

    def on_barge_in(self, callback: Callable[[], None]):
        """Register a callback for barge-in events.

        The callback is called when a barge-in is successfully triggered.
        Use this to clear audio buffers, stop TTS, etc.

        Args:
            callback: Function to call on barge-in.
        """
        self._barge_in_callback = callback

    @property
    def current_user(self) -> Optional[str]:
        """Get the current speaking user."""
        return self._current_user

    @property
    def barge_in_count(self) -> int:
        """Get total barge-in count (for stats)."""
        return self._barge_in_count

    def is_speaking(self) -> bool:
        """Check if bot is currently speaking."""
        return self.state == TurnState.SPEAKING

    def is_listening(self) -> bool:
        """Check if currently listening to a user."""
        return self.state == TurnState.LISTENING

    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self.state == TurnState.PROCESSING

    def is_idle(self) -> bool:
        """Check if idle (not in a conversation turn)."""
        return self.state == TurnState.IDLE
