"""Event/filler system for the Mumble Voice Bot.

Handles soul-themed event responses (greetings, thinking fillers,
interruption acknowledgments, etc.) and conversational event tracking
(channel quiet, long speech, first-time speakers).

Extracted from mumble_tts_bot.py to reduce monolith size.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

logger = logging.getLogger(__name__)


class EventResponder:
    """Generates themed responses for bot events from soul config.

    Events include greetings, thinking fillers, interruption acknowledgments,
    farewells, and more. Responses are drawn from soul config pools with
    random selection and placeholder substitution.

    Attributes:
        soul_config: SoulConfig with events and fallbacks.
    """

    def __init__(self, soul_config: Any = None):
        self.soul_config = soul_config

    def update_soul(self, soul_config: Any) -> None:
        """Update the soul config (e.g., after soul switch)."""
        self.soul_config = soul_config

    def get_filler(self, filler_type: str) -> str | None:
        """Get a random filler phrase from soul config.

        Checks events first, then falls back to fallbacks for compatibility.

        Args:
            filler_type: One of 'thinking', 'still_thinking', 'interrupted'

        Returns:
            Random filler phrase, or None if unavailable.
        """
        fillers = None

        # First try events config
        if self.soul_config and self.soul_config.events:
            fillers = getattr(self.soul_config.events, filler_type, None)

        # Fall back to fallbacks
        if not fillers and self.soul_config and self.soul_config.fallbacks:
            fillers = getattr(self.soul_config.fallbacks, filler_type, None)

        if not fillers:
            return None

        return random.choice(fillers)

    def get_event_response(self, event_type: str, user: str | None = None) -> str | None:
        """Get a random event-triggered response from soul config.

        First checks soul_config.events, then falls back to soul_config.fallbacks
        for compatible event types.

        Args:
            event_type: Event name (e.g., 'user_first_speech', 'interrupted', 'thinking')
            user: Username for {user} placeholder substitution.

        Returns:
            Response string with placeholders filled, or None if disabled/unavailable.
        """
        responses = None

        # First try events config
        if self.soul_config and self.soul_config.events:
            responses = getattr(self.soul_config.events, event_type, None)

        # Fall back to fallbacks for compatible types
        if not responses and self.soul_config and self.soul_config.fallbacks:
            fallback_map = {
                'user_first_speech': 'greetings',
                'user_joined': 'greetings',
                'user_left': 'farewells',
                'thinking': 'thinking',
                'still_thinking': 'still_thinking',
                'interrupted': 'interrupted',
                'rate_limited': 'thinking',
            }
            fallback_key = fallback_map.get(event_type)
            if fallback_key:
                responses = getattr(self.soul_config.fallbacks, fallback_key, None)

        if not responses:
            return None

        response = random.choice(responses)

        # Fill in placeholders
        if user:
            response = response.replace("{user}", user)

        return response


class ChannelActivityTracker:
    """Tracks channel activity for quiet detection and long speech events.

    Attributes:
        quiet_threshold: Seconds of silence before triggering channel_quiet event.
        long_speech_threshold: Seconds of continuous speech before triggering event.
    """

    def __init__(
        self,
        quiet_threshold: float = 60.0,
        long_speech_threshold: float = 15.0,
    ):
        self.quiet_threshold = quiet_threshold
        self.long_speech_threshold = long_speech_threshold

        self._last_channel_activity = time.time()
        self._quiet_event_triggered = False

        # Per-user speech duration tracking
        self._user_speech_time: dict[int, float] = {}

        # First-time speaker tracking
        self._greeted_users: set[str] = set()

    def record_activity(self) -> None:
        """Record that activity occurred in the channel."""
        self._last_channel_activity = time.time()
        self._quiet_event_triggered = False

    def check_channel_quiet(self) -> bool:
        """Check if channel has been quiet long enough to trigger event.

        Returns:
            True if the quiet threshold was just exceeded (fires once).
        """
        if self._quiet_event_triggered:
            return False

        time_since_activity = time.time() - self._last_channel_activity
        if time_since_activity >= self.quiet_threshold:
            self._quiet_event_triggered = True
            logger.info(f"Channel quiet for {time_since_activity:.0f}s")
            return True

        return False

    def check_long_speech(self, user_id: int, user_name: str, speech_duration: float) -> bool:
        """Track speech duration and check if threshold exceeded.

        Args:
            user_id: The user's session ID.
            user_name: The user's name.
            speech_duration: Duration of this speech segment in seconds.

        Returns:
            True if the user's total speech exceeded the threshold (resets counter).
        """
        if user_id not in self._user_speech_time:
            self._user_speech_time[user_id] = 0.0

        self._user_speech_time[user_id] += speech_duration
        total = self._user_speech_time[user_id]

        if total >= self.long_speech_threshold:
            logger.info(f"{user_name} spoke for {total:.1f}s total")
            self._user_speech_time[user_id] = 0.0
            return True

        return False

    def reset_speech_tracking(self, user_id: int) -> None:
        """Reset speech tracking for a user."""
        self._user_speech_time.pop(user_id, None)

    def check_first_time_speaker(self, user_name: str) -> bool:
        """Check if this is the first time we've heard from this user.

        Args:
            user_name: The user's name.

        Returns:
            True if this is their first speech this session.
        """
        if user_name in self._greeted_users:
            return False

        self._greeted_users.add(user_name)
        logger.info(f"First speech detected from: {user_name}")
        return True


def is_message_directed_at_bot(text: str, bot_name: str) -> bool:
    """Check if a message is directed at the bot by name.

    Args:
        text: The user's utterance.
        bot_name: The bot's display name.

    Returns:
        True if the message mentions the bot's name.
    """
    if not bot_name:
        return False
    return bot_name.lower() in text.lower()
