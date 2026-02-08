"""Event handlers for the Mumble voice bot.

These handlers implement MumbleEventHandler to provide clean, typed
handling of Mumble events while keeping the main bot class focused
on voice processing logic.
"""

import asyncio
import random
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable

from mumble_voice_bot.interfaces.events import (
    ConnectedEvent,
    DisconnectedEvent,
    EventType,
    MumbleEventHandler,
    TextMessageEvent,
    UserJoinedEvent,
    UserLeftEvent,
    UserUpdatedEvent,
)
from mumble_voice_bot.logging_config import get_logger

if TYPE_CHECKING:
    from mumble_tts_bot import MumbleTTSBot

logger = get_logger(__name__)


class PresenceHandler(MumbleEventHandler):
    """Tracks user presence and handles join/leave events.

    This handler:
    - Maintains a list of users currently in the bot's channel
    - Generates greetings when users join the channel
    - Cleans up user state when users leave

    Args:
        bot: The MumbleTTSBot instance for accessing Mumble state and TTS.
        greet_on_join: Whether to generate greetings for joining users.
        greeting_generator: Optional async callable that generates greeting text.
                           Receives (user_name: str, time_of_day: str) and returns str.
    """

    def __init__(
        self,
        bot: "MumbleTTSBot",
        greet_on_join: bool = True,
        greeting_generator: Callable[[str, str], Awaitable[str]] = None,
    ):
        self._bot = bot
        self._greet_on_join = greet_on_join
        self._greeting_generator = greeting_generator

        # Track users in our channel: session_id -> user_name
        self._users_in_channel: dict[int, str] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "PresenceHandler"

    def subscribed_events(self) -> set[EventType]:
        return {
            EventType.CONNECTED,
            EventType.USER_CREATED,
            EventType.USER_UPDATED,
            EventType.USER_REMOVED,
        }

    def get_users_in_channel(self) -> list[str]:
        """Get list of usernames currently in the bot's channel."""
        with self._lock:
            return list(self._users_in_channel.values())

    def is_user_in_channel(self, session_id: int) -> bool:
        """Check if a user is in the bot's channel."""
        with self._lock:
            return session_id in self._users_in_channel

    async def on_connected(self, event: ConnectedEvent) -> None:
        """Bot connected to server - scan existing channel users and greet."""
        logger.info("Bot connected, scanning channel users...")

        # Give Mumble a moment to sync user list
        await asyncio.sleep(0.5)

        await self._scan_channel_users()

    async def _scan_channel_users(self) -> None:
        """Scan current channel for existing users and optionally greet."""
        try:
            my_session = self._bot.mumble.users.myself_session
            my_channel = self._bot.mumble.users.myself["channel_id"]
        except Exception as e:
            logger.warning(f"Could not get bot's channel info: {e}")
            return

        # Find all users in our channel
        users_found = []
        for user in self._bot.mumble.users.values():
            session_id = user.get("session")
            channel_id = user.get("channel_id")
            name = user.get("name", "Unknown")

            # Skip ourselves
            if session_id == my_session:
                continue

            # Only track users in our channel
            if channel_id == my_channel:
                with self._lock:
                    self._users_in_channel[session_id] = name
                users_found.append(name)

        if users_found:
            logger.info(f"Found {len(users_found)} user(s) in channel: {', '.join(users_found)}")
            # Generate a greeting for all users
            if self._greet_on_join:
                await self._greet_channel(users_found)
        else:
            logger.info("Channel is empty")

    async def _greet_channel(self, users: list[str]) -> None:
        """Generate and speak a greeting for users already in the channel."""
        time_of_day = self._get_time_of_day()

        if len(users) == 1:
            user_desc = users[0]
        elif len(users) == 2:
            user_desc = f"{users[0]} and {users[1]}"
        else:
            user_desc = f"{', '.join(users[:-1])}, and {users[-1]}"

        # Use LLM if available
        if self._bot.llm:
            try:
                greeting_prompt = (
                    f"You just joined a voice channel. It's {time_of_day}. "
                    f"Already here: {user_desc}. "
                    "Give a brief, casual greeting acknowledging them. One sentence max."
                )
                response = self._bot._generate_oneoff_response_sync(greeting_prompt)
                if response:
                    logger.info(f"Greeting channel: {response}")
                    self._bot.speak(response)
                    return
            except Exception as e:
                logger.warning(f"LLM channel greeting failed: {e}")

        # Fallback greeting
        if len(users) == 1:
            fallback = f"Hey {users[0]}!"
        else:
            fallback = "Hey everyone!"
        logger.info(f"Greeting channel (fallback): {fallback}")
        self._bot.speak(fallback)

    async def on_user_joined(self, event: UserJoinedEvent) -> None:
        """User connected to server - just log it."""
        logger.info(f"User {event.name} connected to server (session={event.session_id})")
        # Don't greet here - wait for them to join our channel

    async def on_user_updated(self, event: UserUpdatedEvent) -> None:
        """User state changed - check for channel moves."""
        # Only care about channel changes
        if "channel_id" not in event.changed_fields:
            return

        try:
            my_session = self._bot.mumble.users.myself_session
            my_channel = self._bot.mumble.users.myself["channel_id"]
        except Exception as e:
            logger.warning(f"Could not get bot's channel: {e}")
            return

        # Skip our own updates
        if event.session_id == my_session:
            return

        new_channel = event.changed_fields["channel_id"]

        with self._lock:
            was_in_channel = event.session_id in self._users_in_channel

        if new_channel == my_channel:
            # User joined our channel
            with self._lock:
                self._users_in_channel[event.session_id] = event.name

            if not was_in_channel:
                logger.info(f"User {event.name} joined our channel")
                if self._greet_on_join:
                    logger.debug(f"Generating greeting for {event.name}")
                    await self._greet_user(event.name, event.session_id)
        else:
            # User left our channel
            with self._lock:
                if event.session_id in self._users_in_channel:
                    del self._users_in_channel[event.session_id]

            if was_in_channel:
                logger.info(f"User {event.name} left our channel")

    async def on_user_left(self, event: UserLeftEvent) -> None:
        """User disconnected from server."""
        with self._lock:
            if event.session_id in self._users_in_channel:
                del self._users_in_channel[event.session_id]

        # Clean up any pending state for this user
        self._cleanup_user_state(event.session_id)

        if event.kicked:
            logger.info(f"User {event.name} was kicked from server")
        elif event.ban:
            logger.info(f"User {event.name} was banned from server")
        else:
            logger.info(f"User {event.name} left server")

    def _cleanup_user_state(self, session_id: int) -> None:
        """Clean up bot state for a departed user."""
        # Clear audio buffers
        self._bot.audio_buffers.pop(session_id, None)
        self._bot.speech_active_until.pop(session_id, None)
        self._bot.speech_start_time.pop(session_id, None)

        # Clear pending transcriptions
        self._bot.pending_text.pop(session_id, None)
        self._bot.pending_text_time.pop(session_id, None)

    async def _greet_user(self, user_name: str, session_id: int) -> None:
        """Generate and speak a greeting for a user."""
        time_of_day = self._get_time_of_day()

        if self._greeting_generator:
            try:
                greeting = await self._greeting_generator(user_name, time_of_day)
                if greeting:
                    logger.info(f"Greeting {user_name}: {greeting}")
                    self._bot.speak(greeting)
                    return
            except Exception as e:
                logger.warning(f"Custom greeting generator failed: {e}")

        # Use bot's LLM if available
        if self._bot.llm:
            try:
                greeting_prompt = (
                    f"{user_name} just joined the voice channel. It's {time_of_day}. "
                    "Give a brief, casual greeting. One sentence max."
                )
                response = self._bot._generate_oneoff_response_sync(greeting_prompt)
                if response:
                    logger.info(f"Greeting {user_name}: {response}")
                    self._bot.speak(response)
                    return
            except Exception as e:
                logger.warning(f"LLM greeting failed: {e}")

        # Fallback to random greeting
        greetings = [
            f"Hey {user_name}!",
            f"Oh hey, {user_name}.",
            f"Yo {user_name}, what's up?",
            f"Hey! {user_name}'s here.",
        ]
        fallback = random.choice(greetings)
        logger.info(f"Greeting {user_name} (fallback): {fallback}")
        self._bot.speak(fallback)

    @staticmethod
    def _get_time_of_day() -> str:
        """Get time of day as a friendly string."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "late night"


class TextCommandHandler(MumbleEventHandler):
    """Handles text messages sent to the bot.

    Processes text messages and generates LLM responses that are
    spoken via TTS.
    """

    def __init__(self, bot: "MumbleTTSBot"):
        self._bot = bot

    @property
    def name(self) -> str:
        return "TextCommandHandler"

    def subscribed_events(self) -> set[EventType]:
        return {EventType.TEXT_RECEIVED}

    async def on_text_message(self, event: TextMessageEvent) -> None:
        """Handle incoming text message."""
        # Ignore our own messages
        try:
            my_session = self._bot.mumble.users.myself_session
            if event.sender_session_id == my_session:
                return
        except Exception:
            pass

        # Check if message is for our channel
        if not self._is_message_for_us(event):
            return

        # Strip HTML from message
        text = self._strip_html(event.message)
        if not text.strip():
            return

        sender_name = event.sender_name or "Someone"
        logger.info(f"Text from {sender_name}: {text}")

        if not self._bot.llm:
            logger.debug("LLM not available - ignoring text message")
            return

        try:
            response = self._bot._generate_response_sync(
                event.sender_session_id,
                text,
                sender_name
            )
            if response:
                self._bot.speak(response)
        except Exception as e:
            logger.error(f"Error generating text response: {e}")

    def _is_message_for_us(self, event: TextMessageEvent) -> bool:
        """Check if a message targets our channel."""
        # Private messages are always for us
        if event.is_private:
            return True

        # If no channels specified, assume it's for us
        if not event.channel_ids:
            return True

        try:
            my_channel = self._bot.mumble.users.myself["channel_id"]
            return my_channel in event.channel_ids
        except Exception:
            return True

    @staticmethod
    def _strip_html(text: str) -> str:
        """Strip HTML tags from text."""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)


class ConnectionHandler(MumbleEventHandler):
    """Handles connection state changes.

    Logs connection events and could be extended to handle
    reconnection logic or announcements.
    """

    def __init__(self, bot: "MumbleTTSBot" = None):
        self._bot = bot

    @property
    def name(self) -> str:
        return "ConnectionHandler"

    def subscribed_events(self) -> set[EventType]:
        return {EventType.CONNECTED, EventType.DISCONNECTED}

    async def on_connected(self, event: ConnectedEvent) -> None:
        """Handle successful connection."""
        logger.info("Connected to Mumble server")

    async def on_disconnected(self, event: DisconnectedEvent) -> None:
        """Handle disconnection."""
        logger.warning("Disconnected from Mumble server")
