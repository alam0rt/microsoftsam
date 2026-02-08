"""Tests for event handlers."""

from unittest.mock import MagicMock

import pytest

from mumble_voice_bot.handlers import (
    ConnectionHandler,
    PresenceHandler,
    TextCommandHandler,
)
from mumble_voice_bot.interfaces.events import (
    ConnectedEvent,
    EventType,
    TextMessageEvent,
    UserLeftEvent,
    UserUpdatedEvent,
)


class MockBot:
    """Mock bot for testing handlers."""

    def __init__(self):
        self.mumble = MagicMock()
        self.mumble.users = MagicMock()
        self.mumble.users.myself_session = 1
        self.mumble.users.myself = {"channel_id": 100}
        self.llm = MagicMock()

        # Track state
        self.audio_buffers = {}
        self.speech_active_until = {}
        self.speech_start_time = {}
        self.pending_text = {}
        self.pending_text_time = {}

        # Mock speak method
        self.speak = MagicMock()

        # Mock LLM response
        self._generate_oneoff_response_sync = MagicMock(return_value="Hello there!")


class TestPresenceHandler:
    """Tests for PresenceHandler."""

    def test_subscribed_events(self):
        """Test that handler subscribes to the correct events."""
        bot = MockBot()
        handler = PresenceHandler(bot, greet_on_join=True)

        events = handler.subscribed_events()
        assert EventType.CONNECTED in events
        assert EventType.USER_CREATED in events
        assert EventType.USER_UPDATED in events
        assert EventType.USER_REMOVED in events

    def test_handler_name(self):
        """Test handler name property."""
        bot = MockBot()
        handler = PresenceHandler(bot, greet_on_join=True)
        assert handler.name == "PresenceHandler"

    @pytest.mark.asyncio
    async def test_on_connected_scans_users(self):
        """Test that on_connected scans existing channel users."""
        bot = MockBot()

        # Mock users in the channel
        bot.mumble.users.values.return_value = [
            {"session": 1, "channel_id": 100, "name": "Bot"},  # Bot itself
            {"session": 2, "channel_id": 100, "name": "Alice"},  # User in channel
            {"session": 3, "channel_id": 200, "name": "Bob"},  # User in different channel
        ]

        handler = PresenceHandler(bot, greet_on_join=True)

        # Trigger connected event
        event = ConnectedEvent()
        await handler.on_connected(event)

        # Should have found Alice
        users = handler.get_users_in_channel()
        assert "Alice" in users
        assert "Bob" not in users
        assert "Bot" not in users

        # Should have generated a greeting
        bot._generate_oneoff_response_sync.assert_called()
        bot.speak.assert_called()

    @pytest.mark.asyncio
    async def test_on_connected_empty_channel(self):
        """Test on_connected with empty channel."""
        bot = MockBot()

        # Only the bot in the channel
        bot.mumble.users.values.return_value = [
            {"session": 1, "channel_id": 100, "name": "Bot"},
        ]

        handler = PresenceHandler(bot, greet_on_join=False)

        event = ConnectedEvent()
        await handler.on_connected(event)

        # Should have no users
        users = handler.get_users_in_channel()
        assert len(users) == 0

    @pytest.mark.asyncio
    async def test_on_user_updated_join_channel(self):
        """Test user joining our channel triggers greeting."""
        bot = MockBot()
        handler = PresenceHandler(bot, greet_on_join=True)

        # User moves to our channel
        event = UserUpdatedEvent(
            session_id=2,
            name="Alice",
            channel_id=100,
            user_id=None,
            muted=False,
            deafened=False,
            suppressed=False,
            self_muted=False,
            self_deafened=False,
            recording=False,
            comment=None,
            changed_fields={"channel_id": 100},  # User moved to channel 100
            raw={},
        )

        await handler.on_user_updated(event)

        # Should have added user
        assert handler.is_user_in_channel(2)
        users = handler.get_users_in_channel()
        assert "Alice" in users

        # Should have greeted
        bot.speak.assert_called()

    @pytest.mark.asyncio
    async def test_on_user_updated_leave_channel(self):
        """Test user leaving our channel is tracked."""
        bot = MockBot()
        handler = PresenceHandler(bot, greet_on_join=False)

        # Add user first
        handler._users_in_channel[2] = "Alice"

        # User moves to different channel
        event = UserUpdatedEvent(
            session_id=2,
            name="Alice",
            channel_id=200,  # Different channel
            user_id=None,
            muted=False,
            deafened=False,
            suppressed=False,
            self_muted=False,
            self_deafened=False,
            recording=False,
            comment=None,
            changed_fields={"channel_id": 200},
            raw={},
        )

        await handler.on_user_updated(event)

        # Should have removed user
        assert not handler.is_user_in_channel(2)
        users = handler.get_users_in_channel()
        assert "Alice" not in users

    @pytest.mark.asyncio
    async def test_on_user_left_cleans_state(self):
        """Test user leaving server cleans up state."""
        bot = MockBot()
        handler = PresenceHandler(bot, greet_on_join=False)

        # Add user and some state
        handler._users_in_channel[2] = "Alice"
        bot.audio_buffers[2] = []
        bot.pending_text[2] = "test"

        event = UserLeftEvent(
            session_id=2,
            name="Alice",
            channel_id=100,
            user_id=None,
            muted=False,
            deafened=False,
            suppressed=False,
            self_muted=False,
            self_deafened=False,
            recording=False,
            comment=None,
            raw={},
            kicked=False,
            ban=False,
        )

        await handler.on_user_left(event)

        # Should have cleaned up
        assert 2 not in handler._users_in_channel
        assert 2 not in bot.audio_buffers
        assert 2 not in bot.pending_text


class TestTextCommandHandler:
    """Tests for TextCommandHandler."""

    def test_subscribed_events(self):
        """Test that handler subscribes to text events."""
        bot = MockBot()
        handler = TextCommandHandler(bot)

        events = handler.subscribed_events()
        assert EventType.TEXT_RECEIVED in events

    def test_handler_name(self):
        """Test handler name property."""
        bot = MockBot()
        handler = TextCommandHandler(bot)
        assert handler.name == "TextCommandHandler"

    @pytest.mark.asyncio
    async def test_on_text_message_generates_response(self):
        """Test text message triggers LLM response."""
        bot = MockBot()
        bot._generate_response_sync = MagicMock(return_value="Response text")
        handler = TextCommandHandler(bot)

        event = TextMessageEvent(
            sender_session_id=2,
            sender_name="Alice",
            message="Hello bot!",
            channel_ids=[100],  # Our channel
            is_private=False,
            raw={},
        )

        await handler.on_text_message(event)

        # Should have generated response
        bot._generate_response_sync.assert_called_once()
        bot.speak.assert_called_with("Response text")

    @pytest.mark.asyncio
    async def test_on_text_message_ignores_own_messages(self):
        """Test that bot ignores its own messages."""
        bot = MockBot()
        handler = TextCommandHandler(bot)

        # Message from the bot itself
        event = TextMessageEvent(
            sender_session_id=1,  # Same as bot's session
            sender_name="Bot",
            message="Hello!",
            channel_ids=[100],
            is_private=False,
            raw={},
        )

        await handler.on_text_message(event)

        # Should not have called speak
        bot.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_text_message_ignores_other_channels(self):
        """Test that bot ignores messages from other channels."""
        bot = MockBot()
        handler = TextCommandHandler(bot)

        # Message from different channel
        event = TextMessageEvent(
            sender_session_id=2,
            sender_name="Alice",
            message="Hello!",
            channel_ids=[200],  # Different channel
            is_private=False,
            raw={},
        )

        await handler.on_text_message(event)

        # Should not have called speak
        bot.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_text_message_handles_private(self):
        """Test that private messages are handled."""
        bot = MockBot()
        bot._generate_response_sync = MagicMock(return_value="Private response")
        handler = TextCommandHandler(bot)

        event = TextMessageEvent(
            sender_session_id=2,
            sender_name="Alice",
            message="Private message",
            channel_ids=[],
            is_private=True,
            raw={},
        )

        await handler.on_text_message(event)

        # Should have responded to private message
        bot._generate_response_sync.assert_called_once()
        bot.speak.assert_called()


class TestConnectionHandler:
    """Tests for ConnectionHandler."""

    def test_subscribed_events(self):
        """Test handler subscribes to connection events."""
        handler = ConnectionHandler()

        events = handler.subscribed_events()
        assert EventType.CONNECTED in events
        assert EventType.DISCONNECTED in events

    def test_handler_name(self):
        """Test handler name property."""
        handler = ConnectionHandler()
        assert handler.name == "ConnectionHandler"
