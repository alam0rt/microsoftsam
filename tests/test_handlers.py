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
    DisconnectedEvent,
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
        self._generate_response_sync = MagicMock(return_value="Response to your message!")
        
        # Mock shared services (always present now)
        self._shared_services = MagicMock()
        self._shared_services.log_event = MagicMock()
        self._shared_services.try_claim_response = MagicMock(return_value=True)


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

    @pytest.mark.asyncio
    async def test_on_connected_logs(self):
        """Test connected event is handled."""
        bot = MockBot()
        handler = ConnectionHandler(bot)

        event = ConnectedEvent()
        # Should not raise
        await handler.on_connected(event)

    @pytest.mark.asyncio
    async def test_on_disconnected_logs(self):
        """Test disconnected event is handled."""
        bot = MockBot()
        handler = ConnectionHandler(bot)

        event = DisconnectedEvent()
        # Should not raise
        await handler.on_disconnected(event)


class TestPresenceHandlerGreetings:
    """Test greeting generation in PresenceHandler."""

    @pytest.mark.asyncio
    async def test_greet_user_with_llm(self):
        """Test that LLM is used for greeting when available."""
        bot = MockBot()
        bot.llm = MagicMock()
        handler = PresenceHandler(bot, greet_on_join=True)

        # Manually trigger greeting
        await handler._greet_user("TestUser", 42)

        # Should have called the LLM greeting
        bot._generate_oneoff_response_sync.assert_called()
        bot.speak.assert_called()

    @pytest.mark.asyncio
    async def test_greet_user_fallback_without_llm(self):
        """Test fallback greeting when LLM is not available."""
        bot = MockBot()
        bot.llm = None  # No LLM
        handler = PresenceHandler(bot, greet_on_join=True)

        await handler._greet_user("TestUser", 42)

        # Should have used fallback greeting
        bot.speak.assert_called()
        call_args = bot.speak.call_args[0][0]
        assert "TestUser" in call_args

    @pytest.mark.asyncio
    async def test_greet_user_with_custom_generator(self):
        """Test custom greeting generator is used when provided."""
        bot = MockBot()

        async def custom_greeting(user_name, time_of_day):
            return f"Custom hello to {user_name}!"

        handler = PresenceHandler(
            bot, greet_on_join=True, greeting_generator=custom_greeting
        )

        await handler._greet_user("TestUser", 42)

        bot.speak.assert_called_with("Custom hello to TestUser!")

    def test_get_time_of_day_morning(self):
        """Test time of day detection for morning."""
        from unittest.mock import MagicMock, patch

        mock_datetime = MagicMock()
        mock_datetime.now.return_value.hour = 9
        with patch('mumble_voice_bot.handlers.datetime', mock_datetime):
            time_str = PresenceHandler._get_time_of_day()
            assert time_str == "morning"

    def test_get_time_of_day_afternoon(self):
        """Test time of day detection for afternoon."""
        from unittest.mock import MagicMock, patch

        mock_datetime = MagicMock()
        mock_datetime.now.return_value.hour = 14
        with patch('mumble_voice_bot.handlers.datetime', mock_datetime):
            time_str = PresenceHandler._get_time_of_day()
            assert time_str == "afternoon"

    def test_get_time_of_day_evening(self):
        """Test time of day detection for evening."""
        from unittest.mock import MagicMock, patch

        mock_datetime = MagicMock()
        mock_datetime.now.return_value.hour = 19
        with patch('mumble_voice_bot.handlers.datetime', mock_datetime):
            time_str = PresenceHandler._get_time_of_day()
            assert time_str == "evening"

    def test_get_time_of_day_late_night(self):
        """Test time of day detection for late night."""
        from unittest.mock import MagicMock, patch

        mock_datetime = MagicMock()
        mock_datetime.now.return_value.hour = 2
        with patch('mumble_voice_bot.handlers.datetime', mock_datetime):
            time_str = PresenceHandler._get_time_of_day()
            assert time_str == "late night"


class TestTextCommandHandlerHTMLStripping:
    """Test HTML stripping in TextCommandHandler."""

    def test_strip_html_removes_tags(self):
        """Test HTML tag removal."""
        result = TextCommandHandler._strip_html("<b>bold</b> and <i>italic</i>")
        assert result == "bold and italic"

    def test_strip_html_handles_links(self):
        """Test link tag removal."""
        result = TextCommandHandler._strip_html('<a href="http://example.com">click</a>')
        assert result == "click"

    def test_strip_html_preserves_text(self):
        """Test plain text is preserved."""
        result = TextCommandHandler._strip_html("plain text")
        assert result == "plain text"

    def test_strip_html_handles_empty(self):
        """Test empty string handling."""
        result = TextCommandHandler._strip_html("")
        assert result == ""


class TestTextCommandHandlerMessageRouting:
    """Test message routing in TextCommandHandler."""

    def test_is_message_for_us_private(self):
        """Test private messages are always for us."""
        bot = MockBot()
        handler = TextCommandHandler(bot)

        event = TextMessageEvent(
            sender_session_id=2,
            sender_name="Alice",
            message="Hello",
            channel_ids=[],
            is_private=True,
            raw={},
        )

        assert handler._is_message_for_us(event) is True

    def test_is_message_for_us_same_channel(self):
        """Test messages in our channel are for us."""
        bot = MockBot()
        handler = TextCommandHandler(bot)

        event = TextMessageEvent(
            sender_session_id=2,
            sender_name="Alice",
            message="Hello",
            channel_ids=[100],  # Bot's channel
            is_private=False,
            raw={},
        )

        assert handler._is_message_for_us(event) is True

    def test_is_message_for_us_different_channel(self):
        """Test messages in other channels are not for us."""
        bot = MockBot()
        handler = TextCommandHandler(bot)

        event = TextMessageEvent(
            sender_session_id=2,
            sender_name="Alice",
            message="Hello",
            channel_ids=[200],  # Different channel
            is_private=False,
            raw={},
        )

        assert handler._is_message_for_us(event) is False

    def test_is_message_for_us_no_channel_specified(self):
        """Test messages with no channel are for us."""
        bot = MockBot()
        handler = TextCommandHandler(bot)

        event = TextMessageEvent(
            sender_session_id=2,
            sender_name="Alice",
            message="Hello",
            channel_ids=[],  # No channel specified
            is_private=False,
            raw={},
        )

        assert handler._is_message_for_us(event) is True


class TestPresenceHandlerChannelGreeting:
    """Test channel greeting (for multiple users)."""

    @pytest.mark.asyncio
    async def test_greet_channel_one_user(self):
        """Test greeting when one user is in channel."""
        bot = MockBot()
        bot.llm = MagicMock()
        handler = PresenceHandler(bot, greet_on_join=True)

        await handler._greet_channel(["Alice"])

        bot._generate_oneoff_response_sync.assert_called()
        call_args = bot._generate_oneoff_response_sync.call_args[0][0]
        assert "Alice" in call_args

    @pytest.mark.asyncio
    async def test_greet_channel_two_users(self):
        """Test greeting when two users are in channel."""
        bot = MockBot()
        bot.llm = MagicMock()
        handler = PresenceHandler(bot, greet_on_join=True)

        await handler._greet_channel(["Alice", "Bob"])

        call_args = bot._generate_oneoff_response_sync.call_args[0][0]
        assert "Alice" in call_args
        assert "Bob" in call_args

    @pytest.mark.asyncio
    async def test_greet_channel_many_users(self):
        """Test greeting when many users are in channel."""
        bot = MockBot()
        bot.llm = MagicMock()
        handler = PresenceHandler(bot, greet_on_join=True)

        await handler._greet_channel(["Alice", "Bob", "Charlie", "David"])

        call_args = bot._generate_oneoff_response_sync.call_args[0][0]
        # Check all names are mentioned
        assert "Alice" in call_args
        assert "Bob" in call_args
        assert "Charlie" in call_args
        assert "David" in call_args

    @pytest.mark.asyncio
    async def test_greet_channel_fallback(self):
        """Test fallback greeting for channel."""
        bot = MockBot()
        bot.llm = None
        handler = PresenceHandler(bot, greet_on_join=True)

        await handler._greet_channel(["Alice", "Bob"])

        bot.speak.assert_called()
        call_args = bot.speak.call_args[0][0]
        assert "everyone" in call_args.lower() or "hey" in call_args.lower()
