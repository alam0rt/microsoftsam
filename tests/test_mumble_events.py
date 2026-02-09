"""Tests for Mumble event dispatcher and adapters.

Tests cover:
- EventDispatcher initialization and configuration
- Handler registration and unregistration
- Event conversion from pymumble callbacks
- Event dispatching to handlers
- Thread safety and async handler scheduling
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from mumble_voice_bot.interfaces.events import (
    ACLReceivedEvent,
    ChannelCreatedEvent,
    ChannelRemovedEvent,
    ChannelUpdatedEvent,
    ConnectedEvent,
    ContextActionEvent,
    DisconnectedEvent,
    EventType,
    MumbleEvent,
    MumbleEventHandler,
    PermissionDeniedEvent,
    SoundReceivedEvent,
    TextMessageEvent,
    UserJoinedEvent,
    UserLeftEvent,
    UserUpdatedEvent,
)
from mumble_voice_bot.providers.mumble_events import (
    PYMUMBLE_TO_EVENT_TYPE,
    EventDispatcher,
)

# --- Mock Handlers ---


class MockEventHandler(MumbleEventHandler):
    """Mock handler that records received events."""

    def __init__(self, name: str = "MockHandler", subscribed: set[EventType] | None = None):
        self._name = name
        self._subscribed = subscribed
        self.events_received: list[MumbleEvent] = []
        self.on_connected_calls = 0
        self.on_disconnected_calls = 0
        self.on_text_message_calls = 0
        self.on_sound_received_calls = 0
        self.on_user_joined_calls = 0
        self.on_user_left_calls = 0
        self.on_user_updated_calls = 0
        self.on_channel_created_calls = 0
        self.on_channel_updated_calls = 0
        self.on_channel_removed_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def subscribed_events(self) -> set[EventType] | None:
        return self._subscribed

    async def on_connected(self, event: ConnectedEvent) -> None:
        self.events_received.append(event)
        self.on_connected_calls += 1

    async def on_disconnected(self, event: DisconnectedEvent) -> None:
        self.events_received.append(event)
        self.on_disconnected_calls += 1

    async def on_text_message(self, event: TextMessageEvent) -> None:
        self.events_received.append(event)
        self.on_text_message_calls += 1

    async def on_sound_received(self, event: SoundReceivedEvent) -> None:
        self.events_received.append(event)
        self.on_sound_received_calls += 1

    async def on_user_joined(self, event: UserJoinedEvent) -> None:
        self.events_received.append(event)
        self.on_user_joined_calls += 1

    async def on_user_left(self, event: UserLeftEvent) -> None:
        self.events_received.append(event)
        self.on_user_left_calls += 1

    async def on_user_updated(self, event: UserUpdatedEvent) -> None:
        self.events_received.append(event)
        self.on_user_updated_calls += 1

    async def on_channel_created(self, event: ChannelCreatedEvent) -> None:
        self.events_received.append(event)
        self.on_channel_created_calls += 1

    async def on_channel_updated(self, event: ChannelUpdatedEvent) -> None:
        self.events_received.append(event)
        self.on_channel_updated_calls += 1

    async def on_channel_removed(self, event: ChannelRemovedEvent) -> None:
        self.events_received.append(event)
        self.on_channel_removed_calls += 1

    async def on_context_action(self, event: ContextActionEvent) -> None:
        self.events_received.append(event)

    async def on_acl_received(self, event: ACLReceivedEvent) -> None:
        self.events_received.append(event)

    async def on_permission_denied(self, event: PermissionDeniedEvent) -> None:
        self.events_received.append(event)


class ExceptionHandler(MumbleEventHandler):
    """Handler that raises exceptions for testing error isolation."""

    @property
    def name(self) -> str:
        return "ExceptionHandler"

    def subscribed_events(self) -> set[EventType] | None:
        return None  # Subscribe to all

    async def on_connected(self, event: ConnectedEvent) -> None:
        raise RuntimeError("Test exception in handler")

    async def on_text_message(self, event: TextMessageEvent) -> None:
        raise ValueError("Handler error")


# --- Fixtures ---


@pytest.fixture
def mock_mumble():
    """Create a mock pymumble Mumble instance."""
    mumble = MagicMock()
    mumble.callbacks = MagicMock()
    mumble.callbacks.add_callback = MagicMock()
    mumble.callbacks.remove_callback = MagicMock()
    mumble.set_receive_sound = MagicMock()
    mumble.users = MagicMock()
    mumble.users.myself = {"session": 1}
    return mumble


@pytest.fixture
def event_loop():
    """Create an event loop for tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def dispatcher(mock_mumble, event_loop):
    """Create an EventDispatcher with mocked Mumble."""
    return EventDispatcher(mock_mumble, loop=event_loop, enable_sound_events=True)


# --- Test Classes ---


class TestEventDispatcherInit:
    """Test EventDispatcher initialization."""

    def test_init_with_all_parameters(self, mock_mumble, event_loop):
        """Test initialization with all parameters."""
        dispatcher = EventDispatcher(
            mock_mumble,
            loop=event_loop,
            enable_sound_events=False,
        )

        assert dispatcher.mumble is mock_mumble
        assert dispatcher._loop is event_loop
        assert dispatcher._enable_sound_events is False
        assert len(dispatcher._handlers) == 0
        assert dispatcher._started is False

    def test_init_minimal_parameters(self, mock_mumble):
        """Test initialization with minimal parameters."""
        dispatcher = EventDispatcher(mock_mumble)

        assert dispatcher.mumble is mock_mumble
        assert dispatcher._loop is None
        assert dispatcher._enable_sound_events is True

    def test_loop_property_creates_loop_if_none(self, mock_mumble):
        """Test loop property creates event loop if none provided."""
        dispatcher = EventDispatcher(mock_mumble, loop=None)

        # Access the loop property
        loop = dispatcher.loop
        assert loop is not None
        assert isinstance(loop, asyncio.AbstractEventLoop)


class TestEventDispatcherHandlerRegistration:
    """Test handler registration and unregistration."""

    def test_register_handler(self, dispatcher):
        """Test registering a handler."""
        handler = MockEventHandler("TestHandler")
        dispatcher.register_handler(handler)

        assert len(dispatcher._handlers) == 1
        assert handler in dispatcher._handlers

    def test_register_duplicate_handler_ignored(self, dispatcher):
        """Test that registering the same handler twice is idempotent."""
        handler = MockEventHandler("TestHandler")
        dispatcher.register_handler(handler)
        dispatcher.register_handler(handler)

        assert len(dispatcher._handlers) == 1

    def test_register_multiple_handlers(self, dispatcher):
        """Test registering multiple different handlers."""
        handler1 = MockEventHandler("Handler1")
        handler2 = MockEventHandler("Handler2")
        handler3 = MockEventHandler("Handler3")

        dispatcher.register_handler(handler1)
        dispatcher.register_handler(handler2)
        dispatcher.register_handler(handler3)

        assert len(dispatcher._handlers) == 3

    def test_unregister_handler(self, dispatcher):
        """Test unregistering a handler."""
        handler = MockEventHandler("TestHandler")
        dispatcher.register_handler(handler)
        dispatcher.unregister_handler(handler)

        assert len(dispatcher._handlers) == 0
        assert handler not in dispatcher._handlers

    def test_unregister_nonexistent_handler_safe(self, dispatcher):
        """Test that unregistering non-existent handler doesn't raise."""
        handler = MockEventHandler("TestHandler")
        # Should not raise
        dispatcher.unregister_handler(handler)
        assert len(dispatcher._handlers) == 0


class TestEventDispatcherStartStop:
    """Test start and stop functionality."""

    def test_start_registers_callbacks(self, dispatcher, mock_mumble):
        """Test that start() registers all pymumble callbacks."""
        dispatcher.start()

        assert dispatcher._started is True
        mock_mumble.set_receive_sound.assert_called_once_with(True)

        # Verify all callbacks were registered
        add_callback = mock_mumble.callbacks.add_callback
        callback_names = [call[0][0] for call in add_callback.call_args_list]

        expected_callbacks = [
            "connected",
            "disconnected",
            "channel_created",
            "channel_updated",
            "channel_remove",
            "user_created",
            "user_updated",
            "user_removed",
            "sound_received",
            "text_received",
            "contextAction_received",
            "acl_received",
            "permission_denied",
        ]

        for name in expected_callbacks:
            assert name in callback_names

    def test_start_idempotent(self, dispatcher, mock_mumble):
        """Test that start() is idempotent."""
        dispatcher.start()
        dispatcher.start()

        # Should only register callbacks once
        assert mock_mumble.callbacks.add_callback.call_count == 13

    def test_start_without_sound_events(self, mock_mumble, event_loop):
        """Test start without sound events enabled."""
        dispatcher = EventDispatcher(
            mock_mumble, loop=event_loop, enable_sound_events=False
        )
        dispatcher.start()

        mock_mumble.set_receive_sound.assert_not_called()

    def test_stop_removes_callbacks(self, dispatcher, mock_mumble):
        """Test that stop() removes all callbacks."""
        dispatcher.start()
        dispatcher.stop()

        assert dispatcher._started is False

        # Verify callbacks were removed
        remove_callback = mock_mumble.callbacks.remove_callback
        callback_names = [call[0][0] for call in remove_callback.call_args_list]

        expected_callbacks = [
            "connected",
            "disconnected",
            "channel_created",
            "channel_updated",
            "channel_remove",
            "user_created",
            "user_updated",
            "user_removed",
            "sound_received",
            "text_received",
            "contextAction_received",
            "acl_received",
            "permission_denied",
        ]

        for name in expected_callbacks:
            assert name in callback_names

    def test_stop_idempotent(self, dispatcher, mock_mumble):
        """Test that stop() is idempotent."""
        dispatcher.start()
        dispatcher.stop()
        dispatcher.stop()

        # Should only remove callbacks once
        assert mock_mumble.callbacks.remove_callback.call_count == 13


class TestEventConversion:
    """Test conversion of pymumble callbacks to events."""

    def test_on_connected_creates_event(self, dispatcher, event_loop):
        """Test connected callback creates ConnectedEvent."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        # Simulate the callback
        dispatcher._on_connected()

        # Wait for async dispatch
        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_connected_calls == 1
        assert len(handler.events_received) == 1
        assert isinstance(handler.events_received[0], ConnectedEvent)

    def test_on_disconnected_creates_event(self, dispatcher, event_loop):
        """Test disconnected callback creates DisconnectedEvent."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        dispatcher._on_disconnected()

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_disconnected_calls == 1
        assert isinstance(handler.events_received[0], DisconnectedEvent)

    def test_on_channel_created_converts_correctly(self, dispatcher, event_loop):
        """Test channel_created callback converts channel data."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        channel_data = {
            "channel_id": 123,
            "name": "Test Channel",
            "parent": 0,
            "description": "A test channel",
            "position": 1,
            "max_users": 10,
        }

        dispatcher._on_channel_created(channel_data)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_channel_created_calls == 1
        event = handler.events_received[0]
        assert isinstance(event, ChannelCreatedEvent)
        assert event.channel_id == 123
        assert event.name == "Test Channel"
        assert event.parent_id == 0
        assert event.description == "A test channel"
        assert event.position == 1
        assert event.max_users == 10

    def test_on_channel_updated_includes_changes(self, dispatcher, event_loop):
        """Test channel_updated callback includes changed fields."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        channel_data = {"channel_id": 123, "name": "Updated Channel"}
        actions = {"name": "Updated Channel"}

        dispatcher._on_channel_updated(channel_data, actions)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_channel_updated_calls == 1
        event = handler.events_received[0]
        assert isinstance(event, ChannelUpdatedEvent)
        assert event.changed_fields == {"name": "Updated Channel"}

    def test_on_user_created_converts_correctly(self, dispatcher, event_loop):
        """Test user_created callback converts user data."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        user_data = {
            "session": 42,
            "name": "TestUser",
            "channel_id": 100,
            "user_id": 1001,
            "mute": False,
            "deaf": False,
            "suppress": False,
            "self_mute": True,
            "self_deaf": False,
            "recording": False,
            "comment": "Hello!",
        }

        dispatcher._on_user_created(user_data)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_user_joined_calls == 1
        event = handler.events_received[0]
        assert isinstance(event, UserJoinedEvent)
        assert event.session_id == 42
        assert event.name == "TestUser"
        assert event.channel_id == 100
        assert event.user_id == 1001
        assert event.self_muted is True

    def test_on_user_removed_extracts_reason(self, dispatcher, event_loop):
        """Test user_removed callback extracts kick/ban info."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        user_data = {"session": 42, "name": "KickedUser", "channel_id": 100}

        # Mock protobuf message
        message = MagicMock()
        message.reason = "Violated rules"
        message.ban = False
        message.actor = 1  # Session of admin who kicked

        dispatcher._on_user_removed(user_data, message)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_user_left_calls == 1
        event = handler.events_received[0]
        assert isinstance(event, UserLeftEvent)
        assert event.reason == "Violated rules"
        assert event.ban is False
        assert event.kicked is True
        assert event.actor_session_id == 1

    def test_on_sound_received_extracts_audio(self, dispatcher, event_loop):
        """Test sound_received callback extracts audio data."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        user_data = {"session": 42, "name": "Speaker"}

        sound_chunk = MagicMock()
        sound_chunk.pcm = b"\x00\x01\x02\x03"
        sound_chunk.sequence = 12345
        sound_chunk.duration = 0.02
        sound_chunk.time = 1234567890.0

        dispatcher._on_sound_received(user_data, sound_chunk)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_sound_received_calls == 1
        event = handler.events_received[0]
        assert isinstance(event, SoundReceivedEvent)
        assert event.session_id == 42
        assert event.user_name == "Speaker"
        assert event.pcm_data == b"\x00\x01\x02\x03"
        assert event.sequence == 12345

    def test_on_text_received_parses_message(self, dispatcher, event_loop, mock_mumble):
        """Test text_received callback parses message correctly."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        # Setup mock users - users[session_id] returns a dict
        mock_user_42 = MagicMock()
        mock_user_42.get.return_value = "Sender"

        mock_users = MagicMock()
        mock_users.__contains__ = MagicMock(return_value=True)
        mock_users.__getitem__ = MagicMock(return_value=mock_user_42)

        # myself is accessed via .myself.get("session")
        mock_myself = MagicMock()
        mock_myself.get.return_value = 1
        mock_users.myself = mock_myself

        mock_mumble.users = mock_users

        message = MagicMock()
        message.actor = 42
        message.message = "Hello bot!"
        message.channel_id = [100]
        message.session = [1]  # Sent to session 1 (the bot)

        dispatcher._on_text_received(message)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_text_message_calls == 1
        event = handler.events_received[0]
        assert isinstance(event, TextMessageEvent)
        assert event.sender_session_id == 42
        assert event.sender_name == "Sender"
        assert event.message == "Hello bot!"
        assert event.is_private is True


class TestEventDispatchFiltering:
    """Test event filtering based on subscribed_events."""

    def test_handler_receives_only_subscribed_events(self, dispatcher, event_loop):
        """Test handler only receives events it subscribed to."""
        # Handler that only subscribes to connected events
        handler = MockEventHandler(
            "ConnectedOnly",
            subscribed={EventType.CONNECTED},
        )
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        # Send multiple events
        dispatcher._on_connected()
        dispatcher._on_disconnected()

        channel_data = {"channel_id": 1, "name": "Test"}
        dispatcher._on_channel_created(channel_data)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        # Should only have received the connected event
        assert handler.on_connected_calls == 1
        assert handler.on_disconnected_calls == 0
        assert handler.on_channel_created_calls == 0

    def test_handler_with_none_subscribed_receives_all(self, dispatcher, event_loop):
        """Test handler with None subscribed_events receives all events."""
        handler = MockEventHandler("AllEvents", subscribed=None)
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        dispatcher._on_connected()
        dispatcher._on_disconnected()

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler.on_connected_calls == 1
        assert handler.on_disconnected_calls == 1


class TestEventDispatchMultipleHandlers:
    """Test dispatching to multiple handlers."""

    def test_dispatch_to_multiple_handlers(self, dispatcher, event_loop):
        """Test event is dispatched to all registered handlers."""
        handler1 = MockEventHandler("Handler1")
        handler2 = MockEventHandler("Handler2")
        handler3 = MockEventHandler("Handler3")

        dispatcher.register_handler(handler1)
        dispatcher.register_handler(handler2)
        dispatcher.register_handler(handler3)
        dispatcher._loop = event_loop

        dispatcher._on_connected()

        event_loop.run_until_complete(asyncio.sleep(0.1))

        assert handler1.on_connected_calls == 1
        assert handler2.on_connected_calls == 1
        assert handler3.on_connected_calls == 1

    def test_handler_exception_isolated(self, dispatcher, event_loop):
        """Test that exception in one handler doesn't affect others."""
        exception_handler = ExceptionHandler()
        normal_handler = MockEventHandler("NormalHandler")

        dispatcher.register_handler(exception_handler)
        dispatcher.register_handler(normal_handler)
        dispatcher._loop = event_loop

        # Should not raise, even though exception_handler raises
        dispatcher._on_connected()

        event_loop.run_until_complete(asyncio.sleep(0.1))

        # Normal handler should still receive the event
        assert normal_handler.on_connected_calls == 1


class TestEventMapping:
    """Test event type mapping."""

    def test_pymumble_to_event_type_mapping(self):
        """Test the mapping from pymumble callbacks to EventType."""
        expected_mappings = {
            "connected": EventType.CONNECTED,
            "disconnected": EventType.DISCONNECTED,
            "channel_created": EventType.CHANNEL_CREATED,
            "channel_updated": EventType.CHANNEL_UPDATED,
            "channel_remove": EventType.CHANNEL_REMOVED,
            "user_created": EventType.USER_CREATED,
            "user_updated": EventType.USER_UPDATED,
            "user_removed": EventType.USER_REMOVED,
            "sound_received": EventType.SOUND_RECEIVED,
            "text_received": EventType.TEXT_RECEIVED,
            "contextAction_received": EventType.CONTEXT_ACTION,
            "acl_received": EventType.ACL_RECEIVED,
            "permission_denied": EventType.PERMISSION_DENIED,
        }

        for callback_name, event_type in expected_mappings.items():
            assert PYMUMBLE_TO_EVENT_TYPE.get(callback_name) == event_type


class TestContextActionAndACL:
    """Test context action and ACL events."""

    def test_on_context_action(self, dispatcher, event_loop):
        """Test context action callback creates ContextActionEvent."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        action = MagicMock()
        action.action = "mute_user"
        action.session = 42
        action.channel_id = 100

        dispatcher._on_context_action(action)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        # Find the context action event
        context_events = [
            e for e in handler.events_received if isinstance(e, ContextActionEvent)
        ]
        assert len(context_events) == 1
        event = context_events[0]
        assert event.action == "mute_user"
        assert event.session_id == 42
        assert event.channel_id == 100

    def test_on_acl_received(self, dispatcher, event_loop):
        """Test ACL received callback creates ACLReceivedEvent."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        acl = MagicMock()
        acl.channel_id = 123

        dispatcher._on_acl_received(acl)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        acl_events = [
            e for e in handler.events_received if isinstance(e, ACLReceivedEvent)
        ]
        assert len(acl_events) == 1
        assert acl_events[0].channel_id == 123

    def test_on_permission_denied(self, dispatcher, event_loop):
        """Test permission denied callback creates PermissionDeniedEvent."""
        handler = MockEventHandler()
        dispatcher.register_handler(handler)
        dispatcher._loop = event_loop

        denied = MagicMock()
        denied.reason = "No permission"
        denied.type = 1
        denied.channel_id = 100
        denied.session = 42
        denied.name = "test_action"

        dispatcher._on_permission_denied(denied)

        event_loop.run_until_complete(asyncio.sleep(0.1))

        denied_events = [
            e for e in handler.events_received if isinstance(e, PermissionDeniedEvent)
        ]
        assert len(denied_events) == 1
        event = denied_events[0]
        assert event.reason == "No permission"
        assert event.channel_id == 100


class TestThreadSafety:
    """Test thread safety of handler registration."""

    def test_concurrent_handler_registration(self, dispatcher):
        """Test that handlers can be registered from multiple threads."""
        handlers = [MockEventHandler(f"Handler{i}") for i in range(10)]
        errors = []

        def register_handler(handler):
            try:
                dispatcher.register_handler(handler)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_handler, args=(h,))
            for h in handlers
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(dispatcher._handlers) == 10
