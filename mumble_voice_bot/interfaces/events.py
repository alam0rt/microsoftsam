"""Abstract interface for Mumble event handling.

Provides a clean abstraction over pymumble's callback system with typed
event payloads and async handler support.

Available Events (from pymumble):
- CONNECTED: Connection to server established
- DISCONNECTED: Connection dropped
- CHANNEL_CREATED/UPDATED/REMOVED: Channel state changes
- USER_CREATED/UPDATED/REMOVED: User presence changes
- SOUND_RECEIVED: Audio data from users
- TEXT_RECEIVED: Chat messages
- CONTEXT_ACTION: Context menu actions
- ACL_RECEIVED: Permission data received
- PERMISSION_DENIED: Action denied by server
"""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """All pymumble event types."""
    CONNECTED = auto()
    DISCONNECTED = auto()
    CHANNEL_CREATED = auto()
    CHANNEL_UPDATED = auto()
    CHANNEL_REMOVED = auto()
    USER_CREATED = auto()
    USER_UPDATED = auto()
    USER_REMOVED = auto()
    SOUND_RECEIVED = auto()
    TEXT_RECEIVED = auto()
    CONTEXT_ACTION = auto()
    ACL_RECEIVED = auto()
    PERMISSION_DENIED = auto()


# =============================================================================
# Event Payload Dataclasses
# =============================================================================

@dataclass
class BaseEvent:
    """Base class for all event payloads."""
    event_type: EventType


@dataclass
class ConnectedEvent(BaseEvent):
    """Fired when connection to Mumble server is established."""
    event_type: EventType = field(default=EventType.CONNECTED, init=False)


@dataclass
class DisconnectedEvent(BaseEvent):
    """Fired when connection to Mumble server is lost."""
    event_type: EventType = field(default=EventType.DISCONNECTED, init=False)


@dataclass
class ChannelEvent(BaseEvent):
    """Base class for channel-related events."""
    channel_id: int
    name: str
    parent_id: int | None = None
    description: str | None = None
    position: int | None = None
    max_users: int | None = None
    raw: Any = field(default=None, repr=False)  # Original pymumble channel object


@dataclass
class ChannelCreatedEvent(ChannelEvent):
    """Fired when a new channel is created."""
    event_type: EventType = field(default=EventType.CHANNEL_CREATED, init=False)


@dataclass
class ChannelUpdatedEvent(ChannelEvent):
    """Fired when a channel's properties change."""
    event_type: EventType = field(default=EventType.CHANNEL_UPDATED, init=False)
    changed_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelRemovedEvent(ChannelEvent):
    """Fired when a channel is deleted."""
    event_type: EventType = field(default=EventType.CHANNEL_REMOVED, init=False)


@dataclass
class UserEvent(BaseEvent):
    """Base class for user-related events."""
    session_id: int
    name: str
    channel_id: int
    user_id: int | None = None  # Registered user ID, if any
    muted: bool = False
    deafened: bool = False
    suppressed: bool = False
    self_muted: bool = False
    self_deafened: bool = False
    recording: bool = False
    comment: str | None = None
    raw: Any = field(default=None, repr=False)  # Original pymumble user object


@dataclass
class UserJoinedEvent(UserEvent):
    """Fired when a user connects to the server."""
    event_type: EventType = field(default=EventType.USER_CREATED, init=False)


@dataclass
class UserUpdatedEvent(UserEvent):
    """Fired when a user's state changes (channel, mute, etc.)."""
    event_type: EventType = field(default=EventType.USER_UPDATED, init=False)
    changed_fields: dict[str, Any] = field(default_factory=dict)
    actor_session_id: int | None = None  # Who performed the action


@dataclass
class UserLeftEvent(UserEvent):
    """Fired when a user disconnects or is kicked/banned."""
    event_type: EventType = field(default=EventType.USER_REMOVED, init=False)
    reason: str | None = None
    ban: bool = False
    kicked: bool = False
    actor_session_id: int | None = None  # Who performed the kick/ban


@dataclass
class SoundReceivedEvent(BaseEvent):
    """Fired when audio data is received from a user."""
    event_type: EventType = field(default=EventType.SOUND_RECEIVED, init=False)
    session_id: int
    user_name: str
    pcm_data: bytes  # Raw PCM: 16-bit signed, mono, 48kHz little-endian
    sequence: int
    duration: float  # Duration in seconds
    timestamp: float  # Receive time
    raw: Any = field(default=None, repr=False)  # Original SoundChunk object


@dataclass
class TextMessageEvent(BaseEvent):
    """Fired when a text message is received."""
    event_type: EventType = field(default=EventType.TEXT_RECEIVED, init=False)
    sender_session_id: int
    message: str
    sender_name: str | None = None
    channel_ids: list[int] = field(default_factory=list)  # Target channels
    recipient_session_ids: list[int] = field(default_factory=list)  # Direct recipients
    is_private: bool = False  # True if message was sent directly to bot
    raw: Any = field(default=None, repr=False)  # Original TextMessage protobuf


@dataclass
class ContextActionEvent(BaseEvent):
    """Fired when a context menu action is triggered."""
    event_type: EventType = field(default=EventType.CONTEXT_ACTION, init=False)
    action: str
    session_id: int | None = None  # Target user session, if applicable
    channel_id: int | None = None  # Target channel, if applicable
    raw: Any = field(default=None, repr=False)


@dataclass
class ACLReceivedEvent(BaseEvent):
    """Fired when ACL/permission data is received."""
    event_type: EventType = field(default=EventType.ACL_RECEIVED, init=False)
    channel_id: int
    raw: Any = field(default=None, repr=False)  # Original ACL protobuf


@dataclass
class PermissionDeniedEvent(BaseEvent):
    """Fired when an action is denied by the server."""
    event_type: EventType = field(default=EventType.PERMISSION_DENIED, init=False)
    reason: str | None = None
    denied_type: int | None = None  # DenyType enum value from protobuf
    channel_id: int | None = None
    session_id: int | None = None
    name: str | None = None  # Name related to denial (e.g., channel name)
    raw: Any = field(default=None, repr=False)


# Type alias for any event
MumbleEvent = (
    ConnectedEvent | DisconnectedEvent |
    ChannelCreatedEvent | ChannelUpdatedEvent | ChannelRemovedEvent |
    UserJoinedEvent | UserUpdatedEvent | UserLeftEvent |
    SoundReceivedEvent | TextMessageEvent | ContextActionEvent |
    ACLReceivedEvent | PermissionDeniedEvent
)


# =============================================================================
# Event Handler Interface
# =============================================================================

class MumbleEventHandler(ABC):
    """Abstract base class for Mumble event handlers.

    Implement this class to create handlers that respond to Mumble events.
    Override only the methods for events you care about - all default
    implementations are no-ops.

    Handler methods can be sync or async. The EventDispatcher will handle
    both transparently.

    Example:
        class GreetingHandler(MumbleEventHandler):
            async def on_user_joined(self, event: UserJoinedEvent) -> None:
                print(f"Welcome, {event.name}!")

            async def on_user_left(self, event: UserLeftEvent) -> None:
                print(f"Goodbye, {event.name}!")
    """

    @property
    def name(self) -> str:
        """Handler name for logging. Override to customize."""
        return self.__class__.__name__

    def subscribed_events(self) -> set[EventType] | None:
        """Return set of events this handler cares about, or None for all.

        Override this to filter which events are dispatched to this handler.
        Returning None (default) means the handler receives all events.

        Example:
            def subscribed_events(self) -> set[EventType]:
                return {EventType.USER_CREATED, EventType.USER_REMOVED}
        """
        return None

    # Connection events
    async def on_connected(self, event: ConnectedEvent) -> None:
        """Called when connected to Mumble server."""
        pass

    async def on_disconnected(self, event: DisconnectedEvent) -> None:
        """Called when disconnected from Mumble server."""
        pass

    # Channel events
    async def on_channel_created(self, event: ChannelCreatedEvent) -> None:
        """Called when a channel is created."""
        pass

    async def on_channel_updated(self, event: ChannelUpdatedEvent) -> None:
        """Called when a channel's properties change."""
        pass

    async def on_channel_removed(self, event: ChannelRemovedEvent) -> None:
        """Called when a channel is deleted."""
        pass

    # User events
    async def on_user_joined(self, event: UserJoinedEvent) -> None:
        """Called when a user connects to the server."""
        pass

    async def on_user_updated(self, event: UserUpdatedEvent) -> None:
        """Called when a user's state changes (channel move, mute, etc.)."""
        pass

    async def on_user_left(self, event: UserLeftEvent) -> None:
        """Called when a user disconnects or is kicked/banned."""
        pass

    # Audio events
    async def on_sound_received(self, event: SoundReceivedEvent) -> None:
        """Called when audio data is received from a user.

        WARNING: This is called very frequently during voice activity.
        Keep handlers fast to avoid audio jitter.
        """
        pass

    # Message events
    async def on_text_message(self, event: TextMessageEvent) -> None:
        """Called when a text message is received."""
        pass

    # Other events
    async def on_context_action(self, event: ContextActionEvent) -> None:
        """Called when a context menu action is triggered."""
        pass

    async def on_acl_received(self, event: ACLReceivedEvent) -> None:
        """Called when ACL/permission data is received."""
        pass

    async def on_permission_denied(self, event: PermissionDeniedEvent) -> None:
        """Called when an action is denied by the server."""
        pass
