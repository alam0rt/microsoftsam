"""Event dispatcher for pymumble callbacks.

Bridges pymumble's callback system to the MumbleEventHandler interface,
providing typed event payloads and async handler support.
"""

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable

from mumble_voice_bot.interfaces.events import (
    ACLReceivedEvent,
    ChannelCreatedEvent,
    ChannelRemovedEvent,
    ChannelUpdatedEvent,
    # Event payloads
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

if TYPE_CHECKING:
    from vendor.botamusique.pymumble_py3 import Mumble

logger = logging.getLogger(__name__)


# Mapping from pymumble callback names to EventType
PYMUMBLE_TO_EVENT_TYPE: dict[str, EventType] = {
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


class EventDispatcher:
    """Dispatches pymumble events to registered MumbleEventHandler instances.

    This class:
    1. Registers with pymumble's callback system
    2. Converts raw pymumble callbacks into typed event objects
    3. Dispatches events to registered handlers (sync or async)

    Thread Safety:
        pymumble callbacks run in the pymumble thread. This dispatcher
        schedules async handlers via asyncio.run_coroutine_threadsafe()
        to avoid blocking the audio loop.

    Example:
        # Setup
        mumble = pymumble.Mumble(host, user)
        loop = asyncio.get_event_loop()
        dispatcher = EventDispatcher(mumble, loop)

        # Register handlers
        dispatcher.register_handler(GreetingHandler())
        dispatcher.register_handler(CommandHandler())

        # Start listening
        dispatcher.start()
        mumble.start()
    """

    def __init__(
        self,
        mumble: "Mumble",
        loop: asyncio.AbstractEventLoop | None = None,
        enable_sound_events: bool = True,
    ):
        """Initialize the event dispatcher.

        Args:
            mumble: The pymumble Mumble instance to listen to.
            loop: The asyncio event loop for async handler dispatch.
                  If None, will try to get the running loop when needed.
            enable_sound_events: Whether to enable audio reception for
                                 SOUND_RECEIVED events. Default True.
        """
        self.mumble = mumble
        self._loop = loop
        self._enable_sound_events = enable_sound_events
        self._handlers: list[MumbleEventHandler] = []
        self._started = False
        self._lock = threading.Lock()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop, creating one if needed."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        return self._loop

    def register_handler(self, handler: MumbleEventHandler) -> None:
        """Register an event handler.

        Args:
            handler: The MumbleEventHandler to register.
        """
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)
                logger.debug(f"Registered event handler: {handler.name}")

    def unregister_handler(self, handler: MumbleEventHandler) -> None:
        """Unregister an event handler.

        Args:
            handler: The MumbleEventHandler to unregister.
        """
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)
                logger.debug(f"Unregistered event handler: {handler.name}")

    def start(self) -> None:
        """Start listening for pymumble events.

        Registers callbacks with pymumble for all event types.
        """
        if self._started:
            return

        # Enable sound reception if needed
        if self._enable_sound_events:
            self.mumble.set_receive_sound(True)

        # Register callbacks for all events
        callbacks = self.mumble.callbacks

        callbacks.add_callback("connected", self._on_connected)
        callbacks.add_callback("disconnected", self._on_disconnected)
        callbacks.add_callback("channel_created", self._on_channel_created)
        callbacks.add_callback("channel_updated", self._on_channel_updated)
        callbacks.add_callback("channel_remove", self._on_channel_removed)
        callbacks.add_callback("user_created", self._on_user_created)
        callbacks.add_callback("user_updated", self._on_user_updated)
        callbacks.add_callback("user_removed", self._on_user_removed)
        callbacks.add_callback("sound_received", self._on_sound_received)
        callbacks.add_callback("text_received", self._on_text_received)
        callbacks.add_callback("contextAction_received", self._on_context_action)
        callbacks.add_callback("acl_received", self._on_acl_received)
        callbacks.add_callback("permission_denied", self._on_permission_denied)

        self._started = True
        logger.info("EventDispatcher started")

    def stop(self) -> None:
        """Stop listening for pymumble events.

        Removes all callbacks registered by this dispatcher.
        """
        if not self._started:
            return

        callbacks = self.mumble.callbacks

        try:
            callbacks.remove_callback("connected", self._on_connected)
            callbacks.remove_callback("disconnected", self._on_disconnected)
            callbacks.remove_callback("channel_created", self._on_channel_created)
            callbacks.remove_callback("channel_updated", self._on_channel_updated)
            callbacks.remove_callback("channel_remove", self._on_channel_removed)
            callbacks.remove_callback("user_created", self._on_user_created)
            callbacks.remove_callback("user_updated", self._on_user_updated)
            callbacks.remove_callback("user_removed", self._on_user_removed)
            callbacks.remove_callback("sound_received", self._on_sound_received)
            callbacks.remove_callback("text_received", self._on_text_received)
            callbacks.remove_callback("contextAction_received", self._on_context_action)
            callbacks.remove_callback("acl_received", self._on_acl_received)
            callbacks.remove_callback("permission_denied", self._on_permission_denied)
        except Exception as e:
            logger.warning(f"Error removing callbacks: {e}")

        self._started = False
        logger.info("EventDispatcher stopped")

    def _dispatch(self, event: MumbleEvent) -> None:
        """Dispatch an event to all registered handlers.

        Runs synchronously in the pymumble thread but schedules
        async handlers via the event loop.
        """
        with self._lock:
            handlers = list(self._handlers)

        for handler in handlers:
            # Check if handler wants this event
            subscribed = handler.subscribed_events()
            if subscribed is not None and event.event_type not in subscribed:
                continue

            # Get the appropriate handler method
            handler_method = self._get_handler_method(handler, event)
            if handler_method is None:
                continue

            try:
                # Schedule async execution
                future = asyncio.run_coroutine_threadsafe(
                    handler_method(event),
                    self.loop
                )
                # Don't wait for result - fire and forget
                # Errors will be logged via the future's exception
                future.add_done_callback(
                    lambda f, h=handler.name: self._handle_dispatch_result(f, h, event)
                )
            except Exception as e:
                logger.error(f"Error dispatching {event.event_type} to {handler.name}: {e}")

    def _handle_dispatch_result(
        self,
        future: asyncio.Future,
        handler_name: str,
        event: MumbleEvent
    ) -> None:
        """Handle the result of an async dispatch."""
        try:
            future.result()
        except Exception as e:
            logger.error(
                f"Handler {handler_name} raised exception for "
                f"{event.event_type.name}: {e}",
                exc_info=True
            )

    def _get_handler_method(
        self,
        handler: MumbleEventHandler,
        event: MumbleEvent
    ) -> Callable | None:
        """Get the handler method for a specific event type."""
        method_map = {
            EventType.CONNECTED: handler.on_connected,
            EventType.DISCONNECTED: handler.on_disconnected,
            EventType.CHANNEL_CREATED: handler.on_channel_created,
            EventType.CHANNEL_UPDATED: handler.on_channel_updated,
            EventType.CHANNEL_REMOVED: handler.on_channel_removed,
            EventType.USER_CREATED: handler.on_user_joined,
            EventType.USER_UPDATED: handler.on_user_updated,
            EventType.USER_REMOVED: handler.on_user_left,
            EventType.SOUND_RECEIVED: handler.on_sound_received,
            EventType.TEXT_RECEIVED: handler.on_text_message,
            EventType.CONTEXT_ACTION: handler.on_context_action,
            EventType.ACL_RECEIVED: handler.on_acl_received,
            EventType.PERMISSION_DENIED: handler.on_permission_denied,
        }
        return method_map.get(event.event_type)

    # =========================================================================
    # pymumble Callback Handlers
    # =========================================================================

    def _on_connected(self) -> None:
        """Handle pymumble connected callback."""
        event = ConnectedEvent()
        logger.debug("Connected to Mumble server")
        self._dispatch(event)

    def _on_disconnected(self) -> None:
        """Handle pymumble disconnected callback."""
        event = DisconnectedEvent()
        logger.debug("Disconnected from Mumble server")
        self._dispatch(event)

    def _on_channel_created(self, channel: Any) -> None:
        """Handle pymumble channel_created callback."""
        event = ChannelCreatedEvent(
            channel_id=channel.get("channel_id", 0),
            name=channel.get("name", ""),
            parent_id=channel.get("parent"),
            description=channel.get("description"),
            position=channel.get("position"),
            max_users=channel.get("max_users"),
            raw=channel,
        )
        logger.debug(f"Channel created: {event.name}")
        self._dispatch(event)

    def _on_channel_updated(self, channel: Any, actions: dict) -> None:
        """Handle pymumble channel_updated callback."""
        event = ChannelUpdatedEvent(
            channel_id=channel.get("channel_id", 0),
            name=channel.get("name", ""),
            parent_id=channel.get("parent"),
            description=channel.get("description"),
            position=channel.get("position"),
            max_users=channel.get("max_users"),
            changed_fields=actions,
            raw=channel,
        )
        logger.debug(f"Channel updated: {event.name}, changes: {list(actions.keys())}")
        self._dispatch(event)

    def _on_channel_removed(self, channel: Any) -> None:
        """Handle pymumble channel_remove callback."""
        event = ChannelRemovedEvent(
            channel_id=channel.get("channel_id", 0),
            name=channel.get("name", ""),
            parent_id=channel.get("parent"),
            description=channel.get("description"),
            position=channel.get("position"),
            max_users=channel.get("max_users"),
            raw=channel,
        )
        logger.debug(f"Channel removed: {event.name}")
        self._dispatch(event)

    def _on_user_created(self, user: Any) -> None:
        """Handle pymumble user_created callback."""
        event = UserJoinedEvent(
            session_id=user.get("session", 0),
            name=user.get("name", "Unknown"),
            channel_id=user.get("channel_id", 0),
            user_id=user.get("user_id"),
            muted=user.get("mute", False),
            deafened=user.get("deaf", False),
            suppressed=user.get("suppress", False),
            self_muted=user.get("self_mute", False),
            self_deafened=user.get("self_deaf", False),
            recording=user.get("recording", False),
            comment=user.get("comment"),
            raw=user,
        )
        logger.debug(f"User joined: {event.name} (session={event.session_id})")
        self._dispatch(event)

    def _on_user_updated(self, user: Any, actions: dict) -> None:
        """Handle pymumble user_updated callback."""
        event = UserUpdatedEvent(
            session_id=user.get("session", 0),
            name=user.get("name", "Unknown"),
            channel_id=user.get("channel_id", 0),
            user_id=user.get("user_id"),
            muted=user.get("mute", False),
            deafened=user.get("deaf", False),
            suppressed=user.get("suppress", False),
            self_muted=user.get("self_mute", False),
            self_deafened=user.get("self_deaf", False),
            recording=user.get("recording", False),
            comment=user.get("comment"),
            changed_fields=actions,
            actor_session_id=actions.get("actor"),
            raw=user,
        )
        logger.debug(f"User updated: {event.name}, changes: {list(actions.keys())}")
        self._dispatch(event)

    def _on_user_removed(self, user: Any, message: Any) -> None:
        """Handle pymumble user_removed callback."""
        # Message is a protobuf with session, reason, ban, actor fields
        reason = getattr(message, "reason", None)
        ban = getattr(message, "ban", False)
        actor = getattr(message, "actor", None)

        event = UserLeftEvent(
            session_id=user.get("session", 0),
            name=user.get("name", "Unknown"),
            channel_id=user.get("channel_id", 0),
            user_id=user.get("user_id"),
            muted=user.get("mute", False),
            deafened=user.get("deaf", False),
            suppressed=user.get("suppress", False),
            self_muted=user.get("self_mute", False),
            self_deafened=user.get("self_deaf", False),
            recording=user.get("recording", False),
            comment=user.get("comment"),
            reason=reason,
            ban=ban,
            kicked=actor is not None and not ban,
            actor_session_id=actor,
            raw=user,
        )
        logger.debug(f"User left: {event.name} (ban={ban}, kicked={event.kicked})")
        self._dispatch(event)

    def _on_sound_received(self, user: Any, sound_chunk: Any) -> None:
        """Handle pymumble sound_received callback.

        WARNING: This is called very frequently. Keep processing minimal.
        """
        event = SoundReceivedEvent(
            session_id=user.get("session", 0),
            user_name=user.get("name", "Unknown"),
            pcm_data=sound_chunk.pcm,
            sequence=sound_chunk.sequence,
            duration=sound_chunk.duration,
            timestamp=sound_chunk.time,
            raw=sound_chunk,
        )
        # Don't log sound events - too noisy
        self._dispatch(event)

    def _on_text_received(self, message: Any) -> None:
        """Handle pymumble text_received callback."""
        # Get sender info
        sender_session = message.actor
        sender_name = None

        # Try to get sender name from users list
        try:
            if hasattr(self.mumble, 'users') and sender_session in self.mumble.users:
                sender_name = self.mumble.users[sender_session].get("name")
        except Exception:
            pass

        # Parse recipient info
        channel_ids = list(message.channel_id) if hasattr(message, 'channel_id') else []
        recipient_ids = list(message.session) if hasattr(message, 'session') else []

        # Check if it's a private message to the bot
        is_private = False
        try:
            my_session = self.mumble.users.myself.get("session")
            is_private = my_session in recipient_ids
        except Exception:
            pass

        event = TextMessageEvent(
            sender_session_id=sender_session,
            sender_name=sender_name,
            message=message.message,
            channel_ids=channel_ids,
            recipient_session_ids=recipient_ids,
            is_private=is_private,
            raw=message,
        )
        logger.debug(f"Text message from {sender_name}: {event.message[:50]}...")
        self._dispatch(event)

    def _on_context_action(self, action: Any) -> None:
        """Handle pymumble contextAction_received callback."""
        event = ContextActionEvent(
            action=getattr(action, "action", ""),
            session_id=getattr(action, "session", None),
            channel_id=getattr(action, "channel_id", None),
            raw=action,
        )
        logger.debug(f"Context action received: {event.action}")
        self._dispatch(event)

    def _on_acl_received(self, acl: Any) -> None:
        """Handle pymumble acl_received callback."""
        event = ACLReceivedEvent(
            channel_id=getattr(acl, "channel_id", 0),
            raw=acl,
        )
        logger.debug(f"ACL received for channel {event.channel_id}")
        self._dispatch(event)

    def _on_permission_denied(self, denied: Any) -> None:
        """Handle pymumble permission_denied callback."""
        event = PermissionDeniedEvent(
            reason=getattr(denied, "reason", None),
            denied_type=getattr(denied, "type", None),
            channel_id=getattr(denied, "channel_id", None),
            session_id=getattr(denied, "session", None),
            name=getattr(denied, "name", None),
            raw=denied,
        )
        logger.warning(f"Permission denied: {event.reason}")
        self._dispatch(event)
