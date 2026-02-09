"""Interfaces for pluggable components."""

from mumble_voice_bot.interfaces.events import (
    ACLReceivedEvent,
    # Event payloads
    BaseEvent,
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
from mumble_voice_bot.interfaces.llm import LLMProvider, LLMResponse
from mumble_voice_bot.interfaces.services import (
    ConversationMessage,
    InteractionConfig,
    MultiPersonaConfig,
    Persona,
    PersonaConfig,
    PersonaIdentity,
    PersonaManager,
    PersonaState,
    SharedServices,
    VoicePrompt,
)
from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.interfaces.tts import TTSProvider, TTSResult, TTSVoice

__all__ = [
    # LLM
    "LLMProvider",
    "LLMResponse",
    # STT
    "STTProvider",
    "STTResult",
    # TTS
    "TTSProvider",
    "TTSResult",
    "TTSVoice",
    # Multi-persona / Shared Services
    "ConversationMessage",
    "InteractionConfig",
    "MultiPersonaConfig",
    "Persona",
    "PersonaConfig",
    "PersonaIdentity",
    "PersonaManager",
    "PersonaState",
    "SharedServices",
    "VoicePrompt",
    # Events
    "EventType",
    "MumbleEventHandler",
    "MumbleEvent",
    "BaseEvent",
    "ConnectedEvent",
    "DisconnectedEvent",
    "ChannelCreatedEvent",
    "ChannelUpdatedEvent",
    "ChannelRemovedEvent",
    "UserJoinedEvent",
    "UserUpdatedEvent",
    "UserLeftEvent",
    "SoundReceivedEvent",
    "TextMessageEvent",
    "ContextActionEvent",
    "ACLReceivedEvent",
    "PermissionDeniedEvent",
]
