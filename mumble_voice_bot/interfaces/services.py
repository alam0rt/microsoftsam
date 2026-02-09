"""Interfaces for shared services and multi-persona bot architecture.

This module defines the abstractions for running multiple bot personas
that share expensive resources (TTS engine, STT engine, LLM client)
while maintaining separate identities (voice, personality, conversation history).

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                    SharedServices                           │
    │  - Single TTS engine (multiple voice prompts)               │
    │  - Single STT engine                                        │
    │  - Single LLM client (different system prompts per persona) │
    └─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   Persona 1   │ │   Persona 2   │ │   Persona N   │
    │  (Knight)     │ │(PotionSeller) │ │   (Custom)    │
    └───────────────┘ └───────────────┘ └───────────────┘
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable

import torch


class PersonaState(Enum):
    """Current state of a bot persona."""

    INITIALIZING = auto()  # Loading voice prompt, connecting
    IDLE = auto()          # Connected, waiting for input
    LISTENING = auto()     # Receiving audio from a user
    THINKING = auto()      # Waiting for LLM response
    SPEAKING = auto()      # Playing TTS audio
    COOLDOWN = auto()      # Post-speech cooldown period
    DISCONNECTED = auto()  # Not connected to Mumble
    ERROR = auto()         # In error state


@dataclass
class VoicePrompt:
    """Encoded voice prompt for TTS voice cloning.

    Attributes:
        name: Identifier for this voice (e.g., "knight", "potion-seller").
        tensors: The encoded voice tensors for the TTS model.
        ref_audio_path: Path to the original reference audio file.
        ref_duration: Duration of reference audio used (seconds).
    """
    name: str
    tensors: dict[str, torch.Tensor]
    ref_audio_path: str | None = None
    ref_duration: float | None = None

    def to_device(self, device: str) -> "VoicePrompt":
        """Move all tensors to the specified device.

        Args:
            device: Target device ('cuda', 'cpu', 'mps').

        Returns:
            New VoicePrompt with tensors on the target device.
        """
        moved_tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in self.tensors.items()
        }
        return VoicePrompt(
            name=self.name,
            tensors=moved_tensors,
            ref_audio_path=self.ref_audio_path,
            ref_duration=self.ref_duration,
        )


@dataclass
class ConversationMessage:
    """A single message in the conversation history.

    Attributes:
        role: Message role ('system', 'user', 'assistant', 'tool').
        content: Message content.
        speaker: Name of the speaker (for multi-user channels).
        timestamp: Unix timestamp when the message was recorded.
        persona_name: Which persona this message is associated with.
    """
    role: str
    content: str
    speaker: str | None = None
    timestamp: float | None = None
    persona_name: str | None = None


@dataclass
class PersonaIdentity:
    """Core identity configuration for a bot persona.

    This is the lightweight, serializable identity that defines who
    the persona is, separate from runtime state.

    Attributes:
        name: Unique identifier for this persona.
        display_name: Human-readable name (shown in Mumble).
        soul_name: Name of the soul to load from souls/ directory.
        system_prompt: LLM system prompt (overrides soul default if set).
        mumble_user: Username in Mumble (defaults to display_name).
        mumble_channel: Channel to join (optional).
    """
    name: str
    display_name: str
    soul_name: str | None = None
    system_prompt: str | None = None
    mumble_user: str | None = None
    mumble_channel: str | None = None

    @property
    def effective_mumble_user(self) -> str:
        """Get the Mumble username to use."""
        return self.mumble_user or self.display_name


@dataclass
class PersonaConfig:
    """Full configuration for a bot persona.

    Extends PersonaIdentity with voice and behavior settings.

    Attributes:
        identity: Core identity configuration.
        voice_prompt: Encoded voice for TTS (loaded at runtime).
        llm_overrides: LLM parameter overrides (temperature, max_tokens, etc).
        max_history_messages: Maximum conversation history to maintain.
        respond_to_other_personas: Whether to respond to other bot personas.
        tts: TTS configuration (ref_audio, num_steps, etc).
        soul_config: Loaded soul configuration object.
        mumble: Mumble connection settings.
    """
    identity: PersonaIdentity
    voice_prompt: VoicePrompt | None = None
    llm_overrides: dict[str, Any] = field(default_factory=dict)
    max_history_messages: int = 20
    respond_to_other_personas: bool = False
    tts: dict[str, Any] = field(default_factory=dict)
    soul_config: Any = None  # SoulConfig from config.py
    mumble: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MumbleClientProtocol(Protocol):
    """Protocol for Mumble client operations.

    This allows us to mock the Mumble client in tests.
    """

    def start(self) -> None:
        """Start the Mumble connection."""
        ...

    def stop(self) -> None:
        """Stop the Mumble connection."""
        ...

    def is_ready(self) -> bool:
        """Check if connected and ready."""
        ...

    def set_receive_sound(self, receive: bool) -> None:
        """Enable or disable audio reception."""
        ...

    @property
    def sound_output(self) -> Any:
        """Get the sound output interface."""
        ...

    @property
    def my_channel(self) -> Any:
        """Get the bot's current channel."""
        ...


@dataclass
class Persona:
    """Runtime state for a single bot persona.

    This combines the static configuration with runtime state like
    conversation history, current state, and the Mumble connection.

    Attributes:
        config: The persona's configuration.
        state: Current runtime state.
        conversation_history: List of conversation messages.
        mumble_client: The Mumble client instance (optional, for testing).
        last_activity_time: Timestamp of last activity.
        current_turn_id: ID of the current conversation turn.
    """
    config: PersonaConfig
    state: PersonaState = PersonaState.INITIALIZING
    conversation_history: list[ConversationMessage] = field(default_factory=list)
    mumble_client: Any = None  # pymumble.Mumble instance
    last_activity_time: float = 0.0
    current_turn_id: int = 0

    @property
    def name(self) -> str:
        """Get the persona's unique identifier."""
        return self.config.identity.name

    @property
    def display_name(self) -> str:
        """Get the persona's display name."""
        return self.config.identity.display_name

    @property
    def voice_prompt(self) -> VoicePrompt | None:
        """Get the persona's voice prompt."""
        return self.config.voice_prompt

    @property
    def system_prompt(self) -> str | None:
        """Get the persona's system prompt."""
        return self.config.identity.system_prompt

    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to the conversation history.

        Automatically trims history to max_history_messages.
        """
        self.conversation_history.append(message)
        max_messages = self.config.max_history_messages
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

    def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Convert conversation history to LLM message format.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]


class TTSServiceProtocol(Protocol):
    """Protocol for TTS service operations."""

    def generate_speech(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        num_steps: int = 4,
    ) -> torch.Tensor:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize.
            voice_prompt: Voice to use for synthesis.
            num_steps: Number of diffusion steps.

        Returns:
            Audio tensor.
        """
        ...

    def generate_speech_streaming(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        num_steps: int = 4,
    ) -> AsyncIterator[torch.Tensor]:
        """Generate speech audio in streaming chunks.

        Args:
            text: Text to synthesize.
            voice_prompt: Voice to use for synthesis.
            num_steps: Number of diffusion steps.

        Yields:
            Audio tensor chunks.
        """
        ...


class STTServiceProtocol(Protocol):
    """Protocol for STT service operations."""

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio_data: Raw PCM audio bytes.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Transcribed text.
        """
        ...


class LLMServiceProtocol(Protocol):
    """Protocol for LLM service operations."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat response.

        Args:
            messages: Conversation history.
            system_prompt: System prompt to use.
            **kwargs: Additional parameters (temperature, max_tokens, etc).

        Returns:
            Generated response text.
        """
        ...

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a chat response with streaming.

        Args:
            messages: Conversation history.
            system_prompt: System prompt to use.
            **kwargs: Additional parameters.

        Yields:
            Response text chunks.
        """
        ...


@dataclass
class SharedServices:
    """Container for shared, expensive resources.

    These services are instantiated once and shared across all personas.
    This saves GPU memory and initialization time.

    Attributes:
        tts: Text-to-speech service.
        stt: Speech-to-text service.
        llm: Language model service.
        device: Compute device for GPU-accelerated services.
        voice_prompts: Pre-loaded voice prompts keyed by name.
    """
    tts: Any  # StreamingLuxTTS or TTSServiceProtocol
    stt: Any  # STTProvider implementation
    llm: Any  # LLMProvider implementation
    device: str = "cuda"
    voice_prompts: dict[str, VoicePrompt] = field(default_factory=dict)

    def get_voice_prompt(self, name: str) -> VoicePrompt | None:
        """Get a voice prompt by name.

        Args:
            name: Voice prompt identifier.

        Returns:
            VoicePrompt if found, None otherwise.
        """
        return self.voice_prompts.get(name)

    def register_voice_prompt(self, voice_prompt: VoicePrompt) -> None:
        """Register a voice prompt for later use.

        Args:
            voice_prompt: The voice prompt to register.
        """
        self.voice_prompts[voice_prompt.name] = voice_prompt


# Type alias for state change callbacks
StateChangeCallback = Callable[[Persona, PersonaState, PersonaState], None]


class PersonaManager(ABC):
    """Abstract base class for managing multiple bot personas.

    Implementations of this class coordinate multiple personas,
    routing audio and managing turn-taking between them.
    """

    @abstractmethod
    def add_persona(self, persona: Persona) -> None:
        """Add a persona to be managed.

        Args:
            persona: The persona to add.
        """
        pass

    @abstractmethod
    def remove_persona(self, name: str) -> Persona | None:
        """Remove a persona by name.

        Args:
            name: The persona's unique identifier.

        Returns:
            The removed persona, or None if not found.
        """
        pass

    @abstractmethod
    def get_persona(self, name: str) -> Persona | None:
        """Get a persona by name.

        Args:
            name: The persona's unique identifier.

        Returns:
            The persona, or None if not found.
        """
        pass

    @abstractmethod
    def list_personas(self) -> list[Persona]:
        """List all managed personas.

        Returns:
            List of all personas.
        """
        pass

    @abstractmethod
    async def start_all(self) -> None:
        """Start all personas (connect to Mumble)."""
        pass

    @abstractmethod
    async def stop_all(self) -> None:
        """Stop all personas (disconnect from Mumble)."""
        pass

    @abstractmethod
    def on_state_change(self, callback: StateChangeCallback) -> None:
        """Register a callback for persona state changes.

        Args:
            callback: Function called with (persona, old_state, new_state).
        """
        pass


@dataclass
class InteractionConfig:
    """Configuration for bot-to-bot interactions.

    Controls how personas interact with each other to prevent
    infinite loops and enable natural conversations.

    Attributes:
        enable_cross_talk: Allow bots to hear and respond to each other.
        response_delay_ms: Delay before responding to another bot (ms).
        max_chain_length: Maximum consecutive bot-to-bot exchanges.
        cooldown_after_chain_ms: Cooldown after max chain reached (ms).
        ignore_own_audio: Prevent bot from hearing its own TTS output.
    """
    enable_cross_talk: bool = True
    response_delay_ms: int = 500
    max_chain_length: int = 5
    cooldown_after_chain_ms: int = 3000
    ignore_own_audio: bool = True


@dataclass
class MultiPersonaConfig:
    """Top-level configuration for multi-persona bot deployment.

    Attributes:
        personas: List of persona configurations.
        shared: Configuration for shared services.
        interaction: Bot-to-bot interaction settings.
        mumble_host: Default Mumble server host.
        mumble_port: Default Mumble server port.
        mumble_password: Default Mumble server password.
    """
    personas: list[PersonaConfig] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)  # LLM, TTS, STT config
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    mumble_host: str = "localhost"
    mumble_port: int = 64738
    mumble_password: str | None = None
