"""Brain protocol and data types for the pluggable bot architecture.

The Brain is the only thing that varies between bot types. MumbleBot
handles all I/O (Mumble connection, VAD, ASR, TTS playback), and the
Brain decides what to say given a complete utterance.

Brain implementations:
- EchoBrain: Clone speaker voice, echo transcript back (parrot)
- LLMBrain: Speech filter -> LLM -> tool loop -> response (full intelligence)
- ReactiveBrain: Fillers, echo fragments, deflections (no LLM)
- AdaptiveBrain: Score utterance -> delegate to LLMBrain or ReactiveBrain
- NullBrain: Always returns None (transcribe-only monitoring)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VoiceConfig:
    """Voice configuration for TTS output.

    Attributes:
        voice_prompt: Encoded voice tensors for TTS (dict or VoicePrompt).
        speed: TTS speech rate (1.0 = normal).
        num_steps: Number of diffusion steps for TTS quality.
    """
    voice_prompt: dict | Any = field(default_factory=dict)
    speed: float = 1.0
    num_steps: int = 4


@dataclass
class Utterance:
    """Complete utterance ready for the brain to process.

    Created by MumbleBot after VAD + ASR completes. Contains everything
    the brain needs to decide how to respond.

    Attributes:
        text: Accumulated ASR transcript.
        user_id: Mumble session ID of the speaker.
        user_name: Mumble display name of the speaker.
        audio_chunks: Raw 48kHz PCM chunks (for voice cloning).
        duration: Audio duration in seconds.
        rms: Average energy level of the audio.
        is_question: Whether the utterance appears to be a question.
        is_directed: Whether the utterance is directed at the bot.
        is_first_speech: Whether this is the user's first speech this session.
    """
    text: str
    user_id: int
    user_name: str
    audio_chunks: list[bytes] = field(default_factory=list)
    duration: float = 0.0
    rms: float = 0.0
    is_question: bool = False
    is_directed: bool = False
    is_first_speech: bool = False


@dataclass
class BotResponse:
    """What the brain wants the bot to say.

    Returned by Brain.process() when the brain decides to respond.

    Attributes:
        text: Text to speak via TTS.
        voice: Voice configuration for TTS output.
            If None, the bot's default voice is used.
        speed: TTS speech rate override (1.0 = normal).
        skip_broadcast: If True, don't broadcast to other bots.
            Used for fillers and greetings that shouldn't trigger responses.
        is_filler: Whether this is a filler/reactive response (not a real answer).
    """
    text: str
    voice: VoiceConfig | None = None
    speed: float = 1.0
    skip_broadcast: bool = False
    is_filler: bool = False


@runtime_checkable
class Brain(Protocol):
    """Pluggable brain -- the only thing that differs between bot types.

    The Brain receives a complete Utterance (after VAD + ASR) and decides
    how to respond. It can return a BotResponse to speak, or None to stay
    silent.

    Brain implementations own:
    - Decision logic (should we respond?)
    - Response generation (what to say)
    - Any state specific to the brain type (conversation history, personality, etc.)

    Brain implementations do NOT own:
    - Mumble I/O
    - Audio processing (VAD, buffering)
    - ASR transcription
    - TTS synthesis and playback
    - Echo avoidance
    """

    def process(self, utterance: Utterance) -> BotResponse | None:
        """Given a complete utterance, decide how to respond.

        This is the core method that each brain type implements differently.

        Args:
            utterance: Complete utterance with text, audio, and metadata.

        Returns:
            BotResponse to speak, or None to stay silent.
        """
        ...

    def on_bot_utterance(self, speaker_name: str, text: str) -> BotResponse | None:
        """Handle an utterance from another bot (for bot-to-bot communication).

        Called when another bot in the channel speaks. The brain can decide
        whether to respond based on its configuration.

        Default behavior is to return None (ignore other bots).

        Args:
            speaker_name: Name of the bot that spoke.
            text: What the bot said.

        Returns:
            BotResponse to speak, or None to stay silent.
        """
        ...

    def on_text_message(self, sender: str, text: str) -> BotResponse | None:
        """Handle a text chat message.

        Args:
            sender: Name of the message sender.
            text: The message text.

        Returns:
            BotResponse to speak, or None to stay silent.
        """
        ...


class NullBrain:
    """Brain that never responds -- transcribe-only monitoring mode."""

    def process(self, utterance: Utterance) -> BotResponse | None:
        return None

    def on_bot_utterance(self, speaker_name: str, text: str) -> BotResponse | None:
        return None

    def on_text_message(self, sender: str, text: str) -> BotResponse | None:
        return None
