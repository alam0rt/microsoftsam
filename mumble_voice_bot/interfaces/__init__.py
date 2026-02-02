"""Interfaces for pluggable components."""

from mumble_voice_bot.interfaces.llm import LLMProvider, LLMResponse
from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.interfaces.tts import TTSProvider, TTSResult, TTSVoice

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "STTProvider",
    "STTResult",
    "TTSProvider",
    "TTSResult",
    "TTSVoice",
]
