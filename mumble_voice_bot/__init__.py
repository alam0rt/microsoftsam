"""Mumble Voice Bot - LLM-powered voice assistant for Mumble."""

from mumble_voice_bot.interfaces.llm import LLMProvider, LLMResponse
from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.interfaces.tts import TTSProvider, TTSResult, TTSVoice
from mumble_voice_bot.providers.openai_llm import OpenAIChatLLM
from mumble_voice_bot.providers.wyoming_stt import WyomingSTT, WyomingSTTSync
from mumble_voice_bot.providers.wyoming_tts import WyomingTTS, WyomingTTSSync
from mumble_voice_bot.config import BotConfig, load_config
from mumble_voice_bot.pipeline import VoicePipeline, PipelineConfig

__all__ = [
    # Interfaces
    "LLMProvider",
    "LLMResponse",
    "STTProvider",
    "STTResult",
    "TTSProvider",
    "TTSResult",
    "TTSVoice",
    # Providers
    "OpenAIChatLLM",
    "WyomingSTT",
    "WyomingSTTSync",
    "WyomingTTS",
    "WyomingTTSSync",
    # Config
    "BotConfig",
    "load_config",
    # Pipeline
    "VoicePipeline",
    "PipelineConfig",
]
