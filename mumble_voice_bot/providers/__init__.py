"""LLM and voice provider implementations."""

from mumble_voice_bot.providers.openai_llm import OpenAIChatLLM
from mumble_voice_bot.providers.wyoming_stt import WyomingSTT, WyomingSTTSync
from mumble_voice_bot.providers.wyoming_tts import WyomingTTS, WyomingTTSSync

# Streaming ASR providers (optional dependencies)
try:
    from mumble_voice_bot.providers.sherpa_nemotron import SherpaNemotronASR, SherpaNemotronConfig
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    SherpaNemotronASR = None
    SherpaNemotronConfig = None

try:
    from mumble_voice_bot.providers.nemotron_stt import NemotronStreamingASR, NemotronConfig
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    NemotronStreamingASR = None
    NemotronConfig = None

__all__ = [
    "OpenAIChatLLM",
    "WyomingSTT",
    "WyomingSTTSync",
    "WyomingTTS",
    "WyomingTTSSync",
    # Streaming ASR (may be None if dependencies not installed)
    "SherpaNemotronASR",
    "SherpaNemotronConfig",
    "NemotronStreamingASR",
    "NemotronConfig",
    "SHERPA_AVAILABLE",
    "NEMO_AVAILABLE",
]
