"""Mumble Voice Bot - LLM-powered voice assistant for Mumble."""

# Extracted modules (Phase 1 refactor)
from mumble_voice_bot.audio import pcm_rms, prepare_for_stt, resample_48k_to_16k
from mumble_voice_bot.config import BotConfig, ModelsConfig, load_config
from mumble_voice_bot.coordination import SharedBotServices

# Brain protocol (Phase 1 refactor)
from mumble_voice_bot.interfaces.brain import BotResponse, Brain, NullBrain, Utterance, VoiceConfig
from mumble_voice_bot.interfaces.llm import LLMProvider, LLMResponse
from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.interfaces.tts import TTSProvider, TTSResult, TTSVoice

# Latency optimization components
from mumble_voice_bot.latency import LatencyLogger, LatencyTracker, TurnLatency
from mumble_voice_bot.logging_config import BotLogger, get_logger, setup_logging

# Performance improvements (Phase 1-3 from docs/perf.md)
from mumble_voice_bot.perf import (
    AdaptivePacer,
    AudioQueueItem,
    BoundedTTSQueue,
    ChunkedTTSProducer,
    DropPolicy,
    LatencyReporter,
    RollingLatencyTracker,
    TTSPlaybackWorker,
    TTSQueueItem,
    TTSSynthesisWorker,
    TurnIdCoordinator,
)
from mumble_voice_bot.phrase_chunker import PhraseChunker, SentenceChunker
from mumble_voice_bot.pipeline import PipelineConfig, VoicePipeline
from mumble_voice_bot.providers.openai_llm import OpenAIChatLLM
from mumble_voice_bot.providers.wyoming_stt import WyomingSTT, WyomingSTTSync
from mumble_voice_bot.providers.wyoming_tts import WyomingTTS, WyomingTTSSync
from mumble_voice_bot.text_processing import is_question, pad_tts_text, sanitize_for_tts, split_into_sentences
from mumble_voice_bot.transcript_stabilizer import StreamingTranscriptBuffer, TranscriptStabilizer
from mumble_voice_bot.turn_controller import TurnController, TurnState
from mumble_voice_bot.utils import get_best_device, strip_html

__all__ = [
    # Interfaces
    "LLMProvider",
    "LLMResponse",
    "STTProvider",
    "STTResult",
    "TTSProvider",
    "TTSResult",
    "TTSVoice",
    # Brain protocol (Phase 1 refactor)
    "Brain",
    "Utterance",
    "BotResponse",
    "VoiceConfig",
    "NullBrain",
    # Extracted modules (Phase 1 refactor)
    "pcm_rms",
    "resample_48k_to_16k",
    "prepare_for_stt",
    "split_into_sentences",
    "pad_tts_text",
    "sanitize_for_tts",
    "is_question",
    "get_best_device",
    "strip_html",
    "SharedBotServices",
    # Providers
    "OpenAIChatLLM",
    "WyomingSTT",
    "WyomingSTTSync",
    "WyomingTTS",
    "WyomingTTSSync",
    # Config
    "BotConfig",
    "ModelsConfig",
    "load_config",
    # Pipeline
    "VoicePipeline",
    "PipelineConfig",
    # Latency optimization
    "TurnLatency",
    "LatencyTracker",
    "LatencyLogger",
    "TurnController",
    "TurnState",
    "TranscriptStabilizer",
    "StreamingTranscriptBuffer",
    "PhraseChunker",
    "SentenceChunker",
    # Performance improvements (docs/perf.md)
    "TurnIdCoordinator",
    "BoundedTTSQueue",
    "DropPolicy",
    "RollingLatencyTracker",
    "LatencyReporter",
    "TTSQueueItem",
    "ChunkedTTSProducer",
    "AudioQueueItem",
    "TTSSynthesisWorker",
    "TTSPlaybackWorker",
    "AdaptivePacer",
    # Logging
    "setup_logging",
    "get_logger",
    "BotLogger",
]
