#!/usr/bin/env python3
"""
Mumble Voice Bot - LLM-powered voice assistant for Mumble.

Listens to voice in a Mumble channel, transcribes with Whisper, generates
responses with an LLM, and speaks back using LuxTTS voice cloning.

Usage:
    # Basic usage with default Ollama endpoint
    python mumble_tts_bot.py --host mumble.example.com --reference voice.wav

    # With custom LLM endpoint
    python mumble_tts_bot.py --host mumble.example.com --reference voice.wav \\
        --llm-endpoint http://localhost:8000/v1/chat/completions \\
        --llm-model Qwen/Qwen3-32B

    # Debug mode to tune VAD threshold
    python mumble_tts_bot.py --host mumble.example.com --reference voice.wav --debug-rms
"""
import argparse
import asyncio
import os
import queue
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Generator, List

# Add vendor paths for pymumble and LuxTTS
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "botamusique"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LuxTTS"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LinaCodec", "src"))

import numpy as np
import pymumble_py3 as pymumble
import torch
from huggingface_hub import snapshot_download
from pymumble_py3.constants import PYMUMBLE_CLBK_SOUNDRECEIVED
from scipy import signal
from zipvoice.luxvoice import LuxTTS

# Import logging first
from mumble_voice_bot.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Import LLM components
try:
    from mumble_voice_bot.config import load_config
    from mumble_voice_bot.providers.openai_llm import OpenAIChatLLM
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    logger.warning(f"LLM modules not available: {e}")

# Import Wyoming STT provider
try:
    from mumble_voice_bot.providers.wyoming_stt import WyomingSTTSync
    WYOMING_AVAILABLE = True
except ImportError:
    WYOMING_AVAILABLE = False
    logger.warning("Wyoming STT not available. Install with: pip install wyoming")

# Import streaming ASR providers
try:
    from mumble_voice_bot.providers.sherpa_nemotron import SherpaNemotronASR, SherpaNemotronConfig
    SHERPA_NEMOTRON_AVAILABLE = True
except ImportError:
    SHERPA_NEMOTRON_AVAILABLE = False
    SherpaNemotronASR = None
    SherpaNemotronConfig = None

try:
    from mumble_voice_bot.providers.nemotron_stt import NemotronConfig, NemotronStreamingASR
    NEMOTRON_NEMO_AVAILABLE = True
except ImportError:
    NEMOTRON_NEMO_AVAILABLE = False
    NemotronStreamingASR = None
    NemotronConfig = None

# Import latency optimization components
try:
    from mumble_voice_bot.latency import LatencyLogger, LatencyTracker
    from mumble_voice_bot.turn_controller import TurnController
    LATENCY_TRACKING_AVAILABLE = True
except ImportError as e:
    LATENCY_TRACKING_AVAILABLE = False
    logger.warning(f"Latency tracking not available: {e}")

# Import event system
try:
    from mumble_voice_bot.handlers import ConnectionHandler, PresenceHandler, TextCommandHandler
    from mumble_voice_bot.providers.mumble_events import EventDispatcher
    EVENT_SYSTEM_AVAILABLE = True
except ImportError as e:
    EVENT_SYSTEM_AVAILABLE = False
    logger.warning(f"Event system not available: {e}")


# =============================================================================
# StreamingLuxTTS - Subclass that adds streaming and fixes upstream issues
# =============================================================================

def split_into_sentences(text: str, max_chars: int = 120) -> List[str]:
    """
    Split text into speakable chunks optimized for streaming TTS.

    Strategy:
    - Split on sentence boundaries first
    - Split long sentences on clause boundaries
    - Ensure minimum chunk size for natural speech

    Args:
        text: The text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks suitable for TTS.
    """
    MIN_CHUNK = 20  # Don't create tiny chunks

    # First pass: split on sentence endings
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())

    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Split long sentences on clause boundaries
            clause_pattern = r'(?<=[,;:])\s+'
            clauses = re.split(clause_pattern, sentence)

            # Merge tiny clauses
            current = ""
            for clause in clauses:
                if len(current) + len(clause) < MIN_CHUNK:
                    current += (" " if current else "") + clause
                else:
                    if current:
                        chunks.append(current)
                    current = clause
            if current:
                chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def _pad_tts_text(text: str, min_chars: int = 80, min_words: int = 12) -> str:
    """Pad text to avoid very short TTS inputs that can crash the vocoder."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    filler = (
        "Let me think for a second and give you a clear answer."
    )

    while len(cleaned) < min_chars or len(cleaned.split()) < min_words:
        cleaned = f"{cleaned} {filler}"

    return cleaned


class StreamingLuxTTS(LuxTTS):
    """Extended LuxTTS with streaming support and bug fixes."""

    def __init__(self, model_path='YatharthS/LuxTTS', device='cuda', threads=4):
        super().__init__(model_path=model_path, device=device, threads=threads)
        self._patch_transcriber_for_english()

    def _patch_transcriber_for_english(self):
        """Force English language detection and enable timestamps for long audio."""
        original_transcriber = self.transcriber

        def english_transcriber(audio, **kwargs):
            result = original_transcriber(
                audio,
                generate_kwargs={"language": "en", "task": "transcribe"},
                return_timestamps=True,  # Required for audio > 30s
                **kwargs
            )
            return result

        self.transcriber = english_transcriber
        print("[TTS] Patched transcriber for English-only mode")

    def generate_speech_streaming(
        self,
        text: str,
        encode_dict: dict,
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        """Stream speech generation by splitting text into sentences."""
        text = _pad_tts_text(text)
        if not text:
            return

        sentences = split_into_sentences(text)

        if len(sentences) <= 1:
            # Pad very short text to avoid vocoder kernel size issues
            # Vocoder kernel needs 7+ frames, requiring substantial text
            padded_text = _pad_tts_text(text)
            if not padded_text:
                return
            wav = self._generate_speech_safe(
                padded_text,
                encode_dict,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth,
            )
            if wav is not None:
                yield wav
            return

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Pad very short sentences to avoid vocoder kernel size issues
            # The vocoder needs at least 7 frames, which requires ~20+ chars
            sentence = _pad_tts_text(sentence)
            if not sentence:
                continue
            wav = self._generate_speech_safe(
                sentence,
                encode_dict,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth,
            )
            if wav is not None:
                yield wav

    def _generate_speech_safe(
        self,
        text: str,
        encode_dict: dict,
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False,
    ) -> torch.Tensor | None:
        try:
            return self.generate_speech(
                text,
                encode_dict,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth,
            )
        except RuntimeError as e:
            message = str(e)
            if "Kernel size" in message or "kernel size" in message or "padded input size" in message:
                padded = _pad_tts_text(text, min_chars=160, min_words=24)
                if padded and padded != text:
                    try:
                        return self.generate_speech(
                            padded,
                            encode_dict,
                            num_steps=num_steps,
                            guidance_scale=guidance_scale,
                            t_shift=t_shift,
                            speed=speed,
                            return_smooth=return_smooth,
                        )
                    except Exception as retry_error:
                        print(f"[TTS] Retry failed after padding: {retry_error}")
                        import traceback
                        traceback.print_exc()
                        return None
            print(f"[TTS] Error generating speech for '{text[:50]}': {e}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"[TTS] Error generating speech for '{text[:50]}': {e}")
            import traceback
            traceback.print_exc()
            return None


# =============================================================================
# Helper functions
# =============================================================================

def pcm_rms(pcm_bytes: bytes) -> int:
    """Calculate RMS of 16-bit PCM audio."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(audio) == 0:
        return 0
    return int(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)


def get_best_device() -> str:
    """Auto-detect the best available compute device."""
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA available: {device_name}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[Device] MPS (Apple Silicon) available")
            return 'mps'
        else:
            print("[Device] Using CPU")
            return 'cpu'
    except ImportError:
        return 'cpu'


def ensure_models_downloaded(device: str = 'cuda') -> None:
    """
    Pre-download all required models before starting the bot.

    This ensures models are cached locally before connecting to Mumble,
    preventing long delays during the first voice interaction.

    Downloads:
    - LuxTTS model (YatharthS/LuxTTS)
    - Whisper model (openai/whisper-base for GPU, whisper-tiny for CPU)
    """
    from transformers import pipeline as hf_pipeline

    print("=" * 60)
    print("[Models] Ensuring all required models are downloaded...")
    print("=" * 60)

    # Download LuxTTS model
    print("[Models] Checking LuxTTS model (YatharthS/LuxTTS)...")
    try:
        model_path = snapshot_download("YatharthS/LuxTTS")
        print(f"[Models] ✓ LuxTTS ready at: {model_path}")
    except Exception as e:
        print(f"[Models] ✗ Failed to download LuxTTS: {e}")
        raise

    # Download Whisper model (used for transcription in TTS pipeline)
    whisper_model = "openai/whisper-base" if device != 'cpu' else "openai/whisper-tiny"
    print(f"[Models] Checking Whisper model ({whisper_model})...")
    try:
        # This will download if not cached, or load from cache
        _ = hf_pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            device='cpu'  # Load on CPU just to verify download, actual device set later
        )
        print(f"[Models] ✓ Whisper ready: {whisper_model}")
    except Exception as e:
        print(f"[Models] ✗ Failed to download Whisper: {e}")
        raise

    print("=" * 60)
    print("[Models] All models ready!")
    print("=" * 60)


# =============================================================================
# MumbleVoiceBot - Main bot class
# =============================================================================

class MumbleVoiceBot:
    """A Mumble bot that listens, thinks with an LLM, and responds with TTS."""

    def __init__(
        self,
        host: str,
        user: str,
        port: int = 64738,
        password: str = '',
        channel: str = None,
        reference_audio: str = 'reference.wav',
        device: str = 'cpu',
        num_steps: int = 4,
        asr_threshold: int = 2000,
        debug_rms: bool = False,
        voices_dir: str = 'voices',
        # LLM configuration
        llm_endpoint: str = None,
        llm_model: str = None,
        llm_api_key: str = None,
        llm_system_prompt: str = None,
        personality: str = None,
        config_file: str = None,
        # STT configuration
        stt_provider: str = "local",  # local, wyoming, sherpa_nemotron, nemotron_nemo
        wyoming_stt_host: str = None,
        wyoming_stt_port: int = 10300,
        # Sherpa Nemotron settings
        sherpa_encoder: str = None,
        sherpa_decoder: str = None,
        sherpa_joiner: str = None,
        sherpa_tokens: str = None,
        sherpa_provider: str = "cuda",
        # NeMo Nemotron settings
        nemotron_model: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        nemotron_chunk_ms: int = 160,
        nemotron_device: str = "cuda",
        # Staleness configuration
        max_response_staleness: float = 5.0,
        # Soul configuration
        soul_config=None,  # SoulConfig object with themed fallbacks
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.channel = channel
        self.device = device
        self.num_steps = num_steps
        self.voices_dir = voices_dir
        self.soul_config = soul_config  # Soul-specific themed responses

        # VAD settings
        self.asr_threshold = asr_threshold
        self.debug_rms = debug_rms
        self._max_rms = 0

        # ASR state per user
        self.audio_buffers = {}  # user_id -> list of PCM bytes
        self.speech_active_until = {}  # user_id -> timestamp
        self.speech_start_time = {}  # user_id -> timestamp
        self.speech_hold_duration = 0.6  # seconds of silence before processing
        self.min_speech_duration = 0.3  # minimum seconds to transcribe
        self.max_speech_duration = 5.0  # force processing after 5 seconds (keeps Whisper fast)

        # Response staleness settings
        self.max_response_staleness = max_response_staleness  # skip responses older than this (seconds)

        # Pending transcriptions (for accumulating long utterances)
        self.pending_text = {}  # user_id -> accumulated text
        self.pending_text_time = {}  # user_id -> timestamp of last text
        self.pending_text_timeout = 1.5  # seconds to wait for more speech before responding

        # Conversation state per user
        self.conversation_history = {}  # user_id -> list of messages
        self.conversation_timeout = 300.0  # 5 minutes
        self.last_conversation_time = {}  # user_id -> timestamp

        # State flags
        self._speaking = threading.Event()
        self._shutdown = threading.Event()

        # Threading
        self._asr_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ASR")  # Allow 2 workers
        self._tts_queue = queue.Queue()
        self._tts_lock = threading.Lock()

        # Latency tracking and turn control
        if LATENCY_TRACKING_AVAILABLE:
            self.latency_logger = LatencyLogger()
            self.turn_controller = TurnController()
            # Register barge-in callback to clear audio output
            self.turn_controller.on_barge_in(self._handle_barge_in)
            print("[Latency] Tracking enabled - logging to latency.jsonl")
        else:
            self.latency_logger = None
            self.turn_controller = None

        # Current latency tracker for in-progress turn
        self._current_tracker: LatencyTracker = None

        # Start TTS worker
        self._tts_worker_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="TTS-Worker"
        )
        self._tts_worker_thread.start()

        # Initialize TTS
        print(f"[TTS] Loading model on {device}...")
        self.tts = StreamingLuxTTS('YatharthS/LuxTTS', device=device, threads=2)

        # Load voice
        os.makedirs(self.voices_dir, exist_ok=True)
        self._load_reference_voice(reference_audio)

        # Initialize STT provider
        self.stt_provider = stt_provider
        self.wyoming_stt = None
        self.streaming_stt = None

        if stt_provider == "wyoming":
            if wyoming_stt_host and WYOMING_AVAILABLE:
                print(f"[STT] Using Wyoming STT at {wyoming_stt_host}:{wyoming_stt_port}")
                self.wyoming_stt = WyomingSTTSync(host=wyoming_stt_host, port=wyoming_stt_port)
            elif not WYOMING_AVAILABLE:
                print("[Warning] Wyoming STT requested but wyoming package not installed")
                print("[STT] Falling back to local Whisper")
                self.stt_provider = "local"
            else:
                print("[Warning] Wyoming STT provider selected but no host configured")
                print("[STT] Falling back to local Whisper")
                self.stt_provider = "local"

        elif stt_provider == "sherpa_nemotron":
            if SHERPA_NEMOTRON_AVAILABLE:
                print(f"[STT] Using Sherpa Nemotron (provider={sherpa_provider})")
                config = SherpaNemotronConfig(
                    encoder_path=sherpa_encoder or "nemotron-encoder.onnx",
                    decoder_path=sherpa_decoder or "nemotron-decoder.onnx",
                    joiner_path=sherpa_joiner or "nemotron-joiner.onnx",
                    tokens_path=sherpa_tokens or "tokens.txt",
                    provider=sherpa_provider,
                )
                self.streaming_stt = SherpaNemotronASR(config)
                # Try to initialize
                if not self.streaming_stt.initialize():
                    print("[Warning] Failed to initialize Sherpa Nemotron")
                    print("[STT] Falling back to local Whisper")
                    self.streaming_stt = None
                    self.stt_provider = "local"
            else:
                print("[Warning] Sherpa Nemotron requested but sherpa-onnx not installed")
                print("[STT] Falling back to local Whisper")
                self.stt_provider = "local"

        elif stt_provider == "nemotron_nemo":
            if NEMOTRON_NEMO_AVAILABLE:
                print(f"[STT] Using NeMo Nemotron ({nemotron_model}, chunk={nemotron_chunk_ms}ms)")
                config = NemotronConfig(
                    model_name=nemotron_model,
                    chunk_size_ms=nemotron_chunk_ms,
                    device=nemotron_device,
                )
                self.streaming_stt = NemotronStreamingASR(config)
                # Note: NeMo model loads lazily on first use
            else:
                print("[Warning] NeMo Nemotron requested but nemo_toolkit not installed")
                print("[STT] Falling back to local Whisper")
                self.stt_provider = "local"

        elif stt_provider == "local":
            print("[STT] Using local Whisper via LuxTTS")
        else:
            print(f"[Warning] Unknown STT provider: {stt_provider}")
            print("[STT] Falling back to local Whisper")
            self.stt_provider = "local"

        # Initialize LLM
        self.llm = None
        if LLM_AVAILABLE:
            self._init_llm(
                endpoint=llm_endpoint,
                model=llm_model,
                api_key=llm_api_key,
                system_prompt=llm_system_prompt,
                personality=personality,
                config_file=config_file,
            )
        else:
            print("[Warning] LLM not available - bot will only transcribe, not respond")

        # Initialize Mumble
        print(f"[Mumble] Connecting to {host}:{port} as '{user}'...")
        self.mumble = pymumble.Mumble(
            host=host,
            user=user,
            port=port,
            password=password,
            reconnect=True,
        )

        # Enable audio reception
        self.mumble.set_receive_sound(True)
        print(f"[VAD] Listening for voice (threshold: {asr_threshold})")
        if self.debug_rms:
            print("[VAD] Debug mode: showing audio levels")

        # Initialize event system for user presence and text messages
        self.event_dispatcher = None
        self.presence_handler = None
        if EVENT_SYSTEM_AVAILABLE:
            self._init_event_system()
        else:
            # Fallback: register callbacks directly (legacy mode)
            print("[Events] Event system not available, using legacy callbacks")
            self._init_legacy_callbacks()

        # Always register sound callback directly (performance-critical)
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_SOUNDRECEIVED,
            self.on_sound_received
        )

    def _init_event_system(self):
        """Initialize the event dispatcher and handlers."""
        # Create event loop for async handlers
        try:
            self._event_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

        # Create dispatcher (don't enable sound events - we handle those directly)
        self.event_dispatcher = EventDispatcher(
            self.mumble,
            self._event_loop,
            enable_sound_events=False,
        )

        # Create and register handlers
        self.presence_handler = PresenceHandler(self, greet_on_join=True)
        self.text_handler = TextCommandHandler(self)
        self.connection_handler = ConnectionHandler(self)

        self.event_dispatcher.register_handler(self.presence_handler)
        self.event_dispatcher.register_handler(self.text_handler)
        self.event_dispatcher.register_handler(self.connection_handler)

        # Start the dispatcher
        self.event_dispatcher.start()
        print("[Events] Event system initialized with handlers: Presence, TextCommand, Connection")

    def _init_legacy_callbacks(self):
        """Initialize legacy callbacks when event system is unavailable."""
        from pymumble_py3.constants import (
            PYMUMBLE_CLBK_TEXTMESSAGERECEIVED,
            PYMUMBLE_CLBK_USERCREATED,
            PYMUMBLE_CLBK_USERREMOVED,
            PYMUMBLE_CLBK_USERUPDATED,
        )
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_TEXTMESSAGERECEIVED,
            self.on_message
        )
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_USERCREATED,
            self.on_user_joined
        )
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_USERUPDATED,
            self.on_user_updated
        )
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_USERREMOVED,
            self.on_user_left
        )

    def _load_reference_voice(self, reference_audio: str):
        """Load or encode the reference voice for TTS."""
        reference_name = os.path.splitext(os.path.basename(reference_audio))[0]
        saved_voice_path = os.path.join(self.voices_dir, f"{reference_name}.pt")

        if os.path.exists(saved_voice_path):
            print(f"[Voice] Loading cached voice: {saved_voice_path}")
            self.voice_prompt = torch.load(
                saved_voice_path, weights_only=False, map_location=self.device
            )
            self.voice_prompt = self._ensure_voice_on_device(self.voice_prompt)
        else:
            print(f"[Voice] Encoding reference: {reference_audio}")
            self.voice_prompt = self.tts.encode_prompt(reference_audio, rms=0.01)
            # Cache for next time
            torch.save(self.voice_prompt, saved_voice_path)
            print(f"[Voice] Cached as '{reference_name}' for faster startup")

    def _ensure_voice_on_device(self, voice_prompt: dict) -> dict:
        """Ensure all tensors in voice_prompt are on the correct device."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in voice_prompt.items()
        }

    def _init_llm(
        self,
        endpoint: str = None,
        model: str = None,
        api_key: str = None,
        system_prompt: str = None,
        personality: str = None,
        config_file: str = None,
    ):
        """Initialize the LLM provider."""
        # Try config file
        config = None
        if config_file or os.path.exists("config.yaml"):
            try:
                config = load_config(config_file)
                print(f"[LLM] Loaded config from {config_file or 'config.yaml'}")
            except Exception as e:
                print(f"[LLM] Config load failed: {e}")

        # CLI args override config
        final_endpoint = endpoint
        final_model = model
        final_api_key = api_key
        final_system_prompt = system_prompt
        final_timeout = 30.0
        final_max_tokens = None
        final_temperature = None

        if config:
            final_endpoint = final_endpoint or config.llm.endpoint
            final_model = final_model or config.llm.model
            final_api_key = final_api_key or config.llm.api_key
            final_system_prompt = final_system_prompt or config.llm.system_prompt
            final_timeout = config.llm.timeout or final_timeout
            final_max_tokens = config.llm.max_tokens
            final_temperature = config.llm.temperature
            if hasattr(config, 'bot') and config.bot.conversation_timeout:
                self.conversation_timeout = config.bot.conversation_timeout

        # Check environment variables for API key if not set
        if not final_api_key:
            final_api_key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('LLM_API_KEY')

        # Defaults
        final_endpoint = final_endpoint or "http://localhost:11434/v1/chat/completions"
        final_model = final_model or "llama3.2:3b"
        final_system_prompt = final_system_prompt or self._load_system_prompt(personality=personality)

        self.llm = OpenAIChatLLM(
            endpoint=final_endpoint,
            model=final_model,
            api_key=final_api_key,
            system_prompt=final_system_prompt,
            timeout=final_timeout,
            max_tokens=final_max_tokens,
            temperature=final_temperature,
        )
        extra_info = []
        if final_max_tokens:
            extra_info.append(f"max_tokens={final_max_tokens}")
        if final_temperature:
            extra_info.append(f"temp={final_temperature}")
        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
        print(f"[LLM] Initialized: {final_model} @ {final_endpoint}{extra_str}")

    def _load_system_prompt(self, prompt_file: str = None, personality: str = None) -> str:
        """Load system prompt from file, optionally combined with a personality."""
        base_prompt = None

        # Try specified file first
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                print(f"[LLM] Loaded prompt from {prompt_file}")
                base_prompt = f.read()

        # Try default locations
        if not base_prompt:
            default_paths = [
                os.path.join(_THIS_DIR, "prompts", "default.md"),
                os.path.join(_THIS_DIR, "prompts", "default.txt"),
                "prompts/default.md",
                "prompts/default.txt",
            ]

            for path in default_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        print(f"[LLM] Loaded prompt from {path}")
                        base_prompt = f.read()
                        break

        # Fallback to inline prompt
        if not base_prompt:
            print("[LLM] Using built-in default prompt")
            base_prompt = self._get_fallback_prompt()

        # Load personality if specified
        if personality:
            personality_prompt = self._load_personality(personality)
            if personality_prompt:
                base_prompt = base_prompt + "\n\n" + "=" * 40 + "\n\n" + personality_prompt

        return base_prompt

    def _load_personality(self, personality: str) -> str:
        """Load a personality file by name."""
        # Check if it's already a path
        if os.path.exists(personality):
            with open(personality, 'r') as f:
                print(f"[LLM] Loaded personality from {personality}")
                return f.read()

        # Try personalities directory
        personality_paths = [
            os.path.join(_THIS_DIR, "personalities", f"{personality}.md"),
            os.path.join(_THIS_DIR, "personalities", f"{personality}.txt"),
            os.path.join(_THIS_DIR, "personalities", personality),
            f"personalities/{personality}.md",
            f"personalities/{personality}.txt",
        ]

        for path in personality_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    print(f"[LLM] Loaded personality: {personality}")
                    return f.read()

        print(f"[LLM] Warning: Personality '{personality}' not found")
        return None

    def _get_fallback_prompt(self) -> str:
        """Fallback prompt if no file is found."""
        return """You are a casual voice assistant in a Mumble voice channel.

Your responses will be spoken by TTS. Never use emojis, symbols, or formatting.
Keep responses to 1-2 sentences. Use casual language and contractions.
Sound like a friend chatting, not a corporate assistant.
Write numbers and symbols as words: "about 5 dollars" not "$5"."""

    # =========================================================================
    # Context & Awareness
    # =========================================================================

    def _get_time_context(self) -> str:
        """Get current time context for the LLM."""
        now = datetime.now()
        hour = now.hour

        # Time of day
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        # Day info
        day_name = now.strftime("%A")
        time_str = now.strftime("%I:%M %p").lstrip("0")

        return f"It's {time_of_day}, {day_name} at {time_str}."

    def _get_channel_context(self) -> str:
        """Get info about who's in the channel."""
        # Use presence handler if available (more efficient, pre-tracked)
        if self.presence_handler:
            users_in_channel = self.presence_handler.get_users_in_channel()
            if users_in_channel:
                return f"Users in channel: {', '.join(users_in_channel)}."
            return "You're alone in the channel."

        # Fallback to direct Mumble query
        try:
            my_channel = self.mumble.users.myself["channel_id"]
            users_in_channel = [
                u["name"] for u in self.mumble.users.values()
                if u["channel_id"] == my_channel and u["session"] != self.mumble.users.myself_session
            ]
            if users_in_channel:
                return f"Users in channel: {', '.join(users_in_channel)}."
            return "You're alone in the channel."
        except Exception:
            return ""

    def _inject_context(self, text: str, user_name: str = None) -> str:
        """Inject time and context into a message for the LLM."""
        context_parts = [self._get_time_context()]

        # Add what the user said
        if user_name:
            context_parts.append(f"[{user_name} says]: {text}")
        else:
            context_parts.append(text)

        return " ".join(context_parts)

    # =========================================================================
    # Conversation Management
    # =========================================================================

    def _get_history(self, user_id: int) -> list[dict]:
        """Get conversation history for a user, clearing if stale."""
        current_time = time.time()

        if user_id in self.last_conversation_time:
            elapsed = current_time - self.last_conversation_time[user_id]
            if elapsed > self.conversation_timeout:
                self.conversation_history.pop(user_id, None)
                print(f"[Chat] Cleared stale history for user {user_id}")

        self.last_conversation_time[user_id] = current_time

        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        return self.conversation_history[user_id]

    def _add_to_history(self, user_id: int, role: str, content: str, user_name: str = None, include_time: bool = False):
        """Add a message to conversation history."""
        history = self._get_history(user_id)

        # For user messages, add context
        if role == "user":
            parts = []

            # Add time context occasionally (first message or every 5th message)
            if include_time or len(history) == 0 or len(history) % 5 == 0:
                parts.append(f"[{self._get_time_context()}]")

            # Add user name
            if user_name:
                parts.append(f"[{user_name} says]: {content}")
            else:
                parts.append(content)

            content = " ".join(parts)

        history.append({"role": role, "content": content})

        # Keep last 20 messages
        if len(history) > 20:
            self.conversation_history[user_id] = history[-20:]

    async def _generate_response(self, user_id: int, text: str, user_name: str = None) -> str:
        """Generate LLM response."""
        self._add_to_history(user_id, "user", text, user_name)
        history = self._get_history(user_id)

        response = await self.llm.chat(history)

        self._add_to_history(user_id, "assistant", response.content)
        return response.content

    def _generate_response_sync(self, user_id: int, text: str, user_name: str = None) -> str:
        """Generate LLM response synchronously."""
        return self._run_coro_sync(self._generate_response(user_id, text, user_name))

    def _generate_oneoff_response_sync(self, prompt: str) -> str:
        """Generate a one-off LLM response without updating history."""
        if not self.llm:
            return ""
        response = self._run_coro_sync(
            self.llm.chat([{"role": "user", "content": prompt}])
        )
        return response.content

    def _run_coro_sync(self, coroutine):
        """Run an async coroutine from sync code safely."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coroutine, loop)
                return future.result(timeout=35.0)
            return loop.run_until_complete(coroutine)
        except RuntimeError:
            return asyncio.run(coroutine)

    # =========================================================================
    # Barge-in Handling
    # =========================================================================

    def _handle_barge_in(self):
        """Handle barge-in: stop current TTS playback."""
        try:
            # Clear the Mumble output buffer
            self.mumble.sound_output.clear()
            print("[Barge-in] Cleared audio output buffer")
        except Exception as e:
            print(f"[Barge-in] Error clearing buffer: {e}")

    # =========================================================================
    # Audio Processing
    # =========================================================================

    def on_sound_received(self, user, sound_chunk):
        """Handle incoming audio from users."""
        user_id = user['session']
        user_name = user.get('name', 'Unknown')

        # Ignore our own audio to avoid feedback loops
        if user_id == self.mumble.users.myself_session:
            return

        rms = pcm_rms(sound_chunk.pcm)
        self._max_rms = max(rms, self._max_rms)

        # Debug display
        if self.debug_rms:
            bar_width = min(rms // 100, 50)
            threshold_pos = min(self.asr_threshold // 100, 50)
            bar = '-' * threshold_pos + '+' * max(0, bar_width - threshold_pos) if rms >= self.asr_threshold else '-' * bar_width
            print(f'\r[{user_name:12}] RMS: {rms:5d} / {self._max_rms:5d}  |{bar:<50}|', end='', flush=True)

        # Barge-in detection: if user speaks while bot is speaking
        if self.turn_controller and self._speaking.is_set() and rms > self.asr_threshold:
            if self.turn_controller.request_barge_in():
                print(f"\n[Barge-in] User {user_name} interrupted bot")

        # Initialize state for new users
        if user_id not in self.audio_buffers:
            self.audio_buffers[user_id] = []
            self.speech_active_until[user_id] = 0
            self.speech_start_time[user_id] = 0

        current_time = time.time()

        # HARD LIMIT: Always check buffer size first, regardless of speech state
        buffer_duration = self._get_buffer_duration(user_id)
        if buffer_duration >= self.max_speech_duration:
            if self.debug_rms:
                print()
            print(f"[ASR] Processing {buffer_duration:.1f}s chunk from {user_name}...")
            audio_data = list(self.audio_buffers[user_id])
            self.audio_buffers[user_id] = []
            # Pass current tracker (may be None if speech just started)
            tracker = self._current_tracker
            self._asr_executor.submit(self._process_speech, user.copy(), user_id, audio_data, True, tracker)
            return  # Don't process this chunk further

        # Speech detection
        if rms > self.asr_threshold:
            if not self.audio_buffers[user_id]:
                self.speech_start_time[user_id] = current_time
                # Start new latency tracker
                if LATENCY_TRACKING_AVAILABLE and self.latency_logger:
                    self._current_tracker = LatencyTracker(str(user_id), self.latency_logger)
                    self._current_tracker.vad_start()
                # Update turn controller
                if self.turn_controller:
                    self.turn_controller.start_listening(str(user_id))

            self.audio_buffers[user_id].append(sound_chunk.pcm)
            self.speech_active_until[user_id] = current_time + self.speech_hold_duration
        else:
            # Below threshold
            if current_time < self.speech_active_until[user_id]:
                # Still in hold period, keep buffering
                self.audio_buffers[user_id].append(sound_chunk.pcm)
            elif self.audio_buffers[user_id]:
                # Speech ended - process and respond
                if self.debug_rms:
                    print()

                # Mark VAD end in latency tracker
                if self._current_tracker:
                    self._current_tracker.vad_end()

                # Update turn controller
                if self.turn_controller:
                    self.turn_controller.start_processing()

                audio_data = list(self.audio_buffers[user_id])
                self.audio_buffers[user_id] = []
                # Pass the current tracker to the processing function
                tracker = self._current_tracker
                self._current_tracker = None
                self._asr_executor.submit(self._process_speech, user.copy(), user_id, audio_data, False, tracker)

    def _get_buffer_duration(self, user_id) -> float:
        """Calculate buffered audio duration in seconds."""
        if user_id not in self.audio_buffers:
            return 0
        total_bytes = sum(len(chunk) for chunk in self.audio_buffers[user_id])
        return total_bytes / (48000 * 2)  # 48kHz, 16-bit mono

    def _process_speech(self, user: dict, user_id: int, audio_chunks: list, is_continuation: bool = False, tracker: 'LatencyTracker' = None):
        """Process speech: transcribe and accumulate, respond on pause."""
        user_name = user.get('name', 'Unknown')

        # Concatenate audio
        pcm_data = b''.join(audio_chunks)
        buffer_duration = len(pcm_data) / (48000 * 2)

        if buffer_duration < self.min_speech_duration:
            # Too short - but check if we have pending text to respond to
            self._maybe_respond(user_id, user_name, tracker=tracker)
            return

        # Convert and resample
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms < 0.02:
            # Too quiet - but check pending
            self._maybe_respond(user_id, user_name, tracker=tracker)
            return

        # Resample 48kHz -> 16kHz for STT
        audio_16k = signal.resample_poly(audio_float, up=1, down=3).astype(np.float32)

        # Normalize for STT
        rms_16k = np.sqrt(np.mean(audio_16k ** 2))
        if rms_16k > 0.001:
            audio_16k = audio_16k * (0.1 / rms_16k)
            audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)

        # Transcribe
        logger.debug(f"ASR transcribing {buffer_duration:.1f}s", extra={"user": user_name, "duration_ms": buffer_duration * 1000})
        start_time = time.time()

        # Mark ASR start in tracker
        if tracker:
            tracker.asr_start()

        try:
            # Convert float32 to int16 PCM bytes for external STT providers
            audio_16k_int16 = (audio_16k * 32767).astype(np.int16)
            pcm_16k_bytes = audio_16k_int16.tobytes()

            # Select STT provider
            if self.stt_provider == "wyoming" and self.wyoming_stt:
                stt_result = self.wyoming_stt.transcribe(
                    audio_data=pcm_16k_bytes,
                    sample_rate=16000,
                    sample_width=2,
                    channels=1,
                    language="en",
                )
                text = stt_result.text.strip()

            elif self.stt_provider in ("sherpa_nemotron", "nemotron_nemo") and self.streaming_stt:
                # Use streaming STT provider (synchronous transcription)
                stt_result = asyncio.run(self.streaming_stt.transcribe(
                    audio_data=pcm_16k_bytes,
                    sample_rate=16000,
                    sample_width=2,
                    channels=1,
                    language="en",
                ))
                text = stt_result.text.strip()

            else:
                # Use local Whisper via LuxTTS transcriber
                result = self.tts.transcriber(audio_16k)
                text = result.get('text', '').strip()

            transcribe_time = time.time() - start_time

            # Mark ASR complete in tracker
            if tracker:
                tracker.asr_final(text)

            if not text or len(text) < 2:
                self._maybe_respond(user_id, user_name, tracker=tracker)
                return

            logger.asr(user_name, text, buffer_duration * 1000, transcribe_time * 1000)

            # Accumulate text
            current_time = time.time()
            if user_id in self.pending_text:
                self.pending_text[user_id] += " " + text
            else:
                self.pending_text[user_id] = text
            self.pending_text_time[user_id] = current_time

            # If this was triggered by silence (not max duration), respond now
            if not is_continuation:
                self._maybe_respond(user_id, user_name, force=True, tracker=tracker)

        except Exception as e:
            print(f"[Error] Processing failed: {e}")

    def _maybe_respond(self, user_id: int, user_name: str, force: bool = False, tracker: 'LatencyTracker' = None):
        """Respond if we have pending text and enough time has passed."""
        if user_id not in self.pending_text:
            return

        current_time = time.time()
        time_since_last = current_time - self.pending_text_time.get(user_id, 0)

        # Respond if forced or if enough time has passed
        if force or time_since_last >= self.pending_text_timeout:
            text = self.pending_text.pop(user_id, "")
            speech_end_time = self.pending_text_time.pop(user_id, current_time)

            if text and self.llm:
                # Track when user finished speaking for staleness detection
                pipeline_start = time.time()

                # Mark LLM start in tracker
                if tracker:
                    tracker.llm_start()

                logger.debug(f'LLM generating response to: "{text[:100]}"')
                llm_start = time.time()

                try:
                    response = self._generate_response_sync(user_id, text, user_name)
                    llm_time = time.time() - llm_start

                    # Mark LLM complete in tracker
                    if tracker:
                        tracker.llm_complete(response)

                    # Check for staleness - if user said something new, abort
                    if user_id in self.pending_text:
                        logger.info("Pipeline abort: user spoke again", extra={"latency_ms": llm_time * 1000})
                        return

                    # Check if cancelled by barge-in
                    if self.turn_controller and self.turn_controller.is_cancelled():
                        logger.info("Pipeline abort: barge-in")
                        return

                    # Check total latency - if too slow, warn
                    total_latency = time.time() - speech_end_time
                    if total_latency > 3.0:
                        logger.warning(f"High latency: {total_latency:.1f}s since user stopped", extra={"latency_ms": total_latency * 1000})

                    logger.llm(len(text), len(response), llm_time * 1000)

                    # Queue TTS with timing metadata and tracker
                    self._tts_queue.put((response, self.voice_prompt, pipeline_start, user_id, tracker))

                except Exception as e:
                    logger.error(f"LLM error: {e}", exc_info=True)

    # =========================================================================
    # TTS
    # =========================================================================

    def _tts_worker(self):
        """Background worker for TTS."""
        logger.info("TTS worker started")
        while not self._shutdown.is_set():
            try:
                item = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Unpack with optional timing metadata and tracker
            tracker = None
            if len(item) == 5:
                text, voice_prompt, pipeline_start, user_id, tracker = item
            elif len(item) == 4:
                text, voice_prompt, pipeline_start, user_id = item
            else:
                text, voice_prompt = item
                pipeline_start = None
                user_id = None

            try:
                # Check staleness before TTS - if user spoke again, skip
                if user_id is not None and user_id in self.pending_text:
                    print("[TTS] Skipping stale response - user spoke again")
                    continue

                # Check if cancelled by barge-in
                if self.turn_controller and self.turn_controller.is_cancelled():
                    print("[TTS] Skipping - cancelled by barge-in")
                    continue

                # Check if response is too old
                if pipeline_start and (time.time() - pipeline_start) > self.max_response_staleness:
                    latency = time.time() - pipeline_start
                    print(f"[TTS] Skipping stale response ({latency:.1f}s old, limit={self.max_response_staleness}s)")
                    continue

                self._speak_sync(text, voice_prompt, pipeline_start, tracker)
            except Exception as e:
                print(f"[TTS] Error: {e}")
            finally:
                self._tts_queue.task_done()

        print("[TTS] Worker stopped")

    def _speak_sync(self, text: str, voice_prompt: dict, pipeline_start: float = None, tracker: 'LatencyTracker' = None):
        """Generate and play speech."""
        text = _pad_tts_text(text)
        if not text:
            return
        self._speaking.set()
        tts_start = time.time()

        # Mark TTS start in tracker
        if tracker:
            tracker.tts_start()

        # Update turn controller
        if self.turn_controller:
            self.turn_controller.start_speaking()

        try:
            print(f"[TTS] Generating: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")

            first_chunk = True
            total_audio_samples = 0

            for wav_chunk in self.tts.generate_speech_streaming(
                text, voice_prompt, num_steps=self.num_steps
            ):
                # Check for barge-in cancellation
                if self.turn_controller and self.turn_controller.is_cancelled():
                    print("[TTS] Cancelled due to barge-in")
                    break

                if first_chunk:
                    tts_first_chunk = time.time() - tts_start
                    if pipeline_start:
                        total_latency = time.time() - pipeline_start
                        print(f"[Timing] First audio chunk: TTS={tts_first_chunk:.2f}s, Total={total_latency:.2f}s")
                    # Mark TTS first audio in tracker
                    if tracker:
                        tracker.tts_first_audio()
                        tracker.playback_start()
                    first_chunk = False

                wav_float = wav_chunk.numpy().squeeze()
                wav_float = np.clip(wav_float, -1.0, 1.0)
                pcm = (wav_float * 32767).astype(np.int16)
                total_audio_samples += len(pcm)
                self.mumble.sound_output.add_sound(pcm.tobytes())

            tts_total = time.time() - tts_start

            # Calculate audio duration (24kHz sample rate for LuxTTS output)
            audio_duration_ms = (total_audio_samples / 24000) * 1000

            # Mark playback end and finalize tracker
            if tracker:
                tracker.playback_end(audio_duration_ms)
                tracker.finalize()

            if pipeline_start:
                pipeline_total = time.time() - pipeline_start
                print(f"[Timing] Complete: TTS={tts_total:.2f}s, Pipeline={pipeline_total:.2f}s")
        finally:
            self._speaking.clear()
            # Reset turn controller to idle
            if self.turn_controller:
                self.turn_controller.reset()

    def speak(self, text: str, blocking: bool = False):
        """Queue text to be spoken."""
        if blocking:
            self._speak_sync(text, self.voice_prompt, time.time(), None)
        else:
            self._tts_queue.put((text, self.voice_prompt, time.time(), None, None))

    # =========================================================================
    # Legacy Callbacks (used when event system is unavailable)
    # =========================================================================

    def on_message(self, message):
        """Handle text messages (legacy fallback - see TextCommandHandler)."""
        # Ignore our own messages to avoid echo
        actor = getattr(message, "actor", None)
        if actor == self.mumble.users.myself_session:
            return

        # Ignore messages that aren't for our channel (if channel-scoped)
        if not self._is_message_for_us(message):
            return

        raw_text = getattr(message, "message", "") or ""
        text = strip_html(raw_text)
        if not text.strip():
            return

        sender = "Someone"
        if hasattr(message, 'actor') and message.actor in self.mumble.users:
            sender = self.mumble.users[message.actor]['name']

        print(f"[Text] {sender}: {text}")

        if not self.llm:
            print("[Text] LLM not available - ignoring text message")
            return

        try:
            response = self._generate_response_sync(actor or 0, text, sender)
            if response:
                self.speak(response)
        except Exception as e:
            print(f"[Text] Error generating response: {e}")

    def on_user_joined(self, user):
        """Handle user connect (legacy fallback - see PresenceHandler)."""
        user_name = user.get("name", "Someone")
        print(f"[Event] {user_name} connected to the server")
        # Don't greet on server connect - wait for them to join our channel

    def on_user_updated(self, user, actions):
        """Handle user updates (legacy fallback - see PresenceHandler)."""
        # Check if they moved to our channel
        if "channel_id" in actions:
            try:
                my_channel = self.mumble.users.myself["channel_id"]
                user_name = user.get("name", "Someone")
                user_session = user.get("session")

                # Skip if it's us
                if user_session == self.mumble.users.myself_session:
                    return

                # They joined our channel!
                if actions["channel_id"] == my_channel:
                    print(f"[Event] {user_name} joined the channel")
                    self._greet_user(user_name, user_session)

            except Exception as e:
                print(f"[Event] Error handling user update: {e}")

    def on_user_left(self, user, message):
        """Handle user disconnect (legacy fallback - see PresenceHandler)."""
        user_name = user.get("name", "Someone")
        user_session = user.get("session")
        print(f"[Event] {user_name} left the server")

        # Clean up user state
        self.audio_buffers.pop(user_session, None)
        self.speech_active_until.pop(user_session, None)
        self.speech_start_time.pop(user_session, None)
        self.pending_text.pop(user_session, None)
        self.pending_text_time.pop(user_session, None)
        # Could say goodbye here if desired

    def _greet_user(self, user_name: str, user_id: int):
        """Generate a greeting for a user who joined the channel."""
        # Get themed fallback greetings from soul config
        if self.soul_config and self.soul_config.fallbacks:
            fallback_greetings = [
                g.format(user=user_name)
                for g in self.soul_config.fallbacks.greetings
            ]
        else:
            fallback_greetings = [
                f"Hey {user_name}!",
                f"Oh hey, {user_name}.",
                f"Yo {user_name}, what's up?",
                f"Hey! {user_name}'s here.",
            ]

        if not self.llm:
            # Simple fallback
            self.speak(random.choice(fallback_greetings))
            return

        # Get time-appropriate greeting via LLM
        now = datetime.now()
        hour = now.hour

        if 5 <= hour < 12:
            time_greeting = "morning"
        elif 12 <= hour < 17:
            time_greeting = "afternoon"
        elif 17 <= hour < 21:
            time_greeting = "evening"
        else:
            time_greeting = "late night"

        # Create a greeting prompt
        greeting_prompt = (
            f"{user_name} just joined the voice channel. It's {time_greeting}. "
            "Give a brief, casual greeting. One sentence max."
        )

        # Generate via LLM (async in background)
        def generate_greeting():
            try:
                response = self._generate_oneoff_response_sync(greeting_prompt)
                if response:
                    self.speak(response)
            except Exception as e:
                print(f"[Greet] Error generating greeting: {e}")
                # Use themed fallback
                self.speak(random.choice(fallback_greetings))

        # Run in background so we don't block
        self._asr_executor.submit(generate_greeting)

    def _is_message_for_us(self, message) -> bool:
        """Check if a text message targets our current channel."""
        try:
            my_channel = self.mumble.users.myself["channel_id"]
        except Exception:
            return True

        channels = getattr(message, "channels", None)
        if channels:
            return my_channel in channels

        trees = getattr(message, "trees", None)
        if trees:
            return my_channel in trees

        return True

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Start the bot."""
        self.mumble.start()
        self.mumble.is_ready()
        print("[Mumble] Connected!")

        if self.channel:
            try:
                channel = self.mumble.channels.find_by_name(self.channel)
                channel.move_in()
                print(f"[Mumble] Joined channel: {self.channel}")
            except Exception as e:
                print(f"[Mumble] Failed to join channel: {e}")

    def run_forever(self):
        """Keep the bot running."""
        print("[Bot] Running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Bot] Shutting down...")
            self._shutdown.set()
            self._tts_queue.join()
            self._asr_executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(
        description='Mumble Voice Bot - LLM-powered voice assistant'
    )

    # Config file (loaded first, CLI args override)
    parser.add_argument('--config', default=None,
                        help='Path to config.yaml')

    # Logging settings
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level')
    parser.add_argument('--log-json', action='store_true',
                        help='Output logs in JSON format')
    parser.add_argument('--log-file', default=None,
                        help='Log file path (JSON format)')

    # Mumble settings
    parser.add_argument('--host', default=None, help='Mumble server')
    parser.add_argument('--port', type=int, default=None, help='Mumble port')
    parser.add_argument('--user', default=None, help='Bot username')
    parser.add_argument('--password', default=None, help='Server password')
    parser.add_argument('--channel', default=None, help='Channel to join')

    # Voice settings
    parser.add_argument('--reference', default=None,
                        help='Reference audio for voice cloning')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Compute device')
    parser.add_argument('--steps', type=int, default=None,
                        help='TTS quality (more steps = better quality, slower)')
    parser.add_argument('--voices-dir', default=None,
                        help='Directory for cached voices')

    # VAD settings
    parser.add_argument('--asr-threshold', type=int, default=None,
                        help='Voice activity threshold (use --debug-rms to tune)')
    parser.add_argument('--debug-rms', action='store_true',
                        help='Show RMS levels for threshold tuning')

    # LLM settings
    parser.add_argument('--llm-endpoint', default=None,
                        help='LLM API endpoint (default: Ollama localhost)')
    parser.add_argument('--llm-model', default=None,
                        help='LLM model name')
    parser.add_argument('--llm-api-key', default=None,
                        help='LLM API key (or use LLM_API_KEY env var)')
    parser.add_argument('--llm-system-prompt', default=None,
                        help='System prompt for the assistant')
    parser.add_argument('--personality', default=None,
                        help='Personality to use (e.g., "imperial", or path to file)')

    # Wyoming STT settings
    parser.add_argument('--wyoming-stt-host', default=None,
                        help='Wyoming STT server host (e.g., localhost). If set, uses Wyoming instead of local Whisper')
    parser.add_argument('--wyoming-stt-port', type=int, default=None,
                        help='Wyoming STT server port (default: 10300)')

    # Model storage settings
    parser.add_argument('--hf-home', default=None,
                        help='HuggingFace home directory (where models are cached)')

    args = parser.parse_args()

    # Setup logging first
    setup_logging(
        level=args.log_level,
        json_output=args.log_json,
        log_file=args.log_file,
    )

    # Load config file if specified
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            logger.info(f"Config loaded from {args.config}")
        except Exception as e:
            logger.error(f"Error loading {args.config}: {e}")

    # Apply model storage paths early, before loading any models
    # CLI --hf-home overrides config
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        logger.info(f"HF_HOME={args.hf_home}")
    elif config and config.models:
        applied = config.models.apply_environment()
        if applied:
            logger.info(f"Models environment: {', '.join(f'{k}={v}' for k, v in applied.items())}")

    # Merge config with CLI args (CLI takes precedence)
    # Mumble settings
    host = args.host or (config.mumble.host if config else None) or 'localhost'
    port = args.port or (config.mumble.port if config else None) or 64738
    user = args.user or (config.mumble.user if config else None) or 'VoiceBot'
    password = args.password or (config.mumble.password if config else None) or ''
    channel = args.channel or (config.mumble.channel if config else None)

    # TTS settings
    reference = args.reference or (config.tts.ref_audio if config else None) or 'reference.wav'
    steps = args.steps or (config.tts.num_steps if config else None) or 4
    voices_dir = args.voices_dir or 'voices'

    # VAD settings
    asr_threshold = args.asr_threshold or (config.bot.asr_threshold if config else None) or 2000

    # LLM settings
    llm_endpoint = args.llm_endpoint or (config.llm.endpoint if config else None)
    llm_model = args.llm_model or (config.llm.model if config else None)
    llm_api_key = args.llm_api_key or (config.llm.api_key if config else None)
    llm_system_prompt = args.llm_system_prompt
    personality = args.personality or (config.llm.personality if config else None)

    # If config has prompt_file, load it
    if config and config.llm.prompt_file and not llm_system_prompt:
        prompt_path = config.llm.prompt_file
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                llm_system_prompt = f.read()
            print(f"[LLM] Loaded prompt from {prompt_path}")

    # STT settings
    stt_provider = (config.stt.provider if config else None) or "local"
    wyoming_stt_host = args.wyoming_stt_host or (config.stt.wyoming_host if config else None)
    wyoming_stt_port = args.wyoming_stt_port or (config.stt.wyoming_port if config else None) or 10300

    # Sherpa Nemotron settings
    sherpa_encoder = (config.stt.sherpa_encoder if config else None)
    sherpa_decoder = (config.stt.sherpa_decoder if config else None)
    sherpa_joiner = (config.stt.sherpa_joiner if config else None)
    sherpa_tokens = (config.stt.sherpa_tokens if config else None)
    sherpa_provider = (config.stt.sherpa_provider if config else None) or "cuda"

    # NeMo Nemotron settings
    nemotron_model = (config.stt.nemotron_model if config else None) or "nvidia/nemotron-speech-streaming-en-0.6b"
    nemotron_chunk_ms = (config.stt.nemotron_chunk_ms if config else None) or 160
    nemotron_device = (config.stt.nemotron_device if config else None) or "cuda"

    # Staleness settings
    max_response_staleness = (config.bot.max_response_staleness if config else None) or 5.0

    # TTS device - allow override from config for memory-constrained GPUs
    tts_device_config = (config.tts.device if config else None) or "auto"
    if args.device != 'auto':
        device = args.device
    elif tts_device_config != 'auto':
        device = tts_device_config
    else:
        device = get_best_device()

    # Ensure all models are downloaded before connecting to Mumble
    ensure_models_downloaded(device=device)

    bot = MumbleVoiceBot(
        host=host,
        user=user,
        port=port,
        password=password,
        channel=channel,
        reference_audio=reference,
        device=device,
        num_steps=steps,
        asr_threshold=asr_threshold,
        debug_rms=args.debug_rms,
        voices_dir=voices_dir,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_system_prompt=llm_system_prompt,
        personality=personality,
        config_file=args.config,
        stt_provider=stt_provider,
        wyoming_stt_host=wyoming_stt_host,
        wyoming_stt_port=wyoming_stt_port,
        sherpa_encoder=sherpa_encoder,
        sherpa_decoder=sherpa_decoder,
        sherpa_joiner=sherpa_joiner,
        sherpa_tokens=sherpa_tokens,
        sherpa_provider=sherpa_provider,
        nemotron_model=nemotron_model,
        nemotron_chunk_ms=nemotron_chunk_ms,
        nemotron_device=nemotron_device,
        max_response_staleness=max_response_staleness,
        soul_config=config.soul_config if config else None,
    )

    bot.start()
    bot.run_forever()


if __name__ == '__main__':
    main()
