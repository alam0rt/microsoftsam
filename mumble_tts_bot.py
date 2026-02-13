#!/usr/bin/env python3
"""
Mumble Voice Bot - LLM-powered voice assistant for Mumble.

Listens to voice in a Mumble channel, transcribes with NeMo Nemotron ASR, generates
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
import json
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

# Import ParrotBot for parrot personas
try:
    from parrot_bot import ParrotBot
    PARROT_BOT_AVAILABLE = True
except ImportError:
    PARROT_BOT_AVAILABLE = False
    ParrotBot = None

# Import LLM components
try:
    from mumble_voice_bot.config import ConfigValidationError, load_config
    from mumble_voice_bot.providers.openai_llm import OpenAIChatLLM
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    ConfigValidationError = Exception  # Fallback
    logger.warning(f"LLM modules not available: {e}")

# Import multi-persona config support
try:
    from mumble_voice_bot.multi_persona_config import (
        is_multi_persona_config,
        load_multi_persona_config,
    )
    MULTI_PERSONA_AVAILABLE = True
except ImportError as e:
    MULTI_PERSONA_AVAILABLE = False
    logger.debug(f"Multi-persona config not available: {e}")

# Import Nemotron NeMo STT provider (only supported STT backend)
try:
    from mumble_voice_bot.providers.nemotron_stt import NemotronConfig, NemotronStreamingASR
    NEMOTRON_NEMO_AVAILABLE = True
except ImportError:
    NEMOTRON_NEMO_AVAILABLE = False
    NemotronStreamingASR = None
    NemotronConfig = None
    logger.warning("NeMo Nemotron STT not available. Install with: pip install nemo_toolkit")

# Import latency optimization components
try:
    from mumble_voice_bot.latency import LatencyLogger, LatencyTracker
    from mumble_voice_bot.turn_controller import TurnController
    LATENCY_TRACKING_AVAILABLE = True
except ImportError as e:
    LATENCY_TRACKING_AVAILABLE = False
    logger.warning(f"Latency tracking not available: {e}")

# Import speech filtering components (echo filter, utterance classifier, turn predictor)
try:
    from mumble_voice_bot.conversation_state import ConversationState, ConversationStateMachine
    from mumble_voice_bot.speech_filter import EchoFilter, TurnPredictor, UtteranceClassifier
    SPEECH_FILTER_AVAILABLE = True
except ImportError as e:
    SPEECH_FILTER_AVAILABLE = False
    EchoFilter = None
    UtteranceClassifier = None
    TurnPredictor = None
    ConversationState = None
    ConversationStateMachine = None
    logger.warning(f"Speech filtering not available: {e}")

# Import performance improvements (Phase 1-3 from docs/perf.md)
try:
    from mumble_voice_bot.perf import (
        BoundedTTSQueue,
        DropPolicy,
        RollingLatencyTracker,
        TTSQueueItem,
        TurnIdCoordinator,
    )
    PERF_AVAILABLE = True
except ImportError as e:
    PERF_AVAILABLE = False
    logger.warning(f"Performance improvements not available: {e}")

# Import event system
try:
    from mumble_voice_bot.handlers import ConnectionHandler, PresenceHandler, TextCommandHandler
    from mumble_voice_bot.providers.mumble_events import EventDispatcher
    EVENT_SYSTEM_AVAILABLE = True
except ImportError as e:
    EVENT_SYSTEM_AVAILABLE = False
    logger.warning(f"Event system not available: {e}")

# Import tool system
try:
    from mumble_voice_bot.tools.souls import SoulsTool

    from mumble_voice_bot.tools import ToolRegistry
    from mumble_voice_bot.tools.sound_effects import SoundEffectsTool
    from mumble_voice_bot.tools.web_search import WebSearchTool
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    ToolRegistry = None
    WebSearchTool = None
    SoulsTool = None
    SoundEffectsTool = None
    logger.warning(f"Tool system not available: {e}")


# =============================================================================
# Imports from extracted modules (Phase 1 refactor)
# =============================================================================
from mumble_voice_bot.audio import pcm_rms  # noqa: F811
from mumble_voice_bot.text_processing import (
    pad_tts_text as _pad_tts_text,
    sanitize_for_tts as _sanitize_for_tts,
    split_into_sentences,
    is_question as _is_question_heuristic,
)
from mumble_voice_bot.utils import strip_html, get_best_device, ensure_models_downloaded
from mumble_voice_bot.providers.luxtts import StreamingLuxTTS
from mumble_voice_bot.coordination import SharedBotServices  # noqa: F811
from mumble_voice_bot.souls import load_system_prompt, load_personality, get_fallback_prompt, switch_soul
from mumble_voice_bot.events import EventResponder, ChannelActivityTracker
from mumble_voice_bot.cli import create_argument_parser, merge_config_with_args


# =============================================================================
# Shared Services Factory
# =============================================================================
# SharedBotServices is now imported from mumble_voice_bot.coordination

def create_shared_services(
    device: str = "auto",
    nemotron_model: str = None,
    nemotron_chunk_ms: int = 160,
    nemotron_device: str = None,
    llm_endpoint: str = None,
    llm_model: str = None,
    llm_api_key: str = None,
    llm_timeout: float = 30.0,
    llm_max_tokens: int = None,
    llm_temperature: float = None,
) -> SharedBotServices:
    """Create shared services for multiple bots.

    This factory creates TTS, STT, and LLM services once, which can then be
    shared across multiple MumbleVoiceBot instances.

    Args:
        device: Compute device ('auto', 'cuda', 'cpu', 'mps').
        nemotron_model: NeMo Nemotron model name.
        nemotron_chunk_ms: Nemotron chunk size in ms.
        nemotron_device: Device for Nemotron (defaults to device).
        llm_endpoint: LLM API endpoint.
        llm_model: LLM model name.
        llm_api_key: LLM API key.
        llm_timeout: LLM request timeout.
        llm_max_tokens: Max tokens for LLM responses.
        llm_temperature: LLM temperature.

    Returns:
        SharedBotServices instance with initialized services.
    """
    # Determine device
    if device == "auto":
        device = get_best_device()

    print(f"[SharedServices] Initializing on {device}")

    # Ensure models downloaded
    ensure_models_downloaded(device=device)

    # Initialize TTS
    print("[SharedServices] Loading TTS model...")
    tts = StreamingLuxTTS('YatharthS/LuxTTS', device=device, threads=2)

    # Initialize STT - NeMo Nemotron is the only supported provider
    if not NEMOTRON_NEMO_AVAILABLE:
        raise RuntimeError("NeMo Nemotron STT required but nemo_toolkit not installed. Install with: pip install nemo_toolkit")

    model = nemotron_model or "nvidia/nemotron-speech-streaming-en-0.6b"
    nemo_device = nemotron_device or device
    print(f"[SharedServices] Loading NeMo Nemotron ({model})...")
    config = NemotronConfig(
        model_name=model,
        chunk_size_ms=nemotron_chunk_ms,
        device=nemo_device,
    )
    stt = NemotronStreamingASR(config)
    if not asyncio.run(stt.initialize()):
        raise RuntimeError(f"Failed to initialize NeMo Nemotron STT ({model})")
    print("[SharedServices] NeMo Nemotron ready!")

    # Initialize LLM
    llm = None
    if LLM_AVAILABLE:
        endpoint = llm_endpoint or "http://localhost:11434/v1/chat/completions"
        model = llm_model or "llama3.2:3b"
        api_key = llm_api_key or os.environ.get('OPENROUTER_API_KEY') or os.environ.get('LLM_API_KEY')

        print(f"[SharedServices] Initializing LLM: {model}")
        llm = OpenAIChatLLM(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            system_prompt="",  # Set per-bot
            timeout=llm_timeout,
            max_tokens=llm_max_tokens,
            temperature=llm_temperature,
        )

    print("[SharedServices] Ready!")
    return SharedBotServices(tts=tts, stt=stt, llm=llm, device=device)


# =============================================================================
# MumbleVoiceBot - Main bot class
# =============================================================================

class MumbleVoiceBot:
    """A Mumble bot that listens, thinks with an LLM, and responds with TTS.

    Can be instantiated with shared TTS/STT/LLM services (for multi-bot mode)
    or will create its own services (single-bot mode).
    """

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
        # NeMo Nemotron STT settings (only supported STT backend)
        nemotron_model: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        nemotron_chunk_ms: int = 160,
        nemotron_device: str = "cuda",
        # Staleness configuration
        max_response_staleness: float = 5.0,
        # Barge-in configuration
        barge_in_enabled: bool = True,
        # Soul configuration
        soul_config=None,  # SoulConfig object with themed fallbacks
        soul_name=None,  # Name of the active soul (for tool queries)
        # Tools configuration
        tools_config=None,  # ToolsConfig for tool settings
        # Shared services (for multi-bot mode)
        shared_tts=None,  # Shared TTS engine
        shared_stt=None,  # Shared STT engine
        shared_llm=None,  # Shared LLM client
        voice_prompt=None,  # Pre-computed voice prompt (tensors)
        shared_echo_filter=None,  # Shared echo filter for multi-bot
        shared_services: SharedBotServices = None,  # Full shared services (for speaking coordination)
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
        self.tools_config = tools_config  # Tool-specific configuration
        self._current_soul_name = soul_name  # Track active soul name for tool queries

        # Per-bot logger with username context
        self.logger = get_logger(f"{__name__}.{user}")

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
        self.max_speech_duration = 5.0  # force processing after 5 seconds (keeps ASR fast)

        # Long speech tracking (for triggering long_speech_ended event)
        self.user_total_speech_time = {}  # user_id -> accumulated speech duration this utterance
        self.long_speech_threshold = 15.0  # seconds - trigger event if user talks longer than this

        # Channel quiet tracking
        self._last_channel_activity = time.time()
        self._quiet_threshold = 60.0  # seconds before triggering channel_quiet event
        self._quiet_timer = None
        self._quiet_event_triggered = False  # Prevent repeated triggers

        # Response staleness settings
        self.max_response_staleness = max_response_staleness  # skip responses older than this (seconds)

        # Barge-in settings
        self.barge_in_enabled = barge_in_enabled  # allow users to interrupt bot

        # Multi-bot coordination (shared echo filter so all bots know what all bots said)
        self._shared_echo_filter = shared_echo_filter

        # SharedBotServices handles the event journal and coordination
        # Always use it - even for single-bot mode (creates one if not provided)
        if shared_services is not None:
            self._shared_services = shared_services
        else:
            # Single-bot mode: create our own SharedBotServices
            self._shared_services = SharedBotServices(
                tts=shared_tts,
                stt=shared_stt,
                llm=shared_llm,
                device=device,
            )

        # Register to receive utterances from other bots (fake ASR)
        self._shared_services.register_utterance_listener(self._on_bot_utterance)

        # Pending transcriptions (for accumulating long utterances)
        self.pending_text = {}  # user_id -> accumulated text
        self.pending_text_time = {}  # user_id -> timestamp of last text
        self.pending_text_timeout = 1.5  # seconds to wait for more speech before responding

        # Conversation state - SHARED channel history (not per-user)
        self.channel_history = []  # List of {"role": str, "content": str, "speaker": str, "time": float}
        self.channel_history_max = 20  # Default, can be overridden by config.llm.context_messages
        self.conversation_timeout = 300.0  # 5 minutes - clear history after inactivity
        self.last_activity_time = time.time()

        # Legacy per-user history (kept for compatibility)
        self.conversation_history = {}  # user_id -> list of messages
        self.last_conversation_time = {}  # user_id -> timestamp

        # State flags
        self._speaking = threading.Event()
        self._shutdown = threading.Event()

        # Threading
        self._asr_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ASR")  # Allow 2 workers
        self._tts_lock = threading.Lock()

        # Speech filtering components (echo filter, utterance classifier, turn predictor)
        self.echo_filter = None
        self.utterance_classifier = None
        self.turn_predictor = None
        self.conversation_state_machine = None
        self._speech_filter_config = None  # Will be populated from config

        # Turn ID coordination and bounded TTS queue (Phase 1 from docs/perf.md)
        if PERF_AVAILABLE:
            self._turn_coordinator = TurnIdCoordinator()
            self._tts_queue = BoundedTTSQueue(
                maxsize=10,
                policy=DropPolicy.DROP_STALE,
                turn_coordinator=self._turn_coordinator,
            )
            self._rolling_latency = RollingLatencyTracker(window_size=100)
            print("[Perf] Using bounded TTS queue with DROP_STALE policy (maxsize=10)")
        else:
            self._turn_coordinator = None
            self._tts_queue = queue.Queue()
            self._rolling_latency = None

        # Latency tracking and turn control
        if LATENCY_TRACKING_AVAILABLE:
            self.latency_logger = LatencyLogger()  # In-memory only
            self.turn_controller = TurnController()
            # Register barge-in callback to speak interruption acknowledgment
            self.turn_controller.on_barge_in(self._on_barge_in)
            print("[Latency] Tracking enabled (in-memory)")
        else:
            self.latency_logger = None
            self.turn_controller = None

        # Current latency tracker for in-progress turn
        self._current_tracker: LatencyTracker = None

        # Filler state tracking
        self._llm_thinking_since: float | None = None  # When LLM call started (for "still thinking" timer)
        self._still_thinking_timer: threading.Timer | None = None

        # Event tracking - users we've already greeted this session
        self._greeted_users: set[str] = set()  # Track who we've said "first speech" greeting to

        # Stats tracking
        self._stats_interval = 30  # Log stats every 30 seconds
        self._asr_count = 0
        self._asr_total_ms = 0
        self._llm_count = 0
        self._llm_total_ms = 0
        self._tts_count = 0
        self._tts_total_ms = 0
        self._stats_lock = threading.Lock()

        # Start stats logger thread
        self._stats_thread = threading.Thread(
            target=self._stats_logger, daemon=True, name="Stats-Logger"
        )
        self._stats_thread.start()

        # Start TTS worker
        self._tts_worker_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="TTS-Worker"
        )
        self._tts_worker_thread.start()

        # Initialize TTS - use shared or create own
        if shared_tts is not None:
            self.tts = shared_tts
            self._owns_tts = False
            print("[TTS] Using shared TTS engine")
        else:
            print(f"[TTS] Loading model on {device}...")
            self.tts = StreamingLuxTTS('YatharthS/LuxTTS', device=device, threads=2)
            self._owns_tts = True

        # Load voice - use provided or load from file
        os.makedirs(self.voices_dir, exist_ok=True)
        if voice_prompt is not None:
            self.voice_prompt = self._ensure_voice_on_device(voice_prompt)
            print("[Voice] Using provided voice prompt")
        else:
            self._load_reference_voice(reference_audio)

        # Initialize STT - use shared or create own (NeMo Nemotron only)
        self.streaming_stt = None
        self._owns_stt = False

        if shared_stt is not None:
            # Use shared STT
            self.streaming_stt = shared_stt
            print("[STT] Using shared STT engine")
        else:
            # Create own NeMo Nemotron STT
            if not NEMOTRON_NEMO_AVAILABLE:
                raise RuntimeError("NeMo Nemotron STT required but nemo_toolkit not installed. Install with: pip install nemo_toolkit")
            print(f"[STT] Using NeMo Nemotron ({nemotron_model}, chunk={nemotron_chunk_ms}ms)")
            config = NemotronConfig(
                model_name=nemotron_model,
                chunk_size_ms=nemotron_chunk_ms,
                device=nemotron_device,
            )
            self.streaming_stt = NemotronStreamingASR(config)
            self._owns_stt = True
            print("[STT] Pre-loading Nemotron model (this may take a moment)...")
            if not asyncio.run(self.streaming_stt.initialize()):
                raise RuntimeError(f"Failed to initialize NeMo Nemotron STT ({nemotron_model})")
            print("[STT] Nemotron model ready!")

        # Initialize speech filters (echo detection, utterance classification, turn prediction)
        self._init_speech_filters()

        # Initialize LLM - use shared or create own
        self.llm = None
        self.tools = None  # Tool registry for function calling
        self._owns_llm = False

        if shared_llm is not None:
            # Use shared LLM - store our system prompt to set per-request
            self.llm = shared_llm
            self._bot_system_prompt = llm_system_prompt or ""
            print(f"[LLM] Using shared LLM client (prompt: {len(self._bot_system_prompt)} chars)")
            # Still initialize tools
            if TOOLS_AVAILABLE:
                self._init_tools()
        elif LLM_AVAILABLE:
            self._owns_llm = True
            self._init_llm(
                endpoint=llm_endpoint,
                model=llm_model,
                api_key=llm_api_key,
                system_prompt=llm_system_prompt,
                personality=personality,
                config_file=config_file,
            )
            # Initialize tools after LLM
            self._init_tools()
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
        # Create a NEW event loop for async handlers (don't use the main thread's loop)
        self._event_loop = asyncio.new_event_loop()

        # Start the event loop in a background thread so async handlers can run
        self._event_loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="EventLoop"
        )
        self._event_loop_thread.start()
        logger.info("Event loop started in background thread")

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

    def _run_event_loop(self):
        """Run the async event loop in a background thread."""
        asyncio.set_event_loop(self._event_loop)
        logger.debug("Event loop thread running")
        self._event_loop.run_forever()
        logger.debug("Event loop thread stopped")

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
                logger.info(f"Loaded LLM config from {config_file or 'config.yaml'}")
            except Exception as e:
                logger.error(f"LLM config load failed: {e}", exc_info=True)

        # CLI args override config
        final_endpoint = endpoint
        final_model = model
        final_api_key = api_key
        final_system_prompt = system_prompt
        final_timeout = 30.0
        final_max_tokens = None
        final_temperature = None
        final_top_p = None
        final_top_k = None
        final_repetition_penalty = None
        final_frequency_penalty = None
        final_presence_penalty = None

        if config:
            final_endpoint = final_endpoint or config.llm.endpoint
            final_model = final_model or config.llm.model
            final_api_key = final_api_key or config.llm.api_key
            # Don't use config system_prompt if personality is set (we'll load prompts/default.md + personality)
            if not personality:
                final_system_prompt = final_system_prompt or config.llm.system_prompt
            final_timeout = config.llm.timeout or final_timeout
            final_max_tokens = config.llm.max_tokens
            final_temperature = config.llm.temperature
            final_top_p = config.llm.top_p
            final_top_k = config.llm.top_k
            final_repetition_penalty = config.llm.repetition_penalty
            final_frequency_penalty = config.llm.frequency_penalty
            final_presence_penalty = config.llm.presence_penalty
            if hasattr(config, 'bot') and config.bot.conversation_timeout:
                self.conversation_timeout = config.bot.conversation_timeout
            # Update context window from config
            if config.llm.context_messages:
                self.channel_history_max = config.llm.context_messages

        # Check environment variables for API key if not set
        if not final_api_key:
            final_api_key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('LLM_API_KEY')

        # Defaults
        final_endpoint = final_endpoint or "http://localhost:11434/v1/chat/completions"
        final_model = final_model or "llama3.2:3b"
        # Always load system prompt with personality if personality is set
        final_system_prompt = final_system_prompt or self._load_system_prompt(personality=personality)

        self.llm = OpenAIChatLLM(
            endpoint=final_endpoint,
            model=final_model,
            api_key=final_api_key,
            system_prompt=final_system_prompt,
            timeout=final_timeout,
            max_tokens=final_max_tokens,
            temperature=final_temperature,
            top_p=final_top_p,
            top_k=final_top_k,
            repetition_penalty=final_repetition_penalty,
            frequency_penalty=final_frequency_penalty,
            presence_penalty=final_presence_penalty,
        )
        extra_info = []
        if final_max_tokens:
            extra_info.append(f"max_tokens={final_max_tokens}")
        if final_temperature:
            extra_info.append(f"temp={final_temperature}")
        if final_top_p:
            extra_info.append(f"top_p={final_top_p}")
        if final_top_k:
            extra_info.append(f"top_k={final_top_k}")
        if final_repetition_penalty:
            extra_info.append(f"rep_penalty={final_repetition_penalty}")
        if final_frequency_penalty:
            extra_info.append(f"freq_penalty={final_frequency_penalty}")
        if final_presence_penalty:
            extra_info.append(f"pres_penalty={final_presence_penalty}")
        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
        print(f"[LLM] Initialized: {final_model} @ {final_endpoint}{extra_str}")

    def _load_system_prompt(self, prompt_file: str = None, personality: str = None) -> str:
        """Load system prompt — delegates to mumble_voice_bot.souls."""
        return load_system_prompt(prompt_file=prompt_file, personality=personality, project_dir=_THIS_DIR)

    def _load_personality(self, personality: str) -> str:
        """Load a personality file — delegates to mumble_voice_bot.souls."""
        return load_personality(personality, project_dir=_THIS_DIR)

    def _get_fallback_prompt(self) -> str:
        """Fallback prompt — delegates to mumble_voice_bot.souls."""
        return get_fallback_prompt()

    def _init_speech_filters(self) -> None:
        """Initialize speech filtering components.

        These components help prevent:
        - Echo detection (bot responding to its own TTS output)
        - Non-meaningful utterances (fillers, very short fragments)
        - Poor turn-taking (responding too early/late)
        """
        if not SPEECH_FILTER_AVAILABLE:
            print("[Speech] Speech filtering not available")
            return

        # Load config from file if available
        config = None
        try:
            config = load_config()
        except Exception:
            pass

        # Get bot config settings
        bot_config = config.bot if config else None

        # Initialize echo filter - use shared if available (multi-bot mode)
        if self._shared_echo_filter is not None:
            self.echo_filter = self._shared_echo_filter
            print("[Speech] Using shared echo filter (multi-bot mode)")
        else:
            echo_enabled = getattr(bot_config, 'echo_filter_enabled', True) if bot_config else True
            if echo_enabled:
                decay_time = getattr(bot_config, 'echo_filter_decay', 3.0) if bot_config else 3.0
                self.echo_filter = EchoFilter(decay_time=decay_time)
                print(f"[Speech] Echo filter enabled (decay={decay_time}s)")
            else:
                print("[Speech] Echo filter disabled")

        # Initialize utterance classifier
        utterance_enabled = getattr(bot_config, 'utterance_filter_enabled', True) if bot_config else True
        if utterance_enabled:
            min_words = getattr(bot_config, 'utterance_min_words', 2) if bot_config else 2
            min_chars = getattr(bot_config, 'utterance_min_chars', 5) if bot_config else 5
            self.utterance_classifier = UtteranceClassifier(
                min_words=min_words,
                min_chars=min_chars,
            )
            print(f"[Speech] Utterance filter enabled (min_words={min_words}, min_chars={min_chars})")
        else:
            print("[Speech] Utterance filter disabled")

        # Initialize turn predictor
        turn_enabled = getattr(bot_config, 'turn_prediction_enabled', True) if bot_config else True
        if turn_enabled:
            base_delay = getattr(bot_config, 'turn_prediction_base_delay', 0.3) if bot_config else 0.3
            max_delay = getattr(bot_config, 'turn_prediction_max_delay', 1.5) if bot_config else 1.5
            self.turn_predictor = TurnPredictor(
                base_delay=base_delay,
                max_delay=max_delay,
            )
            print(f"[Speech] Turn predictor enabled (base={base_delay}s, max={max_delay}s)")
        else:
            print("[Speech] Turn predictor disabled")

        # Initialize conversation state machine
        state_enabled = getattr(bot_config, 'state_machine_enabled', True) if bot_config else True
        if state_enabled:
            cooldown = getattr(bot_config, 'state_machine_cooldown', 0.5) if bot_config else 0.5
            self.conversation_state_machine = ConversationStateMachine(
                cooldown_duration=cooldown,
                on_state_change=self._on_conversation_state_change,
            )
            print(f"[Speech] State machine enabled (cooldown={cooldown}s)")
        else:
            print("[Speech] State machine disabled")

    def _on_conversation_state_change(
        self,
        old_state: "ConversationState",
        new_state: "ConversationState",
        reason: str,
    ) -> None:
        """Callback for conversation state changes.

        Args:
            old_state: Previous state.
            new_state: New state.
            reason: Reason for the transition.
        """
        logger.debug(
            f"Conversation state: {old_state.name} -> {new_state.name}",
            extra={"reason": reason},
        )

    def _init_tools(self) -> None:
        """Initialize the tool registry with available tools.

        Tools enable the LLM to perform actions like web searches.
        The LLM will decide when to use tools based on user queries.
        """
        if not TOOLS_AVAILABLE:
            print("[Tools] Tool system not available")
            return

        self.tools = ToolRegistry()

        # Register web search tool
        self.tools.register(WebSearchTool(max_results=5, timeout=10.0))

        # Register souls tool for personality management
        souls_tool = SoulsTool(
            souls_dir=os.path.join(_THIS_DIR, "souls"),
            switch_callback=self._switch_soul,
            get_current_callback=self._get_current_soul,
        )
        self.tools.register(souls_tool)

        # Register sound effects tool if configured
        if (
            self.tools_config
            and self.tools_config.sound_effects_enabled
            and self.tools_config.sound_effects_dir
            and SoundEffectsTool is not None
        ):
            sound_effects_tool = SoundEffectsTool(
                sounds_dir=self.tools_config.sound_effects_dir,
                play_callback=self._play_sound_effect,
                auto_play=self.tools_config.sound_effects_auto_play,
                sample_rate=48000,  # Mumble uses 48kHz
                enable_web_search=getattr(self.tools_config, 'sound_effects_web_search', True),
                cache_web_sounds=getattr(self.tools_config, 'sound_effects_cache', True),
                verify_ssl=getattr(self.tools_config, 'sound_effects_verify_ssl', True),
            )
            self.tools.register(sound_effects_tool)
            web_status = "with web search" if getattr(self.tools_config, 'sound_effects_web_search', True) else "local only"
            print(f"[Tools] Sound effects enabled ({web_status}) from {self.tools_config.sound_effects_dir}")

        print(f"[Tools] Initialized with {len(self.tools)} tool(s): {self.tools.tool_names}")

    async def _play_sound_effect(self, pcm_bytes: bytes, sample_rate: int) -> None:
        """Play a sound effect through Mumble.

        Args:
            pcm_bytes: Raw PCM audio data (16-bit, mono).
            sample_rate: Sample rate of the audio.
        """
        if not hasattr(self, 'mumble') or not self.mumble:
            raise RuntimeError("Mumble not initialized")

        # Convert sample rate if needed (Mumble expects 48kHz)
        if sample_rate != 48000:
            import numpy as np
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            num_samples = int(len(pcm_array) * 48000 / sample_rate)
            indices = np.linspace(0, len(pcm_array) - 1, num_samples)
            pcm_array = np.interp(indices, np.arange(len(pcm_array)), pcm_array)
            pcm_bytes = pcm_array.astype(np.int16).tobytes()

        # Add sound to Mumble output
        self.mumble.sound_output.add_sound(pcm_bytes)

    async def _check_keyword_tools(self, text: str) -> str | None:
        """Check for keyword-based tool triggers.

        This is a fallback for models that don't reliably generate tool calls.
        Returns a response string if a tool was triggered, None otherwise.
        """
        if not self.tools:
            return None

        text_lower = text.lower()

        # Soul switching keywords
        switch_patterns = [
            "switch to the", "switch to", "change to the", "change to",
            "use the", "be the", "become the", "switch soul to",
        ]

        for pattern in switch_patterns:
            if pattern in text_lower:
                # Extract what comes after the pattern
                idx = text_lower.find(pattern) + len(pattern)
                remainder = text_lower[idx:].strip()

                # Look for soul name - common patterns like "raf soul", "raf", "raf personality"
                soul_name = remainder.split()[0] if remainder.split() else None
                if soul_name:
                    # Clean up common suffixes
                    soul_name = soul_name.rstrip(".,!?")
                    for suffix in ["soul", "personality", "voice", "character"]:
                        if soul_name.endswith(suffix):
                            soul_name = soul_name[:-len(suffix)].strip()

                    if soul_name:
                        logger.info(f"Keyword trigger: switching to soul '{soul_name}'")
                        result = await self.tools.execute("souls", {
                            "action": "switch",
                            "soul_name": soul_name
                        })
                        return f"Switching to {soul_name}. {result}"

        # List souls keywords
        if any(phrase in text_lower for phrase in ["list souls", "what souls", "available souls", "show souls"]):
            logger.info("Keyword trigger: listing souls")
            result = await self.tools.execute("souls", {"action": "list"})
            return f"Here are the available souls: {result}"

        return None

    # =========================================================================
    # Soul Management
    # =========================================================================

    def _get_current_soul(self) -> str | None:
        """Get the name of the currently active soul."""
        if self.soul_config:
            return self._current_soul_name
        return None

    async def _switch_soul(self, soul_name: str, preserve_context: bool = None) -> str:
        """Switch to a different soul/personality.

        This updates:
        - The LLM system prompt
        - The TTS voice reference
        - The soul config for fallbacks

        Args:
            soul_name: Name of the soul directory to switch to.
            preserve_context: Whether to preserve conversation history.
                             If None, uses config.bot.preserve_context_on_switch.

        Returns:
            Success or error message.
        """
        from mumble_voice_bot.config import load_config, load_soul_config

        souls_dir = os.path.join(_THIS_DIR, "souls")
        soul_path = os.path.join(souls_dir, soul_name)

        if not os.path.exists(soul_path):
            return f"Soul '{soul_name}' not found."

        # Determine if we should preserve context
        if preserve_context is None:
            try:
                config = load_config()
                preserve_context = getattr(config.bot, 'preserve_context_on_switch', True)
                max_preserved = getattr(config.bot, 'max_preserved_messages', 10)
            except Exception:
                preserve_context = True
                max_preserved = 10
        else:
            max_preserved = 10

        # Store current context before switch (only user/assistant messages)
        preserved_messages = []
        if preserve_context and self.channel_history:
            preserved_messages = [
                msg for msg in self.channel_history
                if msg.get('role') in ('user', 'assistant')
            ][-max_preserved:]
            logger.info(f"Preserving {len(preserved_messages)} messages across soul switch")

        try:
            # Load the new soul config
            new_soul = load_soul_config(soul_name, souls_dir)
            logger.info(f"Switching to soul: {new_soul.name}")

            # Update TTS voice if different
            if new_soul.voice.ref_audio and new_soul.voice.ref_audio != "reference.wav":
                ref_audio = new_soul.voice.ref_audio
                # Make relative paths absolute to soul directory
                if not os.path.isabs(ref_audio):
                    ref_audio = os.path.join(soul_path, ref_audio)

                # Handle directory - find first audio file
                if os.path.isdir(ref_audio):
                    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
                    for f in sorted(os.listdir(ref_audio)):
                        if os.path.splitext(f)[1].lower() in audio_extensions:
                            ref_audio = os.path.join(ref_audio, f)
                            break

                if os.path.exists(ref_audio):
                    print(f"[Soul] Loading voice: {ref_audio}")
                    self._load_reference_voice(ref_audio)

            # Update LLM system prompt
            if self.llm:
                # Load personality file
                personality_path = os.path.join(soul_path, "personality.md")
                if os.path.exists(personality_path):
                    new_prompt = self._load_system_prompt(personality=personality_path)
                    self.llm.system_prompt = new_prompt
                    print("[Soul] Updated LLM personality")

                # Apply LLM overrides from soul config
                if new_soul.llm:
                    if "temperature" in new_soul.llm:
                        self.llm.temperature = new_soul.llm["temperature"]
                    if "max_tokens" in new_soul.llm:
                        self.llm.max_tokens = new_soul.llm["max_tokens"]

            # Update soul config and name
            self.soul_config = new_soul
            self._current_soul_name = soul_name

            # Update Mumble username to match the new soul
            if self.mumble and self.mumble.is_ready():
                try:
                    self.mumble.users.myself.update_comment(f"Soul: {new_soul.name}")
                    # Note: Mumble doesn't allow renaming after connection,
                    # but we update the comment to show the current soul
                except Exception as e:
                    logger.debug(f"Could not update Mumble comment: {e}")

            # Clear conversation history and restore preserved messages
            self.channel_history = []
            if preserved_messages:
                self.channel_history.extend(preserved_messages)
                logger.info(f"Restored {len(preserved_messages)} messages to new soul context")

            context_msg = f" ({len(preserved_messages)} messages preserved)" if preserved_messages else ""
            return f"Switched to {new_soul.name}. Voice and personality updated.{context_msg}"

        except Exception as e:
            logger.error(f"Failed to switch soul: {e}")
            return f"Error switching to '{soul_name}': {str(e)}"

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
    # Conversation Management - Shared Channel History
    # =========================================================================

    def _get_channel_history(self) -> list[dict]:
        """Get shared channel conversation history, clearing if stale."""
        current_time = time.time()

        # Clear history if conversation went stale
        if current_time - self.last_activity_time > self.conversation_timeout:
            if self.channel_history:
                print(f"[Chat] Cleared stale channel history ({len(self.channel_history)} messages)")
            self.channel_history = []

        self.last_activity_time = current_time
        return self.channel_history

    def _add_to_channel_history(self, role: str, content: str, speaker: str = None):
        """Add a message to shared channel history.

        Args:
            role: "user" for human speakers, "assistant" for bot responses
            content: The raw message content
            speaker: Name of who said it (for user messages)
        """
        history = self._get_channel_history()

        # Format the content based on role
        if role == "user" and speaker:
            formatted_content = f"{speaker}: {content}"
        else:
            formatted_content = content

        history.append({
            "role": role,
            "content": formatted_content,
            "speaker": speaker,
            "time": time.time()
        })

        # Trim to max size
        if len(history) > self.channel_history_max:
            self.channel_history = history[-self.channel_history_max:]

    def _build_llm_messages(self) -> list[dict]:
        """Build LLM message list from the shared event journal.

        Creates a natural conversation flow where the LLM can see
        what everyone said, including context about what's happening
        (interruptions, who's present, etc.).

        Uses the unified journal from SharedBotServices as the single source of truth.
        """
        messages = []

        # Get full journal with all events (not just messages)
        journal = self._shared_services.get_journal_for_llm(max_events=50)

        # Build a rich context block that tells the LLM what's happening
        context_parts = []

        # Time context
        time_ctx = self._get_time_context()
        if time_ctx:
            context_parts.append(f"Current time: {time_ctx}")

        # Channel context (who's present)
        channel_ctx = self._get_channel_context()
        if channel_ctx:
            context_parts.append(f"Channel: {channel_ctx}")

        # Recent events summary (joins/leaves/interruptions in last 60s)
        recent_events = []
        for e in journal:
            if e.get("seconds_ago", 999) > 60:
                continue
            event_type = e.get("event", "")
            speaker = e.get("speaker", "someone")

            if event_type == "user_joined":
                recent_events.append(f"{speaker} just joined")
            elif event_type == "user_left":
                recent_events.append(f"{speaker} just left")
            elif event_type == "interrupted":
                recent_events.append(f"{speaker} was interrupted")
            elif event_type == "started_speaking":
                # Could note who started speaking
                pass

        if recent_events:
            context_parts.append(f"Recent: {', '.join(recent_events[-3:])}")  # Last 3 events

        if context_parts:
            messages.append({
                "role": "system",
                "content": " | ".join(context_parts)
            })

        # Get conversation messages from the journal
        # Pass our bot name so other bots' messages appear as "user" role
        history = self._shared_services.get_recent_messages_for_llm(max_messages=20, bot_name=self.user)
        messages.extend(history)

        return messages

    def _get_history(self, user_id: int) -> list[dict]:
        """Get conversation history for a user, clearing if stale.

        NOTE: This now returns the shared journal formatted for the LLM,
        not per-user history. The user_id is kept for API compatibility.
        """
        return self._build_llm_messages()

    def _add_to_history(self, user_id: int, role: str, content: str, user_name: str = None):
        """Add a message to conversation history via the shared journal.

        This is the single entry point for adding USER messages to context.
        Bot (assistant) messages are logged via broadcast_utterance() instead,
        to avoid double-logging.
        """
        if role == "user":
            self._shared_services.log_event("user_message", user_name, content)
        # NOTE: Don't log assistant messages here - they get logged in broadcast_utterance()
        # to avoid duplication in the journal

    async def _generate_response(self, user_id: int, text: str, user_name: str = None) -> str:
        """Generate LLM response, executing any tool calls.

        This method implements a tool execution loop:
        1. Check for keyword-based tool triggers (fallback for models without tool support)
        2. Send user message + tool definitions to LLM
        3. If LLM returns tool calls, execute them
        4. Send tool results back to LLM
        5. Repeat until LLM returns a text response (max 5 iterations)

        Args:
            user_id: User session ID
            text: The message text to respond to
            user_name: Name of the speaker
        """
        # Check for keyword-based tool triggers first
        # This helps with models that don't reliably use tool calling
        keyword_result = await self._check_keyword_tools(text)
        if keyword_result:
            self._add_to_history(user_id, "user", text, user_name)
            self._add_to_history(user_id, "assistant", keyword_result)
            return keyword_result

        self._add_to_history(user_id, "user", text, user_name)

        # Build messages for LLM
        messages = self._get_history(user_id)

        # Get tool definitions if tools are available
        tools = self.tools.get_definitions() if self.tools else None
        if tools:
            logger.debug(f"Tools available: {[t.get('function', {}).get('name', '?') for t in tools]}")
        else:
            logger.debug("No tools available for LLM request")

        # Set our system prompt on the shared LLM before each request
        # This is critical for multi-bot mode where each bot has different personality
        if hasattr(self, '_bot_system_prompt') and self._bot_system_prompt:
            self.llm.system_prompt = self._bot_system_prompt

        # Tool execution loop
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Call LLM with tools (pass bot name for debug logging)
            try:
                response = await self.llm.chat(messages, tools=tools, bot_name=self.user)
            except Exception as e:
                # Check for rate limiting (HTTP 429)
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str or "too many" in error_str:
                    logger.warning(f"Rate limited by LLM API: {e}")
                    # Trigger rate_limited event for themed response
                    fallback = self._trigger_event('rate_limited')
                    if fallback:
                        return fallback
                    return "I need a moment to collect my thoughts..."
                # Re-raise other errors
                raise

            # If LLM wants to call tools, execute them
            if response.has_tool_calls:
                # Add assistant message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                            }
                        }
                        for tc in response.tool_calls
                    ]
                })

                # Execute each tool call
                for tool_call in response.tool_calls:
                    logger.info(f"Executing tool: {tool_call.name}({tool_call.arguments})")

                    if self.tools:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    else:
                        result = f"Error: Tool '{tool_call.name}' not available"

                    # Add tool result to messages
                    # Use the LLM's tool formatter to format the result correctly
                    if hasattr(self.llm, 'tool_formatter'):
                        tool_msg = self.llm.tool_formatter.format_tool_result(
                            tool_call.id, tool_call.name, result
                        )
                        messages.append(tool_msg)
                    else:
                        # Fallback to OpenAI format
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })

                # Continue loop to get LLM's response to tool results
                continue

            # LLM returned text response - we're done
            if response.content:
                self._add_to_history(user_id, "assistant", response.content)
                return response.content

            # No content and no tool calls - unusual, return empty
            logger.warning("LLM returned empty response with no tool calls")
            return ""

        # Hit max iterations - return what we have or error
        logger.warning(f"Tool loop hit max iterations ({max_iterations})")
        return "Sorry, I got stuck in a loop trying to look that up."

    def _generate_response_sync(self, user_id: int, text: str, user_name: str = None) -> str:
        """Generate LLM response synchronously."""
        return self._run_coro_sync(self._generate_response(user_id, text, user_name))

    def _generate_oneoff_response_sync(self, prompt: str) -> str:
        """Generate a one-off LLM response without updating history."""
        if not self.llm:
            return ""
        # Set our system prompt on the shared LLM before each request
        if hasattr(self, '_bot_system_prompt') and self._bot_system_prompt:
            self.llm.system_prompt = self._bot_system_prompt
        response = self._run_coro_sync(
            self.llm.chat([{"role": "user", "content": prompt}], bot_name=self.user)
        )
        return response.content

    def _run_coro_sync(self, coroutine):
        """Run an async coroutine from sync code safely.

        This handles the case where we're called from:
        1. The main thread (no running loop) - use asyncio.run()
        2. A different thread (event loop running) - use run_coroutine_threadsafe()
        3. The event loop thread itself - CANNOT block, must use a different approach
        """
        # Check if we're in the event loop thread - if so, we can't block
        if hasattr(self, '_event_loop') and self._event_loop.is_running():
            current_thread = threading.current_thread()
            if hasattr(self, '_event_loop_thread') and current_thread == self._event_loop_thread:
                # We're in the event loop thread - can't block here!
                # Schedule on a thread pool and wait
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, coroutine)
                    return future.result(timeout=35.0)
            else:
                # Different thread, use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(coroutine, self._event_loop)
                return future.result(timeout=35.0)

        # No event loop or not running - just run directly
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coroutine, loop)
                return future.result(timeout=35.0)
            return loop.run_until_complete(coroutine)
        except RuntimeError:
            return asyncio.run(coroutine)

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

        # MULTI-BOT: Ignore ALL audio when ANY bot is speaking
        # Other bots receive speech via broadcast_utterance (fake ASR), not real audio
        # This prevents double-processing: real ASR + fake ASR on same speech
        if self._shared_services.any_bot_speaking():
            return  # Bot is talking - ignore audio, we'll get text via fake ASR

        # CRITICAL: Ignore all audio while bot is speaking to prevent feedback
        # When bot speaks, users' microphones pick up the audio and send it back
        # This causes the bot to transcribe its own TTS output as user speech
        if self._speaking.is_set():
            # Barge-in detection: if enabled and user speaks VERY loudly while bot is speaking,
            # trigger interruption via TurnController (sets is_cancelled flag)
            # IMPORTANT: In multi-bot mode, suppress barge-in entirely when ANY bot is speaking
            # (including ourselves). The loud audio is likely bot TTS, not a human interrupting.
            # This prevents: 1) self-interruption from our own echo, 2) bots interrupting each other
            if self.barge_in_enabled:
                # Any bot speaking = ignore (no valid barge-in target)
                if self._shared_services.any_bot_speaking():
                    return  # Bot audio - not a human interruption
                # Allow barge-in from humans
                rms = pcm_rms(sound_chunk.pcm)
                barge_in_threshold = self.asr_threshold * 3  # Much higher threshold for barge-in
                if self.turn_controller and rms > barge_in_threshold:
                    if self.turn_controller.request_barge_in():
                        self.logger.info(f"Barge-in triggered by {user_name} (RMS={rms} > {barge_in_threshold})")
            return  # Don't buffer audio while speaking

        rms = pcm_rms(sound_chunk.pcm)
        self._max_rms = max(rms, self._max_rms)

        # Track unique users sending audio (for debugging)
        if not hasattr(self, '_audio_senders'):
            self._audio_senders = {}
        if user_id not in self._audio_senders:
            self._audio_senders[user_id] = user_name
            self.logger.info(f"First audio from {user_name} (session={user_id}, RMS={rms})")

        # Debug display
        if self.debug_rms:
            bar_width = min(rms // 100, 50)
            threshold_pos = min(self.asr_threshold // 100, 50)
            bar = '-' * threshold_pos + '+' * max(0, bar_width - threshold_pos) if rms >= self.asr_threshold else '-' * bar_width
            print(f'\r[{user_name:12}] RMS: {rms:5d} / {self._max_rms:5d}  |{bar:<50}|', end='', flush=True)

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
        self.logger.debug(f"ASR transcribing {buffer_duration:.1f}s", extra={"user": user_name, "duration_ms": buffer_duration * 1000})
        start_time = time.time()

        # Mark ASR start in tracker
        if tracker:
            tracker.asr_start()

        try:
            # Convert float32 to int16 PCM bytes for STT
            audio_16k_int16 = (audio_16k * 32767).astype(np.int16)
            pcm_16k_bytes = audio_16k_int16.tobytes()

            # Use NeMo Nemotron STT (synchronous transcription)
            stt_result = asyncio.run(self.streaming_stt.transcribe(
                audio_data=pcm_16k_bytes,
                sample_rate=16000,
                sample_width=2,
                channels=1,
                language="en",
            ))
            text = stt_result.text.strip()

            transcribe_time = time.time() - start_time

            # Record ASR stats
            self._record_asr_stat(transcribe_time * 1000)

            # Mark ASR complete in tracker
            if tracker:
                tracker.asr_final(text)

            if not text or len(text) < 2:
                self._maybe_respond(user_id, user_name, tracker=tracker)
                return

            # Log the full transcription with timing
            self.logger.info(f'ASR ({transcribe_time*1000:.0f}ms): "{text}" [from {user_name}, {buffer_duration:.1f}s audio]')
            if transcribe_time > 2.0:
                self.logger.warning(f"ASR slow: {transcribe_time*1000:.0f}ms (>2s)")

            # Record channel activity (for quiet timer) and check for long speech
            self._record_channel_activity()
            self._check_long_speech(user_id, user_name, buffer_duration)

            # --- Speech Filtering ---

            # Echo filter: check if this is the bot's own speech being picked up
            if self.echo_filter and self.echo_filter.is_echo(text):
                self.logger.debug(f"Echo filter: ignoring '{text}' (matches recent bot output)")
                return

            # Utterance classifier: check if this is meaningful speech
            if self.utterance_classifier and not self.utterance_classifier.is_meaningful(text):
                self.logger.debug(f"Utterance filter: ignoring '{text}' (not meaningful)")
                return

            # --- End Speech Filtering ---

            # NOTE: Don't log to journal here - _add_to_history will do it
            # when _generate_response is called. This avoids duplicate entries.

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
            else:
                # Continuous transmit: respond after accumulating enough text
                # This handles users with always-on mic / continuous transmit
                accumulated_text = self.pending_text.get(user_id, "")
                word_count = len(accumulated_text.split())
                if word_count >= 10:  # Respond after ~10 words even during continuous speech
                    logger.info(f"Continuous transmit: responding after {word_count} words")
                    self._maybe_respond(user_id, user_name, force=True, tracker=tracker)

        except Exception as e:
            logger.error(f"Speech processing failed: {e}", exc_info=True)

    def _on_bot_utterance(self, speaker_name: str, text: str) -> None:
        """Handle an utterance broadcast from another bot.

        The utterance is already logged to the shared journal by the speaking
        bot via broadcast_utterance(), so we have context awareness.

        Whether we RESPOND depends on the soul's talks_to_bots setting.
        Default is False to prevent infinite bot-to-bot loops.

        Args:
            speaker_name: Name of the bot that spoke.
            text: What they said.
        """
        # Don't process our own utterances
        if speaker_name == self.user:
            return

        # Ignore very short utterances (likely fillers that shouldn't trigger responses)
        # Strip dots/ellipsis since TTS pads text with periods
        clean_text = text.strip().rstrip('.')
        word_count = len(clean_text.split())
        if word_count < 3:
            self.logger.debug(f"[BOT-HEARD] {self.user} ignoring short utterance from {speaker_name}: '{text}' ({word_count} words)")
            return

        self.logger.info(f"[BOT-HEARD] {self.user} heard {speaker_name}: '{text[:50]}...'" if len(text) > 50 else f"[BOT-HEARD] {self.user} heard {speaker_name}: '{text}'")

        # Check if this bot is configured to talk to other bots
        talks_to_bots = False
        if self.soul_config and hasattr(self.soul_config, 'talks_to_bots'):
            talks_to_bots = self.soul_config.talks_to_bots

        if not talks_to_bots:
            # Just observe for context, don't respond
            self.logger.info(f"[BOT-HEARD] {self.user} NOT responding (talks_to_bots=False)")
            return

        self.logger.info(f"[BOT-HEARD] {self.user} WILL respond (talks_to_bots=True)")

        # Look up the speaker's Mumble session ID (treat bots as real users)
        user_id = self._get_session_id_by_name(speaker_name)

        # Try to claim this response - only one bot should respond
        if not self._shared_services.try_claim_response(user_id, text):
            self.logger.debug(f"Someone else responding to bot: {text[:30]}...")
            return

        # Schedule response in a background thread that waits for speaker to finish
        def _respond_after_speaker_done():
            # Wait until all bots are done speaking (with timeout)
            max_wait = 60  # seconds
            poll_interval = 0.1  # seconds
            waited = 0
            while self._shared_services.any_bot_speaking() and waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval

            # Don't respond if we started speaking while waiting
            if self._speaking.is_set():
                self.logger.info(f"[BOT-HEARD] {self.user} already speaking, skipping response")
                return

            # The bot's message is ALREADY in the journal (from broadcast_utterance).
            # We just need to generate a response - don't add anything to history.
            self.logger.info(f'Generating response for {speaker_name}: "{text}"')

            # Generate response directly - the journal already has the context
            if self.llm:
                # Set our system prompt on the shared LLM
                if hasattr(self, '_bot_system_prompt') and self._bot_system_prompt:
                    self.llm.system_prompt = self._bot_system_prompt

                # Build messages from journal (bot's message is already there)
                messages = self._build_llm_messages()

                # Generate response
                response = self._run_coro_sync(
                    self.llm.chat(messages, bot_name=self.user)
                )

                if response.content:
                    # Speak the response (this will also broadcast it to other bots)
                    self._speak_sync(response.content, self.voice_prompt)

        t = threading.Thread(target=_respond_after_speaker_done, daemon=True)
        t.start()

    def _get_session_id_by_name(self, name: str) -> int:
        """Look up a user's Mumble session ID by their name.

        Args:
            name: The user/bot name to look up.

        Returns:
            Session ID if found, or a consistent positive hash if not found.
        """
        if hasattr(self, 'mumble') and self.mumble and hasattr(self.mumble, 'users'):
            for user in self.mumble.users.values():
                if user.get('name') == name:
                    return user['session']
        # Fallback: consistent positive hash (e.g., bot not yet connected)
        return hash(name) % 1000000

    def _maybe_respond(self, user_id: int, user_name: str, force: bool = False, tracker: 'LatencyTracker' = None):
        """Respond if we have pending text and enough time has passed.

        Args:
            user_id: ID of the user/bot speaking.
            user_name: Name of the speaker.
            force: Force immediate response.
            tracker: Latency tracker.
        """
        if user_id not in self.pending_text:
            return

        current_time = time.time()
        time_since_last = current_time - self.pending_text_time.get(user_id, 0)

        # Use turn predictor if available to determine response timing
        accumulated_text = self.pending_text.get(user_id, "")
        if self.turn_predictor and not force:
            if not self.turn_predictor.should_respond(accumulated_text, time_since_last):
                logger.debug("Turn predictor: waiting for turn completion")
                return

        # Respond if forced or if enough time has passed
        if force or time_since_last >= self.pending_text_timeout:
            text = self.pending_text.pop(user_id, "")
            speech_end_time = self.pending_text_time.pop(user_id, current_time)

            if text and self.llm:
                # Transition state machine to THINKING
                if self.conversation_state_machine:
                    # First ensure we're in LISTENING state
                    if self.conversation_state_machine.state == ConversationState.IDLE:
                        self.conversation_state_machine.transition_sync(
                            ConversationState.LISTENING,
                            reason="user speech detected",
                        )
                    self.conversation_state_machine.transition_sync(
                        ConversationState.THINKING,
                        reason="generating response",
                    )

                # Track when user finished speaking for staleness detection
                pipeline_start = time.time()

                # Get a new turn ID for this user utterance (for stale response dropping)
                turn_id = None
                if self._turn_coordinator:
                    turn_id = self._turn_coordinator.new_turn(str(user_id))

                # Mark LLM start in tracker
                if tracker:
                    tracker.llm_start()

                # Try to claim this response
                # Only one responder should reply to each utterance (natural turn-taking)
                if not self._shared_services.try_claim_response(user_id, text):
                    self.logger.debug(f"Someone else responding to: {text[:30]}...")
                    return

                self.logger.info(f'Generating response for {user_name}: "{text}"')

                # Check for first-time speaker event
                if user_name and self._check_first_time_speaker(user_name):
                    # Trigger first speech event - this speaks a greeting
                    if self._trigger_event('user_first_speech', user_name):
                        # If we spoke a greeting, add a small pause before continuing
                        import time as time_module
                        time_module.sleep(0.3)

                # Speak a thinking filler if this looks like a question
                # This fills the gap while LLM processes and feels more natural
                if self._is_question(text):
                    self.logger.debug("Detected question - speaking thinking filler")
                    self._trigger_event('thinking', user_name) or self._speak_filler('thinking')

                # Start "still thinking" timer in case LLM is slow
                self._llm_thinking_since = time.time()
                self._start_still_thinking_timer(timeout_sec=5.0)

                llm_start = time.time()

                try:
                    response = self._generate_response_sync(user_id, text, user_name)
                    llm_time = time.time() - llm_start

                    # Cancel "still thinking" timer - we got the response
                    self._cancel_still_thinking_timer()

                    # Record LLM stats
                    self._record_llm_stat(llm_time * 1000)

                    # Mark LLM complete in tracker
                    if tracker:
                        tracker.llm_complete(response)

                    # Check for staleness via turn ID (preferred) or pending_text (fallback)
                    if self._turn_coordinator and turn_id:
                        if self._turn_coordinator.is_stale(str(user_id), turn_id):
                            self.logger.info("Pipeline abort: turn is stale (user started new turn)")
                            return
                    elif user_id in self.pending_text:
                        self.logger.info("Pipeline abort: user spoke again", extra={"latency_ms": llm_time * 1000})
                        return

                    # Check if cancelled by barge-in
                    if self.turn_controller and self.turn_controller.is_cancelled():
                        self.logger.info("Pipeline abort: barge-in")
                        return

                    # Check total latency - if too slow, warn
                    total_latency = time.time() - speech_end_time
                    if total_latency > 3.0:
                        self.logger.warning(f"High latency: {total_latency:.1f}s since user stopped", extra={"latency_ms": total_latency * 1000})

                    self.logger.info(f'Queuing TTS: "{response}" (LLM took {llm_time*1000:.0f}ms)')

                    # Queue TTS with timing metadata, tracker, and turn ID
                    self._queue_tts(response, self.voice_prompt, pipeline_start, user_id, tracker, turn_id)

                except Exception as e:
                    self._cancel_still_thinking_timer()
                    logger.error(f"LLM error: {e}", exc_info=True)

    def _queue_tts(self, text: str, voice_prompt: dict, pipeline_start: float, user_id: int, tracker=None, turn_id: int = None):
        """Queue text for TTS synthesis.

        Args:
            text: Text to synthesize.
            voice_prompt: Voice embedding for TTS.
            pipeline_start: Timestamp when pipeline started.
            user_id: User who triggered the response.
            tracker: LatencyTracker for this turn.
            turn_id: Turn ID for staleness checking.
        """
        # Sanitize text: remove emojis and non-speakable characters
        text = _sanitize_for_tts(text)
        if not text.strip():
            self.logger.warning("TTS text empty after sanitization, skipping")
            return

        if PERF_AVAILABLE and turn_id is not None:
            # Use TTSQueueItem for better staleness checking
            item = TTSQueueItem(
                user_id=str(user_id),
                turn_id=turn_id,
                text=text,
                chunk_index=0,
            )
            # Attach extra metadata as attributes
            item.voice_prompt = voice_prompt
            item.pipeline_start = pipeline_start
            item.tracker = tracker
            self._tts_queue.put(item)
        else:
            # Fallback to tuple format
            self._tts_queue.put((text, voice_prompt, pipeline_start, user_id, tracker))

    # =========================================================================
    # TTS
    # =========================================================================

    def _tts_worker(self):
        """Background worker for TTS."""
        self.logger.info("TTS worker started")
        while not self._shutdown.is_set():
            try:
                item = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # Handle None from BoundedTTSQueue timeout
            if item is None:
                continue

            # Unpack item - either TTSQueueItem (new) or tuple (legacy)
            tracker = None
            turn_id = None
            if hasattr(item, 'text'):
                # New TTSQueueItem format
                text = item.text
                voice_prompt = getattr(item, 'voice_prompt', self.voice_prompt)
                pipeline_start = getattr(item, 'pipeline_start', None)
                user_id = item.user_id
                turn_id = item.turn_id
                tracker = getattr(item, 'tracker', None)
            elif isinstance(item, tuple):
                # Legacy tuple format
                if len(item) == 5:
                    text, voice_prompt, pipeline_start, user_id, tracker = item
                elif len(item) == 4:
                    text, voice_prompt, pipeline_start, user_id = item
                else:
                    text, voice_prompt = item
                    pipeline_start = None
                    user_id = None
            else:
                self.logger.warning(f"Unknown TTS queue item type: {type(item)}")
                continue

            try:
                # Check staleness via turn ID (preferred) or pending_text (fallback)
                if self._turn_coordinator and turn_id is not None and user_id is not None:
                    if self._turn_coordinator.is_stale(str(user_id), turn_id):
                        self.logger.info("[TTS] Skipping stale response - turn superseded")
                        continue
                elif user_id is not None and user_id in self.pending_text:
                    print("[TTS] Skipping stale response - user spoke again")
                    continue

                # Check if cancelled by barge-in
                if self.turn_controller and self.turn_controller.is_cancelled():
                    print("[TTS] Skipping - cancelled by barge-in")
                    # Reset turn controller so future TTS works
                    self.turn_controller.reset()
                    continue

                # Check if response is too old
                if pipeline_start and (time.time() - pipeline_start) > self.max_response_staleness:
                    latency = time.time() - pipeline_start
                    print(f"[TTS] Skipping stale response ({latency:.1f}s old, limit={self.max_response_staleness}s)")
                    continue

                self._speak_sync(text, voice_prompt, pipeline_start, tracker)
            except Exception as e:
                self.logger.error(f"TTS worker error: {e}", exc_info=True)
            finally:
                # task_done only exists on standard Queue, not BoundedTTSQueue
                if hasattr(self._tts_queue, 'task_done'):
                    self._tts_queue.task_done()

        print("[TTS] Worker stopped")

    def _speak_sync(self, text: str, voice_prompt: dict, pipeline_start: float = None, tracker: 'LatencyTracker' = None, skip_broadcast: bool = False):
        """Generate and play speech.

        Args:
            text: Text to speak.
            voice_prompt: Voice prompt for TTS.
            pipeline_start: Pipeline start time for latency tracking.
            tracker: Latency tracker.
            skip_broadcast: If True, don't broadcast to other bots. Use for fillers
                           and greetings that shouldn't trigger other bots to respond.
        """
        text = _pad_tts_text(text)
        if not text:
            return

        # Record this output for echo filtering BEFORE speaking
        # This way we'll recognize it if it's picked up by STT during/after playback
        if self.echo_filter:
            self.echo_filter.add_output(text)

        # Set speaking flags BEFORE broadcast so other bots know we're talking
        # (they check any_bot_speaking() to avoid responding while we speak)
        self._speaking.set()
        self._shared_services.bot_started_speaking()

        # Broadcast utterance to other bots (they receive it as "fake ASR")
        # This lets bots hear each other without actual audio processing
        # SKIP broadcast for fillers/greetings that shouldn't trigger responses
        if not skip_broadcast:
            self._shared_services.broadcast_utterance(self.user, text)
        else:
            self.logger.debug(f"[BROADCAST] Skipping broadcast for filler/event: '{text[:40]}...')")

        # Update conversation state machine
        if self.conversation_state_machine:
            self.conversation_state_machine.transition_sync(
                ConversationState.SPEAKING,
                reason="starting TTS",
            )

        # Clear all audio buffers and pending text when starting to speak
        # This prevents processing stale audio that was buffered before we started speaking
        # and also prevents feedback loops from microphones picking up our TTS output
        for user_id in list(self.audio_buffers.keys()):
            self.audio_buffers[user_id] = []
            self.speech_active_until[user_id] = 0
        # Clear pending text to avoid responding to feedback from our own TTS
        self.pending_text.clear()
        self.pending_text_time.clear()

        tts_start = time.time()

        # Mark TTS start in tracker
        if tracker:
            tracker.tts_start()

        # Update turn controller
        if self.turn_controller:
            self.turn_controller.start_speaking()

        try:
            logger.info(f'TTS generating: \"{text[:80]}...\"' if len(text) > 80 else f'TTS generating: \"{text}\"')

            first_chunk = True
            total_audio_samples = 0
            chunk_count = 0

            for wav_chunk in self.tts.generate_speech_streaming(
                text, voice_prompt, num_steps=self.num_steps
            ):
                chunk_count += 1

                # Check for barge-in cancellation
                if self.turn_controller and self.turn_controller.is_cancelled():
                    logger.info("TTS cancelled due to barge-in")
                    break

                if first_chunk:
                    tts_first_chunk = time.time() - tts_start
                    if pipeline_start:
                        total_latency = time.time() - pipeline_start
                        logger.info(f"TTS first audio: {tts_first_chunk*1000:.0f}ms, pipeline total: {total_latency*1000:.0f}ms")
                        if total_latency > 2.0:
                            logger.warning(f"Pipeline slow: {total_latency*1000:.0f}ms (>2s) to first audio")
                    # Mark TTS first audio in tracker
                    if tracker:
                        tracker.tts_first_audio()
                        tracker.playback_start()
                    first_chunk = False

                wav_float = wav_chunk.numpy().squeeze()
                wav_float = np.clip(wav_float, -1.0, 1.0)
                pcm = (wav_float * 32767).astype(np.int16)
                chunk_samples = len(pcm)
                total_audio_samples += chunk_samples
                self.mumble.sound_output.add_sound(pcm.tobytes())

                # Wait for most of the audio to play before generating next chunk
                # This creates natural pauses between sentences and prevents buffer overflow
                # LuxTTS outputs 48kHz audio
                chunk_duration_sec = chunk_samples / 48000
                # Wait for 90% of audio duration to create natural pacing
                # Plus a small fixed delay for sentence pauses
                wait_time = chunk_duration_sec * 0.9 + 0.15  # Extra 150ms between chunks
                if wait_time > 0.1:
                    time.sleep(wait_time)

            tts_total = time.time() - tts_start

            # Record TTS stats
            self._record_tts_stat(tts_total * 1000)

            # Calculate audio duration (48kHz sample rate for LuxTTS output)
            audio_duration_ms = (total_audio_samples / 48000) * 1000

            # Mark playback end and finalize tracker
            if tracker:
                tracker.playback_end(audio_duration_ms)
                tracker.finalize()

            if pipeline_start:
                pipeline_total = time.time() - pipeline_start
                self.logger.info(f"TTS complete: {tts_total*1000:.0f}ms synthesis, {audio_duration_ms:.0f}ms audio, pipeline total: {pipeline_total*1000:.0f}ms")

        finally:
            # Brief delay after synthesis before clearing _speaking flag
            # This helps prevent feedback from tail-end of audio playback
            # Network latency can cause echo to arrive late, so we wait a bit longer
            time.sleep(0.5)

            # Clear any audio that accumulated during TTS playback
            for user_id in list(self.audio_buffers.keys()):
                self.audio_buffers[user_id] = []
                self.speech_active_until[user_id] = 0
            self.pending_text.clear()
            self.pending_text_time.clear()

            self._speaking.clear()
            # Notify shared services that this bot stopped speaking
            self._shared_services.bot_stopped_speaking()
            # Reset turn controller to idle
            if self.turn_controller:
                self.turn_controller.reset()

            # Transition state machine to COOLDOWN then back to IDLE/LISTENING
            if self.conversation_state_machine:
                self.conversation_state_machine.transition_sync(
                    ConversationState.COOLDOWN,
                    reason="TTS finished",
                )
                # After cooldown, go back to IDLE
                self.conversation_state_machine.transition_sync(
                    ConversationState.IDLE,
                    reason="cooldown complete",
                )

    def _stats_logger(self):
        """Background thread that logs stats periodically."""
        logger.info(f"Stats logger started (interval: {self._stats_interval}s)")
        while not self._shutdown.is_set():
            time.sleep(self._stats_interval)
            if self._shutdown.is_set():
                break

            with self._stats_lock:
                asr_count = self._asr_count
                asr_avg = self._asr_total_ms / asr_count if asr_count > 0 else 0
                llm_count = self._llm_count
                llm_avg = self._llm_total_ms / llm_count if llm_count > 0 else 0
                tts_count = self._tts_count
                tts_avg = self._tts_total_ms / tts_count if tts_count > 0 else 0

            tts_queue_size = self._tts_queue.qsize()
            audio_buffers = len(self.audio_buffers)
            pending_text = len(self.pending_text)
            speaking = self._speaking.is_set()

            # Build basic stats line
            stats_line = (
                f"[Stats] ASR: {asr_count} ({asr_avg:.0f}ms avg) | "
                f"LLM: {llm_count} ({llm_avg:.0f}ms avg) | "
                f"TTS: {tts_count} ({tts_avg:.0f}ms avg) | "
                f"Queues: TTS={tts_queue_size}, AudioBuf={audio_buffers}, Pending={pending_text} | "
                f"Speaking: {speaking}"
            )

            # Add rolling latency percentiles if available
            if self._rolling_latency:
                stats = self._rolling_latency.get_stats()
                if stats:
                    perf_parts = []
                    for category, values in stats.items():
                        if values['count'] > 0:
                            perf_parts.append(
                                f"{category}: p50={values['p50']:.0f}ms p95={values['p95']:.0f}ms"
                            )
                    if perf_parts:
                        stats_line += f" | {' | '.join(perf_parts)}"

            # Add queue drop count if available
            if hasattr(self._tts_queue, 'drop_count'):
                drop_count = self._tts_queue.drop_count
                if drop_count > 0:
                    stats_line += f" | Drops: {drop_count}"

            logger.info(stats_line)
        logger.info("Stats logger stopped")

    def _record_asr_stat(self, duration_ms: float):
        """Record ASR processing time for stats."""
        with self._stats_lock:
            self._asr_count += 1
            self._asr_total_ms += duration_ms
        # Also record to rolling tracker
        if self._rolling_latency:
            self._rolling_latency.record("asr", duration_ms)

    def _record_llm_stat(self, duration_ms: float):
        """Record LLM processing time for stats."""
        with self._stats_lock:
            self._llm_count += 1
            self._llm_total_ms += duration_ms
        # Also record to rolling tracker
        if self._rolling_latency:
            self._rolling_latency.record("llm", duration_ms)

    def _record_tts_stat(self, duration_ms: float):
        """Record TTS processing time for stats."""
        with self._stats_lock:
            self._tts_count += 1
            self._tts_total_ms += duration_ms
        # Also record to rolling tracker
        if self._rolling_latency:
            self._rolling_latency.record("tts", duration_ms)

    # =========================================================================
    # Conversational Fillers (thinking, interruption acknowledgment)
    # =========================================================================

    def _get_filler(self, filler_type: str) -> str | None:
        """Get a random filler phrase — delegates to EventResponder."""
        if not hasattr(self, '_event_responder'):
            self._event_responder = EventResponder(self.soul_config)
        return self._event_responder.get_filler(filler_type)

    def _get_event_response(self, event_type: str, user: str = None) -> str | None:
        """Get a random event response — delegates to EventResponder."""
        if not hasattr(self, '_event_responder'):
            self._event_responder = EventResponder(self.soul_config)
        return self._event_responder.get_event_response(event_type, user)

    def _trigger_event(self, event_type: str, user: str = None) -> bool:
        """Trigger an event and speak the response if configured.

        Args:
            event_type: Event name (e.g., 'user_first_speech', 'interrupted')
            user: Username for placeholder substitution.

        Returns:
            True if event was handled (response spoken), False otherwise.
        """
        response = self._get_event_response(event_type, user)
        if not response:
            self.logger.debug(f"[EVENT] {event_type} - no response configured")
            return False

        self.logger.info(f"[EVENT] {event_type} for {user or 'unknown'} - speaking: '{response}'")

        # Speak directly via TTS (bypasses LLM)
        # skip_broadcast=True because events are quick utterances (greetings, fillers)
        # that shouldn't trigger other bots to respond
        try:
            self._speak_sync(response, self.voice_prompt, None, None, skip_broadcast=True)
            return True
        except Exception as e:
            self.logger.warning(f"[EVENT] Failed to speak event response: {e}")
            return False

    def _record_channel_activity(self):
        """Record that activity occurred in the channel (resets quiet timer)."""
        if hasattr(self, '_activity_tracker'):
            self._activity_tracker.record_activity()
        else:
            self._last_channel_activity = time.time()
            self._quiet_event_triggered = False

    def _check_channel_quiet(self):
        """Check if channel has been quiet and trigger event if so."""
        if hasattr(self, '_activity_tracker'):
            if self._activity_tracker.check_channel_quiet():
                self._trigger_event('channel_quiet')
        else:
            if self._quiet_event_triggered:
                return
            time_since_activity = time.time() - self._last_channel_activity
            if time_since_activity >= self._quiet_threshold:
                self._quiet_event_triggered = True
                self.logger.info(f"[EVENT] Channel quiet for {time_since_activity:.0f}s, triggering event")
                self._trigger_event('channel_quiet')

    def _start_quiet_timer(self):
        """Start background thread to check for channel quiet."""
        def quiet_check_loop():
            while self._running:
                time.sleep(10)  # Check every 10 seconds
                if self._running:
                    self._check_channel_quiet()

        self._quiet_timer = threading.Thread(target=quiet_check_loop, daemon=True)
        self._quiet_timer.start()

    def _check_long_speech(self, user_id: int, user_name: str, speech_duration: float):
        """Track speech duration — delegates to ChannelActivityTracker."""
        if hasattr(self, '_activity_tracker'):
            if self._activity_tracker.check_long_speech(user_id, user_name, speech_duration):
                self._trigger_event('long_speech_ended', user_name)
        else:
            if user_id not in self.user_total_speech_time:
                self.user_total_speech_time[user_id] = 0.0
            self.user_total_speech_time[user_id] += speech_duration
            total = self.user_total_speech_time[user_id]
            if total >= self.long_speech_threshold:
                self._trigger_event('long_speech_ended', user_name)
                self.user_total_speech_time[user_id] = 0.0

    def _reset_speech_tracking(self, user_id: int):
        """Reset speech tracking for a user."""
        if hasattr(self, '_activity_tracker'):
            self._activity_tracker.reset_speech_tracking(user_id)
        else:
            self.user_total_speech_time.pop(user_id, None)

    def _check_first_time_speaker(self, user_name: str) -> bool:
        """Check if this is the first time we've heard from this user."""
        if hasattr(self, '_activity_tracker'):
            return self._activity_tracker.check_first_time_speaker(user_name)
        if user_name in self._greeted_users:
            return False
        self._greeted_users.add(user_name)
        return True

    def _is_question(self, text: str) -> bool:
        """Detect if text is likely a question — delegates to text_processing."""
        return _is_question_heuristic(text)

    def _speak_filler(self, filler_type: str):
        """Speak a filler phrase immediately (bypasses queue).

        Args:
            filler_type: One of 'thinking', 'still_thinking', 'interrupted'
        """
        filler = self._get_filler(filler_type)
        if not filler:
            self.logger.info(f"[FILLER] No filler available for type: {filler_type}")
            return

        self.logger.info(f"[FILLER] Speaking ({filler_type}): '{filler}'")

        # Speak synchronously with skip_broadcast=True so other bots don't
        # try to respond to "Hmm..." or "Let me think..."
        try:
            self._speak_sync(filler, self.voice_prompt, None, None, skip_broadcast=True)
        except Exception as e:
            self.logger.warning(f"[FILLER] Failed to speak filler: {e}")

    def _on_barge_in(self):
        """Called when user interrupts the bot (barge-in callback).

        Speaks a brief acknowledgment like "Oh, sorry" and stops speaking.
        """
        self.logger.info("[BARGE-IN] User interrupted bot - stopping speech")

        # Cancel any "still thinking" timer
        if self._still_thinking_timer:
            self._still_thinking_timer.cancel()
            self._still_thinking_timer = None

        # Try to get an interruption response from events config
        response = self._get_event_response('interrupted')
        if response:
            # We could speak a quick acknowledgment after we stop
            # But speaking while being interrupted is weird - log for now
            self.logger.info(f"[BARGE-IN] Would have said: '{response}' (suppressed - being interrupted)")

        # Clear the speaking flag (handled by turn controller)

    def _start_still_thinking_timer(self, timeout_sec: float = 5.0):
        """Start a timer to speak 'still thinking' if LLM is slow.

        Args:
            timeout_sec: Seconds before triggering the filler.
        """
        # Cancel any existing timer
        if self._still_thinking_timer:
            self._still_thinking_timer.cancel()

        self.logger.debug(f"[TIMER] Starting 'still thinking' timer ({timeout_sec}s)")

        def _on_still_thinking():
            # Only fire if we're still waiting for LLM
            if self._llm_thinking_since is not None:
                self.logger.info("[TIMER] LLM taking long - speaking 'still thinking' filler")
                self._speak_filler('still_thinking')

        self._still_thinking_timer = threading.Timer(timeout_sec, _on_still_thinking)
        self._still_thinking_timer.daemon = True
        self._still_thinking_timer.start()

    def _cancel_still_thinking_timer(self):
        """Cancel the 'still thinking' timer."""
        if self._still_thinking_timer:
            self._still_thinking_timer.cancel()
            self._still_thinking_timer = None
        self._llm_thinking_since = None

    def speak(self, text: str, blocking: bool = False):
        """Queue text to be spoken."""
        if blocking:
            self._speak_sync(text, self.voice_prompt, time.time(), None)
        else:
            # Use the new queue helper for consistent handling
            self._queue_tts(text, self.voice_prompt, time.time(), None, None, None)

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
            logger.error(f"Text response generation failed: {e}", exc_info=True)

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
                logger.error(f"Error handling user update: {e}", exc_info=True)

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
                logger.error(f"Greeting generation failed: {e}", exc_info=True)
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
        self._running = True
        self.mumble.start()
        self.mumble.is_ready()
        print("[Mumble] Connected!")

        if self.channel:
            try:
                channel = self.mumble.channels.find_by_name(self.channel)
                channel.move_in()
                print(f"[Mumble] Joined channel: {self.channel}")
            except Exception as e:
                logger.error(f"Failed to join channel '{self.channel}': {e}", exc_info=True)

        # Start quiet timer for channel_quiet events
        self._start_quiet_timer()

    def run_forever(self):
        """Keep the bot running."""
        print("[Bot] Running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Bot] Shutting down...")
            self._running = False
            self._shutdown.set()
            self._tts_queue.join()
            self._asr_executor.shutdown(wait=True)
            # Stop the event loop
            if hasattr(self, '_event_loop') and self._event_loop.is_running():
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)


def run_multi_persona_bot(args):
    """Run the bot in multi-persona mode.

    Loads a multi-persona config and creates multiple MumbleVoiceBot instances
    that share TTS/STT/LLM resources.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        config = load_multi_persona_config(args.config)
        logger.info(f"Loaded multi-persona config with {len(config.personas)} personas")
        for p in config.personas:
            logger.info(f"  - {p.identity.name} ({p.identity.display_name})")
    except Exception as e:
        logger.error(f"Failed to load multi-persona config: {e}")
        print(f"\n❌ Failed to load multi-persona config:\n{e}", file=sys.stderr)
        sys.exit(1)

    # Apply HF_HOME if specified
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        logger.info(f"HF_HOME={args.hf_home}")

    # Determine device
    if args.device != 'auto':
        device = args.device
    else:
        device = get_best_device()

    logger.info(f"Using device: {device}")

    # Get shared config
    shared_config = config.shared or {}
    stt_config = shared_config.get("stt", {})
    llm_config = shared_config.get("llm", {})

    # Create shared services
    print("[Multi-Persona] Creating shared services...")
    shared = create_shared_services(
        device=device,
        nemotron_model=stt_config.get("nemotron_model"),
        nemotron_chunk_ms=stt_config.get("nemotron_chunk_ms", 160),
        nemotron_device=stt_config.get("nemotron_device"),
        llm_endpoint=llm_config.get("endpoint"),
        llm_model=llm_config.get("model"),
        llm_api_key=llm_config.get("api_key"),
        llm_timeout=llm_config.get("timeout", 30.0),
        llm_max_tokens=llm_config.get("max_tokens"),
        llm_temperature=llm_config.get("temperature"),
    )

    # Create bot instances - one per persona
    bots = []
    print(f"[Multi-Persona] Creating {len(config.personas)} bot instances...")

    for persona_config in config.personas:
        identity = persona_config.identity

        # Get Mumble connection info
        mumble_user = identity.mumble_user or identity.display_name or identity.name
        mumble_channel = persona_config.mumble.get("channel") if persona_config.mumble else None

        # Check if this is a parrot persona (no soul, special type)
        is_parrot = identity.name.lower() == "parrot" and persona_config.soul_config is None
        
        if is_parrot:
            # Create ParrotBot instead of MumbleVoiceBot
            if not PARROT_BOT_AVAILABLE:
                print(f"  ERROR: ParrotBot not available for persona '{identity.name}'")
                continue
            
            print(f"  Creating ParrotBot: {identity.name} as '{mumble_user}'")
            bot = ParrotBot(
                host=config.mumble_host or "localhost",
                port=config.mumble_port or 64738,
                user=mumble_user,
                password=config.mumble_password or "",
                channel=mumble_channel,
                device=device,
                shared_tts=shared.tts,
                shared_stt=shared.stt,
            )
            bots.append(bot)
            continue

        # Determine ref_audio path: tts config > soul voice config
        tts_config = persona_config.tts or {}
        ref_audio = tts_config.get("ref_audio")

        if not ref_audio and persona_config.soul_config:
            # Get from soul config
            soul = persona_config.soul_config
            soul_audio_dir = os.path.join(_THIS_DIR, "souls", soul.name, "audio")
            if os.path.isdir(soul_audio_dir):
                # Look for any .wav file in the soul's audio directory
                for f in os.listdir(soul_audio_dir):
                    if f.endswith(".wav"):
                        ref_audio = os.path.join(soul_audio_dir, f)
                        print(f"  Using soul voice: {ref_audio}")
                        break

        # Load voice prompt for this persona
        voice_prompt = None
        if ref_audio and os.path.exists(ref_audio):
            voice_prompt = shared.load_voice(
                name=identity.name,
                audio_path=ref_audio,
                voices_dir="voices",
            )
        else:
            print(f"  Warning: No voice reference for {identity.name}")

        # System prompt was already loaded in config
        system_prompt = identity.system_prompt or ""

        print(f"  Creating bot: {identity.name} as '{mumble_user}'")

        bot = MumbleVoiceBot(
            host=config.mumble_host or "localhost",
            port=config.mumble_port or 64738,
            user=mumble_user,
            password=config.mumble_password or "",
            channel=mumble_channel,
            device=device,
            reference_audio=ref_audio or "reference.wav",
            llm_system_prompt=system_prompt,
            soul_config=persona_config.soul_config,
            soul_name=identity.name,
            # Shared services
            shared_tts=shared.tts,
            shared_stt=shared.stt,
            shared_llm=shared.llm,
            voice_prompt=voice_prompt,
            shared_echo_filter=shared.echo_filter,
            shared_services=shared,  # Full shared services for speaking coordination
        )
        bots.append(bot)

    # Start all bots with staggered timing to avoid simultaneous greetings
    print(f"\n[Multi-Persona] Starting {len(bots)} bots...")
    for i, bot in enumerate(bots):
        bot.start()
        # Stagger startup to avoid simultaneous greetings (wait for first bot to greet)
        if i < len(bots) - 1:
            time.sleep(5)  # 5 seconds between bot joins

    # Print status
    print(f"\n✓ Multi-persona bot running with {len(bots)} bots:")
    for bot in bots:
        print(f"  - {bot.user}")
    print("\nPress Ctrl+C to stop.")

    # Run forever
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Multi-Persona] Shutting down...")
        for bot in bots:
            bot._shutdown.set()
        print("[Multi-Persona] Goodbye!")


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging first
    setup_logging(
        level=args.log_level,
        json_output=args.log_json,
        log_file=args.log_file,
    )

    # Check if this is a multi-persona config
    if args.config and MULTI_PERSONA_AVAILABLE and is_multi_persona_config(args.config):
        logger.info(f"Detected multi-persona config: {args.config}")
        run_multi_persona_bot(args)
        return  # Multi-persona mode handles its own run loop

    # Load config file if specified (single-persona mode)
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            logger.info(f"Config loaded from {args.config}")
        except ConfigValidationError as e:
            logger.error(f"Config validation failed: {e}")
            print(f"\n❌ Config validation failed:\n{e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading {args.config}: {e}")
            print(f"\n❌ Error loading {args.config}: {e}", file=sys.stderr)
            sys.exit(1)

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
    cfg = merge_config_with_args(args, config)

    # Determine device
    tts_device_config = cfg['tts_device_config']
    if args.device != 'auto':
        device = args.device
    elif tts_device_config != 'auto':
        device = tts_device_config
    else:
        device = get_best_device()

    # Ensure all models are downloaded before connecting to Mumble
    ensure_models_downloaded(device=device)

    bot = MumbleVoiceBot(
        host=cfg['host'],
        user=cfg['user'],
        port=cfg['port'],
        password=cfg['password'],
        channel=cfg['channel'],
        reference_audio=cfg['reference'],
        device=device,
        num_steps=cfg['steps'],
        asr_threshold=cfg['asr_threshold'],
        debug_rms=args.debug_rms,
        voices_dir=cfg['voices_dir'],
        llm_endpoint=cfg['llm_endpoint'],
        llm_model=cfg['llm_model'],
        llm_api_key=cfg['llm_api_key'],
        llm_system_prompt=cfg['llm_system_prompt'],
        personality=cfg['personality'],
        config_file=args.config,
        nemotron_model=cfg['nemotron_model'],
        nemotron_chunk_ms=cfg['nemotron_chunk_ms'],
        nemotron_device=cfg['nemotron_device'],
        max_response_staleness=cfg['max_response_staleness'],
        barge_in_enabled=cfg['barge_in_enabled'],
        soul_config=cfg['soul_config'],
        soul_name=cfg['soul_name'],
        tools_config=cfg['tools_config'],
    )

    bot.start()
    bot.run_forever()


if __name__ == '__main__':
    main()
