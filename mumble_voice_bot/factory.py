"""Factory functions for constructing bots from configuration.

This module is the single place that wires together:
- SharedBotServices (TTS, STT, LLM)
- Brain (LLM, Echo)
- MumbleBot

Entry points call these factories to construct the right bot from config.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mumble_voice_bot.coordination import SharedBotServices
from mumble_voice_bot.interfaces.brain import Brain, NullBrain
from mumble_voice_bot.logging_config import get_logger
from mumble_voice_bot.utils import get_best_device

logger = get_logger(__name__)

# Project root (parent of mumble_voice_bot/)
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_shared_services(
    device: str = "auto",
    nemotron_model: str | None = None,
    nemotron_chunk_ms: int = 160,
    nemotron_device: str | None = None,
    llm_endpoint: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout: float = 30.0,
    llm_max_tokens: int | None = None,
    llm_temperature: float | None = None,
) -> SharedBotServices:
    """Create shared services for one or more bots.

    Initializes TTS, STT, and LLM once so multiple bots can share them.

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
        SharedBotServices with initialized TTS, STT, and LLM.
    """
    from mumble_voice_bot.utils import ensure_models_downloaded

    if device == "auto":
        device = get_best_device()

    logger.info(f"Creating shared services on {device}")

    # Ensure models downloaded
    ensure_models_downloaded(device=device)

    # Initialize TTS
    logger.info("Loading TTS model...")
    from mumble_voice_bot.providers.luxtts import StreamingLuxTTS
    tts = StreamingLuxTTS('YatharthS/LuxTTS', device=device, threads=2)

    # Initialize STT — NeMo Nemotron
    try:
        from mumble_voice_bot.providers.nemotron_stt import NemotronConfig, NemotronStreamingASR
    except ImportError:
        raise RuntimeError("NeMo Nemotron STT required. Install with: pip install nemo_toolkit")

    model = nemotron_model or "nvidia/nemotron-speech-streaming-en-0.6b"
    nemo_device = nemotron_device or device
    logger.info(f"Loading NeMo Nemotron ({model})...")
    stt_config = NemotronConfig(model_name=model, chunk_size_ms=nemotron_chunk_ms, device=nemo_device)
    stt = NemotronStreamingASR(stt_config)
    if not asyncio.run(stt.initialize()):
        raise RuntimeError(f"Failed to initialize NeMo Nemotron STT ({model})")
    logger.info("NeMo Nemotron ready")

    # Initialize LLM
    llm = None
    try:
        from mumble_voice_bot.providers.openai_llm import OpenAIChatLLM

        endpoint = llm_endpoint or "http://localhost:11434/v1/chat/completions"
        model_name = llm_model or "llama3.2:3b"
        api_key = llm_api_key or os.environ.get('OPENROUTER_API_KEY') or os.environ.get('LLM_API_KEY')

        logger.info(f"Initializing LLM: {model_name}")
        llm = OpenAIChatLLM(
            endpoint=endpoint,
            model=model_name,
            api_key=api_key,
            system_prompt="",
            timeout=llm_timeout,
            max_tokens=llm_max_tokens,
            temperature=llm_temperature,
        )
    except ImportError:
        logger.warning("LLM modules not available — bot will only transcribe")

    logger.info("Shared services ready")
    return SharedBotServices(tts=tts, stt=stt, llm=llm, device=device)


def create_brain(
    brain_type: str = "llm",
    llm: Any = None,
    bot_name: str = "",
    shared_services: SharedBotServices | None = None,
    soul_config: Any = None,
    system_prompt: str = "",
    tools_config: Any = None,
    brain_power: float = 1.0,
    tts: Any = None,
) -> Brain:
    """Create a Brain from configuration.

    brain_type mapping:
    - "llm" → LLMBrain(brain_power=1.0)  (always uses LLM)
    - "adaptive" → LLMBrain(brain_power=configured)  (mixed routing)
    - "reactive" → LLMBrain(brain_power=0.0)  (pure reactive, no LLM)
    - "echo" → EchoBrain (voice cloning parrot)
    - "null" → NullBrain

    Args:
        brain_type: One of "llm", "echo", "reactive", "adaptive", "null".
        llm: LLM provider (required for "llm", optional for "adaptive").
        bot_name: Bot display name (for journal context).
        shared_services: SharedBotServices (for journal, coordination).
        soul_config: SoulConfig for personality/events.
        system_prompt: LLM system prompt.
        tools_config: ToolsConfig for tool settings.
        brain_power: brain_power for "adaptive" mode (0.0-1.0).
        tts: TTS engine (required for "echo" brain).

    Returns:
        Configured Brain instance.
    """
    # Configure echo filter and utterance classifier for LLM brains
    echo_filter = None
    utterance_classifier = None
    try:
        from mumble_voice_bot.speech_filter import EchoFilter, UtteranceClassifier
        echo_filter = EchoFilter(decay_time=3.0)
        utterance_classifier = UtteranceClassifier(min_words=2, min_chars=5)
    except ImportError:
        pass

    # Tool registry
    tools = None
    if tools_config or brain_type in ("llm", "adaptive"):
        tools = _create_tool_registry(tools_config, soul_config, bot_name)

    if brain_type == "null":
        return NullBrain()

    elif brain_type == "echo":
        from mumble_voice_bot.brains.echo import EchoBrain
        if tts is None:
            raise ValueError("EchoBrain requires a TTS engine")
        return EchoBrain(tts=tts)

    elif brain_type == "reactive":
        # Pure reactive: LLMBrain with brain_power=0.0 (never calls LLM)
        from mumble_voice_bot.brains.llm import LLMBrain
        return LLMBrain(
            llm=None,
            bot_name=bot_name,
            shared_services=shared_services,
            soul_config=soul_config,
            echo_filter=echo_filter,
            utterance_classifier=utterance_classifier,
            brain_power=0.0,
        )

    elif brain_type == "adaptive":
        # Mixed mode: LLMBrain with configurable brain_power
        from mumble_voice_bot.brains.llm import LLMBrain
        return LLMBrain(
            llm=llm,
            bot_name=bot_name,
            shared_services=shared_services,
            tools=tools,
            soul_config=soul_config,
            echo_filter=echo_filter,
            utterance_classifier=utterance_classifier,
            system_prompt=system_prompt,
            talks_to_bots=getattr(soul_config, 'talks_to_bots', False) if soul_config else False,
            brain_power=brain_power,
        )

    else:  # Default: "llm"
        from mumble_voice_bot.brains.llm import LLMBrain

        if llm is None:
            logger.warning("brain_type='llm' but no LLM available — falling back to NullBrain")
            return NullBrain()

        return LLMBrain(
            llm=llm,
            bot_name=bot_name,
            shared_services=shared_services,
            tools=tools,
            soul_config=soul_config,
            echo_filter=echo_filter,
            utterance_classifier=utterance_classifier,
            system_prompt=system_prompt,
            talks_to_bots=getattr(soul_config, 'talks_to_bots', False) if soul_config else False,
            brain_power=1.0,
        )


def _create_tool_registry(tools_config: Any, soul_config: Any, bot_name: str) -> Any:
    """Create and populate a ToolRegistry from config.

    Args:
        tools_config: ToolsConfig with tool settings.
        soul_config: SoulConfig (may have allowed_tools).
        bot_name: Bot name for the souls tool callback.

    Returns:
        ToolRegistry with registered tools, or None.
    """
    try:
        from mumble_voice_bot.tools import ToolRegistry
        from mumble_voice_bot.tools.web_search import WebSearchTool
    except ImportError:
        return None

    registry = ToolRegistry()

    # Web search
    registry.register(WebSearchTool(max_results=5, timeout=10.0))

    # Sound effects
    if tools_config and getattr(tools_config, 'sound_effects_enabled', False):
        try:
            from mumble_voice_bot.tools.sound_effects import SoundEffectsTool
            sound_tool = SoundEffectsTool(
                sounds_dir=tools_config.sound_effects_dir or "sounds",
                auto_play=getattr(tools_config, 'sound_effects_auto_play', True),
                sample_rate=48000,
                enable_web_search=getattr(tools_config, 'sound_effects_web_search', True),
                cache_web_sounds=getattr(tools_config, 'sound_effects_cache', True),
            )
            registry.register(sound_tool)
        except ImportError:
            pass

    # Souls tool
    try:
        from mumble_voice_bot.tools.souls import SoulsTool
        souls_tool = SoulsTool(
            souls_dir=os.path.join(_PROJECT_DIR, "souls"),
            switch_callback=None,  # Wired later by the bot
            get_current_callback=None,
        )
        registry.register(souls_tool)
    except ImportError:
        pass

    # Per-soul allowlisting
    if soul_config and hasattr(soul_config, 'brain_power'):
        allowed = getattr(soul_config, 'allowed_tools', None)
        if allowed:
            registry.set_allowed_tools(allowed)

    logger.info(f"Tool registry: {len(registry)} tool(s): {registry.tool_names}")
    return registry


def create_bot_from_config(
    config: Any,
    args: Any = None,
    shared_services: SharedBotServices | None = None,
    voice_prompt: dict | None = None,
    system_prompt_override: str | None = None,
) -> Any:
    """Create a fully configured MumbleBot from BotConfig.

    This is the main factory for single-persona mode.

    Args:
        config: BotConfig with all settings.
        args: Parsed CLI args (for device, debug_rms overrides).
        shared_services: Pre-created SharedBotServices (for multi-persona).
        voice_prompt: Pre-loaded voice prompt tensors.
        system_prompt_override: Override system prompt (for multi-persona).

    Returns:
        Configured MumbleBot ready to start().
    """
    from mumble_voice_bot.bot import MumbleBot
    from mumble_voice_bot.souls import load_system_prompt

    # Determine device
    device = "auto"
    if args and hasattr(args, 'device') and args.device != 'auto':
        device = args.device
    elif config and config.tts.device and config.tts.device != 'auto':
        device = config.tts.device
    if device == "auto":
        device = get_best_device()

    # Create shared services if not provided
    if shared_services is None:
        shared_services = create_shared_services(
            device=device,
            nemotron_model=config.stt.nemotron_model if config else None,
            nemotron_chunk_ms=config.stt.nemotron_chunk_ms if config else 160,
            nemotron_device=config.stt.nemotron_device if config else None,
            llm_endpoint=config.llm.endpoint if config else None,
            llm_model=config.llm.model if config else None,
            llm_api_key=config.llm.api_key if config else None,
            llm_timeout=config.llm.timeout if config else 30.0,
            llm_max_tokens=config.llm.max_tokens if config else None,
            llm_temperature=config.llm.temperature if config else None,
        )

    # Load system prompt
    system_prompt = system_prompt_override or ""
    if not system_prompt and config:
        personality = config.llm.personality if config.llm else None
        system_prompt = load_system_prompt(personality=personality, project_dir=_PROJECT_DIR)

    # Determine brain type
    brain_type = config.bot.brain_type if config else "llm"
    brain_power = config.bot.brain_power if config else 1.0

    # Per-soul brain_power override
    soul_config = config.soul_config if config else None
    if soul_config and hasattr(soul_config, 'brain_power') and soul_config.brain_power is not None:
        brain_power = soul_config.brain_power

    # Create brain
    brain = create_brain(
        brain_type=brain_type,
        llm=shared_services.llm,
        bot_name=config.mumble.user if config else "VoiceBot",
        shared_services=shared_services,
        soul_config=soul_config,
        system_prompt=system_prompt,
        tools_config=config.tools if config else None,
        brain_power=brain_power,
        tts=shared_services.tts,
    )

    # Load voice prompt if not provided
    if voice_prompt is None and config and config.tts.ref_audio:
        ref_audio = config.tts.ref_audio
        if os.path.exists(ref_audio):
            voice_prompt = shared_services.load_voice(
                name=config.mumble.user or "default",
                audio_path=ref_audio,
            )

    # Create MumbleBot
    bot = MumbleBot(
        host=config.mumble.host if config else "localhost",
        user=config.mumble.user if config else "VoiceBot",
        port=config.mumble.port if config else 64738,
        password=config.mumble.password if config else "",
        channel=config.mumble.channel if config else None,
        brain=brain,
        tts=shared_services.tts,
        stt=shared_services.stt,
        voice_prompt=voice_prompt or {},
        device=device,
        num_steps=config.tts.num_steps if config else 4,
        asr_threshold=config.bot.asr_threshold if config else 2000,
        debug_rms=args.debug_rms if args and hasattr(args, 'debug_rms') else False,
        shared_services=shared_services,
        soul_config=soul_config,
        max_response_staleness=config.bot.max_response_staleness if config else 5.0,
        barge_in_enabled=config.bot.barge_in_enabled if config else False,
    )

    return bot
