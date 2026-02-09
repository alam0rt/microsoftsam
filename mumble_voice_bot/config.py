"""Configuration management for Mumble Voice Bot.

Supports loading configuration from YAML files with environment variable expansion.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values.

    Supports ${VAR_NAME} syntax for environment variable expansion.

    Args:
        value: A config value (string, dict, list, or other).

    Returns:
        The value with environment variables expanded.
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, '')

        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    else:
        return value


@dataclass
class LLMConfig:
    """Configuration for the LLM provider.

    Attributes:
        endpoint: URL to the chat completions API endpoint.
        model: Model identifier to use.
        api_key: Optional API key for authentication.
        system_prompt: System prompt for the assistant persona.
        prompt_file: Path to a prompt file (overrides system_prompt if set).
        personality: Personality name or path (combined with prompt).
        timeout: Request timeout in seconds.
        max_tokens: Maximum tokens in response (optional).
        temperature: Sampling temperature (optional).
        top_p: Nucleus sampling parameter (optional).
        top_k: Top-k sampling parameter (optional).
        repetition_penalty: Penalty for repetition (optional).
        frequency_penalty: Penalty for token frequency (optional, OpenAI-style).
        presence_penalty: Penalty for token presence (optional, OpenAI-style).
        context_messages: Max messages to keep in conversation history.
    """
    endpoint: str = "http://localhost:11434/v1/chat/completions"
    model: str = "llama3.2:3b"
    api_key: str | None = None
    system_prompt: str = (
        "You are a helpful voice assistant in a Mumble voice chat. "
        "Keep responses concise and conversational (1-3 sentences). "
        "Be friendly but not overly verbose - this is voice, not text."
    )
    prompt_file: str | None = None
    personality: str | None = None
    timeout: float = 30.0
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    context_messages: int = 20


@dataclass
class TTSConfig:
    """Configuration for text-to-speech.

    Attributes:
        ref_audio: Path to reference audio for voice cloning.
        ref_duration: Seconds of reference audio to use.
        num_steps: Number of diffusion steps (quality vs speed).
        speed: Speech speed multiplier.
        device: Compute device for TTS ('auto', 'cpu', 'cuda', 'mps').
    """
    ref_audio: str = "reference.wav"
    ref_duration: float = 5.0
    num_steps: int = 4
    speed: float = 1.0
    device: str = "auto"


@dataclass
class MumbleConfig:
    """Configuration for Mumble connection.

    Attributes:
        host: Mumble server hostname.
        port: Mumble server port.
        user: Bot username.
        password: Server password (optional).
        channel: Channel to join (optional).
        certfile: Path to client certificate (optional).
        keyfile: Path to client key (optional).
    """
    host: str = "localhost"
    port: int = 64738
    user: str = "VoiceBot"
    password: str | None = None
    channel: str | None = None
    certfile: str | None = None
    keyfile: str | None = None


@dataclass
class PipelineBotConfig:
    """Configuration for bot behavior.

    Attributes:
        wake_word: Wake word to trigger the bot (None = respond to all speech).
        silence_threshold_ms: Milliseconds of silence before processing speech.
        max_recording_ms: Maximum recording duration in milliseconds.
        asr_threshold: RMS threshold for voice activity detection.
        enable_conversation: Enable LLM conversation mode (vs mimic mode).
        conversation_timeout: Seconds before conversation history is cleared.
        max_response_staleness: Skip responses older than this (seconds).
                               Increase if TTS is being skipped due to slow LLM.
        barge_in_enabled: Allow users to interrupt the bot mid-speech.
                         When False, bot talks over everyone without interruption.
        echo_filter_enabled: Enable echo filtering to ignore bot's own speech.
        echo_filter_decay: How long to remember bot outputs for echo detection (seconds).
        utterance_filter_enabled: Enable filtering of non-meaningful utterances.
        utterance_min_words: Minimum words for a meaningful utterance.
        utterance_min_chars: Minimum characters for a meaningful utterance.
        turn_prediction_enabled: Enable turn prediction for natural response timing.
        turn_prediction_base_delay: Base delay before responding (seconds).
        turn_prediction_max_delay: Maximum delay to wait for turn completion (seconds).
        turn_prediction_threshold: Confidence threshold for turn completion (0-1).
        state_machine_enabled: Enable conversation state machine for turn management.
        state_machine_cooldown: Cooldown duration after speaking (seconds).
    """
    wake_word: str | None = None
    silence_threshold_ms: int = 1500
    max_recording_ms: int = 30000
    asr_threshold: int = 2000
    enable_conversation: bool = True
    conversation_timeout: float = 300.0  # 5 minutes
    max_response_staleness: float = 5.0  # Skip responses older than this
    barge_in_enabled: bool = False  # Disabled by default - bot talks over everyone

    # Echo filtering - prevents responding to bot's own TTS output
    echo_filter_enabled: bool = True
    echo_filter_decay: float = 3.0  # seconds

    # Utterance filtering - filters out non-meaningful speech
    utterance_filter_enabled: bool = True
    utterance_min_words: int = 2
    utterance_min_chars: int = 5

    # Turn prediction - natural response timing
    turn_prediction_enabled: bool = True
    turn_prediction_base_delay: float = 0.3  # seconds
    turn_prediction_max_delay: float = 1.5  # seconds
    turn_prediction_threshold: float = 0.7  # confidence threshold

    # Conversation state machine
    state_machine_enabled: bool = True
    state_machine_cooldown: float = 0.5  # seconds

    # Context preservation on soul switch
    preserve_context_on_switch: bool = True  # Keep conversation history when switching souls
    max_preserved_messages: int = 10  # Maximum messages to preserve across switch


@dataclass
class STTConfig:
    """Configuration for speech-to-text.

    Attributes:
        provider: STT provider to use. Options:
                  - "local" (default): Use local Whisper via LuxTTS
                  - "wyoming": Use Wyoming STT server
                  - "wyoming_streaming": Wyoming with local streaming wrapper
                  - "websocket": WebSocket streaming ASR server
                  - "sherpa_nemotron": Use Nemotron via sherpa-onnx (streaming)
                  - "nemotron_nemo": Use Nemotron via NeMo framework (streaming)
        wyoming_host: Wyoming STT server host.
        wyoming_port: Wyoming STT server port.
        sherpa_encoder: Path to sherpa-onnx encoder model.
        sherpa_decoder: Path to sherpa-onnx decoder model.
        sherpa_joiner: Path to sherpa-onnx joiner model.
        sherpa_tokens: Path to sherpa-onnx tokens file.
        sherpa_provider: ONNX runtime provider ("cuda" or "cpu").
        nemotron_model: HuggingFace model name for NeMo Nemotron.
        nemotron_chunk_ms: Chunk size in ms for streaming (80, 160, 560, 1120).
        nemotron_device: Device for NeMo Nemotron ("cuda" or "cpu").
        websocket_endpoint: WebSocket endpoint for streaming ASR.
        streaming_chunk_ms: Chunk size for streaming ASR (80, 160, 500).
        streaming_stability_window: Partials before text is stable.
        streaming_min_stable_chars: Min chars before emitting stable text.
    """
    provider: str = "local"  # local, wyoming, wyoming_streaming, websocket, sherpa_nemotron, nemotron_nemo

    # Wyoming settings
    wyoming_host: str | None = None
    wyoming_port: int = 10300

    # Sherpa-onnx Nemotron settings
    sherpa_encoder: str | None = None
    sherpa_decoder: str | None = None
    sherpa_joiner: str | None = None
    sherpa_tokens: str | None = None
    sherpa_provider: str = "cuda"

    # NeMo Nemotron settings
    nemotron_model: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    nemotron_chunk_ms: int = 160
    nemotron_device: str = "cuda"

    # WebSocket streaming settings
    websocket_endpoint: str | None = None
    streaming_chunk_ms: int = 160
    streaming_stability_window: int = 2
    streaming_min_stable_chars: int = 10


@dataclass
class StreamingPipelineConfig:
    """Configuration for the streaming voice pipeline.

    Controls early LLM start and ASR/LLM overlap behavior.

    Attributes:
        enabled: Whether to use streaming pipeline (vs batch).
        llm_start_threshold: Minimum stable chars before starting LLM.
        llm_abort_on_change: Abort LLM if transcript changes significantly.
        change_threshold: Characters of change to trigger abort.
        phrase_min_chars: Minimum phrase length for TTS.
        phrase_max_chars: Maximum phrase length before force-flush.
        phrase_timeout_ms: Flush phrase after this delay.
    """
    enabled: bool = False
    llm_start_threshold: int = 50
    llm_abort_on_change: bool = False
    change_threshold: int = 20
    phrase_min_chars: int = 30
    phrase_max_chars: int = 150
    phrase_timeout_ms: int = 400


@dataclass
class ModelsConfig:
    """Configuration for model storage and caching.

    Controls where HuggingFace models and other ML models are stored/cached.
    These settings are applied as environment variables before loading models.

    Attributes:
        hf_home: HuggingFace home directory (sets HF_HOME).
                 This is the root for all HF-related files including cache.
        hf_hub_cache: HuggingFace Hub cache directory (sets HF_HUB_CACHE).
                      Where downloaded model files are stored.
        transformers_cache: Legacy transformers cache (sets TRANSFORMERS_CACHE).
                           For older transformers library versions.
        torch_home: PyTorch home directory (sets TORCH_HOME).
                    Where PyTorch downloads pretrained models.
        xdg_cache_home: XDG cache directory (sets XDG_CACHE_HOME).
                        Fallback cache location on Linux.
    """
    hf_home: str | None = None
    hf_hub_cache: str | None = None
    transformers_cache: str | None = None
    torch_home: str | None = None
    xdg_cache_home: str | None = None

    def apply_environment(self) -> dict[str, str]:
        """Apply model paths as environment variables.

        Call this before loading any HuggingFace or PyTorch models.

        Returns:
            Dict of environment variables that were set.
        """
        applied = {}

        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home
            applied["HF_HOME"] = self.hf_home

        if self.hf_hub_cache:
            os.environ["HF_HUB_CACHE"] = self.hf_hub_cache
            os.environ["HUGGINGFACE_HUB_CACHE"] = self.hf_hub_cache  # Legacy alias
            applied["HF_HUB_CACHE"] = self.hf_hub_cache

        if self.transformers_cache:
            os.environ["TRANSFORMERS_CACHE"] = self.transformers_cache
            applied["TRANSFORMERS_CACHE"] = self.transformers_cache

        if self.torch_home:
            os.environ["TORCH_HOME"] = self.torch_home
            applied["TORCH_HOME"] = self.torch_home

        if self.xdg_cache_home:
            os.environ["XDG_CACHE_HOME"] = self.xdg_cache_home
            applied["XDG_CACHE_HOME"] = self.xdg_cache_home

        return applied


@dataclass
class SoulFallbacks:
    """Fallback responses for when the LLM is unavailable.

    These provide character-appropriate responses without LLM generation.
    Each list is selected from randomly when needed.

    Attributes:
        greetings: Fallback greetings when users join the channel.
        farewells: Fallback farewells when users leave.
        acknowledgments: Brief acknowledgments for commands/requests.
        idle_chatter: Random things the bot might say when idle.
        errors: Responses when something goes wrong.
        thinking: Quick fillers when processing a question (bypasses LLM).
        still_thinking: Fillers when LLM is taking too long.
        interrupted: What to say when barged-in on (user interrupts bot).
    """
    greetings: list[str] = field(default_factory=lambda: [
        "Hey {user}!",
        "Oh hey, {user}.",
        "Hey! {user}'s here.",
    ])
    farewells: list[str] = field(default_factory=lambda: [
        "See ya, {user}.",
        "Later, {user}!",
        "Bye {user}.",
    ])
    acknowledgments: list[str] = field(default_factory=lambda: [
        "Got it.",
        "Okay.",
        "Sure thing.",
    ])
    idle_chatter: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=lambda: [
        "Hmm, something went wrong.",
        "Uh, I'm having trouble with that.",
        "Sorry, I can't do that right now.",
    ])
    # Conversation fillers - short utterances that bypass LLM for natural flow
    thinking: list[str] = field(default_factory=lambda: [
        "Hmm...",
        "Let me think...",
        "Umm...",
        "One sec...",
    ])
    still_thinking: list[str] = field(default_factory=lambda: [
        "Still thinking...",
        "Bear with me...",
        "Hmm, let me see...",
    ])
    interrupted: list[str] = field(default_factory=lambda: [
        "Oh, sorry.",
        "Go ahead.",
        "Yes?",
        "Mm?",
    ])


@dataclass
class SoulEvents:
    """Event-triggered responses that bypass the LLM.

    These are spoken directly via TTS when specific events occur.
    Use {user} placeholder for the username. Use null/empty to disable.

    Attributes:
        user_first_speech: When a user speaks for the first time this session.
        user_joined: When a user joins the channel (if you want TTS, not just log).
        user_left: When a user leaves the channel.
        interrupted: What to say when user interrupts (barge-in).
        thinking: Quick filler when processing a question.
        still_thinking: When LLM is taking too long (>5s).
        wake_word_detected: When wake word is heard.
        tool_started: When a tool starts executing.
        tool_completed: When a tool finishes.
    """
    user_first_speech: list[str] | None = field(default_factory=lambda: [
        "Hey {user}!",
        "Oh, hey {user}.",
    ])
    user_joined: list[str] | None = None  # Disabled by default (can be noisy)
    user_left: list[str] | None = None  # Disabled by default
    interrupted: list[str] | None = field(default_factory=lambda: [
        "Oh, sorry.",
        "Go ahead.",
    ])
    thinking: list[str] | None = field(default_factory=lambda: [
        "Hmm...",
        "Let me think...",
    ])
    still_thinking: list[str] | None = field(default_factory=lambda: [
        "Still thinking...",
        "Bear with me...",
    ])
    wake_word_detected: list[str] | None = None  # e.g., "Yes?"
    tool_started: list[str] | None = None  # e.g., "Let me look that up..."
    tool_completed: list[str] | None = None  # e.g., "Found it!"


@dataclass
class SoulConfig:
    """Soul configuration for character/personality theming.

    A "soul" defines the character, voice, and behavior of the bot.
    This includes voice cloning settings, LLM behavior overrides,
    and fallback responses themed to the character.

    Attributes:
        name: Display name for the soul.
        description: Brief description of the character.
        author: Who created this soul.
        version: Soul configuration version.
        voice: TTS voice settings (overrides main config).
        weights: Custom model weights paths.
        llm: LLM behavior overrides (temperature, max_tokens, etc).
        fallbacks: Fallback responses themed to the character.
        events: Event-triggered responses (bypass LLM, go direct to TTS).
        talks_to_bots: Whether this bot responds to other bots' utterances.
    """
    name: str = "Default Soul"
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    voice: TTSConfig = field(default_factory=TTSConfig)
    weights: dict[str, str | None] = field(default_factory=lambda: {
        "tts_model": None,
        "voice_encoder": None,
    })
    llm: dict[str, Any] = field(default_factory=dict)
    fallbacks: SoulFallbacks = field(default_factory=SoulFallbacks)
    events: SoulEvents = field(default_factory=SoulEvents)
    talks_to_bots: bool = False  # Whether to respond to other bots


@dataclass
class ToolsConfig:
    """Configuration for agent tools.

    Attributes:
        enabled: Whether tools are enabled.
        max_iterations: Maximum tool execution iterations per turn.
        web_search_enabled: Enable web search tool.
        web_search_max_results: Number of web search results.
        web_search_timeout: Web search timeout in seconds.
        sound_effects_enabled: Enable sound effects tool.
        sound_effects_dir: Directory containing sound effect files.
        sound_effects_auto_play: Allow LLM to proactively play sounds.
        sound_effects_web_search: Allow searching MyInstants.com for sounds.
        sound_effects_cache: Cache downloaded sounds locally.
        sound_effects_verify_ssl: Verify SSL certs for web requests (disable if cert issues).
    """
    enabled: bool = True
    max_iterations: int = 5

    # Web search settings
    web_search_enabled: bool = True
    web_search_max_results: int = 5
    web_search_timeout: float = 10.0

    # Sound effects settings
    sound_effects_enabled: bool = False  # Disabled by default
    sound_effects_dir: str | None = None  # Path to sound effects directory
    sound_effects_auto_play: bool = True  # Allow LLM to proactively play sounds
    sound_effects_web_search: bool = True  # Allow searching MyInstants.com
    sound_effects_cache: bool = True  # Cache downloaded sounds locally
    sound_effects_verify_ssl: bool = True  # Verify SSL certs (disable for systems with cert issues)


@dataclass
class BotConfig:
    """Complete bot configuration.

    Attributes:
        llm: LLM provider configuration.
        tts: Text-to-speech configuration.
        stt: Speech-to-text configuration.
        mumble: Mumble connection configuration.
        bot: Bot behavior configuration.
        models: Model storage/cache configuration.
        tools: Tool system configuration.
        soul: Soul name to load (from souls/ directory).
        soul_config: Loaded soul configuration (populated by load_config).
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    mumble: MumbleConfig = field(default_factory=MumbleConfig)
    bot: PipelineBotConfig = field(default_factory=PipelineBotConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    soul: str | None = None  # Soul name to load from souls/ directory
    soul_config: SoulConfig | None = None  # Loaded soul configuration


def load_soul_config(
    soul_name: str,
    souls_dir: str | Path = "souls",
) -> SoulConfig:
    """Load soul configuration from a soul.yaml file.

    Args:
        soul_name: Name of the soul (directory name under souls/).
        souls_dir: Base directory containing soul directories.

    Returns:
        SoulConfig with loaded settings.

    Raises:
        FileNotFoundError: If the soul directory or soul.yaml doesn't exist.
        yaml.YAMLError: If the soul.yaml file is invalid.
    """
    souls_path = Path(souls_dir)
    soul_path = souls_path / soul_name
    soul_yaml = soul_path / "soul.yaml"

    if not soul_yaml.exists():
        raise FileNotFoundError(f"Soul config not found: {soul_yaml}")

    with open(soul_yaml) as f:
        raw_config = yaml.safe_load(f) or {}

    # Expand environment variables
    config_data = _expand_env_vars(raw_config)

    # Build voice config (with soul-relative paths)
    voice_data = config_data.get("voice", {})
    if "ref_audio" in voice_data:
        ref_audio = voice_data["ref_audio"]
        # Make relative paths relative to the soul directory
        if not Path(ref_audio).is_absolute():
            ref_audio_path = soul_path / ref_audio
        else:
            ref_audio_path = Path(ref_audio)

        # If ref_audio points to a directory, find the first audio file in it
        if ref_audio_path.is_dir():
            audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
            audio_files = [
                f for f in sorted(ref_audio_path.iterdir())
                if f.is_file() and f.suffix.lower() in audio_extensions
            ]
            if audio_files:
                ref_audio_path = audio_files[0]
                logger.info(f"Soul using audio: {ref_audio_path.name}")
            else:
                logger.warning(f"No audio files found in {ref_audio_path}")

        voice_data["ref_audio"] = str(ref_audio_path)

    voice = TTSConfig(**{k: v for k, v in voice_data.items() if v is not None})

    # Build fallbacks config
    fallbacks_data = config_data.get("fallbacks", {})
    default_fallbacks = SoulFallbacks()
    fallbacks = SoulFallbacks(
        greetings=fallbacks_data.get("greetings", default_fallbacks.greetings),
        farewells=fallbacks_data.get("farewells", default_fallbacks.farewells),
        acknowledgments=fallbacks_data.get(
            "acknowledgments", default_fallbacks.acknowledgments
        ),
        idle_chatter=fallbacks_data.get("idle_chatter", []),
        errors=fallbacks_data.get("errors", default_fallbacks.errors),
        thinking=fallbacks_data.get("thinking", default_fallbacks.thinking),
        still_thinking=fallbacks_data.get("still_thinking", default_fallbacks.still_thinking),
        interrupted=fallbacks_data.get("interrupted", default_fallbacks.interrupted),
    )

    # Build events config (event-driven TTS responses)
    # NOTE: "on" in YAML is parsed as boolean True, so we check for True as well
    # Can be under "events", "responses", or boolean True (from "on:" in YAML)
    events_data = config_data.get("events", config_data.get("responses", config_data.get(True, {})))
    default_events = SoulEvents()
    events = SoulEvents(
        user_first_speech=events_data.get("user_first_speech", default_events.user_first_speech),
        user_joined=events_data.get("user_joined", default_events.user_joined),
        user_left=events_data.get("user_left", default_events.user_left),
        interrupted=events_data.get("interrupted", default_events.interrupted),
        thinking=events_data.get("thinking", default_events.thinking),
        still_thinking=events_data.get("still_thinking", default_events.still_thinking),
        wake_word_detected=events_data.get("wake_word_detected", default_events.wake_word_detected),
        tool_started=events_data.get("tool_started", default_events.tool_started),
        tool_completed=events_data.get("tool_completed", default_events.tool_completed),
    )

    return SoulConfig(
        name=config_data.get("name", soul_name),
        description=config_data.get("description", ""),
        author=config_data.get("author", ""),
        version=config_data.get("version", "1.0.0"),
        voice=voice,
        weights=config_data.get("weights", {"tts_model": None, "voice_encoder": None}),
        llm=config_data.get("llm", {}),
        fallbacks=fallbacks,
        events=events,
        talks_to_bots=config_data.get("talks_to_bots", False),
    )


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def _get_dataclass_fields(cls) -> set[str]:
    """Get the field names of a dataclass."""
    from dataclasses import fields as dataclass_fields
    return {f.name for f in dataclass_fields(cls)}


def validate_config_section(section_name: str, data: dict, config_class) -> list[str]:
    """Validate a config section against its dataclass.

    Args:
        section_name: Name of the section (for error messages).
        data: The config data dict.
        config_class: The dataclass to validate against.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    valid_fields = _get_dataclass_fields(config_class)
    for key in data.keys():
        if key not in valid_fields:
            errors.append(f"Unknown field '{key}' in {section_name}. Valid fields: {sorted(valid_fields)}")
    return errors


def validate_config_data(config_data: dict, path: str | Path) -> None:
    """Validate config data and raise ConfigValidationError if invalid.

    Args:
        config_data: The parsed config dict.
        path: Path to the config file (for error messages).

    Raises:
        ConfigValidationError: If any validation errors are found.
    """
    errors = []

    # Known top-level sections
    valid_sections = {"llm", "tts", "stt", "mumble", "bot", "models", "tools", "soul"}
    for key in config_data.keys():
        if key not in valid_sections:
            errors.append(f"Unknown top-level section '{key}'. Valid sections: {sorted(valid_sections)}")

    # Validate each section
    section_mapping = {
        "llm": LLMConfig,
        "tts": TTSConfig,
        "stt": STTConfig,
        "mumble": MumbleConfig,
        "bot": PipelineBotConfig,
        "models": ModelsConfig,
        "tools": ToolsConfig,
    }

    for section_name, config_class in section_mapping.items():
        section_data = config_data.get(section_name, {})
        if section_data:
            errors.extend(validate_config_section(section_name, section_data, config_class))

    if errors:
        error_msg = f"Config validation failed for {path}:\n  - " + "\n  - ".join(errors)
        raise ConfigValidationError(error_msg)


def load_config(path: str | Path | None = None) -> BotConfig:
    """Load configuration from a YAML file.

    If no path is provided, looks for config.yaml in the current directory.
    Environment variables in the format ${VAR_NAME} are expanded.

    Args:
        path: Path to the YAML config file.

    Returns:
        BotConfig with loaded settings.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid YAML.
    """
    if path is None:
        path = Path("config.yaml")
    else:
        path = Path(path)

    if not path.exists():
        # Return default config if no file exists
        return BotConfig()

    with open(path) as f:
        raw_config = yaml.safe_load(f) or {}

    # Expand environment variables
    config_data = _expand_env_vars(raw_config)

    # Validate config structure before building objects
    validate_config_data(config_data, path)

    # Build config objects
    llm_data = config_data.get("llm", {})
    tts_data = config_data.get("tts", {})
    stt_data = config_data.get("stt", {})
    mumble_data = config_data.get("mumble", {})
    bot_data = config_data.get("bot", {})
    models_data = config_data.get("models", {})
    tools_data = config_data.get("tools", {})
    soul_name = config_data.get("soul")

    # Load soul configuration if specified
    soul_config = None
    if soul_name:
        # Determine souls directory relative to config file
        souls_dir = path.parent / "souls"
        try:
            soul_config = load_soul_config(soul_name, souls_dir)
            logger.info(f"Loaded soul: {soul_config.name}", extra={"soul": soul_name, "description": soul_config.description})

            # Soul voice settings override main TTS config if present
            if soul_config.voice.ref_audio != "reference.wav":
                if "ref_audio" not in tts_data:
                    tts_data["ref_audio"] = soul_config.voice.ref_audio
            # Apply other soul voice overrides
            for attr in ["ref_duration", "num_steps", "speed", "device"]:
                soul_val = getattr(soul_config.voice, attr)
                default_val = getattr(TTSConfig(), attr)
                if soul_val != default_val and attr not in tts_data:
                    tts_data[attr] = soul_val

            # Soul LLM overrides apply to main config
            for key, value in soul_config.llm.items():
                if key not in llm_data:
                    llm_data[key] = value

            # Auto-load soul's personality.md if no personality is set
            if "personality" not in llm_data:
                personality_path = souls_dir / soul_name / "personality.md"
                if personality_path.exists():
                    llm_data["personality"] = str(personality_path)
                    logger.info(f"Using soul personality: {personality_path}")
        except FileNotFoundError as e:
            logger.warning(f"Soul not found: {e}")

    return BotConfig(
        llm=LLMConfig(**{k: v for k, v in llm_data.items() if v is not None}),
        tts=TTSConfig(**{k: v for k, v in tts_data.items() if v is not None}),
        stt=STTConfig(**{k: v for k, v in stt_data.items() if v is not None}),
        mumble=MumbleConfig(**{k: v for k, v in mumble_data.items() if v is not None}),
        bot=PipelineBotConfig(**{k: v for k, v in bot_data.items() if v is not None}),
        models=ModelsConfig(**{k: v for k, v in models_data.items() if v is not None}),
        tools=ToolsConfig(**{k: v for k, v in tools_data.items() if v is not None}),
        soul=soul_name,
        soul_config=soul_config,
    )


def create_example_config(path: str | Path = "config.yaml") -> None:
    """Create an example configuration file.

    Args:
        path: Path where to write the example config.
    """
    example = """\
# Mumble Voice Bot Configuration
# Environment variables can be used with ${VAR_NAME} syntax

llm:
  # OpenAI-compatible endpoint (works with Ollama, vLLM, OpenAI, etc.)
  endpoint: "http://localhost:11434/v1/chat/completions"
  model: "llama3.2:3b"
  api_key: "${LLM_API_KEY}"  # Optional, set via environment

  # Prompt configuration (choose one):
  # Option 1: Inline system prompt
  system_prompt: |
    You are a helpful voice assistant in a Mumble voice chat.
    Keep responses concise and conversational (1-3 sentences).
    Be friendly but not overly verbose - this is voice, not text.

  # Option 2: Load prompt from file (overrides system_prompt)
  # prompt_file: "prompts/default.md"

  # Option 3: Add a personality on top of the prompt
  # personality: "imperial"  # Loads personalities/imperial.md

  timeout: 30.0
  # max_tokens: 256  # Optional, limit response length
  # temperature: 0.7  # Optional, control randomness

tts:
  ref_audio: "reference.wav"  # Reference audio for voice cloning
  ref_duration: 5.0           # Seconds of reference to use
  num_steps: 4                # Quality vs speed (3-4 recommended)
  speed: 1.0                  # Playback speed

mumble:
  host: "localhost"
  port: 64738
  user: "VoiceBot"
  password: null
  channel: null
  certfile: null
  keyfile: null

bot:
  wake_word: null             # e.g., "hey bot" - if null, responds to all speech
  silence_threshold_ms: 1500  # Silence before processing speech
  max_recording_ms: 30000     # Max speech duration
  asr_threshold: 2000         # RMS threshold for voice activity
  enable_conversation: true   # Enable LLM conversation mode
  conversation_timeout: 300   # Clear history after 5 minutes of inactivity

stt:
  wyoming_host: null          # e.g., "localhost" for wyoming-faster-whisper
  wyoming_port: 10300         # Wyoming STT server port

# Model storage paths (optional)
# Use to specify where HuggingFace and PyTorch models are downloaded/cached
models:
  # hf_home: "/path/to/huggingface"       # HF_HOME - main HuggingFace directory
  # hf_hub_cache: "/path/to/hf/hub"       # HF_HUB_CACHE - downloaded model files
  # transformers_cache: "/path/to/cache"  # TRANSFORMERS_CACHE - legacy location
  # torch_home: "/path/to/torch"          # TORCH_HOME - PyTorch models
  # xdg_cache_home: "/path/to/cache"      # XDG_CACHE_HOME - fallback cache
"""

    with open(path, "w") as f:
        f.write(example)

    print(f"Created example config at {path}")
