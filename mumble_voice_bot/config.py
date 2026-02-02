"""Configuration management for Mumble Voice Bot.

Supports loading configuration from YAML files with environment variable expansion.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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


@dataclass
class TTSConfig:
    """Configuration for text-to-speech.
    
    Attributes:
        ref_audio: Path to reference audio for voice cloning.
        ref_duration: Seconds of reference audio to use.
        num_steps: Number of diffusion steps (quality vs speed).
        speed: Speech speed multiplier.
    """
    ref_audio: str = "reference.wav"
    ref_duration: float = 5.0
    num_steps: int = 4
    speed: float = 1.0


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
    """
    wake_word: str | None = None
    silence_threshold_ms: int = 1500
    max_recording_ms: int = 30000
    asr_threshold: int = 2000
    enable_conversation: bool = True
    conversation_timeout: float = 300.0  # 5 minutes
    max_response_staleness: float = 5.0  # Skip responses older than this


@dataclass
class STTConfig:
    """Configuration for speech-to-text.
    
    Attributes:
        wyoming_host: Wyoming STT server host (None = use local Whisper).
        wyoming_port: Wyoming STT server port.
    """
    wyoming_host: str | None = None
    wyoming_port: int = 10300


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
class BotConfig:
    """Complete bot configuration.
    
    Attributes:
        llm: LLM provider configuration.
        tts: Text-to-speech configuration.
        stt: Speech-to-text configuration.
        mumble: Mumble connection configuration.
        bot: Bot behavior configuration.
        models: Model storage/cache configuration.
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    mumble: MumbleConfig = field(default_factory=MumbleConfig)
    bot: PipelineBotConfig = field(default_factory=PipelineBotConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)


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
    
    # Build config objects
    llm_data = config_data.get("llm", {})
    tts_data = config_data.get("tts", {})
    stt_data = config_data.get("stt", {})
    mumble_data = config_data.get("mumble", {})
    bot_data = config_data.get("bot", {})
    models_data = config_data.get("models", {})
    
    return BotConfig(
        llm=LLMConfig(**{k: v for k, v in llm_data.items() if v is not None}),
        tts=TTSConfig(**{k: v for k, v in tts_data.items() if v is not None}),
        stt=STTConfig(**{k: v for k, v in stt_data.items() if v is not None}),
        mumble=MumbleConfig(**{k: v for k, v in mumble_data.items() if v is not None}),
        bot=PipelineBotConfig(**{k: v for k, v in bot_data.items() if v is not None}),
        models=ModelsConfig(**{k: v for k, v in models_data.items() if v is not None}),
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
