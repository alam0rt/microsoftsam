"""Tests for configuration management.

Tests cover:
- Configuration loading from YAML
- Environment variable expansion
- Default values
- Soul configuration loading
- Config validation
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mumble_voice_bot.config import (
    BotConfig,
    LLMConfig,
    MumbleConfig,
    ModelsConfig,
    PipelineBotConfig,
    SoulConfig,
    SoulFallbacks,
    STTConfig,
    TTSConfig,
    ToolsConfig,
    StreamingPipelineConfig,
    _expand_env_vars,
    create_example_config,
    load_config,
    load_soul_config,
)


# --- Fixtures ---


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory with test configs."""
    return tmp_path


@pytest.fixture
def basic_config_yaml(temp_config_dir):
    """Create a basic config.yaml file."""
    config_content = """
llm:
  endpoint: "http://localhost:8080/v1/chat/completions"
  model: "test-model"
  api_key: "test-key"
  system_prompt: "You are a test assistant."

tts:
  ref_audio: "test.wav"
  num_steps: 3

mumble:
  host: "mumble.example.com"
  port: 64738
  user: "TestBot"

bot:
  wake_word: "hey test"
  silence_threshold_ms: 2000
"""
    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def config_with_env_vars(temp_config_dir):
    """Create a config with environment variables."""
    config_content = """
llm:
  endpoint: "${LLM_ENDPOINT}"
  api_key: "${LLM_API_KEY}"

mumble:
  host: "${MUMBLE_HOST}"
  password: "${MUMBLE_PASSWORD}"
"""
    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def temp_souls_dir(tmp_path):
    """Create a temporary souls directory with test souls."""
    souls_dir = tmp_path / "souls"
    souls_dir.mkdir()

    # Create a test soul
    soul_dir = souls_dir / "test_soul"
    soul_dir.mkdir()

    soul_yaml = """
name: "Test Soul"
description: "A test personality"
author: "Test Author"
version: "1.0.0"

voice:
  ref_audio: "audio/"
  ref_duration: 3.0
  num_steps: 3
  speed: 1.1

llm:
  temperature: 0.8
  max_tokens: 256

fallbacks:
  greetings:
    - "Hello {user}!"
    - "Hey there {user}!"
  errors:
    - "Oops, something went wrong."
"""
    (soul_dir / "soul.yaml").write_text(soul_yaml)
    (soul_dir / "personality.md").write_text("You are a friendly test bot.")

    # Create audio directory with test audio
    audio_dir = soul_dir / "audio"
    audio_dir.mkdir()
    (audio_dir / "reference.wav").write_bytes(b"fake audio data")

    return souls_dir


# --- Test Classes ---


class TestExpandEnvVars:
    """Test environment variable expansion."""

    def test_expand_simple_var(self):
        """Test expanding a simple variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = _expand_env_vars("${TEST_VAR}")
            assert result == "test_value"

    def test_expand_var_in_string(self):
        """Test expanding variable embedded in string."""
        with patch.dict(os.environ, {"HOST": "localhost"}):
            result = _expand_env_vars("http://${HOST}:8080")
            assert result == "http://localhost:8080"

    def test_expand_multiple_vars(self):
        """Test expanding multiple variables."""
        with patch.dict(os.environ, {"USER": "admin", "PASS": "secret"}):
            result = _expand_env_vars("${USER}:${PASS}")
            assert result == "admin:secret"

    def test_expand_missing_var_empty(self):
        """Test that missing variables expand to empty string."""
        result = _expand_env_vars("${NONEXISTENT_VAR}")
        assert result == ""

    def test_expand_in_dict(self):
        """Test expanding in a dictionary."""
        with patch.dict(os.environ, {"API_KEY": "my-key"}):
            result = _expand_env_vars({"key": "${API_KEY}", "other": "static"})
            assert result == {"key": "my-key", "other": "static"}

    def test_expand_in_list(self):
        """Test expanding in a list."""
        with patch.dict(os.environ, {"ITEM": "value"}):
            result = _expand_env_vars(["${ITEM}", "literal"])
            assert result == ["value", "literal"]

    def test_expand_nested(self):
        """Test expanding in nested structures."""
        with patch.dict(os.environ, {"VAR": "expanded"}):
            result = _expand_env_vars(
                {"outer": {"inner": "${VAR}", "list": ["${VAR}"]}}
            )
            assert result == {"outer": {"inner": "expanded", "list": ["expanded"]}}

    def test_expand_non_string_unchanged(self):
        """Test that non-strings are unchanged."""
        result = _expand_env_vars(123)
        assert result == 123

        result = _expand_env_vars(None)
        assert result is None

        result = _expand_env_vars(True)
        assert result is True


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_default_values(self):
        """Test default LLM configuration."""
        config = LLMConfig()

        assert "localhost:11434" in config.endpoint
        assert config.model == "llama3.2:3b"
        assert config.api_key is None
        assert "voice assistant" in config.system_prompt.lower()
        assert config.timeout == 30.0
        assert config.max_tokens is None

    def test_custom_values(self):
        """Test custom LLM configuration."""
        config = LLMConfig(
            endpoint="https://api.openai.com/v1/chat/completions",
            model="gpt-4o",
            api_key="sk-test",
            system_prompt="Custom prompt.",
            timeout=60.0,
            max_tokens=500,
            temperature=0.7,
        )

        assert config.endpoint == "https://api.openai.com/v1/chat/completions"
        assert config.model == "gpt-4o"
        assert config.api_key == "sk-test"
        assert config.max_tokens == 500


class TestTTSConfig:
    """Test TTSConfig dataclass."""

    def test_default_values(self):
        """Test default TTS configuration."""
        config = TTSConfig()

        assert config.ref_audio == "reference.wav"
        assert config.ref_duration == 5.0
        assert config.num_steps == 4
        assert config.speed == 1.0
        assert config.device == "auto"

    def test_custom_values(self):
        """Test custom TTS configuration."""
        config = TTSConfig(
            ref_audio="custom.wav",
            ref_duration=3.0,
            num_steps=3,
            speed=1.2,
            device="cuda",
        )

        assert config.ref_audio == "custom.wav"
        assert config.num_steps == 3


class TestMumbleConfig:
    """Test MumbleConfig dataclass."""

    def test_default_values(self):
        """Test default Mumble configuration."""
        config = MumbleConfig()

        assert config.host == "localhost"
        assert config.port == 64738
        assert config.user == "VoiceBot"
        assert config.password is None
        assert config.channel is None

    def test_custom_values(self):
        """Test custom Mumble configuration."""
        config = MumbleConfig(
            host="mumble.example.com",
            port=12345,
            user="CustomBot",
            password="secret",
            channel="Lobby",
        )

        assert config.host == "mumble.example.com"
        assert config.password == "secret"


class TestSTTConfig:
    """Test STTConfig dataclass."""

    def test_default_values(self):
        """Test default STT configuration."""
        config = STTConfig()

        assert config.provider == "local"
        assert config.wyoming_host is None
        assert config.wyoming_port == 10300

    def test_wyoming_configuration(self):
        """Test Wyoming STT configuration."""
        config = STTConfig(
            provider="wyoming",
            wyoming_host="localhost",
            wyoming_port=10300,
        )

        assert config.provider == "wyoming"
        assert config.wyoming_host == "localhost"


class TestPipelineBotConfig:
    """Test PipelineBotConfig dataclass."""

    def test_default_values(self):
        """Test default bot configuration."""
        config = PipelineBotConfig()

        assert config.wake_word is None
        assert config.silence_threshold_ms == 1500
        assert config.max_recording_ms == 30000
        assert config.enable_conversation is True
        assert config.barge_in_enabled is False

    def test_custom_values(self):
        """Test custom bot configuration."""
        config = PipelineBotConfig(
            wake_word="hey bot",
            silence_threshold_ms=2000,
            barge_in_enabled=True,
        )

        assert config.wake_word == "hey bot"
        assert config.barge_in_enabled is True


class TestModelsConfig:
    """Test ModelsConfig dataclass."""

    def test_default_values(self):
        """Test default models configuration."""
        config = ModelsConfig()

        assert config.hf_home is None
        assert config.hf_hub_cache is None

    def test_apply_environment(self):
        """Test applying model paths as environment variables."""
        config = ModelsConfig(
            hf_home="/custom/hf_home",
            torch_home="/custom/torch_home",
        )

        # Clear any existing vars first
        for var in ["HF_HOME", "TORCH_HOME"]:
            os.environ.pop(var, None)

        applied = config.apply_environment()

        assert os.environ.get("HF_HOME") == "/custom/hf_home"
        assert os.environ.get("TORCH_HOME") == "/custom/torch_home"
        assert applied["HF_HOME"] == "/custom/hf_home"

        # Clean up
        for var in applied:
            os.environ.pop(var, None)


class TestToolsConfig:
    """Test ToolsConfig dataclass."""

    def test_default_values(self):
        """Test default tools configuration."""
        config = ToolsConfig()

        assert config.enabled is True
        assert config.max_iterations == 5
        assert config.web_search_enabled is True

    def test_custom_values(self):
        """Test custom tools configuration."""
        config = ToolsConfig(
            enabled=False,
            max_iterations=3,
            web_search_enabled=False,
        )

        assert config.enabled is False
        assert config.max_iterations == 3


class TestSoulFallbacks:
    """Test SoulFallbacks dataclass."""

    def test_default_values(self):
        """Test default fallbacks."""
        fallbacks = SoulFallbacks()

        assert len(fallbacks.greetings) > 0
        assert "{user}" in fallbacks.greetings[0]
        assert len(fallbacks.errors) > 0

    def test_custom_values(self):
        """Test custom fallbacks."""
        fallbacks = SoulFallbacks(
            greetings=["Custom greeting {user}!"],
            errors=["Custom error."],
        )

        assert fallbacks.greetings == ["Custom greeting {user}!"]
        assert fallbacks.errors == ["Custom error."]


class TestSoulConfig:
    """Test SoulConfig dataclass."""

    def test_default_values(self):
        """Test default soul configuration."""
        config = SoulConfig()

        assert config.name == "Default Soul"
        assert config.description == ""
        assert isinstance(config.voice, TTSConfig)
        assert isinstance(config.fallbacks, SoulFallbacks)


class TestLoadSoulConfig:
    """Test load_soul_config function."""

    def test_load_valid_soul(self, temp_souls_dir):
        """Test loading a valid soul configuration."""
        config = load_soul_config("test_soul", temp_souls_dir)

        assert config.name == "Test Soul"
        assert config.description == "A test personality"
        assert config.author == "Test Author"
        assert config.version == "1.0.0"
        assert config.voice.ref_duration == 3.0
        assert config.voice.num_steps == 3
        assert config.voice.speed == 1.1

    def test_load_soul_with_llm_overrides(self, temp_souls_dir):
        """Test soul LLM configuration overrides."""
        config = load_soul_config("test_soul", temp_souls_dir)

        assert config.llm.get("temperature") == 0.8
        assert config.llm.get("max_tokens") == 256

    def test_load_soul_with_fallbacks(self, temp_souls_dir):
        """Test soul fallback configuration."""
        config = load_soul_config("test_soul", temp_souls_dir)

        assert "Hello {user}!" in config.fallbacks.greetings
        assert "Oops, something went wrong." in config.fallbacks.errors

    def test_load_soul_resolves_audio_directory(self, temp_souls_dir):
        """Test that audio directory is resolved to first file."""
        config = load_soul_config("test_soul", temp_souls_dir)

        # Should have resolved to the actual audio file
        assert "reference.wav" in config.voice.ref_audio

    def test_load_nonexistent_soul_raises(self, temp_souls_dir):
        """Test loading non-existent soul raises error."""
        with pytest.raises(FileNotFoundError):
            load_soul_config("nonexistent", temp_souls_dir)


class TestLoadConfig:
    """Test load_config function."""

    def test_load_basic_config(self, basic_config_yaml):
        """Test loading a basic configuration file."""
        config = load_config(basic_config_yaml)

        assert config.llm.endpoint == "http://localhost:8080/v1/chat/completions"
        assert config.llm.model == "test-model"
        assert config.llm.api_key == "test-key"
        assert config.tts.ref_audio == "test.wav"
        assert config.tts.num_steps == 3
        assert config.mumble.host == "mumble.example.com"
        assert config.bot.wake_word == "hey test"

    def test_load_config_with_env_vars(self, config_with_env_vars):
        """Test loading config with environment variable expansion."""
        env_vars = {
            "LLM_ENDPOINT": "http://expanded:8080",
            "LLM_API_KEY": "expanded-key",
            "MUMBLE_HOST": "expanded-host",
            "MUMBLE_PASSWORD": "expanded-pass",
        }

        with patch.dict(os.environ, env_vars):
            config = load_config(config_with_env_vars)

        assert config.llm.endpoint == "http://expanded:8080"
        assert config.llm.api_key == "expanded-key"
        assert config.mumble.host == "expanded-host"
        assert config.mumble.password == "expanded-pass"

    def test_load_config_missing_file_returns_default(self, temp_config_dir):
        """Test loading non-existent config returns defaults."""
        nonexistent = temp_config_dir / "nonexistent.yaml"
        config = load_config(nonexistent)

        # Should have default values
        assert isinstance(config, BotConfig)
        assert config.llm.model == "llama3.2:3b"

    def test_load_config_with_soul(self, temp_config_dir):
        """Test loading config with soul specified."""
        config_content = """
soul: "test_soul"
llm:
  model: "base-model"
"""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(config_content)

        # Create souls directory as sibling to config
        target_souls_dir = temp_config_dir / "souls"
        target_souls_dir.mkdir()

        # Create the test soul directory structure
        soul_dir = target_souls_dir / "test_soul"
        soul_dir.mkdir()
        
        soul_yaml = """
name: "Test Soul"
description: "A test personality"
author: "Test Author"
version: "1.0.0"

voice:
  ref_audio: "audio/"
  ref_duration: 3.0
  num_steps: 3
  speed: 1.1

llm:
  temperature: 0.8
  max_tokens: 256

fallbacks:
  greetings:
    - "Hello {user}!"
  errors:
    - "Oops, something went wrong."
"""
        (soul_dir / "soul.yaml").write_text(soul_yaml)
        (soul_dir / "personality.md").write_text("You are a friendly test bot.")

        # Create audio directory
        audio_dir = soul_dir / "audio"
        audio_dir.mkdir()
        (audio_dir / "reference.wav").write_bytes(b"fake audio data")

        config = load_config(config_path)

        assert config.soul == "test_soul"
        assert config.soul_config is not None
        assert config.soul_config.name == "Test Soul"

    def test_load_config_none_path(self, temp_config_dir):
        """Test loading config with None path looks for config.yaml."""
        # Create config.yaml in current directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_config_dir)
            config_content = """
llm:
  model: "auto-discovered"
"""
            (temp_config_dir / "config.yaml").write_text(config_content)

            config = load_config(None)
            assert config.llm.model == "auto-discovered"
        finally:
            os.chdir(original_cwd)


class TestCreateExampleConfig:
    """Test create_example_config function."""

    def test_creates_example_file(self, temp_config_dir):
        """Test that example config file is created."""
        example_path = temp_config_dir / "example.yaml"
        create_example_config(example_path)

        assert example_path.exists()
        content = example_path.read_text()

        # Check key sections exist
        assert "llm:" in content
        assert "tts:" in content
        assert "mumble:" in content
        assert "bot:" in content

    def test_example_is_valid_yaml(self, temp_config_dir):
        """Test that example config is valid YAML."""
        import yaml

        example_path = temp_config_dir / "example.yaml"
        create_example_config(example_path)

        # Should parse without error
        with open(example_path) as f:
            parsed = yaml.safe_load(f)

        assert isinstance(parsed, dict)
        assert "llm" in parsed


class TestBotConfig:
    """Test BotConfig dataclass."""

    def test_default_values(self):
        """Test default BotConfig."""
        config = BotConfig()

        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.tts, TTSConfig)
        assert isinstance(config.stt, STTConfig)
        assert isinstance(config.mumble, MumbleConfig)
        assert isinstance(config.bot, PipelineBotConfig)
        assert isinstance(config.models, ModelsConfig)
        assert isinstance(config.tools, ToolsConfig)
        assert config.soul is None
        assert config.soul_config is None


class TestStreamingPipelineConfig:
    """Test StreamingPipelineConfig dataclass."""

    def test_default_values(self):
        """Test default streaming pipeline configuration."""
        config = StreamingPipelineConfig()

        assert config.enabled is False
        assert config.llm_start_threshold == 50
        assert config.llm_abort_on_change is False
        assert config.phrase_min_chars == 30
        assert config.phrase_max_chars == 150
        assert config.phrase_timeout_ms == 400

    def test_custom_values(self):
        """Test custom streaming pipeline configuration."""
        config = StreamingPipelineConfig(
            enabled=True,
            llm_start_threshold=30,
            phrase_min_chars=20,
        )

        assert config.enabled is True
        assert config.llm_start_threshold == 30
        assert config.phrase_min_chars == 20
