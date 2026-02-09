"""Tests for multi-persona configuration loading.

Tests cover:
- Basic config loading
- Config validation (missing fields, invalid structure)
- Persona config building
- Soul integration
- Interaction settings
- Environment variable expansion
"""

from textwrap import dedent

import pytest
import yaml

from mumble_voice_bot.multi_persona_config import (
    MultiPersonaConfigError,
    _build_persona_config,
    _validate_interaction_data,
    _validate_persona_data,
    create_example_multi_persona_config,
    is_multi_persona_config,
    load_multi_persona_config,
)


class TestValidatePersonaData:
    """Tests for persona data validation."""

    def test_valid_persona_with_soul(self):
        """Test validating a valid persona with soul."""
        data = {"name": "knight", "soul": "knight"}
        errors = _validate_persona_data(data, 0)
        assert errors == []

    def test_valid_persona_with_display_name(self):
        """Test validating a valid persona with display_name."""
        data = {"name": "custom", "display_name": "Custom Bot"}
        errors = _validate_persona_data(data, 0)
        assert errors == []

    def test_missing_name(self):
        """Test that missing name is caught."""
        data = {"soul": "knight"}
        errors = _validate_persona_data(data, 0)
        assert any("missing required field 'name'" in e for e in errors)

    def test_missing_soul_and_display_name(self):
        """Test that missing both soul and display_name is caught."""
        data = {"name": "test"}
        errors = _validate_persona_data(data, 0)
        assert any("must specify either 'soul' or 'display_name'" in e for e in errors)

    def test_unknown_field(self):
        """Test that unknown fields are caught."""
        data = {"name": "test", "soul": "knight", "unknown_field": "value"}
        errors = _validate_persona_data(data, 0)
        assert any("unknown field 'unknown_field'" in e for e in errors)

    def test_valid_with_all_fields(self):
        """Test validating a persona with all valid fields."""
        data = {
            "name": "knight",
            "display_name": "Sir Reginald",
            "soul": "knight",
            "system_prompt": "You are a knight.",
            "mumble": {"user": "Knight", "channel": "Tavern"},
            "llm_overrides": {"temperature": 0.7},
            "max_history_messages": 30,
            "respond_to_other_personas": True,
        }
        errors = _validate_persona_data(data, 0)
        assert errors == []


class TestValidateInteractionData:
    """Tests for interaction data validation."""

    def test_valid_interaction(self):
        """Test validating valid interaction config."""
        data = {
            "enable_cross_talk": True,
            "response_delay_ms": 500,
            "max_chain_length": 5,
        }
        errors = _validate_interaction_data(data)
        assert errors == []

    def test_unknown_field(self):
        """Test that unknown fields are caught."""
        data = {"unknown_field": "value"}
        errors = _validate_interaction_data(data)
        assert any("unknown field 'unknown_field'" in e for e in errors)

    def test_invalid_response_delay_type(self):
        """Test that non-integer response_delay_ms is caught."""
        data = {"response_delay_ms": "500"}
        errors = _validate_interaction_data(data)
        assert any("must be an integer" in e for e in errors)

    def test_invalid_max_chain_length(self):
        """Test that max_chain_length < 1 is caught."""
        data = {"max_chain_length": 0}
        errors = _validate_interaction_data(data)
        assert any("must be at least 1" in e for e in errors)


class TestLoadMultiPersonaConfig:
    """Tests for load_multi_persona_config function."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary directory with a souls subdirectory."""
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        return tmp_path

    def test_load_minimal_config(self, temp_config_dir):
        """Test loading a minimal valid config."""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(dedent("""
            personas:
              - name: "test"
                display_name: "Test Bot"
        """))

        config = load_multi_persona_config(config_path)

        assert len(config.personas) == 1
        assert config.personas[0].identity.name == "test"
        assert config.personas[0].identity.display_name == "Test Bot"

    def test_load_full_config(self, temp_config_dir):
        """Test loading a full config with all sections."""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(dedent("""
            shared:
              llm:
                endpoint: "http://localhost:11434/v1"
                model: "llama3.2:3b"
              stt:
                provider: "wyoming"

            personas:
              - name: "knight"
                display_name: "Sir Reginald"
                system_prompt: "You are a knight."
                mumble:
                  user: "Knight"
                  channel: "Tavern"
                llm_overrides:
                  temperature: 0.7
                respond_to_other_personas: true

              - name: "seller"
                display_name: "Potion Seller"
                respond_to_other_personas: true

            interaction:
              enable_cross_talk: true
              response_delay_ms: 500
              max_chain_length: 3

            mumble:
              host: "mumble.example.com"
              port: 64738
        """))

        config = load_multi_persona_config(config_path)

        # Check personas
        assert len(config.personas) == 2
        assert config.personas[0].identity.name == "knight"
        assert config.personas[0].identity.mumble_user == "Knight"
        assert config.personas[0].identity.mumble_channel == "Tavern"
        assert config.personas[0].llm_overrides == {"temperature": 0.7}
        assert config.personas[0].respond_to_other_personas is True

        # Check interaction
        assert config.interaction.enable_cross_talk is True
        assert config.interaction.response_delay_ms == 500
        assert config.interaction.max_chain_length == 3

        # Check Mumble defaults
        assert config.mumble_host == "mumble.example.com"
        assert config.mumble_port == 64738

        # Check shared settings
        assert config.shared["llm"]["model"] == "llama3.2:3b"

    def test_file_not_found(self, temp_config_dir):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_multi_persona_config(temp_config_dir / "nonexistent.yaml")

    def test_validation_error_missing_personas(self, temp_config_dir):
        """Test that missing personas section raises error."""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(dedent("""
            mumble:
              host: "localhost"
        """))

        with pytest.raises(MultiPersonaConfigError) as exc_info:
            load_multi_persona_config(config_path)

        assert "Missing required section 'personas'" in str(exc_info.value)

    def test_validation_error_empty_personas(self, temp_config_dir):
        """Test that empty personas list raises error."""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(dedent("""
            personas: []
        """))

        with pytest.raises(MultiPersonaConfigError) as exc_info:
            load_multi_persona_config(config_path)

        assert "must contain at least one persona" in str(exc_info.value)

    def test_validation_error_duplicate_names(self, temp_config_dir):
        """Test that duplicate persona names raise error."""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(dedent("""
            personas:
              - name: "test"
                display_name: "Test 1"
              - name: "test"
                display_name: "Test 2"
        """))

        with pytest.raises(MultiPersonaConfigError) as exc_info:
            load_multi_persona_config(config_path)

        assert "Duplicate persona names" in str(exc_info.value)

    def test_validation_error_unknown_section(self, temp_config_dir):
        """Test that unknown top-level sections raise error."""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(dedent("""
            personas:
              - name: "test"
                display_name: "Test"
            unknown_section:
              foo: "bar"
        """))

        with pytest.raises(MultiPersonaConfigError) as exc_info:
            load_multi_persona_config(config_path)

        assert "Unknown top-level section 'unknown_section'" in str(exc_info.value)

    def test_environment_variable_expansion(self, temp_config_dir, monkeypatch):
        """Test that environment variables are expanded."""
        monkeypatch.setenv("TEST_BOT_NAME", "EnvBot")
        monkeypatch.setenv("TEST_HOST", "env.example.com")

        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(dedent("""
            personas:
              - name: "test"
                display_name: "${TEST_BOT_NAME}"

            mumble:
              host: "${TEST_HOST}"
        """))

        config = load_multi_persona_config(config_path)

        assert config.personas[0].identity.display_name == "EnvBot"
        assert config.mumble_host == "env.example.com"

    def test_mumble_user_defaults_to_display_name(self, temp_config_dir):
        """Test that mumble user defaults to display_name if not set."""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text(dedent("""
            personas:
              - name: "test"
                display_name: "Test Bot"
        """))

        config = load_multi_persona_config(config_path)

        assert config.personas[0].identity.effective_mumble_user == "Test Bot"


class TestBuildPersonaConfig:
    """Tests for _build_persona_config function."""

    @pytest.fixture
    def temp_souls_dir(self, tmp_path):
        """Create a temporary souls directory with a test soul."""
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()

        # Create a test soul
        test_soul = souls_dir / "test-soul"
        test_soul.mkdir()

        # Create soul.yaml
        soul_yaml = test_soul / "soul.yaml"
        soul_yaml.write_text(dedent("""
            name: "Test Soul"
            description: "A test soul"
            version: "1.0.0"
            voice:
              ref_audio: "audio/reference.wav"
            fallbacks:
              greetings:
                - "Hello, {user}!"
        """))

        # Create personality.md
        personality_md = test_soul / "personality.md"
        personality_md.write_text("You are a test persona.")

        # Create audio directory
        audio_dir = test_soul / "audio"
        audio_dir.mkdir()

        return souls_dir

    def test_build_persona_with_soul(self, temp_souls_dir):
        """Test building persona config from soul."""
        persona_data = {"name": "test", "soul": "test-soul"}

        config = _build_persona_config(persona_data, temp_souls_dir, {})

        assert config.identity.name == "test"
        assert config.identity.soul_name == "test-soul"
        assert config.identity.display_name == "Test Soul"
        assert config.identity.system_prompt == "You are a test persona."

    def test_build_persona_without_soul(self, temp_souls_dir):
        """Test building persona config without soul."""
        persona_data = {
            "name": "custom",
            "display_name": "Custom Bot",
            "system_prompt": "You are custom.",
        }

        config = _build_persona_config(persona_data, temp_souls_dir, {})

        assert config.identity.name == "custom"
        assert config.identity.soul_name is None
        assert config.identity.display_name == "Custom Bot"
        assert config.identity.system_prompt == "You are custom."

    def test_build_persona_mumble_override(self, temp_souls_dir):
        """Test that persona mumble settings override defaults."""
        persona_data = {
            "name": "test",
            "display_name": "Test",
            "mumble": {"user": "OverrideUser", "channel": "OverrideChannel"},
        }
        default_mumble = {"host": "default.com", "user": "DefaultUser"}

        config = _build_persona_config(persona_data, temp_souls_dir, default_mumble)

        assert config.identity.mumble_user == "OverrideUser"
        assert config.identity.mumble_channel == "OverrideChannel"

    def test_build_persona_llm_overrides(self, temp_souls_dir):
        """Test that llm_overrides are preserved."""
        persona_data = {
            "name": "test",
            "display_name": "Test",
            "llm_overrides": {"temperature": 0.9, "max_tokens": 200},
        }

        config = _build_persona_config(persona_data, temp_souls_dir, {})

        assert config.llm_overrides == {"temperature": 0.9, "max_tokens": 200}


class TestIsMultiPersonaConfig:
    """Tests for is_multi_persona_config function."""

    def test_is_multi_persona_true(self, tmp_path):
        """Test detection of multi-persona config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(dedent("""
            personas:
              - name: "test"
                display_name: "Test"
        """))

        assert is_multi_persona_config(config_path) is True

    def test_is_multi_persona_false(self, tmp_path):
        """Test detection of regular config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(dedent("""
            llm:
              model: "llama3.2:3b"
            mumble:
              host: "localhost"
        """))

        assert is_multi_persona_config(config_path) is False

    def test_is_multi_persona_file_not_found(self, tmp_path):
        """Test that nonexistent file returns False."""
        assert is_multi_persona_config(tmp_path / "nonexistent.yaml") is False

    def test_is_multi_persona_invalid_yaml(self, tmp_path):
        """Test that invalid YAML returns False."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("not: valid: yaml: {{")

        assert is_multi_persona_config(config_path) is False


class TestCreateExampleConfig:
    """Tests for create_example_multi_persona_config function."""

    def test_creates_valid_file(self, tmp_path):
        """Test that example config is valid YAML."""
        config_path = tmp_path / "example.yaml"
        create_example_multi_persona_config(config_path)

        assert config_path.exists()

        # Should be valid YAML
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        assert "personas" in config_data
        assert "shared" in config_data
        assert "interaction" in config_data

    def test_example_has_personas(self, tmp_path):
        """Test that example config has persona definitions."""
        config_path = tmp_path / "example.yaml"
        create_example_multi_persona_config(config_path)

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        assert len(config_data["personas"]) >= 2
