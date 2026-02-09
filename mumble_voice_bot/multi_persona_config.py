"""Multi-persona configuration loading and validation.

This module handles loading configurations for running multiple bot personas
that share resources (TTS, STT, LLM) while maintaining separate identities.

Example config file (config.multi.yaml):

    shared:
      llm:
        endpoint: "http://localhost:11434/v1/chat/completions"
        model: "llama3.2:3b"
      stt:
        provider: "wyoming"
        wyoming_host: "localhost"
      tts:
        device: "cuda"

    personas:
      - name: "knight"
        soul: "knight"
        mumble:
          user: "Sir Reginald"
          channel: "Tavern"

      - name: "seller"
        soul: "potion-seller"
        mumble:
          user: "Potion Seller"
          channel: "Tavern"

    interaction:
      enable_cross_talk: true
      response_delay_ms: 500
      max_chain_length: 5

    mumble:
      host: "mumble.example.com"
      port: 64738
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from mumble_voice_bot.config import (
    LLMConfig,
    MumbleConfig,
    STTConfig,
    TTSConfig,
    _expand_env_vars,
    load_soul_config,
)
from mumble_voice_bot.interfaces.services import (
    InteractionConfig,
    MultiPersonaConfig,
    PersonaConfig,
    PersonaIdentity,
)

logger = logging.getLogger(__name__)


class MultiPersonaConfigError(Exception):
    """Raised when multi-persona config validation fails."""
    pass


def _validate_persona_data(persona_data: dict, index: int) -> list[str]:
    """Validate a single persona configuration block.
    
    Args:
        persona_data: The persona config dict.
        index: Index of the persona (for error messages).
        
    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    
    # Required fields
    if "name" not in persona_data:
        errors.append(f"personas[{index}]: missing required field 'name'")
    
    # Either soul or display_name must be present
    if "soul" not in persona_data and "display_name" not in persona_data:
        errors.append(
            f"personas[{index}]: must specify either 'soul' or 'display_name'"
        )
    
    # Valid fields for persona block
    valid_fields = {
        "name", "display_name", "soul", "system_prompt",
        "mumble", "llm_overrides", "max_history_messages",
        "respond_to_other_personas",
    }
    for key in persona_data.keys():
        if key not in valid_fields:
            errors.append(
                f"personas[{index}]: unknown field '{key}'. "
                f"Valid fields: {sorted(valid_fields)}"
            )
    
    return errors


def _validate_interaction_data(interaction_data: dict) -> list[str]:
    """Validate interaction configuration.
    
    Args:
        interaction_data: The interaction config dict.
        
    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    valid_fields = {
        "enable_cross_talk", "response_delay_ms", "max_chain_length",
        "cooldown_after_chain_ms", "ignore_own_audio",
    }
    
    for key in interaction_data.keys():
        if key not in valid_fields:
            errors.append(
                f"interaction: unknown field '{key}'. "
                f"Valid fields: {sorted(valid_fields)}"
            )
    
    # Type validation
    if "response_delay_ms" in interaction_data:
        if not isinstance(interaction_data["response_delay_ms"], int):
            errors.append("interaction.response_delay_ms: must be an integer")
    
    if "max_chain_length" in interaction_data:
        if not isinstance(interaction_data["max_chain_length"], int):
            errors.append("interaction.max_chain_length: must be an integer")
        elif interaction_data["max_chain_length"] < 1:
            errors.append("interaction.max_chain_length: must be at least 1")
    
    return errors


def _validate_multi_persona_config(config_data: dict, path: Path) -> None:
    """Validate multi-persona configuration structure.
    
    Args:
        config_data: The parsed config dict.
        path: Path to the config file (for error messages).
        
    Raises:
        MultiPersonaConfigError: If validation fails.
    """
    errors = []
    
    # Valid top-level sections
    valid_sections = {"shared", "personas", "interaction", "mumble"}
    for key in config_data.keys():
        if key not in valid_sections:
            errors.append(
                f"Unknown top-level section '{key}'. "
                f"Valid sections: {sorted(valid_sections)}"
            )
    
    # personas is required
    if "personas" not in config_data:
        errors.append("Missing required section 'personas'")
    elif not isinstance(config_data["personas"], list):
        errors.append("'personas' must be a list")
    elif len(config_data["personas"]) == 0:
        errors.append("'personas' must contain at least one persona")
    else:
        # Validate each persona
        for i, persona_data in enumerate(config_data["personas"]):
            if not isinstance(persona_data, dict):
                errors.append(f"personas[{i}]: must be a dict")
            else:
                errors.extend(_validate_persona_data(persona_data, i))
        
        # Check for duplicate names
        names = [p.get("name") for p in config_data["personas"] if p.get("name")]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            errors.append(f"Duplicate persona names: {set(duplicates)}")
    
    # Validate interaction config if present
    if "interaction" in config_data:
        if not isinstance(config_data["interaction"], dict):
            errors.append("'interaction' must be a dict")
        else:
            errors.extend(_validate_interaction_data(config_data["interaction"]))
    
    if errors:
        error_msg = f"Multi-persona config validation failed for {path}:\n  - " + "\n  - ".join(errors)
        raise MultiPersonaConfigError(error_msg)


def _build_persona_config(
    persona_data: dict,
    souls_dir: Path,
    default_mumble: dict,
) -> PersonaConfig:
    """Build a PersonaConfig from raw config data.
    
    Args:
        persona_data: The persona config dict.
        souls_dir: Path to the souls directory.
        default_mumble: Default Mumble settings to use.
        
    Returns:
        PersonaConfig instance.
    """
    name = persona_data["name"]
    soul_name = persona_data.get("soul")
    
    # Try to load soul if specified
    soul_config = None
    if soul_name:
        try:
            soul_config = load_soul_config(soul_name, souls_dir)
            logger.info(f"Loaded soul '{soul_name}' for persona '{name}'")
        except FileNotFoundError:
            logger.warning(f"Soul '{soul_name}' not found for persona '{name}'")
    
    # Determine display name
    display_name = persona_data.get("display_name")
    if not display_name:
        display_name = soul_config.name if soul_config else name
    
    # Determine system prompt (persona override > soul personality.md)
    system_prompt = persona_data.get("system_prompt")
    if not system_prompt and soul_name:
        personality_path = souls_dir / soul_name / "personality.md"
        if personality_path.exists():
            with open(personality_path) as f:
                system_prompt = f.read()
            logger.info(f"Loaded personality from {personality_path}")
    
    # Mumble settings (merge with defaults)
    mumble_data = persona_data.get("mumble", {})
    effective_mumble = {**default_mumble, **mumble_data}
    
    # If no explicit mumble user, use display_name
    mumble_user = effective_mumble.get("user") or display_name
    mumble_channel = effective_mumble.get("channel")
    
    # Build identity
    identity = PersonaIdentity(
        name=name,
        display_name=display_name,
        soul_name=soul_name,
        system_prompt=system_prompt,
        mumble_user=mumble_user,
        mumble_channel=mumble_channel,
    )
    
    # Build config
    return PersonaConfig(
        identity=identity,
        voice_prompt=None,  # Loaded at runtime from soul
        llm_overrides=persona_data.get("llm_overrides", {}),
        max_history_messages=persona_data.get("max_history_messages", 20),
        respond_to_other_personas=persona_data.get("respond_to_other_personas", False),
    )


def load_multi_persona_config(path: str | Path) -> MultiPersonaConfig:
    """Load multi-persona configuration from a YAML file.
    
    Args:
        path: Path to the YAML config file.
        
    Returns:
        MultiPersonaConfig with loaded settings.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid YAML.
        MultiPersonaConfigError: If config validation fails.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path) as f:
        raw_config = yaml.safe_load(f) or {}
    
    # Expand environment variables
    config_data = _expand_env_vars(raw_config)
    
    # Validate structure
    _validate_multi_persona_config(config_data, path)
    
    # Parse shared settings
    shared_data = config_data.get("shared", {})
    
    # Parse default Mumble settings
    mumble_data = config_data.get("mumble", {})
    mumble_host = mumble_data.get("host", "localhost")
    mumble_port = mumble_data.get("port", 64738)
    mumble_password = mumble_data.get("password")
    
    # Parse interaction settings
    interaction_data = config_data.get("interaction", {})
    interaction = InteractionConfig(
        enable_cross_talk=interaction_data.get("enable_cross_talk", True),
        response_delay_ms=interaction_data.get("response_delay_ms", 500),
        max_chain_length=interaction_data.get("max_chain_length", 5),
        cooldown_after_chain_ms=interaction_data.get("cooldown_after_chain_ms", 3000),
        ignore_own_audio=interaction_data.get("ignore_own_audio", True),
    )
    
    # Build persona configs
    souls_dir = path.parent / "souls"
    personas = []
    for persona_data in config_data["personas"]:
        persona_config = _build_persona_config(
            persona_data,
            souls_dir,
            mumble_data,
        )
        personas.append(persona_config)
    
    return MultiPersonaConfig(
        personas=personas,
        shared=shared_data,
        interaction=interaction,
        mumble_host=mumble_host,
        mumble_port=mumble_port,
        mumble_password=mumble_password,
    )


def create_example_multi_persona_config(path: str | Path = "config.multi.yaml") -> None:
    """Create an example multi-persona configuration file.
    
    Args:
        path: Path where to write the example config.
    """
    example = """\
# Multi-Persona Bot Configuration
# This config allows running multiple bot personas that share TTS/STT/LLM resources
# while maintaining separate identities (voice, personality, conversation history).
#
# Environment variables can be used with ${VAR_NAME} syntax

# =============================================================================
# Shared Services Configuration
# =============================================================================
# These expensive resources are loaded once and shared across all personas.
# This saves GPU memory and initialization time.

shared:
  llm:
    # OpenAI-compatible endpoint (works with Ollama, vLLM, OpenAI, etc.)
    endpoint: "http://localhost:11434/v1/chat/completions"
    model: "llama3.2:3b"
    api_key: "${LLM_API_KEY}"  # Optional, set via environment
    timeout: 30.0
    # temperature: 0.7  # Can be overridden per-persona

  stt:
    provider: "wyoming"
    wyoming_host: "localhost"
    wyoming_port: 10300

  tts:
    device: "cuda"
    num_steps: 4

# =============================================================================
# Persona Definitions
# =============================================================================
# Each persona gets their own:
# - Mumble connection (separate user)
# - Voice (from soul's ref_audio)
# - System prompt / personality
# - Conversation history

personas:
  # Knight persona - loads from souls/knight/
  - name: "knight"
    soul: "knight"  # Loads soul.yaml, personality.md, audio/ from souls/knight/
    mumble:
      user: "Sir Reginald"  # Mumble username
      channel: "Tavern"     # Channel to join (optional)
    llm_overrides:
      temperature: 0.7
    respond_to_other_personas: true  # Can engage in bot-to-bot conversation

  # Potion Seller persona - loads from souls/potion-seller/
  - name: "seller"
    soul: "potion-seller"
    mumble:
      user: "Potion Seller"
      channel: "Tavern"
    llm_overrides:
      temperature: 0.8
      max_tokens: 150
    respond_to_other_personas: true

  # Custom persona without a soul directory
  # - name: "custom"
  #   display_name: "Custom Bot"
  #   system_prompt: |
  #     You are a helpful assistant.
  #   mumble:
  #     user: "CustomBot"

# =============================================================================
# Bot-to-Bot Interaction Settings
# =============================================================================
# Controls how personas interact with each other to prevent infinite loops
# and enable natural conversations.

interaction:
  enable_cross_talk: true        # Allow bots to hear and respond to each other
  response_delay_ms: 500         # Delay before responding to another bot
  max_chain_length: 5            # Max consecutive bot-to-bot exchanges
  cooldown_after_chain_ms: 3000  # Cooldown after max chain is reached
  ignore_own_audio: true         # Prevent bot from hearing its own TTS output

# =============================================================================
# Default Mumble Connection Settings
# =============================================================================
# These are used as defaults for all personas unless overridden.

mumble:
  host: "localhost"
  port: 64738
  password: null
"""

    path = Path(path)
    with open(path, "w") as f:
        f.write(example)
    
    print(f"Created example multi-persona config at {path}")


def is_multi_persona_config(path: str | Path) -> bool:
    """Check if a config file is a multi-persona config.
    
    Multi-persona configs have a 'personas' section at the top level.
    
    Args:
        path: Path to the config file.
        
    Returns:
        True if the file appears to be a multi-persona config.
    """
    path = Path(path)
    
    if not path.exists():
        return False
    
    try:
        with open(path) as f:
            config_data = yaml.safe_load(f) or {}
        return "personas" in config_data
    except yaml.YAMLError:
        return False
