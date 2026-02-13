"""Soul/personality management for the Mumble Voice Bot.

Handles loading system prompts, personalities, and switching between souls.
Extracted from mumble_tts_bot.py to reduce monolith size.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Project root directory (parent of mumble_voice_bot/)
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_system_prompt(
    prompt_file: str | None = None,
    personality: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Load system prompt from file, optionally combined with a personality.

    Search order:
    1. Explicit prompt_file path
    2. prompts/default.md or prompts/default.txt
    3. Built-in fallback prompt

    If personality is specified, it's appended after the base prompt.

    Args:
        prompt_file: Explicit path to a prompt file.
        personality: Personality name or path to personality file.
        project_dir: Project root directory (defaults to repo root).

    Returns:
        The assembled system prompt string.
    """
    base_dir = project_dir or _PROJECT_DIR
    base_prompt = None

    # Try specified file first
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            logger.info(f"Loaded prompt from {prompt_file}")
            base_prompt = f.read()

    # Try default locations
    if not base_prompt:
        default_paths = [
            os.path.join(base_dir, "prompts", "default.md"),
            os.path.join(base_dir, "prompts", "default.txt"),
            "prompts/default.md",
            "prompts/default.txt",
        ]

        for path in default_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    logger.info(f"Loaded prompt from {path}")
                    base_prompt = f.read()
                    break

    # Fallback to inline prompt
    if not base_prompt:
        logger.info("Using built-in default prompt")
        base_prompt = get_fallback_prompt()

    # Load personality if specified
    if personality:
        personality_prompt = load_personality(personality, project_dir=base_dir)
        if personality_prompt:
            base_prompt = base_prompt + "\n\n" + "=" * 40 + "\n\n" + personality_prompt

    return base_prompt


def load_personality(personality: str, project_dir: str | None = None) -> str | None:
    """Load a personality file by name or path.

    Args:
        personality: Personality name (e.g., "imperial") or path to file.
        project_dir: Project root directory.

    Returns:
        Personality prompt text, or None if not found.
    """
    base_dir = project_dir or _PROJECT_DIR

    # Check if it's already a path
    if os.path.exists(personality):
        with open(personality, 'r') as f:
            logger.info(f"Loaded personality from {personality}")
            return f.read()

    # Try personalities directory
    personality_paths = [
        os.path.join(base_dir, "personalities", f"{personality}.md"),
        os.path.join(base_dir, "personalities", f"{personality}.txt"),
        os.path.join(base_dir, "personalities", personality),
        f"personalities/{personality}.md",
        f"personalities/{personality}.txt",
    ]

    for path in personality_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                logger.info(f"Loaded personality: {personality}")
                return f.read()

    logger.warning(f"Personality '{personality}' not found")
    return None


def get_fallback_prompt() -> str:
    """Return the built-in fallback system prompt."""
    return """You are a casual voice assistant in a Mumble voice channel.

Your responses will be spoken by TTS. Never use emojis, symbols, or formatting.
Keep responses to 1-2 sentences. Use casual language and contractions.
Sound like a friend chatting, not a corporate assistant.
Write numbers and symbols as words: "about 5 dollars" not "$5"."""


async def switch_soul(
    soul_name: str,
    souls_dir: str | None = None,
    llm: Any = None,
    tts_voice_loader: Any = None,
    mumble_client: Any = None,
    channel_history: list | None = None,
    current_soul_config: Any = None,
    preserve_context: bool | None = None,
    project_dir: str | None = None,
) -> tuple[str, Any]:
    """Switch to a different soul/personality.

    Updates the LLM system prompt, TTS voice, and soul config.

    Args:
        soul_name: Name of the soul directory to switch to.
        souls_dir: Path to the souls directory.
        llm: LLM provider to update system prompt on.
        tts_voice_loader: Callback to load a new voice reference.
        mumble_client: Mumble client for updating comments.
        channel_history: Current channel history (modified in place).
        current_soul_config: Current soul config for fallback.
        preserve_context: Whether to preserve conversation history.
        project_dir: Project root directory.

    Returns:
        Tuple of (success/error message, new SoulConfig or None).
    """
    from mumble_voice_bot.config import load_config, load_soul_config

    base_dir = project_dir or _PROJECT_DIR
    if souls_dir is None:
        souls_dir = os.path.join(base_dir, "souls")
    soul_path = os.path.join(souls_dir, soul_name)

    if not os.path.exists(soul_path):
        return f"Soul '{soul_name}' not found.", None

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

    # Store current context before switch
    preserved_messages = []
    if preserve_context and channel_history:
        preserved_messages = [
            msg for msg in channel_history
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
            if not os.path.isabs(ref_audio):
                ref_audio = os.path.join(soul_path, ref_audio)

            # Handle directory - find first audio file
            if os.path.isdir(ref_audio):
                audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
                for f in sorted(os.listdir(ref_audio)):
                    if os.path.splitext(f)[1].lower() in audio_extensions:
                        ref_audio = os.path.join(ref_audio, f)
                        break

            if os.path.exists(ref_audio) and tts_voice_loader:
                logger.info(f"Loading voice: {ref_audio}")
                tts_voice_loader(ref_audio)

        # Update LLM system prompt
        if llm:
            personality_path = os.path.join(soul_path, "personality.md")
            if os.path.exists(personality_path):
                new_prompt = load_system_prompt(personality=personality_path, project_dir=base_dir)
                llm.system_prompt = new_prompt
                logger.info("Updated LLM personality")

            # Apply LLM overrides from soul config
            if new_soul.llm:
                if "temperature" in new_soul.llm:
                    llm.temperature = new_soul.llm["temperature"]
                if "max_tokens" in new_soul.llm:
                    llm.max_tokens = new_soul.llm["max_tokens"]

        # Update Mumble comment
        if mumble_client:
            try:
                if hasattr(mumble_client, 'is_ready') and mumble_client.is_ready():
                    mumble_client.users.myself.update_comment(f"Soul: {new_soul.name}")
            except Exception as e:
                logger.debug(f"Could not update Mumble comment: {e}")

        # Clear and restore channel history
        if channel_history is not None:
            channel_history.clear()
            if preserved_messages:
                channel_history.extend(preserved_messages)
                logger.info(f"Restored {len(preserved_messages)} messages to new soul context")

        context_msg = f" ({len(preserved_messages)} messages preserved)" if preserved_messages else ""
        return f"Switched to {new_soul.name}. Voice and personality updated.{context_msg}", new_soul

    except Exception as e:
        logger.error(f"Failed to switch soul: {e}")
        return f"Error switching to '{soul_name}': {str(e)}", None
