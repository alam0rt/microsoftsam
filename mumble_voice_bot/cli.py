"""CLI argument parsing for the Mumble Voice Bot.

Extracted from mumble_tts_bot.py to reduce monolith size.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the voice bot.

    Returns:
        Configured ArgumentParser.
    """
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

    # Model storage settings
    parser.add_argument('--hf-home', default=None,
                        help='HuggingFace home directory (where models are cached)')

    return parser


def merge_config_with_args(args: argparse.Namespace, config) -> dict:
    """Merge YAML config with CLI arguments, where CLI takes precedence.

    Args:
        args: Parsed CLI arguments.
        config: Loaded BotConfig (or None).

    Returns:
        Dict with all resolved configuration values.
    """
    import os

    # Mumble settings
    host = args.host or (config.mumble.host if config else None) or 'localhost'
    port = args.port or (config.mumble.port if config else None) or 64738
    soul_name_for_user = config.soul_config.name if (config and config.soul_config) else None
    user = args.user or (config.mumble.user if config else None) or soul_name_for_user or 'VoiceBot'
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
            logger.info(f"Loaded prompt from {prompt_path}")

    # NeMo Nemotron STT settings
    nemotron_model = (config.stt.nemotron_model if config else None) or "nvidia/nemotron-speech-streaming-en-0.6b"
    nemotron_chunk_ms = (config.stt.nemotron_chunk_ms if config else None) or 160
    nemotron_device = (config.stt.nemotron_device if config else None) or "cuda"

    # Staleness settings
    max_response_staleness = (config.bot.max_response_staleness if config else None) or 5.0

    # Barge-in settings
    barge_in_enabled = config.bot.barge_in_enabled if config else False

    # TTS device
    tts_device_config = (config.tts.device if config else None) or "auto"

    return {
        'host': host,
        'port': port,
        'user': user,
        'password': password,
        'channel': channel,
        'reference': reference,
        'steps': steps,
        'voices_dir': voices_dir,
        'asr_threshold': asr_threshold,
        'llm_endpoint': llm_endpoint,
        'llm_model': llm_model,
        'llm_api_key': llm_api_key,
        'llm_system_prompt': llm_system_prompt,
        'personality': personality,
        'nemotron_model': nemotron_model,
        'nemotron_chunk_ms': nemotron_chunk_ms,
        'nemotron_device': nemotron_device,
        'max_response_staleness': max_response_staleness,
        'barge_in_enabled': barge_in_enabled,
        'tts_device_config': tts_device_config,
        'soul_config': config.soul_config if config else None,
        'soul_name': config.soul if config else None,
        'tools_config': config.tools if config else None,
    }
