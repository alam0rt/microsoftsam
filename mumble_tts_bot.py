#!/usr/bin/env python3
"""Mumble Voice Bot — LLM-powered voice assistant for Mumble.

Listens to voice in a Mumble channel, transcribes with NeMo Nemotron ASR,
generates responses with an LLM, and speaks back using LuxTTS voice cloning.

This is the main entry point. It parses config, constructs a MumbleBot
with the appropriate Brain, and starts it.

Usage:
    # Basic usage with config file
    python mumble_tts_bot.py --config config.yaml

    # With CLI overrides
    python mumble_tts_bot.py --host mumble.example.com --reference voice.wav \\
        --llm-endpoint http://localhost:8000/v1/chat/completions \\
        --llm-model Qwen/Qwen3-32B

    # Debug mode to tune VAD threshold
    python mumble_tts_bot.py --config config.yaml --debug-rms
"""

import os
import sys
import time

# Add vendor paths for pymumble and LuxTTS before any vendor imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "botamusique"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LuxTTS"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LinaCodec", "src"))

from mumble_voice_bot.cli import create_argument_parser, merge_config_with_args
from mumble_voice_bot.logging_config import get_logger, setup_logging
from mumble_voice_bot.utils import get_best_device

logger = get_logger(__name__)


def run_single_bot(args, config):
    """Run the bot in single-persona mode.

    Args:
        args: Parsed CLI arguments.
        config: Loaded BotConfig (may be None).
    """
    from mumble_voice_bot.factory import create_shared_services
    from mumble_voice_bot.souls import load_system_prompt

    # Merge CLI args with config
    cfg = merge_config_with_args(args, config)

    # Determine device
    tts_device = cfg['tts_device_config']
    if args.device != 'auto':
        device = args.device
    elif tts_device != 'auto':
        device = tts_device
    else:
        device = get_best_device()

    # Create shared services (TTS, STT, LLM)
    shared = create_shared_services(
        device=device,
        nemotron_model=cfg['nemotron_model'],
        nemotron_chunk_ms=cfg['nemotron_chunk_ms'],
        nemotron_device=cfg['nemotron_device'],
        llm_endpoint=cfg['llm_endpoint'],
        llm_model=cfg['llm_model'],
        llm_api_key=cfg['llm_api_key'],
    )

    # Load system prompt
    system_prompt = cfg['llm_system_prompt'] or ""
    if not system_prompt:
        system_prompt = load_system_prompt(
            personality=cfg['personality'], project_dir=_THIS_DIR
        )

    # Load voice prompt
    voice_prompt = None
    ref_audio = cfg['reference']
    if ref_audio and os.path.exists(ref_audio):
        voice_prompt = shared.load_voice(
            name=cfg['user'], audio_path=ref_audio, voices_dir=cfg['voices_dir'],
        )

    # Determine brain type and power
    brain_type = config.bot.brain_type if config else "llm"
    brain_power = config.bot.brain_power if config else 1.0
    soul_config = cfg['soul_config']

    if soul_config and hasattr(soul_config, 'brain_power') and soul_config.brain_power is not None:
        brain_power = soul_config.brain_power

    # Create brain
    from mumble_voice_bot.factory import create_brain

    brain = create_brain(
        brain_type=brain_type,
        llm=shared.llm,
        bot_name=cfg['user'],
        shared_services=shared,
        soul_config=soul_config,
        system_prompt=system_prompt,
        tools_config=cfg['tools_config'],
        brain_power=brain_power,
        tts=shared.tts,
    )

    # Create and start MumbleBot
    from mumble_voice_bot.bot import MumbleBot

    bot = MumbleBot(
        host=cfg['host'],
        user=cfg['user'],
        port=cfg['port'],
        password=cfg['password'],
        channel=cfg['channel'],
        brain=brain,
        tts=shared.tts,
        stt=shared.stt,
        voice_prompt=voice_prompt or {},
        device=device,
        num_steps=cfg['steps'],
        asr_threshold=cfg['asr_threshold'],
        debug_rms=args.debug_rms,
        shared_services=shared,
        soul_config=soul_config,
        max_response_staleness=cfg['max_response_staleness'],
        barge_in_enabled=cfg['barge_in_enabled'],
    )

    bot.start()
    bot.run_forever()


def run_multi_persona_bot(args):
    """Run the bot in multi-persona mode.

    Loads a multi-persona config and creates multiple MumbleBot instances
    that share TTS/STT/LLM resources.

    Args:
        args: Parsed CLI arguments.
    """
    from mumble_voice_bot.bot import MumbleBot
    from mumble_voice_bot.factory import create_brain, create_shared_services
    from mumble_voice_bot.multi_persona_config import load_multi_persona_config

    try:
        config = load_multi_persona_config(args.config)
        logger.info(f"Loaded multi-persona config with {len(config.personas)} personas")
    except Exception as e:
        logger.error(f"Failed to load multi-persona config: {e}")
        print(f"\nFailed to load multi-persona config:\n{e}", file=sys.stderr)
        sys.exit(1)

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    device = args.device if args.device != 'auto' else get_best_device()

    # Create shared services
    shared_cfg = config.shared or {}
    stt_cfg = shared_cfg.get("stt", {})
    llm_cfg = shared_cfg.get("llm", {})

    shared = create_shared_services(
        device=device,
        nemotron_model=stt_cfg.get("nemotron_model"),
        nemotron_chunk_ms=stt_cfg.get("nemotron_chunk_ms", 160),
        nemotron_device=stt_cfg.get("nemotron_device"),
        llm_endpoint=llm_cfg.get("endpoint"),
        llm_model=llm_cfg.get("model"),
        llm_api_key=llm_cfg.get("api_key"),
        llm_timeout=llm_cfg.get("timeout", 30.0),
        llm_max_tokens=llm_cfg.get("max_tokens"),
        llm_temperature=llm_cfg.get("temperature"),
    )

    # Create bot instances
    bots = []
    for persona in config.personas:
        identity = persona.identity
        mumble_user = identity.mumble_user or identity.display_name or identity.name
        mumble_channel = persona.mumble.get("channel") if persona.mumble else None

        # Determine brain type
        is_parrot = identity.name.lower() == "parrot" and persona.soul_config is None
        brain_type = "echo" if is_parrot else "llm"

        # Load voice
        ref_audio = (persona.tts or {}).get("ref_audio")
        if not ref_audio and persona.soul_config:
            soul_audio_dir = os.path.join(_THIS_DIR, "souls", persona.soul_config.name, "audio")
            if os.path.isdir(soul_audio_dir):
                for f in os.listdir(soul_audio_dir):
                    if f.endswith(".wav"):
                        ref_audio = os.path.join(soul_audio_dir, f)
                        break

        voice_prompt = None
        if ref_audio and os.path.exists(ref_audio):
            voice_prompt = shared.load_voice(name=identity.name, audio_path=ref_audio)

        # Create brain
        brain = create_brain(
            brain_type=brain_type,
            llm=shared.llm,
            bot_name=mumble_user,
            shared_services=shared,
            soul_config=persona.soul_config,
            system_prompt=identity.system_prompt or "",
            brain_power=getattr(persona.soul_config, 'brain_power', 1.0) or 1.0 if persona.soul_config else 1.0,
            tts=shared.tts,
        )

        bot = MumbleBot(
            host=config.mumble_host or "localhost",
            user=mumble_user,
            port=config.mumble_port or 64738,
            password=config.mumble_password or "",
            channel=mumble_channel,
            brain=brain,
            tts=shared.tts,
            stt=shared.stt,
            voice_prompt=voice_prompt or {},
            device=device,
            shared_services=shared,
            soul_config=persona.soul_config,
        )
        bots.append(bot)
        logger.info(f"  Created bot: {identity.name} as '{mumble_user}' (brain={brain_type})")

    # Start all bots with staggered timing
    print(f"\nStarting {len(bots)} bots...")
    for i, bot in enumerate(bots):
        bot.start()
        if i < len(bots) - 1:
            time.sleep(5)

    print(f"\nMulti-persona bot running with {len(bots)} bots:")
    for bot in bots:
        print(f"  - {bot.user}")
    print("\nPress Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        for bot in bots:
            bot.shutdown()


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        json_output=args.log_json,
        log_file=args.log_file,
    )

    # Check for multi-persona config
    try:
        from mumble_voice_bot.multi_persona_config import is_multi_persona_config
        if args.config and is_multi_persona_config(args.config):
            logger.info(f"Detected multi-persona config: {args.config}")
            run_multi_persona_bot(args)
            return
    except ImportError:
        pass

    # Load config
    config = None
    if args.config:
        try:
            from mumble_voice_bot.config import load_config
            config = load_config(args.config)
            logger.info(f"Config loaded from {args.config}")
        except Exception as e:
            logger.error(f"Config error: {e}")
            print(f"\nConfig error: {e}", file=sys.stderr)
            sys.exit(1)

    # Apply model storage paths
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    elif config and config.models:
        config.models.apply_environment()

    run_single_bot(args, config)


if __name__ == '__main__':
    main()
