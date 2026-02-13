#!/usr/bin/env python3
"""Parrot Bot — A minimal Mumble bot that echoes back what users say in their own voice.

No LLM — just ASR + voice cloning TTS. This is now a thin wrapper around
MumbleBot + EchoBrain.

Usage:
    python parrot_bot.py --host mumble.example.com --user ParrotBot
"""

import argparse
import asyncio
import logging
import os
import sys

# Add vendor paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "botamusique"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LuxTTS"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LinaCodec", "src"))

logger = logging.getLogger(__name__)


class ParrotBot:
    """Backward-compatible ParrotBot wrapper around MumbleBot + EchoBrain.

    This class exists for backward compatibility with multi-persona configs
    that reference ParrotBot directly. New code should use:

        MumbleBot(brain=EchoBrain(tts))

    Attributes:
        user: Bot username.
    """

    def __init__(
        self,
        host: str,
        user: str = "ParrotBot",
        port: int = 64738,
        password: str = None,
        channel: str = None,
        device: str = "cuda",
        asr_threshold: int = 1500,
        min_speech_duration: float = 0.5,
        silence_timeout: float = 1.5,
        speech_hold_duration: float = 0.3,
        debug_rms: bool = False,
        shared_tts=None,
        shared_stt=None,
    ):
        from mumble_voice_bot.bot import MumbleBot
        from mumble_voice_bot.brains.echo import EchoBrain

        self.user = user

        # Use provided or create TTS
        tts = shared_tts
        if tts is None:
            from mumble_voice_bot.providers.luxtts import StreamingLuxTTS
            tts = StreamingLuxTTS(device=device)

        # Use provided or create STT
        stt = shared_stt
        if stt is None:
            from mumble_voice_bot.providers.nemotron_stt import NemotronConfig, NemotronStreamingASR
            stt_config = NemotronConfig(
                model_name="nvidia/nemotron-speech-streaming-en-0.6b",
                chunk_size_ms=160,
                device=device,
            )
            stt = NemotronStreamingASR(stt_config)
            asyncio.run(stt.initialize())

        brain = EchoBrain(tts=tts)

        self._bot = MumbleBot(
            host=host,
            user=user,
            port=port,
            password=password or "",
            channel=channel,
            brain=brain,
            tts=tts,
            stt=stt,
            device=device,
            asr_threshold=asr_threshold,
            debug_rms=debug_rms,
            min_speech_duration=min_speech_duration,
            speech_hold_duration=speech_hold_duration,
        )

        # Expose _shutdown for multi-persona compatibility
        self._shutdown = self._bot._shutdown

    def start(self):
        """Connect and start listening."""
        self._bot.start()

    def run_forever(self):
        """Run until interrupted."""
        self._bot.run_forever()


def main():
    parser = argparse.ArgumentParser(
        description="Parrot Bot - Echo back what users say in their own voice"
    )
    parser.add_argument("--host", required=True, help="Mumble server hostname")
    parser.add_argument("--port", type=int, default=64738, help="Mumble server port")
    parser.add_argument("--user", default="ParrotBot", help="Bot username")
    parser.add_argument("--password", help="Server password")
    parser.add_argument("--channel", help="Channel to join")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for ML models")
    parser.add_argument("--asr-threshold", type=int, default=2000,
                        help="RMS threshold for VAD")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    bot = ParrotBot(
        host=args.host,
        user=args.user,
        port=args.port,
        password=args.password,
        channel=args.channel,
        device=args.device,
        asr_threshold=args.asr_threshold,
    )
    bot.start()
    bot.run_forever()


if __name__ == "__main__":
    main()
