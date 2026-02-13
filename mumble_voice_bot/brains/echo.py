"""EchoBrain - Clone speaker voice and echo transcript back.

This is the brain used by ParrotBot. It receives an utterance,
clones the speaker's voice from their audio, and echoes the
transcribed text back in their own voice.

No LLM is used -- this is a pure ASR + voice cloning pipeline.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np

from mumble_voice_bot.interfaces.brain import BotResponse, Utterance, VoiceConfig

logger = logging.getLogger(__name__)


class EchoBrain:
    """Brain that echoes back what users say in their own cloned voice.

    Attributes:
        tts: TTS engine (StreamingLuxTTS or SimpleLuxTTS) for voice cloning.
    """

    def __init__(self, tts: Any):
        """Initialize EchoBrain.

        Args:
            tts: TTS engine with encode_prompt() method for voice cloning.
        """
        self.tts = tts

    def process(self, utterance: Utterance) -> BotResponse | None:
        """Clone speaker's voice and echo their transcript.

        Args:
            utterance: Complete utterance with audio chunks for voice cloning.

        Returns:
            BotResponse with the transcript and cloned voice, or None on failure.
        """
        if not utterance.text or len(utterance.text) < 2:
            return None

        if not utterance.audio_chunks:
            logger.warning("EchoBrain: No audio chunks for voice cloning")
            return None

        # Clone voice from the user's audio
        try:
            voice_prompt = self._clone_voice(utterance.audio_chunks)
        except Exception as e:
            logger.error(f"Voice cloning failed for {utterance.user_name}: {e}")
            return None

        logger.info(f"EchoBrain: Echoing '{utterance.text[:50]}' as {utterance.user_name}")

        return BotResponse(
            text=utterance.text,
            voice=VoiceConfig(voice_prompt=voice_prompt),
        )

    def on_bot_utterance(self, speaker_name: str, text: str) -> BotResponse | None:
        """EchoBrain ignores other bots."""
        return None

    def on_text_message(self, sender: str, text: str) -> BotResponse | None:
        """EchoBrain ignores text messages (needs audio for voice cloning)."""
        return None

    def _clone_voice(self, audio_chunks: list[bytes]) -> dict:
        """Clone a voice from raw PCM audio chunks.

        Writes audio to a temp file, encodes the voice prompt, and returns
        the voice tensors for TTS.

        Args:
            audio_chunks: List of raw 48kHz 16-bit PCM chunks.

        Returns:
            Voice prompt dict with tensors for TTS.
        """
        import soundfile as sf

        # Concatenate and convert to float
        pcm_data = b''.join(audio_chunks)
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Limit to 5 seconds for voice cloning
        max_samples = int(5.0 * 48000)
        if len(audio_float) > max_samples:
            audio_float = audio_float[:max_samples]

        # Write to temp file for TTS encoder
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio_float, 48000)
            temp_path = f.name

        try:
            voice_prompt = self.tts.encode_prompt(temp_path, rms=0.01)
            return voice_prompt
        finally:
            os.unlink(temp_path)
