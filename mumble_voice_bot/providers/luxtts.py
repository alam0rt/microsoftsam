"""StreamingLuxTTS - Extended LuxTTS with streaming support and bug fixes.

Extracted from mumble_tts_bot.py and parrot_bot.py to eliminate duplication.
Both bots had their own StreamingLuxTTS implementations.
"""

from __future__ import annotations

import logging
from typing import Generator

import torch

from mumble_voice_bot.text_processing import pad_tts_text, split_into_sentences

logger = logging.getLogger(__name__)

# LuxTTS is imported from the vendor directory (must be on sys.path)
try:
    from zipvoice.luxvoice import LuxTTS

    LUXTTS_AVAILABLE = True
except ImportError:
    LUXTTS_AVAILABLE = False
    LuxTTS = None
    logger.warning("LuxTTS not available (vendor/LuxTTS not on sys.path)")


class StreamingLuxTTS(LuxTTS):
    """Extended LuxTTS with streaming support and bug fixes.

    Key improvements over base LuxTTS:
    - Streaming generation: yields audio sentence-by-sentence
    - English-only transcriber patch for reliable voice cloning
    - Safe generation with automatic retry on vocoder kernel size errors
    - Text padding to avoid vocoder crashes on very short inputs
    """

    def __init__(self, model_path: str = 'YatharthS/LuxTTS', device: str = 'cuda', threads: int = 4):
        if not LUXTTS_AVAILABLE:
            raise ImportError("LuxTTS not available. Ensure vendor/LuxTTS is on sys.path.")
        super().__init__(model_path=model_path, device=device, threads=threads)
        self._patch_transcriber_for_english()

    def _patch_transcriber_for_english(self):
        """Force English language detection and enable timestamps for long audio."""
        original_transcriber = self.transcriber

        def english_transcriber(audio, **kwargs):
            result = original_transcriber(
                audio,
                generate_kwargs={"language": "en", "task": "transcribe"},
                return_timestamps=True,  # Required for audio > 30s
                **kwargs,
            )
            return result

        self.transcriber = english_transcriber
        logger.info("Patched transcriber for English-only mode")

    def generate_speech_streaming(
        self,
        text: str,
        encode_dict: dict,
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """Stream speech generation by splitting text into sentences.

        Yields audio tensors sentence-by-sentence for low-latency playback.

        Args:
            text: Text to synthesize.
            encode_dict: Voice prompt tensors from encode_prompt().
            num_steps: Diffusion steps (more = higher quality, slower).
            guidance_scale: Classifier-free guidance scale.
            t_shift: Timestep shift for diffusion.
            speed: Speech rate (1.0 = normal).
            return_smooth: Whether to return smoothed audio.

        Yields:
            Audio tensors (one per sentence).
        """
        text = pad_tts_text(text)
        if not text:
            return

        sentences = split_into_sentences(text)

        if len(sentences) <= 1:
            padded_text = pad_tts_text(text)
            if not padded_text:
                return
            wav = self._generate_speech_safe(
                padded_text,
                encode_dict,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth,
            )
            if wav is not None:
                yield wav
            return

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            padded = pad_tts_text(sentence)
            if not padded:
                continue
            wav = self._generate_speech_safe(
                padded,
                encode_dict,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth,
            )
            if wav is not None:
                yield wav

    def _generate_speech_safe(
        self,
        text: str,
        encode_dict: dict,
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False,
    ) -> torch.Tensor | None:
        """Generate speech with error recovery for vocoder kernel issues.

        If the vocoder fails due to kernel size issues (text too short),
        automatically retries with padded text.

        Args:
            text: Text to synthesize.
            encode_dict: Voice prompt tensors.
            num_steps: Diffusion steps.
            guidance_scale: CFG scale.
            t_shift: Timestep shift.
            speed: Speech rate.
            return_smooth: Whether to return smoothed audio.

        Returns:
            Audio tensor, or None on failure.
        """
        try:
            return self.generate_speech(
                text,
                encode_dict,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth,
            )
        except RuntimeError as e:
            message = str(e)
            if "Kernel size" in message or "kernel size" in message or "padded input size" in message:
                padded = pad_tts_text(text, min_chars=160)
                if padded and padded != text:
                    try:
                        return self.generate_speech(
                            padded,
                            encode_dict,
                            num_steps=num_steps,
                            guidance_scale=guidance_scale,
                            t_shift=t_shift,
                            speed=speed,
                            return_smooth=return_smooth,
                        )
                    except Exception as retry_error:
                        logger.error(f"TTS retry failed after padding: {retry_error}", exc_info=True)
                        return None
            logger.error(f"TTS error for '{text[:50]}': {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"TTS error for '{text[:50]}': {e}", exc_info=True)
            return None


class SimpleLuxTTS:
    """Simplified LuxTTS wrapper for standalone use (e.g., ParrotBot).

    Wraps the base LuxTTS without the streaming extensions.
    Used when the full StreamingLuxTTS is not needed.
    """

    def __init__(self, device: str = "cuda"):
        if not LUXTTS_AVAILABLE:
            raise ImportError("LuxTTS not available. Ensure vendor/LuxTTS is on sys.path.")
        self.device = device
        logger.info(f"Loading LuxTTS on {device}...")
        self.tts = LuxTTS(device=device)
        logger.info("LuxTTS ready")

    def encode_prompt(self, audio_path: str, rms: float = 0.01, duration: float = 5.0) -> dict:
        """Encode a reference audio file for voice cloning."""
        return self.tts.encode_prompt(audio_path, duration=duration, rms=rms)

    def generate_speech(self, text: str, voice_prompt: dict, num_steps: int = 4, speed: float = 1.0):
        """Generate speech from text using the voice prompt."""
        return self.tts.generate_speech(text, voice_prompt, num_steps=num_steps, speed=speed)
