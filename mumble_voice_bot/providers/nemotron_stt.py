"""Streaming ASR using NVIDIA Nemotron-Speech via NeMo.

This provider uses the full NeMo framework for inference.
For lighter deployment, consider sherpa_nemotron.py instead.

Model: nvidia/nemotron-speech-streaming-en-0.6b
Time-to-first-token: ~24ms (80ms chunk)
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.transcript_stabilizer import TranscriptStabilizer

logger = logging.getLogger(__name__)

# Lazy import NeMo to allow graceful fallback
try:
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    import torch
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    torch = None
    nemo_asr = None
    sf = None


@dataclass
class NemotronConfig:
    """Configuration for Nemotron streaming ASR.

    Attributes:
        model_name: HuggingFace model identifier.
        chunk_size_ms: Processing chunk size in milliseconds.
                      Options: 80, 160, 560, 1120
                      Smaller = lower latency, slightly lower accuracy.
        device: Compute device ("cuda" or "cpu").
    """
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    chunk_size_ms: int = 160  # 80, 160, 560, or 1120
    device: str = "cuda"


class NemotronStreamingASR(STTProvider):
    """
    Streaming ASR using NVIDIA Nemotron-Speech via NeMo.

    Provides partial results with ~24-160ms latency depending on chunk size.

    Features:
    - True streaming ASR with cache-aware architecture
    - Configurable latency/accuracy tradeoff via chunk_size_ms
    - Built-in punctuation and casing
    - English-only but highly optimized for voice agents

    Note: This requires the full NeMo framework. For lighter deployment,
    use SherpaNemotronASR which uses ONNX runtime.

    Usage:
        config = NemotronConfig(chunk_size_ms=160)
        asr = NemotronStreamingASR(config)
        await asr.initialize()

        async for text, is_final in asr.transcribe_streaming(audio_stream):
            print(f"{'FINAL' if is_final else 'PARTIAL'}: {text}")
    """

    def __init__(self, config: NemotronConfig = None):
        """Initialize the NeMo Nemotron ASR.

        Args:
            config: Configuration for the model.
        """
        import threading
        self.config = config or NemotronConfig()
        self.model = None
        self.stabilizer = TranscriptStabilizer()
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._transcribe_lock = threading.Lock()  # Protect model from concurrent access

    async def initialize(self) -> bool:
        """Load the model (do this once at startup).

        Returns:
            True if initialization succeeded.
        """
        # Use lock to prevent concurrent initialization attempts
        async with self._init_lock:
            if self._initialized:
                return True

            if not NEMO_AVAILABLE:
                logger.error("NeMo not available. Install with: pip install nemo_toolkit[asr]")
                return False

            try:
                logger.info(f"Loading Nemotron model: {self.config.model_name}")

                # Load model in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None,
                    lambda: nemo_asr.models.ASRModel.from_pretrained(
                        self.config.model_name
                    ).to(self.config.device)
                )
                self.model.eval()

                self._initialized = True
                logger.info(f"Nemotron initialized (chunk_size={self.config.chunk_size_ms}ms)")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize Nemotron: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False

    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> AsyncIterator[tuple[str, bool]]:
        """Transcribe audio stream, yielding partial results.

        Note: This implementation collects audio and transcribes in chunks
        since the NeMo streaming API requires special handling.
        For true low-latency streaming, consider using sherpa_nemotron.py.

        Args:
            audio_stream: Async iterator yielding PCM audio bytes (16-bit).
            sample_rate: Audio sample rate (should be 16000).

        Yields:
            Tuples of (text, is_final) where text is stable transcript
            and is_final indicates end of utterance.
        """
        if not self._initialized:
            if not await self.initialize():
                raise RuntimeError("Failed to initialize model")

        self.stabilizer.reset()

        # Collect audio chunks and transcribe periodically
        audio_buffer = []
        chunk_samples = int(self.config.chunk_size_ms * sample_rate / 1000)
        chunk_bytes = chunk_samples * 2  # 16-bit audio = 2 bytes per sample

        accumulated_bytes = b""

        async for chunk_data in audio_stream:
            accumulated_bytes += chunk_data

            # Process when we have enough data
            while len(accumulated_bytes) >= chunk_bytes:
                # Extract chunk
                chunk = accumulated_bytes[:chunk_bytes]
                accumulated_bytes = accumulated_bytes[chunk_bytes:]

                # Append to buffer
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                audio_buffer.append(audio_int16)

                # Transcribe accumulated audio
                if len(audio_buffer) > 0:
                    full_audio = np.concatenate(audio_buffer)
                    transcription = await self._transcribe_numpy(full_audio, sample_rate)

                    if transcription:
                        stable_delta, _, _ = self.stabilizer.update(transcription)
                        if stable_delta:
                            yield stable_delta, False

        # Process remaining audio
        if accumulated_bytes:
            audio_int16 = np.frombuffer(accumulated_bytes, dtype=np.int16)
            audio_buffer.append(audio_int16)

        # Final transcription
        if audio_buffer:
            full_audio = np.concatenate(audio_buffer)
            final_transcription = await self._transcribe_numpy(full_audio, sample_rate)
            if final_transcription:
                final_text = self.stabilizer.finalize(final_transcription)
                if final_text:
                    yield final_text, True

    async def _transcribe_numpy(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe numpy audio array.

        Args:
            audio: Audio as int16 numpy array.
            sample_rate: Sample rate of audio.

        Returns:
            Transcribed text.
        """
        # Convert to float32
        audio_float = audio.astype(np.float32) / 32768.0

        # NeMo's transcribe() expects file paths or numpy arrays
        # Use numpy array directly via transcribe()
        loop = asyncio.get_event_loop()

        def do_transcribe():
            # Use lock to prevent concurrent model access (causes CUDA graph corruption)
            with self._transcribe_lock:
                with torch.no_grad():
                    try:
                        # Try to transcribe numpy array directly (NeMo 2.0+)
                        # This avoids slow disk I/O from temp file creation
                        result = self.model.transcribe([audio_float])
                        if not result:
                            return ""

                        text = result[0]

                        # Handle nested lists (NeMo sometimes returns [[text]])
                        while isinstance(text, list) and len(text) > 0:
                            text = text[0]

                        # Handle Hypothesis objects (have .text attribute)
                        if hasattr(text, 'text'):
                            text = text.text

                        # Ensure we return a string
                        return str(text).strip() if text else ""
                    except Exception as e:
                        # Fallback to temp file if numpy transcription fails
                        logger.warning(f"Direct numpy transcription failed, using temp file: {e}")
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                            temp_path = f.name
                            sf.write(temp_path, audio_float, sample_rate)

                        try:
                            result = self.model.transcribe([temp_path])
                            if not result:
                                return ""

                            text = result[0]
                            while isinstance(text, list) and len(text) > 0:
                                text = text[0]
                            if hasattr(text, 'text'):
                                text = text.text
                            return str(text).strip() if text else ""
                        finally:
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)

        return await loop.run_in_executor(None, do_transcribe)

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Non-streaming transcription.

        Args:
            audio_data: Raw PCM audio bytes.
            sample_rate: Audio sample rate.
            sample_width: Bytes per sample.
            channels: Number of audio channels.
            language: Language hint (ignored, English-only).

        Returns:
            STTResult with transcribed text.
        """
        if not self._initialized:
            if not await self.initialize():
                raise RuntimeError("Failed to initialize model")

        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        text = await self._transcribe_numpy(audio_int16, sample_rate)

        duration = len(audio_data) / (sample_rate * sample_width * channels)

        return STTResult(
            text=text,
            language="en",
            duration=duration,
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Transcribe streaming audio data.

        Args:
            audio_stream: Async iterator yielding audio chunks.
            sample_rate: Audio sample rate.
            sample_width: Bytes per sample.
            channels: Number of audio channels.
            language: Language hint (ignored).

        Returns:
            STTResult with final transcription.
        """
        full_text = ""

        async for text, is_final in self.transcribe_streaming(audio_stream, sample_rate):
            full_text += text

        return STTResult(
            text=full_text,
            language="en",
        )

    async def is_available(self) -> bool:
        """Check if NeMo is available and model is loaded.

        Returns:
            True if ready for transcription.
        """
        if not NEMO_AVAILABLE:
            return False

        if not self._initialized:
            return await self.initialize()

        return self.model is not None
