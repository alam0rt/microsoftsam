"""Streaming ASR using Nemotron via sherpa-onnx.

This is the recommended provider for NixOS deployments due to lighter
dependencies compared to full NeMo framework.

Model: nvidia/nemotron-speech-streaming-en-0.6b (ONNX format)
Time-to-first-token: ~24ms (80ms chunk)
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
import logging

import numpy as np

from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.transcript_stabilizer import TranscriptStabilizer

logger = logging.getLogger(__name__)

# Lazy import sherpa_onnx to allow graceful fallback
try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    sherpa_onnx = None


@dataclass
class SherpaNemotronConfig:
    """Config for sherpa-onnx Nemotron.
    
    Download the model files from:
    https://github.com/k2-fsa/sherpa-onnx/releases
    
    Look for: sherpa-onnx-streaming-zipformer-en-* or similar
    
    Attributes:
        encoder_path: Path to encoder ONNX model.
        decoder_path: Path to decoder ONNX model.
        joiner_path: Path to joiner ONNX model.
        tokens_path: Path to tokens.txt vocabulary file.
        chunk_size: Number of frames per chunk (affects latency).
        provider: ONNX runtime provider ("cuda" or "cpu").
        num_threads: Number of threads for inference.
    """
    encoder_path: str = "nemotron-encoder.onnx"
    decoder_path: str = "nemotron-decoder.onnx"
    joiner_path: str = "nemotron-joiner.onnx"
    tokens_path: str = "tokens.txt"
    chunk_size: int = 8  # frames
    provider: str = "cuda"  # or "cpu"
    num_threads: int = 4


class SherpaNemotronASR(STTProvider):
    """Lightweight Nemotron streaming ASR via sherpa-onnx.
    
    This provider uses sherpa-onnx for inference, which has lighter
    dependencies than the full NeMo framework. Good for NixOS deployments.
    
    Features:
    - True streaming ASR with partial results
    - ~24ms time-to-first-token
    - Transcript stabilization for reliable LLM input
    - GPU or CPU inference
    
    Usage:
        config = SherpaNemotronConfig(
            encoder_path="/path/to/encoder.onnx",
            decoder_path="/path/to/decoder.onnx",
            joiner_path="/path/to/joiner.onnx",
            tokens_path="/path/to/tokens.txt",
        )
        asr = SherpaNemotronASR(config)
        
        # Streaming transcription
        async for text, is_final in asr.transcribe_streaming(audio_stream):
            print(f"{'FINAL' if is_final else 'PARTIAL'}: {text}")
    """
    
    def __init__(self, config: SherpaNemotronConfig = None):
        """Initialize the sherpa-onnx Nemotron ASR.
        
        Args:
            config: Configuration for the model. If None, uses defaults.
        """
        self.config = config or SherpaNemotronConfig()
        self.recognizer = None
        self.stream = None
        self.stabilizer = TranscriptStabilizer()
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the recognizer.
        
        Returns:
            True if initialization succeeded.
        """
        if not SHERPA_AVAILABLE:
            logger.error("sherpa-onnx not available. Install with: pip install sherpa-onnx")
            return False
        
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=self.config.encoder_path,
                decoder=self.config.decoder_path,
                joiner=self.config.joiner_path,
                tokens=self.config.tokens_path,
                num_threads=self.config.num_threads,
                provider=self.config.provider,
            )
            self._initialized = True
            logger.info(f"Initialized sherpa-onnx Nemotron ASR (provider={self.config.provider})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize sherpa-onnx: {e}")
            return False
    
    def start_stream(self):
        """Start a new recognition stream."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize recognizer")
        
        self.stream = self.recognizer.create_stream()
        self.stabilizer.reset()
    
    def feed_audio(self, samples: list[float]) -> Optional[str]:
        """Feed audio samples, return partial transcript if available.
        
        Args:
            samples: List of float audio samples (16kHz, normalized -1 to 1).
            
        Returns:
            Stable transcript delta if available, None otherwise.
        """
        if self.stream is None:
            self.start_stream()
        
        self.stream.accept_waveform(16000, samples)
        
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        
        result = self.recognizer.get_result(self.stream)
        if result.text:
            stable_delta, _, _ = self.stabilizer.update(result.text)
            return stable_delta if stable_delta else None
        return None
    
    def finalize(self) -> str:
        """Finalize and get final transcript.
        
        Returns:
            Any remaining text not yet emitted.
        """
        if self.stream is None:
            return ""
        
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        
        final = self.recognizer.get_result(self.stream).text
        remaining = self.stabilizer.finalize(final)
        self.stream = None
        return remaining
    
    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> AsyncIterator[tuple[str, bool]]:
        """Transcribe audio stream, yielding partial results.
        
        Args:
            audio_stream: Async iterator yielding PCM audio bytes (16-bit).
            sample_rate: Audio sample rate (should be 16000).
            
        Yields:
            Tuples of (text, is_final) where text is stable transcript
            and is_final indicates end of utterance.
        """
        self.start_stream()
        
        async for chunk_bytes in audio_stream:
            audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
            samples = (audio_int16.astype(np.float32) / 32768.0).tolist()
            
            partial = self.feed_audio(samples)
            if partial:
                yield partial, False
        
        final = self.finalize()
        if final:
            yield final, True
    
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Non-streaming transcription (for compatibility).
        
        Args:
            audio_data: Raw PCM audio bytes.
            sample_rate: Audio sample rate.
            sample_width: Bytes per sample.
            channels: Number of audio channels.
            language: Language hint (ignored, English-only).
            
        Returns:
            STTResult with transcribed text.
        """
        self.start_stream()
        
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        samples = (audio_int16.astype(np.float32) / 32768.0).tolist()
        
        self.feed_audio(samples)
        text = self.finalize()
        
        # Calculate duration
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
        """Transcribe streaming audio data (required by STTProvider).
        
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
        total_bytes = 0
        
        async for text, is_final in self.transcribe_streaming(audio_stream, sample_rate):
            full_text += text
            
        return STTResult(
            text=full_text,
            language="en",
        )
    
    async def is_available(self) -> bool:
        """Check if sherpa-onnx is available and model is loaded.
        
        Returns:
            True if ready for transcription.
        """
        if not SHERPA_AVAILABLE:
            return False
        
        if not self._initialized:
            return self.initialize()
        
        return self.recognizer is not None
