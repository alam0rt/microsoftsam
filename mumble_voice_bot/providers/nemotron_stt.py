"""Streaming ASR using NVIDIA Nemotron-Speech via NeMo.

This provider uses the full NeMo framework for inference.
For lighter deployment, consider sherpa_nemotron.py instead.

Model: nvidia/nemotron-speech-streaming-en-0.6b
Time-to-first-token: ~24ms (80ms chunk)
"""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
import logging

import numpy as np

from mumble_voice_bot.interfaces.stt import STTProvider, STTResult
from mumble_voice_bot.transcript_stabilizer import TranscriptStabilizer

logger = logging.getLogger(__name__)

# Lazy import NeMo to allow graceful fallback
try:
    import torch
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    torch = None
    nemo_asr = None


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
        self.config = config or NemotronConfig()
        self.model = None
        self.stabilizer = TranscriptStabilizer()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Load the model (do this once at startup).
        
        Returns:
            True if initialization succeeded.
        """
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
            
            # Configure chunk size for streaming
            chunk_samples = int(self.config.chunk_size_ms * 16)  # 16kHz sample rate
            self.model.change_decoding_strategy(
                decoding_cfg={"strategy": "greedy", "chunk_size": chunk_samples}
            )
            
            self._initialized = True
            logger.info(f"Nemotron initialized (chunk_size={self.config.chunk_size_ms}ms)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Nemotron: {e}")
            return False
    
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
        if not self._initialized:
            if not await self.initialize():
                raise RuntimeError("Failed to initialize model")
        
        self.stabilizer.reset()
        cache = None
        last_transcription = ""
        
        async for chunk_bytes in audio_stream:
            audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
            audio_float = torch.from_numpy(
                audio_int16.astype(np.float32) / 32768.0
            ).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                transcription, cache = self.model.transcribe_streaming(
                    audio_float, cache=cache, return_hypotheses=False
                )
            
            if transcription and transcription[0]:
                last_transcription = transcription[0]
                stable_delta, _, _ = self.stabilizer.update(transcription[0])
                if stable_delta:
                    yield stable_delta, False
        
        # Finalize
        if cache is not None:
            final_text = self.stabilizer.finalize(last_transcription)
            if final_text:
                yield final_text, True
    
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
        audio_float = torch.from_numpy(
            audio_int16.astype(np.float32) / 32768.0
        ).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            transcriptions = self.model.transcribe([audio_float])
        
        text = transcriptions[0] if transcriptions else ""
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
