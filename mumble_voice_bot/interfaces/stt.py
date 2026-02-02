"""Abstract interface for Speech-to-Text providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class STTResult:
    """Result from speech-to-text transcription.
    
    Attributes:
        text: The transcribed text.
        language: Detected or specified language (if available).
        duration: Audio duration in seconds (if available).
        confidence: Confidence score 0-1 (if available).
    """
    text: str
    language: str | None = None
    duration: float | None = None
    confidence: float | None = None


class STTProvider(ABC):
    """Abstract base class for Speech-to-Text providers.
    
    All STT implementations (Whisper, Wyoming, etc.) should inherit
    from this class and implement the transcribe() method.
    """
    
    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Transcribe audio data to text.
        
        Args:
            audio_data: Raw PCM audio bytes.
            sample_rate: Audio sample rate in Hz (default: 16000).
            sample_width: Bytes per sample, 2 = 16-bit (default: 2).
            channels: Number of audio channels (default: 1).
            language: Optional language hint (e.g., "en").
        
        Returns:
            STTResult containing the transcribed text and metadata.
        
        Raises:
            Exception: If transcription fails.
        """
        pass
    
    @abstractmethod
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
            sample_rate: Audio sample rate in Hz.
            sample_width: Bytes per sample.
            channels: Number of audio channels.
            language: Optional language hint.
        
        Returns:
            STTResult containing the transcribed text.
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the STT service is available.
        
        Returns:
            True if the service is reachable and responding.
        """
        pass
