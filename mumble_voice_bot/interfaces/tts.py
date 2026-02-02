"""Abstract interface for Text-to-Speech providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis.
    
    Attributes:
        audio: Raw PCM audio bytes.
        sample_rate: Audio sample rate in Hz.
        sample_width: Bytes per sample (2 = 16-bit).
        channels: Number of audio channels.
        duration: Audio duration in seconds (if available).
    """
    audio: bytes
    sample_rate: int = 48000
    sample_width: int = 2
    channels: int = 1
    duration: float | None = None


@dataclass 
class TTSVoice:
    """Information about a TTS voice.
    
    Attributes:
        name: Voice identifier.
        description: Human-readable description.
        languages: List of supported language codes.
    """
    name: str
    description: str | None = None
    languages: list[str] | None = None


class TTSProvider(ABC):
    """Abstract base class for Text-to-Speech providers.
    
    All TTS implementations (LuxTTS, Wyoming/Piper, etc.) should inherit
    from this class and implement the synthesize() method.
    """
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
    ) -> TTSResult:
        """Synthesize text to audio.
        
        Args:
            text: Text to synthesize.
            voice: Optional voice identifier.
        
        Returns:
            TTSResult containing the audio data and format info.
        
        Raises:
            Exception: If synthesis fails.
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Synthesize text to audio, yielding chunks as they're generated.
        
        Args:
            text: Text to synthesize.
            voice: Optional voice identifier.
        
        Yields:
            Raw PCM audio chunks.
        """
        pass
    
    @abstractmethod
    async def get_voices(self) -> list[TTSVoice]:
        """Get available voices.
        
        Returns:
            List of available TTSVoice objects.
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the TTS service is available.
        
        Returns:
            True if the service is reachable and responding.
        """
        pass
