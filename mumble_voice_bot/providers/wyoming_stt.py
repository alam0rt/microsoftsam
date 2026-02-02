"""Wyoming protocol STT provider.

This provider connects to a Wyoming-compatible STT server like wyoming-faster-whisper.

Usage:
    from mumble_voice_bot.providers.wyoming_stt import WyomingSTT
    
    stt = WyomingSTT(host="localhost", port=10300)
    result = await stt.transcribe(audio_data, sample_rate=16000)
    print(result.text)
"""

import asyncio
from typing import AsyncIterator

from wyoming.client import AsyncTcpClient
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info

from mumble_voice_bot.interfaces.stt import STTProvider, STTResult


class WyomingSTT(STTProvider):
    """STT provider using Wyoming protocol (e.g., wyoming-faster-whisper).
    
    Connects to a Wyoming-compatible STT server over TCP and sends audio
    for transcription using the Wyoming protocol.
    
    Attributes:
        host: Server hostname.
        port: Server port (default: 10300 for faster-whisper).
    """
    
    def __init__(self, host: str = "localhost", port: int = 10300):
        """Initialize the Wyoming STT provider.
        
        Args:
            host: Wyoming STT server hostname.
            port: Wyoming STT server port.
        """
        self.host = host
        self.port = port
    
    async def _get_client(self) -> AsyncTcpClient:
        """Create and connect a new client."""
        client = AsyncTcpClient(self.host, self.port)
        await client.connect()
        return client
    
    async def get_info(self) -> Info:
        """Get server capabilities.
        
        Returns:
            Wyoming Info object with server details.
        """
        client = await self._get_client()
        try:
            await client.write_event(Describe().event())
            event = await client.read_event()
            return Info.from_event(event)
        finally:
            await client.disconnect()
    
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
            sample_width: Bytes per sample (default: 2 for 16-bit).
            channels: Number of audio channels (default: 1).
            language: Optional language hint (e.g., "en").
        
        Returns:
            STTResult with transcribed text.
        """
        client = await self._get_client()
        
        try:
            # Send transcribe request with optional language
            await client.write_event(
                Transcribe(language=language).event()
            )
            
            # Send audio start
            await client.write_event(
                AudioStart(
                    rate=sample_rate,
                    width=sample_width,
                    channels=channels,
                ).event()
            )
            
            # Send audio chunks (1 second each for efficiency)
            chunk_size = sample_rate * sample_width * channels
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await client.write_event(
                    AudioChunk(
                        audio=chunk,
                        rate=sample_rate,
                        width=sample_width,
                        channels=channels,
                    ).event()
                )
            
            # Send audio stop
            await client.write_event(AudioStop().event())
            
            # Wait for transcript
            event = await client.read_event()
            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                # Calculate duration from audio data
                duration = len(audio_data) / (sample_rate * sample_width * channels)
                return STTResult(
                    text=transcript.text,
                    language=language,
                    duration=duration,
                )
            
            return STTResult(text="", language=language)
            
        finally:
            await client.disconnect()
    
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
            STTResult with transcribed text.
        """
        client = await self._get_client()
        
        try:
            await client.write_event(
                Transcribe(language=language).event()
            )
            
            await client.write_event(
                AudioStart(
                    rate=sample_rate,
                    width=sample_width,
                    channels=channels,
                ).event()
            )
            
            total_bytes = 0
            async for chunk in audio_stream:
                total_bytes += len(chunk)
                await client.write_event(
                    AudioChunk(
                        audio=chunk,
                        rate=sample_rate,
                        width=sample_width,
                        channels=channels,
                    ).event()
                )
            
            await client.write_event(AudioStop().event())
            
            event = await client.read_event()
            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                duration = total_bytes / (sample_rate * sample_width * channels)
                return STTResult(
                    text=transcript.text,
                    language=language,
                    duration=duration,
                )
            
            return STTResult(text="", language=language)
            
        finally:
            await client.disconnect()
    
    async def is_available(self) -> bool:
        """Check if the Wyoming STT server is available.
        
        Returns:
            True if the server is reachable and responding.
        """
        try:
            info = await self.get_info()
            return info.asr is not None and len(info.asr) > 0
        except Exception:
            return False


class WyomingSTTSync:
    """Synchronous wrapper around WyomingSTT for non-async code.
    
    This is useful for integrating with synchronous codebases while
    still using the Wyoming protocol under the hood.
    """
    
    def __init__(self, host: str = "localhost", port: int = 10300):
        """Initialize the synchronous Wyoming STT wrapper.
        
        Args:
            host: Wyoming STT server hostname.
            port: Wyoming STT server port.
        """
        self._async_provider = WyomingSTT(host=host, port=port)
    
    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # We're in an async context, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=60.0)
        else:
            # Create a new event loop
            return asyncio.run(coro)
    
    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Transcribe audio data to text (synchronous).
        
        Args:
            audio_data: Raw PCM audio bytes.
            sample_rate: Audio sample rate in Hz.
            sample_width: Bytes per sample.
            channels: Number of audio channels.
            language: Optional language hint.
        
        Returns:
            STTResult with transcribed text.
        """
        return self._run_async(
            self._async_provider.transcribe(
                audio_data=audio_data,
                sample_rate=sample_rate,
                sample_width=sample_width,
                channels=channels,
                language=language,
            )
        )
    
    def is_available(self) -> bool:
        """Check if the server is available (synchronous).
        
        Returns:
            True if the server is responding.
        """
        return self._run_async(self._async_provider.is_available())
