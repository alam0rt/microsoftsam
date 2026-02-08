"""Wyoming protocol TTS client provider.

This provider connects to a Wyoming-compatible TTS server like wyoming-piper
or our wyoming_tts_server (LuxTTS wrapper).

Usage:
    from mumble_voice_bot.providers.wyoming_tts import WyomingTTS

    tts = WyomingTTS(host="localhost", port=10400)
    result = await tts.synthesize("Hello world")
    # result.audio contains raw PCM bytes
"""

import asyncio
from typing import AsyncIterator

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize

from mumble_voice_bot.interfaces.tts import TTSProvider, TTSResult, TTSVoice


class WyomingTTS(TTSProvider):
    """TTS provider using Wyoming protocol.

    Connects to a Wyoming-compatible TTS server over TCP and requests
    speech synthesis using the Wyoming protocol.

    Attributes:
        host: Server hostname.
        port: Server port (default: 10400 for TTS).
    """

    def __init__(self, host: str = "localhost", port: int = 10400):
        """Initialize the Wyoming TTS provider.

        Args:
            host: Wyoming TTS server hostname.
            port: Wyoming TTS server port.
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
            TTSResult with audio data and format info.
        """
        client = await self._get_client()

        try:
            await client.write_event(Synthesize(text=text, voice=voice).event())

            audio_chunks = []
            sample_rate = 48000
            sample_width = 2
            channels = 1

            while True:
                event = await client.read_event()

                if AudioStart.is_type(event.type):
                    audio_start = AudioStart.from_event(event)
                    sample_rate = audio_start.rate
                    sample_width = audio_start.width
                    channels = audio_start.channels
                    continue

                if AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    audio_chunks.append(chunk.audio)
                    continue

                if AudioStop.is_type(event.type):
                    break

            audio_data = b"".join(audio_chunks)
            duration = len(audio_data) / (sample_rate * sample_width * channels)

            return TTSResult(
                audio=audio_data,
                sample_rate=sample_rate,
                sample_width=sample_width,
                channels=channels,
                duration=duration,
            )

        finally:
            await client.disconnect()

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Synthesize text to audio, yielding chunks as they arrive.

        Args:
            text: Text to synthesize.
            voice: Optional voice identifier.

        Yields:
            Raw PCM audio chunks.
        """
        client = await self._get_client()

        try:
            await client.write_event(Synthesize(text=text, voice=voice).event())

            while True:
                event = await client.read_event()

                if AudioStart.is_type(event.type):
                    continue

                if AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    yield chunk.audio
                    continue

                if AudioStop.is_type(event.type):
                    break

        finally:
            await client.disconnect()

    async def get_voices(self) -> list[TTSVoice]:
        """Get available voices from the server.

        Returns:
            List of available TTSVoice objects.
        """
        try:
            info = await self.get_info()
            voices = []

            if info.tts:
                for program in info.tts:
                    if program.voices:
                        for voice in program.voices:
                            voices.append(TTSVoice(
                                name=voice.name,
                                description=voice.description,
                                languages=voice.languages,
                            ))

            return voices
        except Exception:
            return []

    async def is_available(self) -> bool:
        """Check if the Wyoming TTS server is available.

        Returns:
            True if the server is reachable and responding.
        """
        try:
            info = await self.get_info()
            return info.tts is not None and len(info.tts) > 0
        except Exception:
            return False


class WyomingTTSSync:
    """Synchronous wrapper around WyomingTTS for non-async code.

    This is useful for integrating with synchronous codebases while
    still using the Wyoming protocol under the hood.
    """

    def __init__(self, host: str = "localhost", port: int = 10400):
        """Initialize the synchronous Wyoming TTS wrapper.

        Args:
            host: Wyoming TTS server hostname.
            port: Wyoming TTS server port.
        """
        self._async_provider = WyomingTTS(host=host, port=port)

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context, use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=120.0)
        else:
            # Create a new event loop
            return asyncio.run(coro)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
    ) -> TTSResult:
        """Synthesize text to audio (synchronous).

        Args:
            text: Text to synthesize.
            voice: Optional voice identifier.

        Returns:
            TTSResult with audio data.
        """
        return self._run_async(
            self._async_provider.synthesize(text=text, voice=voice)
        )

    def get_voices(self) -> list[TTSVoice]:
        """Get available voices (synchronous).

        Returns:
            List of available voices.
        """
        return self._run_async(self._async_provider.get_voices())

    def is_available(self) -> bool:
        """Check if the server is available (synchronous).

        Returns:
            True if the server is responding.
        """
        return self._run_async(self._async_provider.is_available())
