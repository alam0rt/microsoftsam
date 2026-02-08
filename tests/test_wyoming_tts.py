"""Tests for Wyoming TTS client provider."""
from unittest.mock import AsyncMock, patch

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop

from mumble_voice_bot.interfaces.tts import TTSResult, TTSVoice
from mumble_voice_bot.providers.wyoming_tts import WyomingTTS, WyomingTTSSync


class TestWyomingTTS:
    """Test suite for WyomingTTS async provider."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return WyomingTTS(host="localhost", port=10400)

    @pytest.mark.asyncio
    async def test_synthesize(self, provider, test_audio_48k_pcm):
        """Test text synthesis."""
        events = [
            AudioStart(rate=48000, width=2, channels=1).event(),
            AudioChunk(audio=test_audio_48k_pcm, rate=48000, width=2, channels=1).event(),
            AudioStop().event(),
        ]

        mock_client = AsyncMock()
        mock_client.write_event = AsyncMock()
        mock_client.read_event = AsyncMock(side_effect=events)
        mock_client.disconnect = AsyncMock()

        with patch.object(provider, '_get_client', return_value=mock_client):
            result = await provider.synthesize("Hello world")

            assert isinstance(result, TTSResult)
            assert result.audio == test_audio_48k_pcm
            assert result.sample_rate == 48000
            assert result.sample_width == 2
            assert result.channels == 1

    @pytest.mark.asyncio
    async def test_synthesize_multiple_chunks(self, provider, test_audio_48k_pcm):
        """Test synthesis with multiple audio chunks."""
        chunk_size = len(test_audio_48k_pcm) // 3
        chunks = [
            test_audio_48k_pcm[i:i + chunk_size]
            for i in range(0, len(test_audio_48k_pcm), chunk_size)
        ]

        events = [
            AudioStart(rate=48000, width=2, channels=1).event(),
            *[AudioChunk(audio=c, rate=48000, width=2, channels=1).event() for c in chunks],
            AudioStop().event(),
        ]

        mock_client = AsyncMock()
        mock_client.write_event = AsyncMock()
        mock_client.read_event = AsyncMock(side_effect=events)
        mock_client.disconnect = AsyncMock()

        with patch.object(provider, '_get_client', return_value=mock_client):
            result = await provider.synthesize("Hello world")

            # Audio should be concatenated
            expected_audio = b"".join(chunks)
            assert result.audio == expected_audio

    @pytest.mark.asyncio
    async def test_synthesize_stream(self, provider, test_audio_48k_pcm):
        """Test streaming synthesis."""
        chunk_size = len(test_audio_48k_pcm) // 3
        chunks = [
            test_audio_48k_pcm[i:i + chunk_size]
            for i in range(0, len(test_audio_48k_pcm), chunk_size)
        ]

        events = [
            AudioStart(rate=48000, width=2, channels=1).event(),
            *[AudioChunk(audio=c, rate=48000, width=2, channels=1).event() for c in chunks],
            AudioStop().event(),
        ]

        mock_client = AsyncMock()
        mock_client.write_event = AsyncMock()
        mock_client.read_event = AsyncMock(side_effect=events)
        mock_client.disconnect = AsyncMock()

        with patch.object(provider, '_get_client', return_value=mock_client):
            received_chunks = []
            async for chunk in provider.synthesize_stream("Hello world"):
                received_chunks.append(chunk)

            assert len(received_chunks) == len(chunks)
            for received, expected in zip(received_chunks, chunks):
                assert received == expected

    @pytest.mark.asyncio
    async def test_get_voices(self, provider, mock_wyoming_tts_info):
        """Test getting available voices."""
        mock_client = AsyncMock()
        mock_client.write_event = AsyncMock()
        mock_client.read_event = AsyncMock(return_value=mock_wyoming_tts_info.event())
        mock_client.disconnect = AsyncMock()

        with patch.object(provider, '_get_client', return_value=mock_client):
            voices = await provider.get_voices()

            assert len(voices) == 1
            assert voices[0].name == "cloned"
            assert "en" in voices[0].languages

    @pytest.mark.asyncio
    async def test_is_available_success(self, provider, mock_wyoming_tts_info):
        """Test server availability check when server is up."""
        mock_client = AsyncMock()
        mock_client.write_event = AsyncMock()
        mock_client.read_event = AsyncMock(return_value=mock_wyoming_tts_info.event())
        mock_client.disconnect = AsyncMock()

        with patch.object(provider, '_get_client', return_value=mock_client):
            result = await provider.is_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_connection_error(self, provider):
        """Test server availability check when server is down."""
        with patch.object(provider, '_get_client', side_effect=ConnectionRefusedError()):
            result = await provider.is_available()
            assert result is False


class TestWyomingTTSSync:
    """Test suite for WyomingTTSSync synchronous wrapper."""

    @pytest.fixture
    def provider(self):
        """Create a sync provider instance."""
        return WyomingTTSSync(host="localhost", port=10400)

    def test_synthesize_sync(self, provider, test_audio_48k_pcm):
        """Test synchronous synthesis."""
        mock_result = TTSResult(
            audio=test_audio_48k_pcm,
            sample_rate=48000,
            sample_width=2,
            channels=1,
        )

        with patch.object(
            provider._async_provider,
            'synthesize',
            new_callable=AsyncMock,
            return_value=mock_result
        ):
            result = provider.synthesize("Hello world")

            assert result.audio == test_audio_48k_pcm
            assert result.sample_rate == 48000

    def test_get_voices_sync(self, provider):
        """Test synchronous voice listing."""
        mock_voices = [
            TTSVoice(name="cloned", description="Test voice", languages=["en"])
        ]

        with patch.object(
            provider._async_provider,
            'get_voices',
            new_callable=AsyncMock,
            return_value=mock_voices
        ):
            voices = provider.get_voices()

            assert len(voices) == 1
            assert voices[0].name == "cloned"

    def test_is_available_sync(self, provider):
        """Test synchronous availability check."""
        with patch.object(
            provider._async_provider,
            'is_available',
            new_callable=AsyncMock,
            return_value=True
        ):
            result = provider.is_available()
            assert result is True


class TestWyomingTTSIntegration:
    """Integration tests requiring a running Wyoming TTS server.

    These tests are skipped by default. Set RUN_INTEGRATION_TESTS=1
    environment variable and ensure a Wyoming TTS server is running
    on localhost:10400.
    """

    @pytest.fixture
    def live_provider(self):
        """Create a provider for integration tests."""
        return WyomingTTS(host="localhost", port=10400)

    @pytest.mark.skipif(
        True,  # Change to check for RUN_INTEGRATION_TESTS env var
        reason="Integration test - requires running Wyoming TTS server"
    )
    @pytest.mark.asyncio
    async def test_real_server_info(self, live_provider):
        """Test against a real Wyoming server."""
        info = await live_provider.get_info()
        assert info.tts is not None
        assert len(info.tts) > 0

    @pytest.mark.skipif(
        True,
        reason="Integration test - requires running Wyoming TTS server"
    )
    @pytest.mark.asyncio
    async def test_real_synthesis(self, live_provider):
        """Test real synthesis."""
        result = await live_provider.synthesize("Hello, this is a test.")
        assert isinstance(result, TTSResult)
        assert len(result.audio) > 0
        assert result.sample_rate > 0
