"""Tests for Wyoming STT provider."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from wyoming.asr import Transcript
from wyoming.info import Info

from mumble_voice_bot.providers.wyoming_stt import WyomingSTT, WyomingSTTSync
from mumble_voice_bot.interfaces.stt import STTResult


class TestWyomingSTT:
    """Test suite for WyomingSTT async provider."""
    
    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return WyomingSTT(host="localhost", port=10300)
    
    @pytest.mark.asyncio
    async def test_transcribe(self, provider, test_audio_16k_pcm):
        """Test transcribing audio."""
        expected_text = "Hello, world!"
        mock_transcript = Transcript(text=expected_text)
        
        mock_client = AsyncMock()
        mock_client.write_event = AsyncMock()
        mock_client.read_event = AsyncMock(return_value=mock_transcript.event())
        mock_client.disconnect = AsyncMock()
        
        with patch.object(provider, '_get_client', return_value=mock_client):
            result = await provider.transcribe(
                audio_data=test_audio_16k_pcm,
                sample_rate=16000,
                language="en",
            )
            
            assert isinstance(result, STTResult)
            assert result.text == expected_text
            assert result.language == "en"
            # Verify correct events were sent (transcribe, audio-start, chunks, audio-stop)
            assert mock_client.write_event.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_transcribe_empty_result(self, provider, test_audio_16k_pcm):
        """Test handling of empty transcription result."""
        # Simulate non-transcript event
        mock_event = MagicMock()
        mock_event.type = "unknown"
        
        mock_client = AsyncMock()
        mock_client.write_event = AsyncMock()
        mock_client.read_event = AsyncMock(return_value=mock_event)
        mock_client.disconnect = AsyncMock()
        
        with patch.object(provider, '_get_client', return_value=mock_client):
            result = await provider.transcribe(
                audio_data=test_audio_16k_pcm,
                sample_rate=16000,
            )
            
            assert result.text == ""
    
    @pytest.mark.asyncio
    async def test_is_available_success(self, provider, mock_wyoming_info):
        """Test server availability check when server is up."""
        mock_client = AsyncMock()
        mock_client.write_event = AsyncMock()
        mock_client.read_event = AsyncMock(return_value=mock_wyoming_info.event())
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


class TestWyomingSTTSync:
    """Test suite for WyomingSTTSync synchronous wrapper."""
    
    @pytest.fixture
    def provider(self):
        """Create a sync provider instance."""
        return WyomingSTTSync(host="localhost", port=10300)
    
    def test_transcribe_sync(self, provider, test_audio_16k_pcm):
        """Test synchronous transcription."""
        expected_text = "Test transcription"
        mock_result = STTResult(text=expected_text, language="en")
        
        with patch.object(
            provider._async_provider, 
            'transcribe', 
            new_callable=AsyncMock,
            return_value=mock_result
        ):
            result = provider.transcribe(
                audio_data=test_audio_16k_pcm,
                sample_rate=16000,
            )
            
            assert result.text == expected_text
    
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


class TestWyomingSTTIntegration:
    """Integration tests requiring a running Wyoming STT server.
    
    These tests are skipped by default. Set RUN_INTEGRATION_TESTS=1
    environment variable and ensure a Wyoming STT server is running
    on localhost:10300.
    """
    
    @pytest.fixture
    def live_provider(self):
        """Create a provider for integration tests."""
        return WyomingSTT(host="localhost", port=10300)
    
    @pytest.mark.skipif(
        True,  # Change to check for RUN_INTEGRATION_TESTS env var
        reason="Integration test - requires running Wyoming STT server"
    )
    @pytest.mark.asyncio
    async def test_real_server_info(self, live_provider):
        """Test against a real Wyoming server."""
        info = await live_provider.get_info()
        assert info.asr is not None
        assert len(info.asr) > 0
    
    @pytest.mark.skipif(
        True,
        reason="Integration test - requires running Wyoming STT server"
    )
    @pytest.mark.asyncio
    async def test_real_transcription(self, live_provider, test_audio_16k_pcm):
        """Test real transcription (won't get meaningful text from sine wave)."""
        result = await live_provider.transcribe(test_audio_16k_pcm)
        assert isinstance(result, STTResult)
        assert isinstance(result.text, str)
