"""End-to-end integration tests for Wyoming protocol components."""
import pytest
import asyncio
import os

from mumble_voice_bot.providers.wyoming_stt import WyomingSTT
from mumble_voice_bot.providers.wyoming_tts import WyomingTTS
from mumble_voice_bot.interfaces.stt import STTResult
from mumble_voice_bot.interfaces.tts import TTSResult


# Check if integration tests should run
RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"
WYOMING_STT_HOST = os.environ.get("WYOMING_STT_HOST", "localhost")
WYOMING_STT_PORT = int(os.environ.get("WYOMING_STT_PORT", "10300"))
WYOMING_TTS_HOST = os.environ.get("WYOMING_TTS_HOST", "localhost")
WYOMING_TTS_PORT = int(os.environ.get("WYOMING_TTS_PORT", "10400"))


@pytest.fixture
def stt_provider():
    """Create STT provider for integration tests."""
    return WyomingSTT(host=WYOMING_STT_HOST, port=WYOMING_STT_PORT)


@pytest.fixture
def tts_provider():
    """Create TTS provider for integration tests."""
    return WyomingTTS(host=WYOMING_TTS_HOST, port=WYOMING_TTS_PORT)


class TestEndToEnd:
    """End-to-end integration tests.
    
    These tests require running Wyoming servers:
    - wyoming-faster-whisper on WYOMING_STT_HOST:WYOMING_STT_PORT
    - wyoming-luxtts (or similar) on WYOMING_TTS_HOST:WYOMING_TTS_PORT
    
    Set RUN_INTEGRATION_TESTS=1 to enable these tests.
    """
    
    @pytest.mark.skipif(not RUN_INTEGRATION, reason="Integration tests disabled")
    @pytest.mark.asyncio
    async def test_stt_server_available(self, stt_provider):
        """Test that STT server is available."""
        is_available = await stt_provider.is_available()
        assert is_available, f"STT server not available at {WYOMING_STT_HOST}:{WYOMING_STT_PORT}"
    
    @pytest.mark.skipif(not RUN_INTEGRATION, reason="Integration tests disabled")
    @pytest.mark.asyncio
    async def test_tts_server_available(self, tts_provider):
        """Test that TTS server is available."""
        is_available = await tts_provider.is_available()
        assert is_available, f"TTS server not available at {WYOMING_TTS_HOST}:{WYOMING_TTS_PORT}"
    
    @pytest.mark.skipif(not RUN_INTEGRATION, reason="Integration tests disabled")
    @pytest.mark.asyncio
    async def test_stt_transcription(self, stt_provider, test_audio_16k_pcm):
        """Test actual transcription through Wyoming STT."""
        result = await stt_provider.transcribe(
            audio_data=test_audio_16k_pcm,
            sample_rate=16000,
            language="en",
        )
        
        assert isinstance(result, STTResult)
        assert isinstance(result.text, str)
        # Note: sine wave won't produce meaningful text
    
    @pytest.mark.skipif(not RUN_INTEGRATION, reason="Integration tests disabled")
    @pytest.mark.asyncio
    async def test_tts_synthesis(self, tts_provider):
        """Test actual synthesis through Wyoming TTS."""
        result = await tts_provider.synthesize(
            text="Hello, this is a test of the text to speech system."
        )
        
        assert isinstance(result, TTSResult)
        assert len(result.audio) > 0
        assert result.sample_rate > 0
        assert result.duration > 0
    
    @pytest.mark.skipif(not RUN_INTEGRATION, reason="Integration tests disabled")
    @pytest.mark.asyncio
    async def test_tts_streaming(self, tts_provider):
        """Test streaming synthesis through Wyoming TTS."""
        chunks = []
        async for chunk in tts_provider.synthesize_stream(
            text="Testing streaming synthesis."
        ):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        total_audio = b"".join(chunks)
        assert len(total_audio) > 0
    
    @pytest.mark.skipif(not RUN_INTEGRATION, reason="Integration tests disabled")
    @pytest.mark.asyncio
    async def test_roundtrip_tts_to_stt(self, stt_provider, tts_provider):
        """Test TTS -> STT roundtrip.
        
        Generate speech with TTS, then transcribe it with STT.
        The transcription should roughly match the input text.
        """
        input_text = "Hello world"
        
        # Generate speech
        tts_result = await tts_provider.synthesize(input_text)
        assert len(tts_result.audio) > 0
        
        # The TTS output is 48kHz, but STT expects 16kHz
        # In a real scenario, you'd resample here
        # For now, we'll just verify the audio was generated
        
        # If we had resampling, we could do:
        # stt_result = await stt_provider.transcribe(
        #     audio_data=resampled_audio,
        #     sample_rate=16000,
        # )
        # assert "hello" in stt_result.text.lower()


class TestProviderInterfaces:
    """Tests for provider interface compliance."""
    
    @pytest.mark.asyncio
    async def test_stt_provider_interface(self, stt_provider):
        """Test that STT provider implements the interface correctly."""
        from mumble_voice_bot.interfaces.stt import STTProvider
        
        assert isinstance(stt_provider, STTProvider)
        
        # Check all required methods exist
        assert hasattr(stt_provider, 'transcribe')
        assert hasattr(stt_provider, 'transcribe_stream')
        assert hasattr(stt_provider, 'is_available')
        
        # Check they're callable
        assert callable(stt_provider.transcribe)
        assert callable(stt_provider.transcribe_stream)
        assert callable(stt_provider.is_available)
    
    @pytest.mark.asyncio
    async def test_tts_provider_interface(self, tts_provider):
        """Test that TTS provider implements the interface correctly."""
        from mumble_voice_bot.interfaces.tts import TTSProvider
        
        assert isinstance(tts_provider, TTSProvider)
        
        # Check all required methods exist
        assert hasattr(tts_provider, 'synthesize')
        assert hasattr(tts_provider, 'synthesize_stream')
        assert hasattr(tts_provider, 'get_voices')
        assert hasattr(tts_provider, 'is_available')
        
        # Check they're callable
        assert callable(tts_provider.synthesize)
        assert callable(tts_provider.synthesize_stream)
        assert callable(tts_provider.get_voices)
        assert callable(tts_provider.is_available)
