"""Tests for the voice pipeline orchestration.

Tests cover:
- Pipeline initialization
- Audio processing (transcription)
- LLM response generation
- TTS synthesis
- Wake word filtering
- Conversation history management
- Streaming pipeline
- Interruption handling
"""

import asyncio
import time
from typing import AsyncIterator

import pytest

from mumble_voice_bot.interfaces.llm import LLMResponse
from mumble_voice_bot.pipeline import (
    PipelineConfig,
    PipelineResult,
    TranscriptionResult,
    VoicePipeline,
)

# --- Mock Providers ---


class MockWhisperTranscriber:
    """Mock Whisper transcriber."""

    def __init__(self, responses: list[dict] | None = None):
        self.responses = responses or [{"text": "Hello there", "language": "en"}]
        self.call_count = 0
        self.last_audio = None

    def __call__(self, audio):
        self.last_audio = audio
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockLLM:
    """Mock LLM provider."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Hello! How can I help you?"]
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None

    async def chat(
        self,
        messages: list[dict],
        context: dict | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        self.last_tools = tools
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return LLMResponse(content=response)

    async def chat_stream(
        self, messages: list[dict], context: dict | None = None
    ) -> AsyncIterator[str]:
        self.last_messages = messages
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        # Simulate streaming by yielding words
        words = response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.01)
            yield word if i == 0 else f" {word}"


class MockTTS:
    """Mock LuxTTS synthesizer."""

    def __init__(self, audio_result: bytes = b"mock_audio_data"):
        self.audio_result = audio_result
        self.call_count = 0
        self.last_text = None

    def generate_speech(self, text: str, voice_prompt: dict, num_steps: int = 4):
        self.last_text = text
        self.call_count += 1
        return self.audio_result


class MockStreamingTTS(MockTTS):
    """Mock TTS with streaming support."""

    def generate_speech_streaming(
        self, text: str, voice_prompt: dict, num_steps: int = 4
    ):
        self.last_text = text
        self.call_count += 1
        # Yield chunks
        for i in range(3):
            yield f"chunk_{i}".encode()


# --- Fixtures ---


@pytest.fixture
def mock_whisper():
    """Create mock Whisper transcriber."""
    return MockWhisperTranscriber()


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    return MockLLM()


@pytest.fixture
def mock_tts():
    """Create mock TTS synthesizer."""
    return MockTTS()


@pytest.fixture
def pipeline(mock_whisper, mock_llm, mock_tts):
    """Create a VoicePipeline with mocked components."""
    return VoicePipeline(
        whisper=mock_whisper,
        llm=mock_llm,
        luxtts=mock_tts,
    )


@pytest.fixture
def test_audio():
    """Generate test audio bytes."""
    return b"\x00\x01" * 8000  # 1 second at 16kHz, 16-bit


# --- Test Classes ---


class TestPipelineInit:
    """Test VoicePipeline initialization."""

    def test_init_with_all_providers(self, mock_whisper, mock_llm, mock_tts):
        """Test initialization with all providers."""
        config = PipelineConfig(
            wake_word="hey assistant",
            silence_threshold_ms=2000,
            max_recording_ms=60000,
            max_history_turns=20,
            history_timeout=600.0,
        )

        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
            config=config,
        )

        assert pipeline.whisper is mock_whisper
        assert pipeline.llm is mock_llm
        assert pipeline.luxtts is mock_tts
        assert pipeline.config.wake_word == "hey assistant"
        assert pipeline.config.silence_threshold_ms == 2000

    def test_init_with_default_config(self, mock_whisper, mock_llm, mock_tts):
        """Test initialization with default config."""
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
        )

        assert pipeline.config is not None
        assert pipeline.config.wake_word is None
        assert pipeline.config.silence_threshold_ms == 1500


class TestWakeWordFiltering:
    """Test wake word detection and filtering."""

    def test_should_respond_empty_text(self, pipeline):
        """Test that empty text is rejected."""
        should_respond, text = pipeline._should_respond("")
        assert should_respond is False
        assert text == ""

    def test_should_respond_whitespace_text(self, pipeline):
        """Test that whitespace-only text is rejected."""
        should_respond, text = pipeline._should_respond("   \n\t   ")
        assert should_respond is False
        assert text == ""

    def test_should_respond_no_wake_word_configured(self, pipeline):
        """Test that any text passes when no wake word configured."""
        should_respond, text = pipeline._should_respond("Hello there")
        assert should_respond is True
        assert text == "Hello there"

    def test_should_respond_wake_word_present(self, mock_whisper, mock_llm, mock_tts):
        """Test wake word detection when configured."""
        config = PipelineConfig(wake_word="hey assistant")
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
            config=config,
        )

        should_respond, text = pipeline._should_respond(
            "Hey assistant, what time is it?"
        )
        assert should_respond is True
        assert text == ", what time is it?"

    def test_should_respond_wake_word_missing(self, mock_whisper, mock_llm, mock_tts):
        """Test rejection when wake word is missing."""
        config = PipelineConfig(wake_word="hey assistant")
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
            config=config,
        )

        should_respond, text = pipeline._should_respond("What time is it?")
        assert should_respond is False

    def test_should_respond_wake_word_case_insensitive(
        self, mock_whisper, mock_llm, mock_tts
    ):
        """Test that wake word matching is case insensitive."""
        config = PipelineConfig(wake_word="Hey Assistant")
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
            config=config,
        )

        should_respond, text = pipeline._should_respond("HEY ASSISTANT tell me a joke")
        assert should_respond is True


class TestConversationHistory:
    """Test conversation history management."""

    def test_get_history_creates_new(self, pipeline):
        """Test that get_history creates new history for unknown user."""
        history = pipeline._get_history("user1")
        assert history == []
        assert "user1" in pipeline._conversation_history

    def test_add_to_history(self, pipeline):
        """Test adding messages to history."""
        pipeline._add_to_history("user1", "user", "Hello")
        pipeline._add_to_history("user1", "assistant", "Hi there!")

        history = pipeline._get_history("user1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

    def test_history_trim_to_max_turns(self, mock_whisper, mock_llm, mock_tts):
        """Test that history is trimmed to max_history_turns."""
        config = PipelineConfig(max_history_turns=2)  # 2 turns = 4 messages
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
            config=config,
        )

        # Add 4 turns (8 messages)
        for i in range(4):
            pipeline._add_to_history("user1", "user", f"User message {i}")
            pipeline._add_to_history("user1", "assistant", f"Response {i}")

        history = pipeline._get_history("user1")
        # Should only have 4 messages (2 turns)
        assert len(history) == 4
        # Should keep the most recent
        assert history[0]["content"] == "User message 2"

    def test_history_timeout_clears(self, mock_whisper, mock_llm, mock_tts):
        """Test that stale history is cleared after timeout."""
        config = PipelineConfig(history_timeout=0.1)  # 100ms timeout
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
            config=config,
        )

        # Add some history
        pipeline._add_to_history("user1", "user", "Hello")
        pipeline._add_to_history("user1", "assistant", "Hi!")

        # Wait for timeout
        time.sleep(0.15)

        # History should be cleared on next access
        history = pipeline._get_history("user1")
        assert len(history) == 0

    def test_clear_history_single_user(self, pipeline):
        """Test clearing history for a single user."""
        pipeline._add_to_history("user1", "user", "Hello")
        pipeline._add_to_history("user2", "user", "Hi")

        pipeline.clear_history("user1")

        assert pipeline._get_history("user1") == []
        assert len(pipeline._get_history("user2")) == 1

    def test_clear_history_all_users(self, pipeline):
        """Test clearing all history."""
        pipeline._add_to_history("user1", "user", "Hello")
        pipeline._add_to_history("user2", "user", "Hi")

        pipeline.clear_history()

        assert len(pipeline._conversation_history) == 0


class TestTranscription:
    """Test audio transcription."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_result(self, pipeline, test_audio):
        """Test that transcribe returns TranscriptionResult."""
        result = await pipeline.transcribe(test_audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello there"
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_passes_audio_to_whisper(
        self, pipeline, mock_whisper, test_audio
    ):
        """Test that audio is passed to Whisper."""
        await pipeline.transcribe(test_audio)

        assert mock_whisper.call_count == 1
        assert mock_whisper.last_audio is test_audio


class TestLLMResponse:
    """Test LLM response generation."""

    @pytest.mark.asyncio
    async def test_generate_response_simple(self, pipeline, mock_llm):
        """Test simple response generation."""
        response = await pipeline.generate_response("Hello", user_id="user1")

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_response_includes_history(self, pipeline, mock_llm):
        """Test that response generation includes conversation history."""
        # First turn
        await pipeline.generate_response("Hello", user_id="user1")

        # Second turn
        await pipeline.generate_response("How are you?", user_id="user1")

        # Check that second call included history
        assert mock_llm.call_count == 2

        # Note: last_messages holds a reference to the internal history list,
        # which gets mutated after the chat call returns. So we see 4 messages:
        # Hello, response1, How are you?, response2
        messages = mock_llm.last_messages
        assert len(messages) == 4  # Both turns complete with responses


class TestTTSSynthesis:
    """Test TTS synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self, pipeline, mock_tts):
        """Test that synthesize returns audio."""
        voice_prompt = {"encoded": "mock_encoding"}
        audio = await pipeline.synthesize("Hello!", voice_prompt)

        assert audio == b"mock_audio_data"
        assert mock_tts.call_count == 1
        assert mock_tts.last_text == "Hello!"

    @pytest.mark.asyncio
    async def test_synthesize_streaming(self, mock_whisper, mock_llm):
        """Test streaming synthesis."""
        mock_tts = MockStreamingTTS()
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
        )

        voice_prompt = {"encoded": "mock"}
        chunks = []
        async for chunk in pipeline.synthesize_streaming("Hello!", voice_prompt):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == b"chunk_0"


class TestFullPipelineProcessing:
    """Test full pipeline processing."""

    @pytest.mark.asyncio
    async def test_process_audio_full_pipeline(
        self, pipeline, mock_whisper, mock_llm, mock_tts, test_audio
    ):
        """Test full audio processing through the pipeline."""
        voice_prompt = {"encoded": "mock"}
        result = await pipeline.process_audio(
            test_audio,
            sample_rate=16000,
            user_id="user1",
            voice_prompt=voice_prompt,
        )

        assert isinstance(result, PipelineResult)
        assert result.transcription.text == "Hello there"
        assert result.llm_response.content == "Hello! How can I help you?"
        assert result.audio == b"mock_audio_data"
        assert "transcription" in result.latency
        assert "llm" in result.latency
        assert "tts" in result.latency
        assert "total" in result.latency

    @pytest.mark.asyncio
    async def test_process_audio_empty_transcription(
        self, mock_llm, mock_tts, test_audio
    ):
        """Test pipeline returns None for empty transcription."""
        mock_whisper = MockWhisperTranscriber(responses=[{"text": "", "language": "en"}])
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
        )

        result = await pipeline.process_audio(test_audio, sample_rate=16000)

        assert result is None
        assert mock_llm.call_count == 0  # LLM should not be called

    @pytest.mark.asyncio
    async def test_process_audio_without_voice_prompt(
        self, pipeline, mock_tts, test_audio
    ):
        """Test pipeline without TTS (no voice prompt)."""
        result = await pipeline.process_audio(
            test_audio,
            sample_rate=16000,
            user_id="user1",
            voice_prompt=None,  # No TTS
        )

        assert result.audio is None
        assert mock_tts.call_count == 0
        assert "tts" not in result.latency


class TestStreamingPipeline:
    """Test streaming pipeline processing."""

    @pytest.mark.asyncio
    async def test_process_audio_streaming(
        self, mock_whisper, mock_llm, test_audio
    ):
        """Test streaming audio processing."""
        mock_tts = MockStreamingTTS()
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
        )

        voice_prompt = {"encoded": "mock"}
        events = []

        async for event_type, data in pipeline.process_audio_streaming(
            test_audio,
            sample_rate=16000,
            user_id="user1",
            voice_prompt=voice_prompt,
        ):
            events.append((event_type, data))

        # Check event types
        event_types = [e[0] for e in events]
        assert "transcription" in event_types
        assert "llm_first_token" in event_types
        assert "llm_chunk" in event_types
        assert "complete" in event_types

    @pytest.mark.asyncio
    async def test_process_audio_streaming_empty_transcription(
        self, mock_llm, test_audio
    ):
        """Test streaming pipeline returns early for empty transcription."""
        mock_whisper = MockWhisperTranscriber(responses=[{"text": "", "language": "en"}])
        mock_tts = MockStreamingTTS()
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
        )

        events = []
        async for event_type, data in pipeline.process_audio_streaming(
            test_audio, sample_rate=16000
        ):
            events.append((event_type, data))

        assert len(events) == 0


class TestPipelineWithTools:
    """Test pipeline with tool support."""

    @pytest.mark.asyncio
    async def test_pipeline_maintains_tool_responses(self, pipeline, mock_llm):
        """Test that tool responses are added to history correctly."""
        # Simulate a conversation with tools
        mock_llm.responses = ["I'll search for that.", "Here are the results."]

        # First turn
        await pipeline.generate_response("Search for news", user_id="user1")

        # Simulate tool execution by adding to history manually
        pipeline._add_to_history("user1", "tool", "Tool result: Breaking news...")

        # Second turn with tool result
        await pipeline.generate_response("What did you find?", user_id="user1")

        history = pipeline._get_history("user1")
        # Should have: user, assistant, tool, user, assistant
        assert len(history) == 5


class TestPipelineEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_process_audio_with_wake_word(
        self, mock_llm, mock_tts, test_audio
    ):
        """Test pipeline with wake word configured."""
        mock_whisper = MockWhisperTranscriber(
            responses=[{"text": "hey bot what time is it", "language": "en"}]
        )
        config = PipelineConfig(wake_word="hey bot")
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
            config=config,
        )

        voice_prompt = {"encoded": "mock"}
        result = await pipeline.process_audio(
            test_audio,
            sample_rate=16000,
            voice_prompt=voice_prompt,
        )

        # Should process - wake word was present
        assert result is not None
        # Wake word should be stripped
        assert result.transcription.text == "what time is it"

    @pytest.mark.asyncio
    async def test_process_audio_missing_wake_word(
        self, mock_llm, mock_tts, test_audio
    ):
        """Test pipeline rejects audio without wake word."""
        mock_whisper = MockWhisperTranscriber(
            responses=[{"text": "what time is it", "language": "en"}]
        )
        config = PipelineConfig(wake_word="hey bot")
        pipeline = VoicePipeline(
            whisper=mock_whisper,
            llm=mock_llm,
            luxtts=mock_tts,
            config=config,
        )

        result = await pipeline.process_audio(test_audio, sample_rate=16000)

        # Should return None - no wake word
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_users_isolated_history(self, pipeline):
        """Test that multiple users have isolated history."""
        await pipeline.generate_response("User1 message", user_id="user1")
        await pipeline.generate_response("User2 message", user_id="user2")

        history1 = pipeline._get_history("user1")
        history2 = pipeline._get_history("user2")

        assert len(history1) == 2
        assert len(history2) == 2
        assert history1[0]["content"] == "User1 message"
        assert history2[0]["content"] == "User2 message"


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.wake_word is None
        assert config.silence_threshold_ms == 1500
        assert config.max_recording_ms == 30000
        assert config.max_history_turns == 10
        assert config.history_timeout == 300.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            wake_word="hey there",
            silence_threshold_ms=3000,
            max_recording_ms=60000,
            max_history_turns=5,
            history_timeout=120.0,
        )

        assert config.wake_word == "hey there"
        assert config.silence_threshold_ms == 3000
        assert config.max_recording_ms == 60000
        assert config.max_history_turns == 5
        assert config.history_timeout == 120.0


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = TranscriptionResult(text="Hello")

        assert result.text == "Hello"
        assert result.language is None
        assert result.duration == 0.0

    def test_all_values(self):
        """Test with all values."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=1.5,
        )

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 1.5


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = PipelineResult(
            transcription=TranscriptionResult(text="Hello"),
            llm_response=LLMResponse(content="Hi there!"),
        )

        assert result.audio is None
        assert result.latency == {}

    def test_all_values(self):
        """Test with all values."""
        result = PipelineResult(
            transcription=TranscriptionResult(text="Hello"),
            llm_response=LLMResponse(content="Hi!"),
            audio=b"audio_data",
            latency={"transcription": 0.1, "llm": 0.2, "tts": 0.3},
        )

        assert result.audio == b"audio_data"
        assert result.latency["transcription"] == 0.1
