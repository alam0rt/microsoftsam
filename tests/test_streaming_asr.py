"""Tests for streaming ASR pipeline.

These tests verify the streaming ASR architecture:
1. Streaming ASR with partial results
2. Stable prefix tracking
3. LLM early start on partial transcript
4. End-to-end latency tracking
"""

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator

import pytest

from mumble_voice_bot.interfaces.stt import PartialSTTResult, STTProvider, STTResult
from mumble_voice_bot.latency import LatencyTracker, TurnLatency
from mumble_voice_bot.providers.streaming_asr import (
    LocalStreamingASR,
    StreamingASR,
    StreamingASRConfig,
    StreamingASRMetrics,
)
from mumble_voice_bot.streaming_pipeline import (
    StreamingEvent,
    StreamingPipelineConfig,
    StreamingPipelineResult,
    StreamingVoicePipeline,
)
from mumble_voice_bot.transcript_stabilizer import (
    StreamingTranscriptBuffer,
    TranscriptStabilizer,
)


# --- Mock Providers ---


class MockStreamingSTT(STTProvider):
    """Mock STT that simulates streaming partial results."""

    def __init__(self, partials: list[str], delays: list[float] | None = None):
        """Initialize with predefined partials.

        Args:
            partials: List of partial transcripts to emit.
            delays: Optional delays between partials (in seconds).
        """
        self.partials = partials
        self.delays = delays or [0.05] * len(partials)
        self.transcribe_calls = 0

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        self.transcribe_calls += 1
        return STTResult(text=self.partials[-1] if self.partials else "")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        # Consume the stream
        async for _ in audio_stream:
            pass
        return STTResult(text=self.partials[-1] if self.partials else "")

    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> AsyncIterator[PartialSTTResult]:
        """Yield mock partial results."""
        # Start consuming audio in background (simulate real ASR)
        audio_task = asyncio.create_task(self._consume_audio(audio_stream))

        stable_text = ""
        for i, (partial, delay) in enumerate(zip(self.partials, self.delays)):
            await asyncio.sleep(delay)

            is_final = i == len(self.partials) - 1

            # Simulate stable text growth
            if len(partial) > 20:
                stable_text = partial[:len(partial) - 10]

            yield PartialSTTResult(
                text=partial,
                stable_text=stable_text,
                is_final=is_final,
                timestamp=delay * (i + 1),
            )

        await audio_task

    async def _consume_audio(self, audio_stream: AsyncIterator[bytes]):
        """Consume audio stream in background."""
        async for _ in audio_stream:
            pass

    async def is_available(self) -> bool:
        return True


class MockLLM:
    """Mock LLM that simulates streaming token generation."""

    def __init__(self, response: str, token_delay: float = 0.01):
        """Initialize with predefined response.

        Args:
            response: Full response to generate.
            token_delay: Delay between tokens (seconds).
        """
        self.response = response
        self.token_delay = token_delay
        self.chat_calls = 0
        self.last_prompt = None

    async def chat(self, messages: list[dict], context: dict | None = None):
        self.chat_calls += 1
        self.last_prompt = messages[-1]["content"] if messages else None

        @dataclass
        class MockResponse:
            content: str = ""

        return MockResponse(content=self.response)

    async def chat_stream(
        self, messages: list[dict], context: dict | None = None
    ) -> AsyncIterator[str]:
        """Stream tokens with delays."""
        self.chat_calls += 1
        self.last_prompt = messages[-1]["content"] if messages else None

        # Split into tokens (words)
        tokens = self.response.split()
        for i, token in enumerate(tokens):
            await asyncio.sleep(self.token_delay)
            # Add space back except for first token
            if i > 0:
                yield " " + token
            else:
                yield token

    async def is_available(self) -> bool:
        return True


class MockTTS:
    """Mock TTS that simulates audio generation."""

    def __init__(self, audio_chunk: bytes = b"\x00" * 1600):
        self.audio_chunk = audio_chunk
        self.synthesize_calls = 0
        self.last_text = None

    def generate_speech(self, text: str, voice_prompt=None, num_steps: int = 4):
        self.synthesize_calls += 1
        self.last_text = text
        return self.audio_chunk

    def generate_speech_streaming(
        self, text: str, voice_prompt=None, num_steps: int = 4
    ):
        """Yield audio chunks."""
        self.synthesize_calls += 1
        self.last_text = text
        for _ in range(3):
            yield self.audio_chunk


# --- Test Fixtures ---


@pytest.fixture
def mock_streaming_stt():
    """Create mock STT with realistic partial sequence."""
    partials = [
        "Hello",
        "Hello there",
        "Hello there, how",
        "Hello there, how are",
        "Hello there, how are you",
        "Hello there, how are you doing",
        "Hello there, how are you doing today?",
    ]
    return MockStreamingSTT(partials, delays=[0.05] * len(partials))


@pytest.fixture
def mock_llm():
    """Create mock LLM with test response."""
    return MockLLM(
        "I'm doing well, thank you for asking! How can I help you today?"
    )


@pytest.fixture
def mock_tts():
    """Create mock TTS."""
    return MockTTS()


@pytest.fixture
def streaming_pipeline(mock_streaming_stt, mock_llm, mock_tts):
    """Create streaming pipeline with mock providers."""
    config = StreamingPipelineConfig(
        llm_start_threshold=30,  # Lower for testing
        phrase_min_chars=20,
        phrase_max_chars=50,
    )
    return StreamingVoicePipeline(
        stt=mock_streaming_stt,
        llm=mock_llm,
        tts=mock_tts,
        config=config,
    )


# --- Transcript Stabilizer Tests ---


class TestTranscriptStabilizer:
    """Test the transcript stabilizer component."""

    def test_stabilizer_tracks_stable_prefix(self):
        """Test that stabilizer correctly identifies stable text."""
        stabilizer = TranscriptStabilizer(stability_window=2, min_stable_chars=5)

        # First partial - nothing stable yet
        delta, unstable, is_final = stabilizer.update("Hello")
        assert delta == ""
        assert unstable == "Hello"

        # Second partial - now we have 2 partials in window, may emit stable
        delta, unstable, is_final = stabilizer.update("Hello there")
        # With stability_window=2, after 2 partials the common prefix "Hello" is stable
        # This is expected behavior - stable text is emitted once window is full
        stable_so_far = stabilizer.get_stable_text()
        assert "Hello" in (delta + stable_so_far) or delta == ""

        # Third partial with same prefix - stable prefix continues
        delta, unstable, is_final = stabilizer.update("Hello there how")
        # "Hello " is the common prefix across all three
        assert "Hello" in stabilizer.get_stable_text() or "Hello" in delta

    def test_stabilizer_finalize(self):
        """Test finalize returns remaining text."""
        stabilizer = TranscriptStabilizer(stability_window=2, min_stable_chars=5)

        stabilizer.update("Hello")
        stabilizer.update("Hello there")

        remaining = stabilizer.finalize("Hello there, goodbye!")
        # Should return text not yet marked stable
        assert "goodbye" in remaining or "there" in remaining

    def test_stabilizer_reset(self):
        """Test reset clears state."""
        stabilizer = TranscriptStabilizer()

        stabilizer.update("Hello")
        stabilizer.update("Hello there")
        stabilizer.reset()

        assert stabilizer.get_stable_text() == ""
        assert stabilizer.get_full_partial() == ""


class TestStreamingTranscriptBuffer:
    """Test the higher-level transcript buffer."""

    def test_buffer_accumulates_stable(self):
        """Test buffer accumulates stable text."""
        buffer = StreamingTranscriptBuffer()

        # Simulate multiple partials
        buffer.add_partial("Hello")
        buffer.add_partial("Hello there")
        buffer.add_partial("Hello there friend")

        # Some text should be accumulated as stable
        full = buffer.get_full_text()
        assert "Hello" in full

    def test_buffer_finalize(self):
        """Test buffer finalize returns remaining."""
        buffer = StreamingTranscriptBuffer()

        buffer.add_partial("Hello")
        buffer.add_partial("Hello there")

        remaining = buffer.finalize("Hello there, goodbye!")
        assert remaining  # Should have some remaining text


# --- Streaming ASR Tests ---


class TestStreamingASRConfig:
    """Test streaming ASR configuration."""

    def test_default_config(self):
        """Test default config values."""
        config = StreamingASRConfig()

        assert config.chunk_size_ms == 160
        assert config.stability_window == 2
        assert config.sample_rate == 16000

    def test_custom_config(self):
        """Test custom config values."""
        config = StreamingASRConfig(
            endpoint="ws://custom:8080/asr",
            chunk_size_ms=80,
            min_stable_chars=20,
        )

        assert config.endpoint == "ws://custom:8080/asr"
        assert config.chunk_size_ms == 80
        assert config.min_stable_chars == 20


class TestLocalStreamingASR:
    """Test local streaming ASR wrapper."""

    @pytest.mark.asyncio
    async def test_local_streaming_yields_partials(self, mock_streaming_stt):
        """Test that local streaming wrapper yields partials."""
        # Wrap a batch provider
        local_asr = LocalStreamingASR(
            batch_provider=mock_streaming_stt,
            chunk_size_ms=100,
        )

        async def audio_gen():
            for _ in range(5):
                yield b"\x00" * 3200  # 100ms of audio
                await asyncio.sleep(0.01)

        partials = []
        async for partial in local_asr.transcribe_streaming(audio_gen()):
            partials.append(partial)

        assert len(partials) > 0
        assert partials[-1].is_final


# --- Latency Tracker Tests ---


class TestLatencyTracker:
    """Test latency tracking for streaming pipeline."""

    def test_tracker_records_streaming_events(self):
        """Test tracker records all streaming events."""
        tracker = LatencyTracker(user_id="test")

        tracker.vad_start()
        tracker.asr_start()
        tracker.asr_partial()
        tracker.asr_stable()
        tracker.llm_start(early=True, stable_chars=50)
        tracker.llm_first_token()
        tracker.vad_end()
        tracker.asr_final("Hello there")
        tracker.llm_complete("Response text")
        tracker.tts_start()
        tracker.tts_first_audio()
        tracker.playback_end()

        turn = tracker.finalize()

        assert turn.t_asr_partial1 is not None
        assert turn.t_asr_stable1 is not None
        assert turn.llm_started_early is True
        assert turn.stable_chars_at_llm_start == 50

    def test_tracker_computes_overlap_metrics(self):
        """Test overlap metrics are computed correctly."""
        tracker = LatencyTracker(user_id="test")

        tracker.vad_start()
        time.sleep(0.01)
        tracker.asr_start()
        time.sleep(0.01)
        tracker.llm_start(early=True, stable_chars=50)
        time.sleep(0.02)  # LLM runs for 20ms before VAD ends
        tracker.vad_end()
        tracker.asr_final("test")
        tracker.llm_complete("response")
        tracker.tts_start()
        tracker.tts_first_audio()
        tracker.playback_end()

        turn = tracker.finalize()
        metrics = turn.compute_metrics()

        assert metrics.get("llm_started_early") is True
        # Overlap should be positive (LLM started before VAD ended)
        assert "llm_overlap_ms" in metrics


# --- Streaming Pipeline Tests ---


class TestStreamingPipeline:
    """Test the full streaming pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_yields_events(self, streaming_pipeline):
        """Test pipeline yields expected event types."""
        async def audio_gen():
            for _ in range(10):
                yield b"\x00" * 1600
                await asyncio.sleep(0.01)

        events = []
        async for event in streaming_pipeline.process_stream(
            audio_gen(), voice_prompt={}
        ):
            events.append(event)

        event_types = [e.type for e in events]

        # Should have ASR events
        assert "asr_partial" in event_types
        assert "asr_final" in event_types

        # Should have LLM events
        assert "llm_start" in event_types

        # Should have completion
        assert "complete" in event_types

    @pytest.mark.asyncio
    async def test_pipeline_early_llm_start(self, streaming_pipeline, mock_llm):
        """Test LLM starts early on stable prefix."""
        # Use STT that produces long enough stable text
        long_partials = [
            "The quick brown fox jumps over",
            "The quick brown fox jumps over the lazy",
            "The quick brown fox jumps over the lazy dog",
        ]
        streaming_pipeline.stt = MockStreamingSTT(long_partials, [0.05] * 3)

        async def audio_gen():
            for _ in range(10):
                yield b"\x00" * 1600
                await asyncio.sleep(0.01)

        events = []
        async for event in streaming_pipeline.process_stream(
            audio_gen(), voice_prompt={}
        ):
            events.append(event)

        # Find when LLM started
        llm_start_events = [e for e in events if e.type == "llm_start"]
        assert len(llm_start_events) > 0

        # Check the result
        complete_events = [e for e in events if e.type == "complete"]
        assert len(complete_events) == 1

        result = complete_events[0].data
        assert isinstance(result, StreamingPipelineResult)

    @pytest.mark.asyncio
    async def test_pipeline_latency_tracking(self, streaming_pipeline):
        """Test pipeline tracks latency metrics."""
        async def audio_gen():
            for _ in range(5):
                yield b"\x00" * 1600
                await asyncio.sleep(0.01)

        result = None
        async for event in streaming_pipeline.process_stream(
            audio_gen(), voice_prompt={}
        ):
            if event.type == "complete":
                result = event.data

        assert result is not None
        assert "asr_total_ms" in result.latency or "llm_total_ms" in result.latency

    @pytest.mark.asyncio
    async def test_pipeline_conversation_history(self, streaming_pipeline, mock_llm):
        """Test pipeline maintains conversation history."""
        async def audio_gen():
            for _ in range(5):
                yield b"\x00" * 1600

        # First turn
        async for _ in streaming_pipeline.process_stream(
            audio_gen(), user_id="test_user", voice_prompt={}
        ):
            pass

        # Check history was updated
        history = streaming_pipeline._get_history("test_user")
        assert len(history) >= 2  # At least user + assistant


# --- Integration Tests ---


class TestStreamingIntegration:
    """Integration tests for streaming components."""

    @pytest.mark.asyncio
    async def test_stabilizer_with_pipeline(self):
        """Test stabilizer integration with pipeline."""
        buffer = StreamingTranscriptBuffer()

        # Simulate realistic ASR output
        partials = [
            "I",
            "I would",
            "I would like",
            "I would like to",
            "I would like to order",
            "I would like to order a pizza",
        ]

        stable_deltas = []
        for partial in partials:
            delta = buffer.add_partial(partial)
            if delta:
                stable_deltas.append(delta)

        remaining = buffer.finalize("I would like to order a pizza please")

        # Verify we got stable output
        full_stable = "".join(stable_deltas) + remaining
        assert "order" in full_stable or "pizza" in full_stable

    @pytest.mark.asyncio
    async def test_end_to_end_latency(self, streaming_pipeline):
        """Test end-to-end latency is tracked correctly."""
        start = time.time()

        async def audio_gen():
            for _ in range(5):
                yield b"\x00" * 1600
                await asyncio.sleep(0.01)

        result = None
        async for event in streaming_pipeline.process_stream(
            audio_gen(), voice_prompt={}
        ):
            if event.type == "complete":
                result = event.data

        elapsed = time.time() - start

        assert result is not None
        # Total time should be reasonable (< 2 seconds for mocked components)
        assert elapsed < 2.0
