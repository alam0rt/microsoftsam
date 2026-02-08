"""Streaming voice pipeline with ASR/LLM overlap for minimal latency.

This module implements the streaming pipeline architecture from streaming-plan.md:

1. Streaming ASR with partial results (24-80ms TTFT)
2. Stable prefix tracking for early LLM start
3. LLM kickoff on stable prefix (>50 chars threshold)
4. Phrase-based TTS streaming for low TTFA

Target metrics:
- ASR TTFT: 24-80ms (vs 1500-3000ms batch)
- Total TTFA: 400-800ms (vs 2100-4000ms)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from mumble_voice_bot.interfaces.llm import LLMProvider, LLMResponse
from mumble_voice_bot.interfaces.stt import PartialSTTResult, STTProvider
from mumble_voice_bot.latency import LatencyTracker, LatencyLogger
from mumble_voice_bot.phrase_chunker import PhraseChunker
from mumble_voice_bot.transcript_stabilizer import StreamingTranscriptBuffer

logger = logging.getLogger(__name__)


@dataclass
class StreamingPipelineConfig:
    """Configuration for the streaming pipeline.

    Attributes:
        llm_start_threshold: Minimum stable chars before starting LLM.
        llm_abort_on_change: Abort LLM if transcript changes significantly.
        change_threshold: Characters of change to trigger abort.
        phrase_min_chars: Minimum phrase length for TTS.
        phrase_max_chars: Maximum phrase length before force-flush.
        phrase_timeout_ms: Flush phrase after this delay.
        max_history_turns: Max conversation history turns.
        history_timeout: Seconds before clearing history.
    """

    llm_start_threshold: int = 50
    llm_abort_on_change: bool = False
    change_threshold: int = 20
    phrase_min_chars: int = 30
    phrase_max_chars: int = 150
    phrase_timeout_ms: int = 400
    max_history_turns: int = 10
    history_timeout: float = 300.0


@dataclass
class StreamingPipelineResult:
    """Result from a streaming pipeline run.

    Attributes:
        transcript: Final transcription text.
        response: Full LLM response text.
        latency: Latency breakdown in milliseconds.
        llm_started_early: Whether LLM started on partial transcript.
        llm_aborted: Whether LLM was aborted due to transcript change.
    """

    transcript: str
    response: str
    latency: dict = field(default_factory=dict)
    llm_started_early: bool = False
    llm_aborted: bool = False


@dataclass
class StreamingEvent:
    """Event from the streaming pipeline.

    Attributes:
        type: Event type (asr_partial, asr_final, llm_start, llm_chunk, etc.)
        data: Event-specific data.
        timestamp: Event timestamp (seconds since epoch).
    """

    type: str
    data: Any
    timestamp: float = field(default_factory=time.time)


class StreamingVoicePipeline:
    """
    Voice pipeline with overlapped ASR/LLM/TTS for minimal latency.

    Key features:
    - True streaming ASR with partial results
    - LLM starts generating on stable transcript prefix
    - TTS starts on first complete phrase
    - Full latency instrumentation

    Usage:
        pipeline = StreamingVoicePipeline(stt, llm, tts)

        async for event in pipeline.process_stream(audio_stream):
            if event.type == "tts_audio":
                play(event.data)
            elif event.type == "asr_partial":
                display_partial(event.data)
    """

    def __init__(
        self,
        stt: STTProvider,
        llm: LLMProvider,
        tts: Any,  # TTS provider with synthesize_streaming
        config: StreamingPipelineConfig | None = None,
        latency_logger: LatencyLogger | None = None,
    ):
        """Initialize the streaming pipeline.

        Args:
            stt: Streaming-capable STT provider.
            llm: LLM provider with chat_stream support.
            tts: TTS provider with streaming synthesis.
            config: Pipeline configuration.
            latency_logger: Optional latency logger for metrics.
        """
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.config = config or StreamingPipelineConfig()
        self._latency_logger = latency_logger

        # Per-user conversation history
        self._history: dict[str, list[dict]] = {}
        self._last_activity: dict[str, float] = {}

    def _get_history(self, user_id: str) -> list[dict]:
        """Get conversation history for a user."""
        now = time.time()

        # Check timeout
        if user_id in self._last_activity:
            if now - self._last_activity[user_id] > self.config.history_timeout:
                self._history.pop(user_id, None)

        self._last_activity[user_id] = now

        if user_id not in self._history:
            self._history[user_id] = []

        return self._history[user_id]

    def _add_to_history(self, user_id: str, role: str, content: str):
        """Add message to conversation history."""
        history = self._get_history(user_id)
        history.append({"role": role, "content": content})

        # Trim to max
        max_msgs = self.config.max_history_turns * 2
        if len(history) > max_msgs:
            self._history[user_id] = history[-max_msgs:]

    async def process_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        user_id: str = "default",
        sample_rate: int = 16000,
        voice_prompt: Any = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamingEvent]:
        """
        Process audio stream with overlapped ASR/LLM/TTS.

        This is the main entry point for streaming voice processing.
        It yields events as each pipeline stage progresses.

        Args:
            audio_stream: Async iterator yielding PCM audio chunks.
            user_id: User identifier for conversation history.
            sample_rate: Audio sample rate in Hz.
            voice_prompt: Voice prompt for TTS.
            system_prompt: Optional system prompt override.

        Yields:
            StreamingEvent with pipeline progress and outputs.

        Event types:
            asr_partial: Partial transcript (PartialSTTResult)
            asr_stable: New stable text added
            asr_final: Final transcript (str)
            llm_start: LLM generation started (str - prompt used)
            llm_chunk: LLM token chunk (str)
            llm_complete: Full LLM response (str)
            tts_phrase: Phrase sent to TTS (str)
            tts_audio: Audio chunk from TTS
            tts_first_audio: First audio timestamp
            complete: StreamingPipelineResult
            error: Exception if something fails
        """
        tracker = LatencyTracker(user_id, self._latency_logger)
        tracker.vad_start()

        transcript_buffer = StreamingTranscriptBuffer()
        llm_task: asyncio.Task | None = None
        llm_started_on = ""
        llm_aborted = False
        full_transcript = ""
        full_response = ""
        first_audio_emitted = False

        # Queues for coordinating stages
        phrase_queue: asyncio.Queue[str | None] = asyncio.Queue()

        try:
            # Start ASR streaming
            tracker.asr_start()
            asr_started = False

            async for partial in self.stt.transcribe_streaming(
                audio_stream, sample_rate
            ):
                if not asr_started:
                    tracker.asr_partial()
                    asr_started = True

                yield StreamingEvent("asr_partial", partial)

                # Track stable text
                if partial.stable_text and partial.stable_text != transcript_buffer.get_accumulated():
                    delta = transcript_buffer.add_partial(partial.text)
                    if delta:
                        yield StreamingEvent("asr_stable", delta)

                    # Check if we should start LLM early
                    stable_len = len(transcript_buffer.get_accumulated())
                    if (
                        llm_task is None
                        and stable_len >= self.config.llm_start_threshold
                    ):
                        llm_started_on = transcript_buffer.get_accumulated()
                        logger.info(
                            f"Starting LLM early on {stable_len} stable chars"
                        )
                        yield StreamingEvent("llm_start", llm_started_on)
                        tracker.llm_start()

                        # Start LLM generation in background
                        llm_task = asyncio.create_task(
                            self._run_llm_streaming(
                                llm_started_on,
                                user_id,
                                phrase_queue,
                                tracker,
                                system_prompt,
                            )
                        )

                if partial.is_final:
                    full_transcript = partial.text
                    tracker.asr_final(full_transcript)
                    tracker.vad_end()
                    yield StreamingEvent("asr_final", full_transcript)
                    break

            # If LLM hasn't started yet, start it now with full transcript
            if llm_task is None and full_transcript:
                yield StreamingEvent("llm_start", full_transcript)
                tracker.llm_start()

                llm_task = asyncio.create_task(
                    self._run_llm_streaming(
                        full_transcript,
                        user_id,
                        phrase_queue,
                        tracker,
                        system_prompt,
                    )
                )
            elif llm_task and self.config.llm_abort_on_change:
                # Check if transcript changed significantly
                change = len(full_transcript) - len(llm_started_on)
                if change > self.config.change_threshold:
                    logger.warning(
                        f"Transcript changed by {change} chars, aborting LLM"
                    )
                    llm_task.cancel()
                    llm_aborted = True

                    # Restart with full transcript
                    yield StreamingEvent("llm_start", full_transcript)
                    tracker.llm_start()

                    llm_task = asyncio.create_task(
                        self._run_llm_streaming(
                            full_transcript,
                            user_id,
                            phrase_queue,
                            tracker,
                            system_prompt,
                        )
                    )

            # Process TTS for phrases as they arrive
            if llm_task:
                # Create a queue for TTS events
                tts_event_queue: asyncio.Queue[StreamingEvent | None] = asyncio.Queue()

                # Start TTS processing in background
                tts_task = asyncio.create_task(
                    self._run_tts_to_queue(
                        phrase_queue, voice_prompt, tracker, tts_event_queue
                    )
                )

                # Wait for LLM to complete
                try:
                    full_response = await llm_task
                    yield StreamingEvent("llm_complete", full_response)
                except asyncio.CancelledError:
                    full_response = ""

                # Signal TTS to finish
                await phrase_queue.put(None)

                # Wait for TTS to complete and yield all events
                await tts_task

                # Drain TTS event queue
                while not tts_event_queue.empty():
                    event = tts_event_queue.get_nowait()
                    if event is not None:
                        if event.type == "tts_first_audio" and not first_audio_emitted:
                            first_audio_emitted = True
                            tracker.tts_first_audio()
                        yield event

            # Update history
            if full_transcript:
                self._add_to_history(user_id, "user", full_transcript)
            if full_response:
                self._add_to_history(user_id, "assistant", full_response)

            # Compute final latency
            tracker.playback_end()
            turn = tracker.finalize()

            yield StreamingEvent(
                "complete",
                StreamingPipelineResult(
                    transcript=full_transcript,
                    response=full_response,
                    latency=turn.compute_metrics(),
                    llm_started_early=bool(llm_started_on),
                    llm_aborted=llm_aborted,
                ),
            )

        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}")
            yield StreamingEvent("error", e)

    async def _run_llm_streaming(
        self,
        prompt: str,
        user_id: str,
        phrase_queue: asyncio.Queue,
        tracker: LatencyTracker,
        system_prompt: str | None = None,
    ) -> str:
        """Run LLM streaming and feed phrases to queue.

        Args:
            prompt: User prompt text.
            user_id: User identifier for history.
            phrase_queue: Queue for TTS phrases.
            tracker: Latency tracker.
            system_prompt: Optional system prompt.

        Returns:
            Full LLM response text.
        """
        history = self._get_history(user_id).copy()
        history.append({"role": "user", "content": prompt})

        if system_prompt:
            history.insert(0, {"role": "system", "content": system_prompt})

        chunker = PhraseChunker(
            min_chars=self.config.phrase_min_chars,
            max_chars=self.config.phrase_max_chars,
            timeout_ms=self.config.phrase_timeout_ms,
        )

        full_response = ""
        first_token = True

        async for token in self.llm.chat_stream(history):
            if first_token:
                tracker.llm_first_token()
                first_token = False

            full_response += token

            # Check for complete phrase
            phrase = chunker.add(token)
            if phrase:
                await phrase_queue.put(phrase)

        # Flush remaining text
        remaining = chunker.flush()
        if remaining:
            await phrase_queue.put(remaining)

        tracker.llm_complete(full_response)
        return full_response

    async def _run_tts_to_queue(
        self,
        phrase_queue: asyncio.Queue,
        voice_prompt: Any,
        tracker: LatencyTracker,
        event_queue: asyncio.Queue,
    ) -> None:
        """Run TTS on phrases and put events into queue.

        Args:
            phrase_queue: Queue of phrases to synthesize.
            voice_prompt: Voice prompt for TTS.
            tracker: Latency tracker.
            event_queue: Queue to put TTS events into.
        """
        first_audio = True

        while True:
            phrase = await phrase_queue.get()
            if phrase is None:
                break

            await event_queue.put(StreamingEvent("tts_phrase", phrase))
            tracker.tts_start()

            if voice_prompt is not None and hasattr(self.tts, "generate_speech_streaming"):
                async for audio_chunk in self._synthesize_phrase(phrase, voice_prompt):
                    if first_audio:
                        first_audio = False
                        await event_queue.put(StreamingEvent("tts_first_audio", time.time()))
                    await event_queue.put(StreamingEvent("tts_audio", audio_chunk))

        # Signal completion
        await event_queue.put(None)

    async def _run_tts_streaming(
        self,
        phrase_queue: asyncio.Queue,
        voice_prompt: Any,
        tracker: LatencyTracker,
    ) -> AsyncIterator[StreamingEvent]:
        """Run TTS on phrases from queue.

        Args:
            phrase_queue: Queue of phrases to synthesize.
            voice_prompt: Voice prompt for TTS.
            tracker: Latency tracker.

        Yields:
            StreamingEvent for TTS outputs.
        """
        first_audio = True

        while True:
            phrase = await phrase_queue.get()
            if phrase is None:
                break

            yield StreamingEvent("tts_phrase", phrase)
            tracker.tts_start()

            if voice_prompt is not None and hasattr(self.tts, "generate_speech_streaming"):
                async for audio_chunk in self._synthesize_phrase(phrase, voice_prompt):
                    if first_audio:
                        first_audio = False
                        yield StreamingEvent("tts_first_audio", time.time())
                    yield StreamingEvent("tts_audio", audio_chunk)

    async def _synthesize_phrase(
        self, phrase: str, voice_prompt: Any
    ) -> AsyncIterator[Any]:
        """Synthesize a phrase using TTS.

        Args:
            phrase: Text to synthesize.
            voice_prompt: Voice prompt for TTS.

        Yields:
            Audio chunks from TTS.
        """
        loop = asyncio.get_event_loop()

        if hasattr(self.tts, "generate_speech_streaming"):
            import queue

            audio_queue = queue.Queue()

            def generate():
                try:
                    for chunk in self.tts.generate_speech_streaming(
                        phrase, voice_prompt
                    ):
                        audio_queue.put(chunk)
                finally:
                    audio_queue.put(None)

            future = loop.run_in_executor(None, generate)

            while True:
                try:
                    chunk = await loop.run_in_executor(
                        None, lambda: audio_queue.get(timeout=0.1)
                    )
                    if chunk is None:
                        break
                    yield chunk
                except Exception:
                    if future.done():
                        while not audio_queue.empty():
                            chunk = audio_queue.get_nowait()
                            if chunk is not None:
                                yield chunk
                        break
        else:
            # Non-streaming fallback
            audio = await loop.run_in_executor(
                None,
                lambda: self.tts.generate_speech(phrase, voice_prompt),
            )
            yield audio


async def create_streaming_pipeline(
    stt_config: dict | None = None,
    llm_config: dict | None = None,
    tts: Any = None,
    pipeline_config: StreamingPipelineConfig | None = None,
) -> StreamingVoicePipeline:
    """
    Factory function to create a streaming pipeline with configured providers.

    Args:
        stt_config: STT provider configuration.
        llm_config: LLM provider configuration.
        tts: TTS provider instance.
        pipeline_config: Pipeline configuration.

    Returns:
        Configured StreamingVoicePipeline instance.
    """
    from mumble_voice_bot.providers.streaming_asr import (
        LocalStreamingASR,
        StreamingASR,
        StreamingASRConfig,
    )
    from mumble_voice_bot.providers.wyoming_stt import WyomingSTT

    # Configure STT
    stt_config = stt_config or {}
    stt_type = stt_config.get("type", "local")

    if stt_type == "websocket":
        stt = StreamingASR(
            StreamingASRConfig(
                endpoint=stt_config.get("endpoint", "ws://localhost:50051/asr"),
                chunk_size_ms=stt_config.get("chunk_size_ms", 160),
            )
        )
    elif stt_type == "wyoming":
        base_stt = WyomingSTT(
            host=stt_config.get("host", "localhost"),
            port=stt_config.get("port", 10300),
        )
        stt = LocalStreamingASR(
            base_stt,
            chunk_size_ms=stt_config.get("chunk_size_ms", 500),
        )
    else:
        raise ValueError(f"Unknown STT type: {stt_type}")

    # Configure LLM
    llm_config = llm_config or {}
    from mumble_voice_bot.providers.openai_llm import OpenAILLM

    llm = OpenAILLM(
        endpoint=llm_config.get("endpoint", "http://localhost:11434/v1"),
        model=llm_config.get("model", "llama3.2:3b"),
        api_key=llm_config.get("api_key"),
    )

    return StreamingVoicePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        config=pipeline_config,
    )
