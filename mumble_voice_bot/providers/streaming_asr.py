"""Streaming ASR provider with WebSocket support and stable prefix tracking.

This provider implements true streaming ASR with:
- Chunked audio ingestion (configurable chunk size)
- Partial transcript emission
- Stable prefix tracking for early LLM start
- Support for NVIDIA Riva/Nemotron-style WebSocket APIs

Usage:
    config = StreamingASRConfig(
        endpoint="ws://localhost:50051/asr/streaming",
        chunk_size_ms=160,
    )
    asr = StreamingASR(config)

    async for partial in asr.transcribe_streaming(audio_stream):
        if partial.stable_text:
            # Start LLM with stable prefix
            await llm.start(partial.stable_text)
        if partial.is_final:
            # Finalize with complete transcript
            await llm.complete(partial.text)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Any

from mumble_voice_bot.interfaces.stt import (
    PartialSTTResult,
    STTProvider,
    STTResult,
)
from mumble_voice_bot.transcript_stabilizer import (
    StreamingTranscriptBuffer,
    TranscriptStabilizer,
)

logger = logging.getLogger(__name__)

# Try to import websockets
try:
    import websockets
    from websockets.asyncio.client import connect as ws_connect

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    ws_connect = None


@dataclass
class StreamingASRConfig:
    """Configuration for streaming ASR.

    Attributes:
        endpoint: WebSocket URL for streaming ASR service.
        chunk_size_ms: Audio chunk size in milliseconds (80, 160, 560, 1120).
        stability_window: Number of consistent partials before text is "stable".
        min_stable_chars: Minimum characters before emitting stable text.
        language: Language code (e.g., "en-US").
        sample_rate: Expected audio sample rate.
        encoding: Audio encoding ("LINEAR16", "OPUS", etc.).
        enable_partial_results: Request partial results from server.
        api_key: Optional API key for authentication.
        connect_timeout: WebSocket connection timeout in seconds.
        read_timeout: Timeout for reading responses in seconds.
    """

    endpoint: str = "ws://localhost:50051/asr/streaming"
    chunk_size_ms: int = 160
    stability_window: int = 2
    min_stable_chars: int = 10
    language: str = "en-US"
    sample_rate: int = 16000
    encoding: str = "LINEAR16"
    enable_partial_results: bool = True
    api_key: str | None = None
    connect_timeout: float = 10.0
    read_timeout: float = 30.0


@dataclass
class StreamingASRMetrics:
    """Metrics for a streaming ASR session.

    Attributes:
        start_time: Session start timestamp.
        first_audio_time: First audio chunk received timestamp.
        first_partial_time: First partial result timestamp.
        final_time: Final result timestamp.
        total_audio_bytes: Total audio bytes processed.
        total_chunks: Number of audio chunks processed.
        total_partials: Number of partial results received.
    """

    start_time: float = 0.0
    first_audio_time: float | None = None
    first_partial_time: float | None = None
    final_time: float | None = None
    total_audio_bytes: int = 0
    total_chunks: int = 0
    total_partials: int = 0

    def compute_latencies(self) -> dict:
        """Compute latency metrics in milliseconds."""
        metrics = {}

        if self.first_audio_time and self.first_partial_time:
            metrics["ttfp_ms"] = (
                self.first_partial_time - self.first_audio_time
            ) * 1000

        if self.first_audio_time and self.final_time:
            metrics["total_asr_ms"] = (
                self.final_time - self.first_audio_time
            ) * 1000

        if self.total_chunks > 0:
            metrics["chunks"] = self.total_chunks
            metrics["partials"] = self.total_partials

        return metrics


class StreamingASR(STTProvider):
    """Streaming ASR provider with WebSocket support.

    This provider connects to a WebSocket-based streaming ASR service
    (like NVIDIA Riva or NeMo Nemotron) and provides:

    - True streaming with partial results every chunk
    - Stable prefix tracking for early LLM kickoff
    - Latency metrics for optimization

    For local NeMo-based streaming, see NemotronStreamingASR.
    """

    def __init__(
        self,
        config: StreamingASRConfig | None = None,
        on_partial: Callable[[PartialSTTResult], Any] | None = None,
    ):
        """Initialize the streaming ASR provider.

        Args:
            config: Configuration for the ASR service.
            on_partial: Optional callback for partial results.
        """
        self.config = config or StreamingASRConfig()
        self._on_partial = on_partial
        self._stabilizer = TranscriptStabilizer(
            stability_window=self.config.stability_window,
            min_stable_chars=self.config.min_stable_chars,
        )
        self._metrics = StreamingASRMetrics()

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Non-streaming transcription (collects all partials).

        Args:
            audio_data: Raw PCM audio bytes.
            sample_rate: Audio sample rate.
            sample_width: Bytes per sample.
            channels: Number of audio channels.
            language: Language hint.

        Returns:
            STTResult with final transcription.
        """

        async def audio_generator():
            chunk_bytes = int(
                self.config.chunk_size_ms * sample_rate * sample_width / 1000
            )
            for i in range(0, len(audio_data), chunk_bytes):
                yield audio_data[i : i + chunk_bytes]

        final_text = ""
        async for partial in self.transcribe_streaming(
            audio_generator(), sample_rate, sample_width, channels, language
        ):
            if partial.is_final:
                final_text = partial.text

        duration = len(audio_data) / (sample_rate * sample_width * channels)
        return STTResult(text=final_text, language=language, duration=duration)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Transcribe streaming audio, returning only final result.

        Args:
            audio_stream: Async iterator yielding audio chunks.
            sample_rate: Audio sample rate.
            sample_width: Bytes per sample.
            channels: Number of audio channels.
            language: Language hint.

        Returns:
            STTResult with final transcription.
        """
        final_text = ""
        async for partial in self.transcribe_streaming(
            audio_stream, sample_rate, sample_width, channels, language
        ):
            if partial.is_final:
                final_text = partial.text

        return STTResult(text=final_text, language=language)

    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> AsyncIterator[PartialSTTResult]:
        """Transcribe streaming audio with partial results.

        This is the main streaming method. It yields partial results
        as audio is processed, with stable prefix tracking.

        Args:
            audio_stream: Async iterator yielding audio chunks.
            sample_rate: Audio sample rate.
            sample_width: Bytes per sample.
            channels: Number of audio channels.
            language: Language hint.

        Yields:
            PartialSTTResult with current transcript and stability info.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError(
                "websockets library required for StreamingASR. "
                "Install with: pip install websockets"
            )

        self._metrics = StreamingASRMetrics(start_time=time.time())
        self._stabilizer.reset()

        # Build connection parameters
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Connect to WebSocket
        try:
            async with ws_connect(
                self.config.endpoint,
                additional_headers=headers,
                open_timeout=self.config.connect_timeout,
            ) as websocket:
                # Send config message
                config_msg = {
                    "type": "config",
                    "language": language or self.config.language,
                    "sample_rate": sample_rate,
                    "encoding": self.config.encoding,
                    "enable_partial_results": self.config.enable_partial_results,
                }
                await websocket.send(json.dumps(config_msg))

                # Create tasks for sending audio and receiving results
                send_task = asyncio.create_task(
                    self._send_audio(websocket, audio_stream, sample_rate, sample_width)
                )
                receive_task = asyncio.create_task(
                    self._receive_results(websocket)
                )

                # Yield results as they come in
                try:
                    async for partial in self._process_stream(
                        send_task, receive_task
                    ):
                        if self._on_partial:
                            self._on_partial(partial)
                        yield partial
                finally:
                    send_task.cancel()
                    receive_task.cancel()

        except Exception as e:
            logger.error(f"WebSocket streaming ASR error: {e}")
            # Yield empty final result on error
            yield PartialSTTResult(text="", stable_text="", is_final=True)

    async def _send_audio(
        self,
        websocket,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int,
        sample_width: int,
    ) -> None:
        """Send audio chunks to WebSocket."""
        chunk_bytes = int(
            self.config.chunk_size_ms * sample_rate * sample_width / 1000
        )
        buffer = b""

        async for chunk in audio_stream:
            if self._metrics.first_audio_time is None:
                self._metrics.first_audio_time = time.time()

            buffer += chunk
            self._metrics.total_audio_bytes += len(chunk)

            # Send when we have enough data
            while len(buffer) >= chunk_bytes:
                audio_chunk = buffer[:chunk_bytes]
                buffer = buffer[chunk_bytes:]

                # Send as binary frame
                await websocket.send(audio_chunk)
                self._metrics.total_chunks += 1

        # Send remaining buffer
        if buffer:
            await websocket.send(buffer)
            self._metrics.total_chunks += 1

        # Send end-of-stream signal
        await websocket.send(json.dumps({"type": "end"}))

    async def _receive_results(self, websocket) -> AsyncIterator[dict]:
        """Receive results from WebSocket."""
        while True:
            try:
                msg = await asyncio.wait_for(
                    websocket.recv(), timeout=self.config.read_timeout
                )

                if isinstance(msg, str):
                    data = json.loads(msg)
                    yield data
                    if data.get("is_final") or data.get("type") == "final":
                        break
            except asyncio.TimeoutError:
                logger.warning("ASR receive timeout")
                break
            except Exception as e:
                logger.error(f"ASR receive error: {e}")
                break

    async def _process_stream(
        self, send_task: asyncio.Task, receive_task: asyncio.Task
    ) -> AsyncIterator[PartialSTTResult]:
        """Process the stream, yielding stabilized partial results."""
        results_queue = asyncio.Queue()
        receive_done = asyncio.Event()

        async def collect_results():
            try:
                async for result in receive_task:
                    await results_queue.put(result)
            finally:
                receive_done.set()
                await results_queue.put(None)  # Sentinel

        collector = asyncio.create_task(collect_results())

        try:
            while True:
                result = await results_queue.get()
                if result is None:
                    break

                if self._metrics.first_partial_time is None:
                    self._metrics.first_partial_time = time.time()
                self._metrics.total_partials += 1

                # Extract text from result (handle different API formats)
                text = result.get("text") or result.get("transcript", "")
                is_final = result.get("is_final") or result.get("type") == "final"
                timestamp = result.get("timestamp", 0.0)

                if is_final:
                    self._metrics.final_time = time.time()
                    remaining = self._stabilizer.finalize(text)
                    yield PartialSTTResult(
                        text=text,
                        stable_text=text,
                        is_final=True,
                        timestamp=timestamp,
                    )
                    logger.debug(
                        f"ASR metrics: {self._metrics.compute_latencies()}"
                    )
                else:
                    # Process partial through stabilizer
                    stable_delta, unstable, _ = self._stabilizer.update(text)
                    stable_text = self._stabilizer.get_stable_text()

                    yield PartialSTTResult(
                        text=text,
                        stable_text=stable_text,
                        is_final=False,
                        timestamp=timestamp,
                    )

        finally:
            collector.cancel()

    async def is_available(self) -> bool:
        """Check if the ASR service is available."""
        if not WEBSOCKETS_AVAILABLE:
            return False

        try:
            async with ws_connect(
                self.config.endpoint,
                open_timeout=self.config.connect_timeout,
            ) as websocket:
                # Send a ping
                await websocket.ping()
                return True
        except Exception:
            return False


class LocalStreamingASR(STTProvider):
    """Local streaming ASR using an existing STT provider with chunking.

    This wraps a batch STT provider to provide streaming-like behavior
    by re-transcribing accumulated audio on each chunk. Not as efficient
    as true streaming but works with any STT provider.

    Best for testing or when WebSocket streaming isn't available.
    """

    def __init__(
        self,
        batch_provider: STTProvider,
        chunk_size_ms: int = 500,
        stability_window: int = 2,
        min_stable_chars: int = 10,
    ):
        """Initialize local streaming ASR.

        Args:
            batch_provider: Underlying batch STT provider.
            chunk_size_ms: How often to re-transcribe (ms).
            stability_window: Partials before text is stable.
            min_stable_chars: Min chars before emitting stable text.
        """
        self._provider = batch_provider
        self._chunk_size_ms = chunk_size_ms
        self._buffer = StreamingTranscriptBuffer(
            stabilizer=TranscriptStabilizer(
                stability_window=stability_window,
                min_stable_chars=min_stable_chars,
            )
        )

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Delegate to underlying provider."""
        return await self._provider.transcribe(
            audio_data, sample_rate, sample_width, channels, language
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> STTResult:
        """Delegate to underlying provider."""
        return await self._provider.transcribe_stream(
            audio_stream, sample_rate, sample_width, channels, language
        )

    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: str | None = None,
    ) -> AsyncIterator[PartialSTTResult]:
        """Provide streaming-like behavior with batch re-transcription.

        This collects audio and periodically re-transcribes the full
        buffer, using the stabilizer to emit stable prefixes.

        Args:
            audio_stream: Async iterator yielding audio chunks.
            sample_rate: Audio sample rate.
            sample_width: Bytes per sample.
            channels: Number of channels.
            language: Language hint.

        Yields:
            PartialSTTResult with current transcript and stability info.
        """
        self._buffer.reset()
        audio_buffer = b""
        chunk_bytes = int(self._chunk_size_ms * sample_rate * sample_width / 1000)
        last_transcribe = 0.0
        start_time = time.time()

        async for chunk in audio_stream:
            audio_buffer += chunk

            # Re-transcribe periodically
            now = time.time()
            if (
                len(audio_buffer) >= chunk_bytes
                and now - last_transcribe >= self._chunk_size_ms / 1000
            ):
                last_transcribe = now

                # Transcribe accumulated audio
                result = await self._provider.transcribe(
                    audio_buffer, sample_rate, sample_width, channels, language
                )

                if result.text:
                    # Update stabilizer
                    stable_delta = self._buffer.add_partial(result.text)
                    stable_text = self._buffer.get_accumulated()

                    yield PartialSTTResult(
                        text=result.text,
                        stable_text=stable_text,
                        is_final=False,
                        timestamp=now - start_time,
                    )

        # Final transcription
        if audio_buffer:
            result = await self._provider.transcribe(
                audio_buffer, sample_rate, sample_width, channels, language
            )
            remaining = self._buffer.finalize(result.text)

            yield PartialSTTResult(
                text=result.text,
                stable_text=result.text,
                is_final=True,
                timestamp=time.time() - start_time,
            )

    async def is_available(self) -> bool:
        """Delegate to underlying provider."""
        return await self._provider.is_available()
