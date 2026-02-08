"""Performance improvements for the voice pipeline (docs/perf.md Phase 1 & 2).

This module implements:
1. TurnIdCoordinator - Authoritative turn ID per user utterance
2. BoundedTTSQueue - Bounded queue with drop policies
3. RollingLatencyTracker - Rolling percentile metrics (p50/p95/p99)
4. ChunkedTTSProducer - Streaming LLM -> sentence chunking with turn IDs
"""

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Turn ID System
# =============================================================================


@dataclass
class TurnIdCoordinator:
    """Manages authoritative turn IDs per user.

    Each user utterance gets a monotonically increasing turn_id.
    This enables:
    - Dropping stale TTS responses when a newer user utterance arrives
    - Tracking which responses belong to which user turn
    - Preventing race conditions between LLM/TTS and new user input

    Usage:
        coord = TurnIdCoordinator()

        # When user finishes speaking
        turn_id = coord.new_turn("user1")
        # Pass turn_id to LLM -> TTS pipeline

        # Before playing TTS audio
        if coord.is_stale("user1", turn_id):
            discard_audio()  # User has started a new turn
        else:
            play_audio()

    Thread-safe: All operations are protected by a lock.
    """

    _turn_ids: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_current_turn(self, user_id: str) -> int:
        """Get the current turn ID for a user.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            Current turn ID (0 if user has no turns yet).
        """
        with self._lock:
            return self._turn_ids.get(user_id, 0)

    def new_turn(self, user_id: str) -> int:
        """Start a new turn for a user.

        Call this when a user utterance is finalized.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            The new turn ID (monotonically increasing).
        """
        with self._lock:
            current = self._turn_ids.get(user_id, 0)
            new_id = current + 1
            self._turn_ids[user_id] = new_id
            return new_id

    def is_stale(self, user_id: str, turn_id: int) -> bool:
        """Check if a turn ID is stale (superseded by a newer turn).

        Args:
            user_id: Unique identifier for the user.
            turn_id: The turn ID to check.

        Returns:
            True if turn_id < current turn (stale), False if current.
        """
        with self._lock:
            current = self._turn_ids.get(user_id, 0)
            return turn_id < current


# =============================================================================
# 2. Bounded Queue with Drop Policy
# =============================================================================


class DropPolicy(Enum):
    """Policy for handling full queues."""
    DROP_OLDEST = "drop_oldest"    # Remove oldest item when full
    DROP_STALE = "drop_stale"      # Remove items with old turn_id
    DROP_NEWEST = "drop_newest"    # Reject new items when full (default queue behavior)


@dataclass
class BoundedTTSQueue:
    """Bounded queue for TTS items with configurable drop policy.

    Prevents unbounded backlog by enforcing a maximum size and
    dropping items according to the configured policy.

    Drop policies:
    - DROP_OLDEST: Discard the oldest item when full
    - DROP_STALE: Discard items with turn_id < current turn_id
    - DROP_NEWEST: Reject new items when full (like standard queue)

    For DROP_STALE, a TurnIdCoordinator must be provided.

    Usage:
        queue = BoundedTTSQueue(maxsize=10, policy=DropPolicy.DROP_OLDEST)

        # Producer
        queue.put(TTSItem(user_id="user1", turn_id=1, text="Hello"))

        # Consumer
        item = queue.get(timeout=1.0)
        if item:
            synthesize_and_play(item)

    Thread-safe: All operations are protected by locks and conditions.
    """

    maxsize: int = 10
    policy: DropPolicy = DropPolicy.DROP_OLDEST
    turn_coordinator: Optional[TurnIdCoordinator] = None

    _queue: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _not_empty: threading.Condition = field(default=None)
    _drop_count: int = 0

    def __post_init__(self):
        """Initialize threading primitives."""
        if self._not_empty is None:
            self._not_empty = threading.Condition(self._lock)

    def put(self, item: Any) -> bool:
        """Add an item to the queue.

        If the queue is full, applies the drop policy.
        If DROP_STALE, items with stale turn_id are dropped on insert.

        Args:
            item: Item to add (must have user_id and turn_id attributes for DROP_STALE).

        Returns:
            True if item was added, False if dropped.
        """
        with self._lock:
            # DROP_STALE: check if this item is already stale
            if self.policy == DropPolicy.DROP_STALE and self.turn_coordinator:
                if hasattr(item, 'user_id') and hasattr(item, 'turn_id'):
                    if self.turn_coordinator.is_stale(item.user_id, item.turn_id):
                        self._drop_count += 1
                        return False

            # Handle full queue according to policy
            if len(self._queue) >= self.maxsize:
                if self.policy == DropPolicy.DROP_OLDEST:
                    self._queue.popleft()
                    self._drop_count += 1
                elif self.policy == DropPolicy.DROP_NEWEST:
                    self._drop_count += 1
                    return False
                elif self.policy == DropPolicy.DROP_STALE:
                    # Try to drop stale items first
                    dropped = self._drop_stale_items()
                    if not dropped and len(self._queue) >= self.maxsize:
                        # No stale items, drop oldest
                        self._queue.popleft()
                        self._drop_count += 1

            self._queue.append(item)
            self._not_empty.notify()
            return True

    def _drop_stale_items(self) -> bool:
        """Drop all stale items from queue (caller must hold lock).

        Returns:
            True if at least one item was dropped.
        """
        if not self.turn_coordinator:
            return False

        dropped_any = False
        new_queue = deque()

        for item in self._queue:
            if hasattr(item, 'user_id') and hasattr(item, 'turn_id'):
                if self.turn_coordinator.is_stale(item.user_id, item.turn_id):
                    self._drop_count += 1
                    dropped_any = True
                    continue
            new_queue.append(item)

        self._queue = new_queue
        return dropped_any

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Remove and return an item from the queue.

        If DROP_STALE policy is active, stale items are skipped.

        Args:
            timeout: Maximum seconds to wait for an item.

        Returns:
            The next item, or None if timeout expired.
        """
        deadline = time.time() + timeout if timeout else None

        with self._not_empty:
            while True:
                # Try to get a non-stale item
                item = self._get_valid_item()
                if item is not None:
                    return item

                # Queue is empty or only has stale items
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(remaining)
                else:
                    self._not_empty.wait()

    def _get_valid_item(self) -> Optional[Any]:
        """Get next valid (non-stale) item (caller must hold lock)."""
        while self._queue:
            item = self._queue.popleft()

            # Check if item is stale (DROP_STALE policy)
            if self.policy == DropPolicy.DROP_STALE and self.turn_coordinator:
                if hasattr(item, 'user_id') and hasattr(item, 'turn_id'):
                    if self.turn_coordinator.is_stale(item.user_id, item.turn_id):
                        self._drop_count += 1
                        continue  # Skip stale item

            return item
        return None

    def get_nowait(self) -> Optional[Any]:
        """Remove and return an item without blocking.

        Returns:
            The next item, or None if queue is empty.
        """
        with self._lock:
            return self._get_valid_item()

    def qsize(self) -> int:
        """Return the current queue size."""
        with self._lock:
            return len(self._queue)

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    @property
    def drop_count(self) -> int:
        """Return total number of dropped items."""
        with self._lock:
            return self._drop_count


# =============================================================================
# 3. Rolling Latency Metrics
# =============================================================================


@dataclass
class RollingLatencyTracker:
    """Track latencies in rolling windows with percentile calculation.

    Keeps the last N samples per category (ASR, LLM, TTS) and
    provides percentile metrics (p50, p95, p99).

    Usage:
        tracker = RollingLatencyTracker(window_size=100)

        # Record latencies
        tracker.record("asr", asr_latency_ms)
        tracker.record("llm", llm_latency_ms)
        tracker.record("tts", tts_latency_ms)

        # Get percentiles
        p50 = tracker.percentile("asr", 50)
        p95 = tracker.percentile("llm", 95)
        p99 = tracker.percentile("tts", 99)

        # Get all stats
        stats = tracker.get_stats()

    Thread-safe: All operations are protected by a lock.
    """

    window_size: int = 100
    _samples: Dict[str, deque] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, category: str, latency_ms: float) -> None:
        """Record a latency sample.

        Args:
            category: Category name (e.g., "asr", "llm", "tts").
            latency_ms: Latency in milliseconds.
        """
        with self._lock:
            if category not in self._samples:
                self._samples[category] = deque(maxlen=self.window_size)
            self._samples[category].append(latency_ms)

    def count(self, category: str) -> int:
        """Get number of samples in a category.

        Args:
            category: Category name.

        Returns:
            Number of samples.
        """
        with self._lock:
            if category not in self._samples:
                return 0
            return len(self._samples[category])

    def percentile(self, category: str, p: float) -> float:
        """Calculate a percentile for a category.

        Args:
            category: Category name.
            p: Percentile (0-100).

        Returns:
            Percentile value, or 0.0 if no samples.
        """
        with self._lock:
            if category not in self._samples or not self._samples[category]:
                return 0.0

            samples = sorted(self._samples[category])
            n = len(samples)

            if n == 1:
                return samples[0]

            # Calculate percentile index
            k = (p / 100.0) * (n - 1)
            f = int(k)
            c = f + 1 if f + 1 < n else f

            # Linear interpolation
            if f == c:
                return samples[f]
            return samples[f] + (k - f) * (samples[c] - samples[f])

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all categories.

        Returns:
            Dict mapping category -> {p50, p95, p99, count}.
        """
        with self._lock:
            stats = {}
            for category in self._samples:
                stats[category] = {
                    "p50": self._percentile_unlocked(category, 50),
                    "p95": self._percentile_unlocked(category, 95),
                    "p99": self._percentile_unlocked(category, 99),
                    "count": len(self._samples[category]),
                }
            return stats

    def _percentile_unlocked(self, category: str, p: float) -> float:
        """Calculate percentile without lock (caller must hold lock)."""
        if category not in self._samples or not self._samples[category]:
            return 0.0

        samples = sorted(self._samples[category])
        n = len(samples)

        if n == 1:
            return samples[0]

        k = (p / 100.0) * (n - 1)
        f = int(k)
        c = f + 1 if f + 1 < n else f

        if f == c:
            return samples[f]
        return samples[f] + (k - f) * (samples[c] - samples[f])


@dataclass
class LatencyReporter:
    """Formats and logs latency statistics.

    Usage:
        reporter = LatencyReporter(tracker, queue=tts_queue)

        # Get formatted stats string
        print(reporter.format_stats())

        # Start periodic logging
        reporter.start_periodic_logging(interval_seconds=30)
    """

    tracker: RollingLatencyTracker
    queue: Optional[BoundedTTSQueue] = None
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _logging_thread: Optional[threading.Thread] = None

    def format_stats(self) -> str:
        """Format current statistics as a string.

        Returns:
            Formatted statistics string.
        """
        stats = self.tracker.get_stats()
        parts = []

        for category, values in stats.items():
            part = (
                f"{category.upper()}: "
                f"p50={values['p50']:.0f}ms "
                f"p95={values['p95']:.0f}ms "
                f"p99={values['p99']:.0f}ms "
                f"(n={values['count']})"
            )
            parts.append(part)

        if self.queue is not None:
            parts.append(f"Queue drops: {self.queue.drop_count}")

        return " | ".join(parts)

    def log_stats(self) -> None:
        """Log current statistics."""
        stats_str = self.format_stats()
        logger.info(f"Latency stats: {stats_str}")

    def start_periodic_logging(self, interval_seconds: float = 30.0) -> None:
        """Start periodic logging of statistics.

        Args:
            interval_seconds: Seconds between log entries.
        """
        if self._logging_thread is not None:
            return

        self._stop_event.clear()

        def log_loop():
            while not self._stop_event.wait(interval_seconds):
                self.log_stats()

        self._logging_thread = threading.Thread(target=log_loop, daemon=True)
        self._logging_thread.start()

    def stop_periodic_logging(self) -> None:
        """Stop periodic logging."""
        self._stop_event.set()
        if self._logging_thread:
            self._logging_thread.join(timeout=1.0)
            self._logging_thread = None


# =============================================================================
# 4. Streaming LLM -> Sentence Chunking with Turn IDs (Phase 2)
# =============================================================================


@dataclass
class TTSQueueItem:
    """Item for the TTS queue with turn ID tracking.

    Attributes:
        user_id: User identifier.
        turn_id: Turn ID for staleness checking.
        text: Text to synthesize.
        chunk_index: Index of this chunk within the turn.
        timestamp: When the item was created.
    """

    user_id: str
    turn_id: int
    text: str
    chunk_index: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChunkedTTSProducer:
    """Produces TTS queue items from streaming LLM tokens with turn IDs.

    This class bridges streaming LLM output with the bounded TTS queue,
    implementing sentence chunking and attaching turn IDs to each chunk.

    Features:
    - Sentence boundary detection (. ! ?)
    - Clause boundary detection for long sentences (, ; :)
    - Maximum chunk length enforcement
    - Turn ID attachment for stale chunk dropping

    Usage:
        producer = ChunkedTTSProducer(turn_coordinator=coord)
        turn_id = coord.new_turn("user1")

        # Stream tokens from LLM
        async for token in llm.chat_stream(messages):
            for chunk in producer.add_token("user1", turn_id, token):
                queue.put(chunk)

        # Flush remaining text
        final = producer.flush("user1", turn_id)
        if final:
            queue.put(final)

    Thread-safe: Uses per-user state tracking.
    """

    turn_coordinator: Optional[TurnIdCoordinator] = None
    min_chars: int = 30
    max_chars: int = 150

    _buffers: Dict[str, str] = field(default_factory=dict)
    _chunk_indices: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Patterns for sentence boundaries
    _sentence_end: re.Pattern = field(
        default_factory=lambda: re.compile(r'[.!?]+\s*$')
    )
    _clause_end: re.Pattern = field(
        default_factory=lambda: re.compile(r'[,;:]+\s*$')
    )

    def _get_buffer_key(self, user_id: str, turn_id: int) -> str:
        """Generate buffer key for user+turn combination."""
        return f"{user_id}:{turn_id}"

    def add_token(
        self,
        user_id: str,
        turn_id: int,
        token: str,
    ) -> List[TTSQueueItem]:
        """Add a token and return any complete chunks.

        Args:
            user_id: User identifier.
            turn_id: Current turn ID.
            token: LLM token to add.

        Returns:
            List of complete TTSQueueItem chunks (may be empty).
        """
        with self._lock:
            key = self._get_buffer_key(user_id, turn_id)

            if key not in self._buffers:
                self._buffers[key] = ""
                self._chunk_indices[key] = 0

            self._buffers[key] += token
            buffer = self._buffers[key]
            chunks = []

            # Check for sentence end
            if len(buffer) >= self.min_chars and self._sentence_end.search(buffer):
                chunks.append(self._emit_chunk(user_id, turn_id, key))

            # Check for clause end with longer buffer
            elif len(buffer) >= self.min_chars * 2 and self._clause_end.search(buffer):
                chunks.append(self._emit_chunk(user_id, turn_id, key))

            # Force emit at max length
            elif len(buffer) >= self.max_chars:
                chunks.append(self._emit_chunk(user_id, turn_id, key))

            return chunks

    def _emit_chunk(self, user_id: str, turn_id: int, key: str) -> TTSQueueItem:
        """Emit a chunk from buffer (caller must hold lock)."""
        text = self._buffers[key].strip()
        self._buffers[key] = ""
        chunk_index = self._chunk_indices[key]
        self._chunk_indices[key] += 1

        return TTSQueueItem(
            user_id=user_id,
            turn_id=turn_id,
            text=text,
            chunk_index=chunk_index,
        )

    def flush(self, user_id: str, turn_id: int) -> Optional[TTSQueueItem]:
        """Flush any remaining buffered text.

        Args:
            user_id: User identifier.
            turn_id: Current turn ID.

        Returns:
            TTSQueueItem with remaining text, or None if buffer empty.
        """
        with self._lock:
            key = self._get_buffer_key(user_id, turn_id)

            if key not in self._buffers or not self._buffers[key].strip():
                return None

            return self._emit_chunk(user_id, turn_id, key)

    def chunk_text(
        self,
        user_id: str,
        turn_id: int,
        text: str,
    ) -> Iterator[TTSQueueItem]:
        """Chunk a complete text into TTS queue items.

        Convenience method for non-streaming use cases.

        Args:
            user_id: User identifier.
            turn_id: Current turn ID.
            text: Complete text to chunk.

        Yields:
            TTSQueueItem for each chunk.
        """
        # Reset buffer for this user+turn
        with self._lock:
            key = self._get_buffer_key(user_id, turn_id)
            self._buffers[key] = ""
            self._chunk_indices[key] = 0

        # Feed text token by token (simulate streaming)
        words = text.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            for chunk in self.add_token(user_id, turn_id, token):
                yield chunk

        # Flush remaining
        final = self.flush(user_id, turn_id)
        if final:
            yield final

    def reset(self, user_id: str, turn_id: Optional[int] = None) -> None:
        """Reset buffers for a user.

        Args:
            user_id: User identifier.
            turn_id: Specific turn to reset, or None for all turns.
        """
        with self._lock:
            if turn_id is not None:
                key = self._get_buffer_key(user_id, turn_id)
                self._buffers.pop(key, None)
                self._chunk_indices.pop(key, None)
            else:
                # Clear all buffers for this user
                keys_to_remove = [
                    k for k in self._buffers.keys()
                    if k.startswith(f"{user_id}:")
                ]
                for key in keys_to_remove:
                    self._buffers.pop(key, None)
                    self._chunk_indices.pop(key, None)


# =============================================================================
# 5. Split Synthesis/Playback Workers (Phase 3)
# =============================================================================


@dataclass
class AudioQueueItem:
    """Item for the audio playback queue.

    Attributes:
        user_id: User identifier.
        turn_id: Turn ID for staleness checking.
        data: Audio bytes to play.
        duration_ms: Duration of audio in milliseconds.
        chunk_index: Index of this chunk within the turn.
        timestamp: When the item was created.
    """

    user_id: str
    turn_id: int
    data: bytes
    duration_ms: float
    chunk_index: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TTSSynthesisWorker:
    """Worker that synthesizes text to audio.

    This worker reads TTSQueueItem from a text queue, synthesizes
    audio using the provided function, and puts AudioQueueItem
    into an audio queue for playback.

    Features:
    - Skips stale items (turn_id < current)
    - Runs in background thread
    - Coordinates with TurnIdCoordinator

    Usage:
        worker = TTSSynthesisWorker(
            text_queue=text_queue,
            audio_queue=audio_queue,
            synthesize_fn=tts.synthesize,
            turn_coordinator=coord,
        )
        worker.start()
        # ... later ...
        worker.stop()
    """

    text_queue: BoundedTTSQueue
    audio_queue: BoundedTTSQueue
    synthesize_fn: Callable[[str], bytes]
    turn_coordinator: Optional[TurnIdCoordinator] = None
    estimate_duration_fn: Optional[Callable[[str], float]] = None

    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def _estimate_duration(self, text: str) -> float:
        """Estimate audio duration for text."""
        if self.estimate_duration_fn:
            return self.estimate_duration_fn(text)
        # Default: ~150ms per word
        words = len(text.split())
        return words * 150.0

    def run_once(self) -> bool:
        """Process one item from the queue.

        Returns:
            True if an item was processed, False if queue empty or stopped.
        """
        item = self.text_queue.get(timeout=0.1)

        if item is None:
            return False

        # Check if stale
        if self.turn_coordinator and hasattr(item, 'user_id') and hasattr(item, 'turn_id'):
            if self.turn_coordinator.is_stale(item.user_id, item.turn_id):
                logger.debug(f"Skipping stale synthesis for turn {item.turn_id}")
                return True  # Item processed (skipped)

        try:
            # Synthesize
            audio_data = self.synthesize_fn(item.text)
            duration_ms = self._estimate_duration(item.text)

            # Put audio into queue
            audio_item = AudioQueueItem(
                user_id=item.user_id,
                turn_id=item.turn_id,
                data=audio_data,
                duration_ms=duration_ms,
                chunk_index=item.chunk_index,
            )
            self.audio_queue.put(audio_item)
            return True

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return True

    def run(self) -> None:
        """Run the worker loop until stopped."""
        logger.info("Synthesis worker started")
        while not self._stop_event.is_set():
            self.run_once()
        logger.info("Synthesis worker stopped")

    def start(self) -> None:
        """Start the worker in a background thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None


@dataclass
class TTSPlaybackWorker:
    """Worker that plays audio chunks.

    This worker reads AudioQueueItem from an audio queue and
    plays them using the provided playback function.

    Features:
    - Skips stale audio (turn_id < current)
    - Adaptive pacing based on queue length
    - Runs in background thread

    Usage:
        worker = TTSPlaybackWorker(
            audio_queue=audio_queue,
            play_fn=mumble.send_audio,
            turn_coordinator=coord,
        )
        worker.start()
    """

    audio_queue: BoundedTTSQueue
    play_fn: Callable[[bytes, float], None]
    turn_coordinator: Optional[TurnIdCoordinator] = None
    pacer: Optional["AdaptivePacer"] = None

    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def run_once(self) -> bool:
        """Process one item from the queue.

        Returns:
            True if an item was processed, False if queue empty or stopped.
        """
        item = self.audio_queue.get(timeout=0.1)

        if item is None:
            return False

        # Check if stale
        if self.turn_coordinator and hasattr(item, 'user_id') and hasattr(item, 'turn_id'):
            if self.turn_coordinator.is_stale(item.user_id, item.turn_id):
                logger.debug(f"Skipping stale playback for turn {item.turn_id}")
                return True  # Item processed (skipped)

        try:
            # Play audio
            self.play_fn(item.data, item.duration_ms)

            # Adaptive pacing
            if self.pacer:
                delay_ms = self.pacer.get_delay(self.audio_queue.qsize())
                time.sleep(delay_ms / 1000.0)

            return True

        except Exception as e:
            logger.error(f"Playback error: {e}")
            return True

    def run(self) -> None:
        """Run the worker loop until stopped."""
        logger.info("Playback worker started")
        while not self._stop_event.is_set():
            self.run_once()
        logger.info("Playback worker stopped")

    def start(self) -> None:
        """Start the worker in a background thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None


# =============================================================================
# 6. Adaptive Pacing (Phase 3)
# =============================================================================


@dataclass
class AdaptivePacer:
    """Adaptive pacing for playback based on queue length.

    When the queue is long (backlog), use minimal delay to drain faster.
    When the queue is empty/short, use longer delay for natural cadence.

    The delay scales linearly between min and max based on queue length:
    - queue_length >= threshold: min_delay_ms
    - queue_length == 0: max_delay_ms
    - in between: linear interpolation

    Usage:
        pacer = AdaptivePacer(min_delay_ms=10, max_delay_ms=200)

        while playing:
            play_chunk()
            delay = pacer.get_delay(queue.qsize())
            time.sleep(delay / 1000.0)
    """

    min_delay_ms: float = 10.0
    max_delay_ms: float = 200.0
    queue_threshold: int = 5

    def get_delay(self, queue_length: int) -> float:
        """Calculate delay based on queue length.

        Args:
            queue_length: Current number of items in queue.

        Returns:
            Delay in milliseconds.
        """
        if queue_length >= self.queue_threshold:
            return self.min_delay_ms

        if queue_length == 0:
            return self.max_delay_ms

        # Linear interpolation
        ratio = queue_length / self.queue_threshold
        delay_range = self.max_delay_ms - self.min_delay_ms
        return self.max_delay_ms - (ratio * delay_range)
