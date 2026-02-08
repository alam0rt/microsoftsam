"""Tests for performance improvements (docs/perf.md Phase 1).

Tests for:
1. Turn ID system - ensures stale responses are dropped
2. Bounded queues with drop policy - prevents backlog
3. Rolling latency metrics - tracks p50/p95/p99

TDD: Write tests first, then implement.
"""

import threading
import time
from dataclasses import dataclass

import pytest

# =============================================================================
# 1. Turn ID System Tests
# =============================================================================


class TestTurnIdCoordinator:
    """Tests for turn ID coordination (perf.md Section 1)."""

    def test_initial_state(self):
        """Coordinator starts with turn_id 0 for all users."""
        from mumble_voice_bot.perf import TurnIdCoordinator

        coord = TurnIdCoordinator()
        assert coord.get_current_turn("user1") == 0
        assert coord.get_current_turn("user2") == 0

    def test_new_turn_increments_id(self):
        """new_turn() increments the turn ID for a user."""
        from mumble_voice_bot.perf import TurnIdCoordinator

        coord = TurnIdCoordinator()
        turn_id = coord.new_turn("user1")
        assert turn_id == 1
        assert coord.get_current_turn("user1") == 1

    def test_multiple_turns_increment_separately(self):
        """Each new turn increments monotonically per user."""
        from mumble_voice_bot.perf import TurnIdCoordinator

        coord = TurnIdCoordinator()
        assert coord.new_turn("user1") == 1
        assert coord.new_turn("user1") == 2
        assert coord.new_turn("user1") == 3
        assert coord.get_current_turn("user1") == 3

    def test_users_have_independent_turn_ids(self):
        """Different users have independent turn ID sequences."""
        from mumble_voice_bot.perf import TurnIdCoordinator

        coord = TurnIdCoordinator()
        assert coord.new_turn("alice") == 1
        assert coord.new_turn("bob") == 1
        assert coord.new_turn("alice") == 2
        assert coord.get_current_turn("alice") == 2
        assert coord.get_current_turn("bob") == 1

    def test_is_stale_for_old_turn_id(self):
        """is_stale() returns True for old turn IDs."""
        from mumble_voice_bot.perf import TurnIdCoordinator

        coord = TurnIdCoordinator()
        coord.new_turn("user1")  # turn 1
        coord.new_turn("user1")  # turn 2

        assert coord.is_stale("user1", 1) is True
        assert coord.is_stale("user1", 2) is False

    def test_is_stale_for_unknown_user(self):
        """is_stale() behavior for unknown user (default turn 0)."""
        from mumble_voice_bot.perf import TurnIdCoordinator

        coord = TurnIdCoordinator()
        # User doesn't exist, current is 0
        # turn_id=1 is NOT stale (it's newer than current=0)
        assert coord.is_stale("unknown", 1) is False
        # Turn 0 is current for unknown user (not stale)
        assert coord.is_stale("unknown", 0) is False

    def test_thread_safety(self):
        """TurnIdCoordinator is thread-safe."""
        from mumble_voice_bot.perf import TurnIdCoordinator

        coord = TurnIdCoordinator()
        errors = []

        def worker(user_id: str, count: int):
            try:
                for _ in range(count):
                    coord.new_turn(user_id)
                    coord.get_current_turn(user_id)
                    coord.is_stale(user_id, 0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("user1", 100)),
            threading.Thread(target=worker, args=("user2", 100)),
            threading.Thread(target=worker, args=("user1", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # user1 should have 200 turns (from two threads)
        assert coord.get_current_turn("user1") == 200
        assert coord.get_current_turn("user2") == 100


# =============================================================================
# 2. Bounded Queue Tests
# =============================================================================


@dataclass
class MockTTSItem:
    """Mock TTS queue item for testing."""
    user_id: str
    turn_id: int
    text: str
    timestamp: float = 0.0


class TestBoundedTTSQueue:
    """Tests for bounded TTS queue with drop policy (perf.md Section 2)."""

    def test_basic_enqueue_dequeue(self):
        """Basic FIFO behavior for queue within bounds."""
        from mumble_voice_bot.perf import BoundedTTSQueue

        queue = BoundedTTSQueue(maxsize=5)
        item1 = MockTTSItem("user1", 1, "Hello")
        item2 = MockTTSItem("user1", 1, "World")

        queue.put(item1)
        queue.put(item2)

        assert queue.get() == item1
        assert queue.get() == item2

    def test_queue_respects_maxsize(self):
        """Queue does not exceed maxsize."""
        from mumble_voice_bot.perf import BoundedTTSQueue

        queue = BoundedTTSQueue(maxsize=3)

        for i in range(10):
            item = MockTTSItem("user1", 1, f"Item {i}")
            queue.put(item)

        assert queue.qsize() <= 3

    def test_drop_oldest_policy(self):
        """With drop-oldest policy, oldest items are dropped when full."""
        from mumble_voice_bot.perf import BoundedTTSQueue, DropPolicy

        queue = BoundedTTSQueue(maxsize=3, policy=DropPolicy.DROP_OLDEST)

        queue.put(MockTTSItem("user1", 1, "A"))
        queue.put(MockTTSItem("user1", 1, "B"))
        queue.put(MockTTSItem("user1", 1, "C"))
        queue.put(MockTTSItem("user1", 1, "D"))  # Drops A

        items = []
        while not queue.empty():
            items.append(queue.get().text)

        assert items == ["B", "C", "D"]

    def test_drop_stale_policy(self):
        """With drop-stale policy, items with old turn_id are dropped."""
        from mumble_voice_bot.perf import BoundedTTSQueue, DropPolicy, TurnIdCoordinator

        coord = TurnIdCoordinator()
        coord.new_turn("user1")  # turn 1
        coord.new_turn("user1")  # turn 2

        queue = BoundedTTSQueue(maxsize=5, policy=DropPolicy.DROP_STALE, turn_coordinator=coord)

        queue.put(MockTTSItem("user1", 1, "Old"))  # stale - dropped
        queue.put(MockTTSItem("user1", 2, "Current"))  # valid

        items = []
        while not queue.empty():
            items.append(queue.get().text)

        assert items == ["Current"]

    def test_drop_count_tracking(self):
        """Queue tracks number of dropped items."""
        from mumble_voice_bot.perf import BoundedTTSQueue, DropPolicy

        queue = BoundedTTSQueue(maxsize=2, policy=DropPolicy.DROP_OLDEST)

        for i in range(5):
            queue.put(MockTTSItem("user1", 1, f"Item {i}"))

        assert queue.drop_count >= 3

    def test_get_blocks_when_empty(self):
        """get() blocks when queue is empty (with timeout)."""
        from mumble_voice_bot.perf import BoundedTTSQueue

        queue = BoundedTTSQueue(maxsize=5)

        start = time.time()
        result = queue.get(timeout=0.1)
        elapsed = time.time() - start

        assert result is None
        assert elapsed >= 0.1

    def test_get_nowait_raises_when_empty(self):
        """get_nowait() returns None when queue is empty."""
        from mumble_voice_bot.perf import BoundedTTSQueue

        queue = BoundedTTSQueue(maxsize=5)
        assert queue.get_nowait() is None

    def test_thread_safe_operations(self):
        """Queue operations are thread-safe."""
        from mumble_voice_bot.perf import BoundedTTSQueue

        queue = BoundedTTSQueue(maxsize=100)
        errors = []
        produced = []
        consumed = []

        def producer():
            try:
                for i in range(50):
                    item = MockTTSItem("user1", 1, f"Item {i}")
                    queue.put(item)
                    produced.append(i)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                for _ in range(50):
                    item = queue.get(timeout=1.0)
                    if item:
                        consumed.append(item.text)
            except Exception as e:
                errors.append(e)

        p_thread = threading.Thread(target=producer)
        c_thread = threading.Thread(target=consumer)

        p_thread.start()
        c_thread.start()
        p_thread.join()
        c_thread.join()

        assert len(errors) == 0
        assert len(consumed) == 50


class TestBoundedQueueDropStaleOnDequeue:
    """Tests for drop-stale behavior on dequeue (perf.md detail)."""

    def test_stale_items_dropped_on_get(self):
        """Stale items are dropped when dequeued, not just on enqueue."""
        from mumble_voice_bot.perf import BoundedTTSQueue, DropPolicy, TurnIdCoordinator

        coord = TurnIdCoordinator()
        coord.new_turn("user1")  # turn 1

        queue = BoundedTTSQueue(maxsize=5, policy=DropPolicy.DROP_STALE, turn_coordinator=coord)

        # Add items for turn 1
        queue.put(MockTTSItem("user1", 1, "Turn 1 A"))
        queue.put(MockTTSItem("user1", 1, "Turn 1 B"))

        # User starts new turn before items are processed
        coord.new_turn("user1")  # turn 2
        queue.put(MockTTSItem("user1", 2, "Turn 2"))

        # All turn 1 items should be dropped on get
        item = queue.get()
        assert item.text == "Turn 2"
        assert item.turn_id == 2


# =============================================================================
# 3. Rolling Latency Metrics Tests
# =============================================================================


class TestRollingLatencyTracker:
    """Tests for rolling latency metrics (perf.md Section 5)."""

    def test_record_latency(self):
        """Can record latency samples."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=100)
        tracker.record("asr", 150.0)
        tracker.record("asr", 200.0)

        assert tracker.count("asr") == 2

    def test_rolling_window_limits_samples(self):
        """Tracker only keeps last N samples."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=5)

        for i in range(10):
            tracker.record("llm", float(i * 100))

        assert tracker.count("llm") == 5

    def test_percentile_p50(self):
        """Calculates p50 (median) correctly."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=100)

        # 100, 200, 300, 400, 500 -> median is 300
        for v in [100, 200, 300, 400, 500]:
            tracker.record("tts", float(v))

        assert tracker.percentile("tts", 50) == 300.0

    def test_percentile_p95(self):
        """Calculates p95 correctly."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=100)

        # 100 samples from 1 to 100
        for i in range(1, 101):
            tracker.record("asr", float(i))

        # p95 should be around 95
        p95 = tracker.percentile("asr", 95)
        assert 94 <= p95 <= 96

    def test_percentile_p99(self):
        """Calculates p99 correctly."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=100)

        for i in range(1, 101):
            tracker.record("llm", float(i))

        p99 = tracker.percentile("llm", 99)
        assert 98 <= p99 <= 100

    def test_multiple_categories(self):
        """Tracks multiple latency categories independently."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=100)

        tracker.record("asr", 100.0)
        tracker.record("llm", 500.0)
        tracker.record("tts", 200.0)

        assert tracker.percentile("asr", 50) == 100.0
        assert tracker.percentile("llm", 50) == 500.0
        assert tracker.percentile("tts", 50) == 200.0

    def test_empty_category_returns_zero(self):
        """Empty category returns 0 for percentiles."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=100)
        assert tracker.percentile("unknown", 50) == 0.0

    def test_get_stats_returns_all_percentiles(self):
        """get_stats() returns p50, p95, p99 for all categories."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=100)

        for i in range(1, 101):
            tracker.record("asr", float(i))
            tracker.record("llm", float(i * 2))

        stats = tracker.get_stats()

        assert "asr" in stats
        assert "p50" in stats["asr"]
        assert "p95" in stats["asr"]
        assert "p99" in stats["asr"]
        assert "count" in stats["asr"]

        assert "llm" in stats
        assert stats["llm"]["p50"] == pytest.approx(stats["asr"]["p50"] * 2, rel=0.1)

    def test_thread_safety(self):
        """RollingLatencyTracker is thread-safe."""
        from mumble_voice_bot.perf import RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=1000)
        errors = []

        def recorder(category: str):
            try:
                for i in range(100):
                    tracker.record(category, float(i))
                    tracker.percentile(category, 50)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=recorder, args=("asr",)),
            threading.Thread(target=recorder, args=("llm",)),
            threading.Thread(target=recorder, args=("tts",)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestRollingLatencyReporter:
    """Tests for periodic latency reporting."""

    def test_reporter_logs_stats(self):
        """Reporter logs statistics at intervals."""
        from mumble_voice_bot.perf import LatencyReporter, RollingLatencyTracker

        tracker = RollingLatencyTracker(window_size=100)
        for i in range(10):
            tracker.record("asr", float(i * 10))

        reporter = LatencyReporter(tracker)
        stats = reporter.format_stats()

        assert "ASR" in stats  # Category is uppercased in output
        assert "p50" in stats

    def test_reporter_includes_drop_count(self):
        """Reporter includes queue drop count if provided."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            DropPolicy,
            LatencyReporter,
            RollingLatencyTracker,
        )

        tracker = RollingLatencyTracker(window_size=100)
        queue = BoundedTTSQueue(maxsize=2, policy=DropPolicy.DROP_OLDEST)

        # Fill and overflow queue
        for i in range(5):
            queue.put(MockTTSItem("user1", 1, f"Item {i}"))

        reporter = LatencyReporter(tracker, queue=queue)
        stats = reporter.format_stats()

        assert "drops" in stats.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestTurnIdWithQueue:
    """Integration tests for Turn ID + Bounded Queue."""

    def test_older_responses_never_speak(self):
        """Acceptance: Older responses never speak after newer user utterance."""
        from mumble_voice_bot.perf import BoundedTTSQueue, DropPolicy, TurnIdCoordinator

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(maxsize=10, policy=DropPolicy.DROP_STALE, turn_coordinator=coord)

        # User says something (turn 1)
        turn1 = coord.new_turn("user1")
        queue.put(MockTTSItem("user1", turn1, "Response to turn 1, part A"))
        queue.put(MockTTSItem("user1", turn1, "Response to turn 1, part B"))

        # User interrupts before we process (turn 2)
        turn2 = coord.new_turn("user1")
        queue.put(MockTTSItem("user1", turn2, "Response to turn 2"))

        # Process queue - should only get turn 2 response
        responses = []
        while not queue.empty():
            item = queue.get()
            if item:
                responses.append(item.text)

        assert responses == ["Response to turn 2"]

    def test_queue_bounded_under_load(self):
        """Acceptance: Under heavy load, queue size stays bounded."""
        from mumble_voice_bot.perf import BoundedTTSQueue, DropPolicy

        queue = BoundedTTSQueue(maxsize=5, policy=DropPolicy.DROP_OLDEST)

        # Simulate heavy load
        for i in range(100):
            queue.put(MockTTSItem("user1", 1, f"Item {i}"))
            assert queue.qsize() <= 5

        # Queue never exceeded maxsize
        assert queue.qsize() <= 5
        # Significant drops occurred
        assert queue.drop_count >= 95


# =============================================================================
# Phase 2: Streaming LLM + Sentence Chunking with Turn ID Tests
# =============================================================================


@dataclass
class TTSQueueItem:
    """TTS queue item with turn ID for testing."""
    user_id: str
    turn_id: int
    text: str
    chunk_index: int = 0


class TestSentenceChunkingWithTurnId:
    """Tests for sentence chunking that attaches turn_id (perf.md Section 4)."""

    def test_chunks_inherit_turn_id(self):
        """Each sentence chunk inherits the correct turn_id."""
        from mumble_voice_bot.perf import ChunkedTTSProducer, TurnIdCoordinator

        coord = TurnIdCoordinator()
        producer = ChunkedTTSProducer(turn_coordinator=coord, min_chars=20)

        turn_id = coord.new_turn("user1")
        chunks = list(producer.chunk_text(
            "user1",
            turn_id,
            "Hello world, this is a test. How are you doing today? I am doing fine thank you."
        ))

        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.turn_id == turn_id
            assert chunk.user_id == "user1"

    def test_multiple_turns_have_distinct_ids(self):
        """Chunks from different turns have different turn_ids."""
        from mumble_voice_bot.perf import ChunkedTTSProducer, TurnIdCoordinator

        coord = TurnIdCoordinator()
        producer = ChunkedTTSProducer(turn_coordinator=coord)

        turn1 = coord.new_turn("user1")
        chunks1 = list(producer.chunk_text("user1", turn1, "First message."))

        turn2 = coord.new_turn("user1")
        chunks2 = list(producer.chunk_text("user1", turn2, "Second message."))

        assert chunks1[0].turn_id == turn1
        assert chunks2[0].turn_id == turn2
        assert turn1 != turn2


class TestBargeInCancellation:
    """Tests for barge-in cancellation (perf.md Section 4 acceptance)."""

    def test_barge_in_cancels_pending_chunks(self):
        """On barge-in, pending chunks for old turn are cancelled."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            ChunkedTTSProducer,
            DropPolicy,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(
            maxsize=20,
            policy=DropPolicy.DROP_STALE,
            turn_coordinator=coord,
        )
        producer = ChunkedTTSProducer(turn_coordinator=coord)

        # Start first turn - generate chunks
        turn1 = coord.new_turn("user1")
        for chunk in producer.chunk_text(
            "user1", turn1,
            "This is a long response. It has multiple sentences. "
            "We will speak all of them. Unless interrupted."
        ):
            queue.put(chunk)

        # Simulate barge-in: new turn starts
        turn2 = coord.new_turn("user1")

        # Add response for new turn
        for chunk in producer.chunk_text("user1", turn2, "New response."):
            queue.put(chunk)

        # All old chunks should be dropped, only new turn chunks remain
        results = []
        while not queue.empty():
            item = queue.get()
            if item:
                results.append((item.turn_id, item.text))

        assert all(turn_id == turn2 for turn_id, _ in results)
        assert any("New response" in text for _, text in results)

    def test_chunks_dropped_on_dequeue_after_barge_in(self):
        """Stale chunks are dropped when dequeued (lazy dropping)."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            DropPolicy,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(
            maxsize=10,
            policy=DropPolicy.DROP_STALE,
            turn_coordinator=coord,
        )

        # Add chunks for turn 1
        turn1 = coord.new_turn("user1")
        queue.put(TTSQueueItem("user1", turn1, "Chunk A", 0))
        queue.put(TTSQueueItem("user1", turn1, "Chunk B", 1))
        queue.put(TTSQueueItem("user1", turn1, "Chunk C", 2))

        # Barge-in: new turn before dequeue
        turn2 = coord.new_turn("user1")
        queue.put(TTSQueueItem("user1", turn2, "New chunk", 0))

        # Dequeue - old chunks should be skipped
        item = queue.get()
        assert item.turn_id == turn2
        assert item.text == "New chunk"


class TestStreamingLLMTTSIntegration:
    """Integration tests for streaming LLM -> chunking -> TTS queue."""

    def test_ttfa_with_streaming(self):
        """First audio begins within target latency with streaming."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            ChunkedTTSProducer,
            DropPolicy,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(maxsize=20, policy=DropPolicy.DROP_STALE, turn_coordinator=coord)
        producer = ChunkedTTSProducer(turn_coordinator=coord)

        turn_id = coord.new_turn("user1")

        # Simulate streaming LLM tokens -> chunking
        start_time = time.time()
        tokens = ["Hello ", "world. ", "This ", "is ", "a ", "test. ", "More ", "text."]

        first_chunk_time = None
        for token in tokens:
            chunks = producer.add_token("user1", turn_id, token)
            for chunk in chunks:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                queue.put(chunk)

        # Flush remaining
        final = producer.flush("user1", turn_id)
        if final:
            queue.put(final)

        # First chunk should be available very quickly (< 100ms for test)
        if first_chunk_time:
            latency = (first_chunk_time - start_time) * 1000
            assert latency < 100, f"First chunk latency {latency}ms too high"

    def test_all_chunks_have_correct_turn_id(self):
        """All chunks from a streaming session have correct turn_id."""
        from mumble_voice_bot.perf import ChunkedTTSProducer, TurnIdCoordinator

        coord = TurnIdCoordinator()
        producer = ChunkedTTSProducer(turn_coordinator=coord)

        turn_id = coord.new_turn("user1")
        all_chunks = []

        # Stream tokens
        for token in ["The ", "quick ", "brown ", "fox. ", "Jumps ", "over. ", "Done."]:
            chunks = producer.add_token("user1", turn_id, token)
            all_chunks.extend(chunks)

        # Flush
        final = producer.flush("user1", turn_id)
        if final:
            all_chunks.append(final)

        assert len(all_chunks) >= 2
        for chunk in all_chunks:
            assert chunk.turn_id == turn_id


# =============================================================================
# Phase 3: Split Synthesis/Playback Workers + Adaptive Pacing
# =============================================================================


@dataclass
class MockAudioChunk:
    """Mock audio chunk for testing."""
    data: bytes
    duration_ms: float
    turn_id: int
    chunk_index: int


class TestTTSSynthesisWorker:
    """Tests for TTS synthesis worker (perf.md Section 3)."""

    def test_synthesis_produces_audio_chunks(self):
        """Synthesis worker produces audio chunks from text items."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            TTSQueueItem,
            TTSSynthesisWorker,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        text_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)
        audio_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)

        # Mock TTS function
        def mock_synthesize(text: str) -> bytes:
            return f"audio:{text}".encode()

        worker = TTSSynthesisWorker(
            text_queue=text_queue,
            audio_queue=audio_queue,
            synthesize_fn=mock_synthesize,
            turn_coordinator=coord,
        )

        # Add text item
        turn_id = coord.new_turn("user1")
        text_queue.put(TTSQueueItem("user1", turn_id, "Hello world", 0))
        text_queue.put(None)  # Signal to stop

        # Run worker
        worker.run_once()

        # Check audio output
        audio_item = audio_queue.get_nowait()
        assert audio_item is not None
        assert audio_item.turn_id == turn_id
        assert b"Hello world" in audio_item.data

    def test_synthesis_skips_stale_items(self):
        """Synthesis worker skips items with stale turn_id."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            DropPolicy,
            TTSQueueItem,
            TTSSynthesisWorker,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        text_queue = BoundedTTSQueue(
            maxsize=10,
            policy=DropPolicy.DROP_STALE,
            turn_coordinator=coord,
        )
        audio_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)

        synthesize_calls = []

        def mock_synthesize(text: str) -> bytes:
            synthesize_calls.append(text)
            return f"audio:{text}".encode()

        worker = TTSSynthesisWorker(
            text_queue=text_queue,
            audio_queue=audio_queue,
            synthesize_fn=mock_synthesize,
            turn_coordinator=coord,
        )

        # Add items for turn 1
        turn1 = coord.new_turn("user1")
        text_queue.put(TTSQueueItem("user1", turn1, "Old message", 0))

        # New turn (makes turn1 stale)
        turn2 = coord.new_turn("user1")
        text_queue.put(TTSQueueItem("user1", turn2, "New message", 0))
        text_queue.put(None)

        # Run worker
        worker.run_once()
        worker.run_once()

        # Only new message should be synthesized
        assert "New message" in synthesize_calls
        assert "Old message" not in synthesize_calls

    def test_synthesis_worker_thread_safe(self):
        """Synthesis worker can run in a thread."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            TTSQueueItem,
            TTSSynthesisWorker,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        text_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)
        audio_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)

        def mock_synthesize(text: str) -> bytes:
            time.sleep(0.01)  # Simulate work
            return f"audio:{text}".encode()

        worker = TTSSynthesisWorker(
            text_queue=text_queue,
            audio_queue=audio_queue,
            synthesize_fn=mock_synthesize,
            turn_coordinator=coord,
        )

        # Start worker thread
        worker_thread = threading.Thread(target=worker.run, daemon=True)
        worker_thread.start()

        # Add items
        turn_id = coord.new_turn("user1")
        for i in range(5):
            text_queue.put(TTSQueueItem("user1", turn_id, f"Message {i}", i))

        # Signal stop
        text_queue.put(None)
        worker_thread.join(timeout=2.0)

        # Verify all items processed
        assert audio_queue.qsize() == 5


class TestTTSPlaybackWorker:
    """Tests for TTS playback worker (perf.md Section 3)."""

    def test_playback_consumes_audio_chunks(self):
        """Playback worker consumes audio chunks in order."""
        from mumble_voice_bot.perf import (
            AudioQueueItem,
            BoundedTTSQueue,
            TTSPlaybackWorker,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        audio_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)

        played = []

        def mock_play(data: bytes, duration_ms: float):
            played.append(data)

        worker = TTSPlaybackWorker(
            audio_queue=audio_queue,
            play_fn=mock_play,
            turn_coordinator=coord,
        )

        # Add audio items
        turn_id = coord.new_turn("user1")
        audio_queue.put(AudioQueueItem("user1", turn_id, b"chunk1", 100.0, 0))
        audio_queue.put(AudioQueueItem("user1", turn_id, b"chunk2", 100.0, 1))
        audio_queue.put(None)  # Signal stop

        # Run worker
        worker.run_once()
        worker.run_once()

        assert played == [b"chunk1", b"chunk2"]

    def test_playback_skips_stale_audio(self):
        """Playback worker skips audio with stale turn_id."""
        from mumble_voice_bot.perf import (
            AudioQueueItem,
            BoundedTTSQueue,
            DropPolicy,
            TTSPlaybackWorker,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        audio_queue = BoundedTTSQueue(
            maxsize=10,
            policy=DropPolicy.DROP_STALE,
            turn_coordinator=coord,
        )

        played = []

        def mock_play(data: bytes, duration_ms: float):
            played.append(data)

        worker = TTSPlaybackWorker(
            audio_queue=audio_queue,
            play_fn=mock_play,
            turn_coordinator=coord,
        )

        # Add items for turn 1
        turn1 = coord.new_turn("user1")
        audio_queue.put(AudioQueueItem("user1", turn1, b"old_audio", 100.0, 0))

        # New turn
        turn2 = coord.new_turn("user1")
        audio_queue.put(AudioQueueItem("user1", turn2, b"new_audio", 100.0, 0))
        audio_queue.put(None)

        # Run worker
        worker.run_once()
        worker.run_once()

        # Only new audio played
        assert b"new_audio" in played
        assert b"old_audio" not in played


class TestAdaptivePacer:
    """Tests for adaptive pacing (perf.md Section 6)."""

    def test_pacer_returns_short_delay_when_queue_full(self):
        """Pacer returns minimal delay when queue has backlog."""
        from mumble_voice_bot.perf import AdaptivePacer

        pacer = AdaptivePacer(
            min_delay_ms=10,
            max_delay_ms=200,
            queue_threshold=5,
        )

        # Queue is long - should return min delay
        delay = pacer.get_delay(queue_length=10)
        assert delay == 10

    def test_pacer_returns_long_delay_when_queue_empty(self):
        """Pacer returns longer delay when queue is empty for natural cadence."""
        from mumble_voice_bot.perf import AdaptivePacer

        pacer = AdaptivePacer(
            min_delay_ms=10,
            max_delay_ms=200,
            queue_threshold=5,
        )

        # Queue is empty - should return max delay
        delay = pacer.get_delay(queue_length=0)
        assert delay == 200

    def test_pacer_scales_linearly(self):
        """Pacer scales delay linearly based on queue length."""
        from mumble_voice_bot.perf import AdaptivePacer

        pacer = AdaptivePacer(
            min_delay_ms=0,
            max_delay_ms=100,
            queue_threshold=10,
        )

        # At threshold/2, should be ~50ms
        delay = pacer.get_delay(queue_length=5)
        assert 45 <= delay <= 55

    def test_pacer_clamps_to_min_above_threshold(self):
        """Pacer clamps to min delay above threshold."""
        from mumble_voice_bot.perf import AdaptivePacer

        pacer = AdaptivePacer(
            min_delay_ms=10,
            max_delay_ms=200,
            queue_threshold=5,
        )

        # Way above threshold
        delay = pacer.get_delay(queue_length=100)
        assert delay == 10


class TestSplitPipelineIntegration:
    """Integration tests for split synthesis/playback pipeline."""

    def test_end_to_end_split_pipeline(self):
        """Full pipeline: text -> synthesis -> playback."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            TTSPlaybackWorker,
            TTSQueueItem,
            TTSSynthesisWorker,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        text_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)
        audio_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)

        played = []

        def mock_synthesize(text: str) -> bytes:
            return f"audio:{text}".encode()

        def mock_play(data: bytes, duration_ms: float):
            played.append(data)

        synthesis_worker = TTSSynthesisWorker(
            text_queue=text_queue,
            audio_queue=audio_queue,
            synthesize_fn=mock_synthesize,
            turn_coordinator=coord,
        )

        playback_worker = TTSPlaybackWorker(
            audio_queue=audio_queue,
            play_fn=mock_play,
            turn_coordinator=coord,
        )

        # Add text
        turn_id = coord.new_turn("user1")
        text_queue.put(TTSQueueItem("user1", turn_id, "Hello", 0))
        text_queue.put(TTSQueueItem("user1", turn_id, "World", 1))

        # Run synthesis
        synthesis_worker.run_once()
        synthesis_worker.run_once()

        # Run playback
        playback_worker.run_once()
        playback_worker.run_once()

        assert len(played) == 2
        assert b"Hello" in played[0]
        assert b"World" in played[1]

    def test_ttfa_improves_with_split_workers(self):
        """TTFA: playback can start while synthesis continues."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            TTSPlaybackWorker,
            TTSQueueItem,
            TTSSynthesisWorker,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        text_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)
        audio_queue = BoundedTTSQueue(maxsize=10, turn_coordinator=coord)

        first_play_time = None

        def mock_synthesize(text: str) -> bytes:
            time.sleep(0.05)  # 50ms per synthesis
            return f"audio:{text}".encode()

        def mock_play(data: bytes, duration_ms: float):
            nonlocal first_play_time
            if first_play_time is None:
                first_play_time = time.time()

        synthesis_worker = TTSSynthesisWorker(
            text_queue=text_queue,
            audio_queue=audio_queue,
            synthesize_fn=mock_synthesize,
            turn_coordinator=coord,
        )

        playback_worker = TTSPlaybackWorker(
            audio_queue=audio_queue,
            play_fn=mock_play,
            turn_coordinator=coord,
        )

        # Start timing
        start_time = time.time()

        # Add 3 text items
        turn_id = coord.new_turn("user1")
        for i in range(3):
            text_queue.put(TTSQueueItem("user1", turn_id, f"Message {i}", i))

        # Run synthesis for first item
        synthesis_worker.run_once()

        # Playback can start before all synthesis completes
        playback_worker.run_once()

        # Check that first play happened before all synthesis would complete
        # (3 items * 50ms = 150ms total synthesis)
        if first_play_time:
            ttfa = (first_play_time - start_time) * 1000
            # TTFA should be ~50ms (one synthesis), not 150ms
            assert ttfa < 100, f"TTFA {ttfa}ms should be < 100ms"


# =============================================================================
# Integration Tests (Section B from perf.md)
# =============================================================================


class TestTTFAUnderLoad:
    """Integration tests for TTFA under load (perf.md Section B.1)."""

    def test_ttfa_with_10_rapid_requests(self):
        """Simulate 10 rapid requests, assert TTFA < 2.0s."""
        from mumble_voice_bot.perf import (
            AudioQueueItem,
            BoundedTTSQueue,
            DropPolicy,
            TTSQueueItem,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        text_queue = BoundedTTSQueue(
            maxsize=20,
            policy=DropPolicy.DROP_STALE,
            turn_coordinator=coord,
        )
        audio_queue = BoundedTTSQueue(
            maxsize=20,
            policy=DropPolicy.DROP_STALE,
            turn_coordinator=coord,
        )

        ttfa_measurements = []

        def synthesize_and_queue(item: TTSQueueItem) -> None:
            """Simulate synthesis and add to audio queue."""
            time.sleep(0.02)  # 20ms synthesis time
            audio_data = f"audio:{item.text}".encode()
            audio_queue.put(AudioQueueItem(
                item.user_id, item.turn_id, audio_data, 100.0, item.chunk_index
            ))

        # Simulate 10 rapid requests
        for request_num in range(10):
            start_time = time.time()
            turn_id = coord.new_turn("user1")

            # Add text chunks to queue
            for i in range(3):
                text_queue.put(TTSQueueItem(
                    "user1", turn_id, f"Request {request_num} chunk {i}.", i
                ))

            # Process first chunk (simulate synthesis worker)
            item = text_queue.get(timeout=0.5)
            if item:
                synthesize_and_queue(item)

            # Measure TTFA
            if not audio_queue.empty():
                ttfa = (time.time() - start_time) * 1000
                ttfa_measurements.append(ttfa)
                # Consume the audio item
                audio_queue.get_nowait()

            # Small gap between requests
            time.sleep(0.01)

        # Assert TTFA < 2000ms for all requests
        assert len(ttfa_measurements) > 0, "No TTFA measurements recorded"
        for i, ttfa in enumerate(ttfa_measurements):
            assert ttfa < 2000, f"Request {i} TTFA {ttfa:.0f}ms exceeds 2000ms"

    def test_ttfa_stable_under_sustained_load(self):
        """TTFA remains stable (no upward drift) under sustained load."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            DropPolicy,
            RollingLatencyTracker,
            TTSQueueItem,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(
            maxsize=10,
            policy=DropPolicy.DROP_OLDEST,
            turn_coordinator=coord,
        )
        tracker = RollingLatencyTracker(window_size=50)

        # Simulate sustained load
        for i in range(100):
            start = time.time()
            turn_id = coord.new_turn("user1")
            queue.put(TTSQueueItem("user1", turn_id, f"Message {i}", 0))

            # Simulate processing
            item = queue.get(timeout=0.1)
            if item:
                latency = (time.time() - start) * 1000
                tracker.record("ttfa", latency)

        # Check that p95 is reasonable (not drifting up)
        stats = tracker.get_stats()
        assert "ttfa" in stats
        assert stats["ttfa"]["p95"] < 100, f"p95 TTFA {stats['ttfa']['p95']:.0f}ms too high"


class TestBacklogRecovery:
    """Integration tests for backlog recovery (perf.md Section B.3)."""

    def test_queue_bounded_with_slow_tts(self):
        """Queue doesn't grow beyond max even with slow TTS."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            DropPolicy,
            TTSQueueItem,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(
            maxsize=5,
            policy=DropPolicy.DROP_OLDEST,
            turn_coordinator=coord,
        )

        # Rapidly add items (faster than "TTS" can process)
        turn_id = coord.new_turn("user1")
        max_observed_size = 0

        for i in range(50):
            queue.put(TTSQueueItem("user1", turn_id, f"Message {i}", i))
            max_observed_size = max(max_observed_size, queue.qsize())

            # Simulate slow TTS occasionally consuming
            if i % 10 == 0:
                queue.get(timeout=0.01)

        # Queue never exceeded maxsize
        assert max_observed_size <= 5, f"Queue grew to {max_observed_size}"
        assert queue.drop_count > 0, "Expected drops due to bounded queue"

    def test_newer_turns_prioritized_in_backlog(self):
        """During backlog, newer turns are prioritized over old."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            DropPolicy,
            TTSQueueItem,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(
            maxsize=10,
            policy=DropPolicy.DROP_STALE,
            turn_coordinator=coord,
        )

        # Create backlog with turn 1
        turn1 = coord.new_turn("user1")
        for i in range(8):
            queue.put(TTSQueueItem("user1", turn1, f"Old message {i}", i))

        # User interrupts with turn 2
        turn2 = coord.new_turn("user1")
        for i in range(3):
            queue.put(TTSQueueItem("user1", turn2, f"New message {i}", i))

        # Process queue - should only get turn 2 items
        processed = []
        while not queue.empty():
            item = queue.get(timeout=0.1)
            if item:
                processed.append((item.turn_id, item.text))

        # All processed items should be from turn 2
        assert all(tid == turn2 for tid, _ in processed), "Old turn items were processed"
        assert len(processed) == 3, f"Expected 3 items, got {len(processed)}"

    def test_backlog_drains_with_adaptive_pacing(self):
        """Backlog drains faster with adaptive pacing."""
        from mumble_voice_bot.perf import (
            AdaptivePacer,
            BoundedTTSQueue,
            TTSQueueItem,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(maxsize=20, turn_coordinator=coord)
        pacer = AdaptivePacer(min_delay_ms=1, max_delay_ms=50, queue_threshold=5)

        # Create backlog
        turn_id = coord.new_turn("user1")
        for i in range(15):
            queue.put(TTSQueueItem("user1", turn_id, f"Message {i}", i))

        # Drain with adaptive pacing
        start_time = time.time()
        while not queue.empty():
            item = queue.get(timeout=0.1)
            if item:
                delay = pacer.get_delay(queue.qsize())
                time.sleep(delay / 1000.0)

        drain_time = (time.time() - start_time) * 1000

        # With adaptive pacing, should drain faster than max_delay * count
        # (50ms * 15 = 750ms max if no adaptation)
        # With adaptation, early items have ~1ms delay, so much faster
        assert drain_time < 500, f"Drain took {drain_time:.0f}ms, expected < 500ms"


class TestQueueStability:
    """Performance benchmark tests for queue stability (perf.md Section C.2)."""

    def test_queue_length_bounded_under_steady_load(self):
        """Queue length stays bounded under steady load."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            DropPolicy,
            TTSQueueItem,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(
            maxsize=10,
            policy=DropPolicy.DROP_OLDEST,
            turn_coordinator=coord,
        )

        queue_lengths = []
        turn_id = coord.new_turn("user1")

        # Steady load: produce slightly faster than consume
        for i in range(100):
            # Produce 2 items
            queue.put(TTSQueueItem("user1", turn_id, f"Msg {i}a", i * 2))
            queue.put(TTSQueueItem("user1", turn_id, f"Msg {i}b", i * 2 + 1))

            # Consume 1 item
            queue.get(timeout=0.001)

            queue_lengths.append(queue.qsize())

        # Queue should never exceed maxsize
        assert max(queue_lengths) <= 10
        # Drops should have occurred
        assert queue.drop_count > 0

    def test_drop_rate_tracking(self):
        """Track and report drop rate under load."""
        from mumble_voice_bot.perf import (
            BoundedTTSQueue,
            DropPolicy,
            LatencyReporter,
            RollingLatencyTracker,
            TTSQueueItem,
            TurnIdCoordinator,
        )

        coord = TurnIdCoordinator()
        queue = BoundedTTSQueue(
            maxsize=5,
            policy=DropPolicy.DROP_OLDEST,
            turn_coordinator=coord,
        )
        tracker = RollingLatencyTracker(window_size=100)
        reporter = LatencyReporter(tracker, queue=queue)

        # Generate load
        turn_id = coord.new_turn("user1")
        for i in range(50):
            queue.put(TTSQueueItem("user1", turn_id, f"Message {i}", i))
            tracker.record("processing", 10.0)

        # Check reporter includes drop count
        stats = reporter.format_stats()
        assert "drops" in stats.lower()
        assert queue.drop_count >= 45  # 50 items into queue of 5
