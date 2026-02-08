"""Tests for turn-taking and barge-in control.

Tests the TurnController from plan-human.md:
- Turn state machine transitions
- Barge-in detection while TTS active
- Interruption delay thresholds
- Callback registration and invocation
- Generation tracking for audio discard
- Interruption classification
- Interruption metrics
"""

import threading
import time
from unittest.mock import MagicMock

from mumble_voice_bot.turn_controller import (
    GenerationTracker,
    InterruptionClassifier,
    InterruptionHandler,
    InterruptionMetrics,
    InterruptionType,
    TurnController,
    TurnState,
)


class TestTurnControllerStates:
    """Tests for turn state machine transitions."""

    def test_initial_state_is_idle(self):
        """Controller starts in IDLE state."""
        controller = TurnController()
        assert controller.state == TurnState.IDLE
        assert controller.is_idle()

    def test_start_listening_transitions_from_idle(self):
        """IDLE -> LISTENING when user starts speaking."""
        controller = TurnController()
        controller.start_listening("user_1")

        assert controller.state == TurnState.LISTENING
        assert controller.is_listening()
        assert controller.current_user == "user_1"

    def test_start_processing_transitions_from_listening(self):
        """LISTENING -> PROCESSING when speech ends."""
        controller = TurnController()
        controller.start_listening("user_1")
        controller.start_processing()

        assert controller.state == TurnState.PROCESSING
        assert controller.is_processing()

    def test_start_speaking_transitions_from_processing(self):
        """PROCESSING -> SPEAKING when bot responds."""
        controller = TurnController()
        controller.start_listening("user_1")
        controller.start_processing()
        controller.start_speaking()

        assert controller.state == TurnState.SPEAKING
        assert controller.is_speaking()

    def test_reset_returns_to_idle(self):
        """Any state -> IDLE on reset."""
        controller = TurnController()
        controller.start_listening("user_1")
        controller.start_processing()
        controller.start_speaking()

        controller.reset()

        assert controller.state == TurnState.IDLE
        assert controller.is_idle()
        assert controller.current_user is None

    def test_full_conversation_cycle(self):
        """Test complete conversation turn cycle."""
        controller = TurnController()

        # User speaks
        controller.start_listening("alice")
        assert controller.state == TurnState.LISTENING

        # User stops, processing begins
        controller.start_processing()
        assert controller.state == TurnState.PROCESSING

        # Bot starts responding
        controller.start_speaking()
        assert controller.state == TurnState.SPEAKING

        # Bot finishes
        controller.reset()
        assert controller.state == TurnState.IDLE


class TestBargeInDetection:
    """Tests for barge-in detection (plan-human.md Phase A & B)."""

    def test_barge_in_only_during_speaking(self):
        """Barge-in only triggers when bot is SPEAKING."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0  # Disable delay for testing

        # Should fail in IDLE state
        assert controller.request_barge_in() is False

        # Should fail in LISTENING state
        controller.start_listening("user_1")
        assert controller.request_barge_in() is False

        # Should fail in PROCESSING state
        controller.start_processing()
        assert controller.request_barge_in() is False

        # Should succeed in SPEAKING state
        controller.start_speaking()
        assert controller.request_barge_in() is True

    def test_barge_in_transitions_to_listening(self):
        """Barge-in sets state to LISTENING (SPEAKING -> LISTENING)."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        controller.start_speaking()
        controller.request_barge_in()

        assert controller.state == TurnState.LISTENING
        assert controller.is_listening()

    def test_barge_in_sets_cancel_flag(self):
        """Barge-in sets the cancellation flag for TTS."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        controller.start_speaking()
        assert controller.is_cancelled() is False

        controller.request_barge_in()
        assert controller.is_cancelled() is True

    def test_cancel_flag_clears_on_new_turn(self):
        """Cancel flag is cleared when starting new turn."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        # Trigger barge-in
        controller.start_speaking()
        controller.request_barge_in()
        assert controller.is_cancelled() is True

        # Start new listening turn - should clear flag
        controller.start_listening("user_1")
        assert controller.is_cancelled() is False

    def test_cancel_flag_clears_on_reset(self):
        """Cancel flag is cleared on reset."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        controller.start_speaking()
        controller.request_barge_in()
        assert controller.is_cancelled() is True

        controller.reset()
        assert controller.is_cancelled() is False

    def test_barge_in_increments_count(self):
        """Barge-in count is tracked for metrics."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        assert controller.barge_in_count == 0

        controller.start_speaking()
        controller.request_barge_in()
        assert controller.barge_in_count == 1

        # Start new speaking turn and barge-in again
        controller.start_speaking()
        controller.request_barge_in()
        assert controller.barge_in_count == 2


class TestBargeInDelay:
    """Tests for barge-in delay threshold (plan-human.md Section 3.A1)."""

    def test_barge_in_blocked_during_delay(self):
        """Barge-in is blocked for barge_in_delay_ms after speaking starts."""
        controller = TurnController()
        controller.barge_in_delay_ms = 200  # 200ms delay

        controller.start_speaking()

        # Immediate barge-in should fail (too quick)
        assert controller.request_barge_in() is False
        assert controller.state == TurnState.SPEAKING

    def test_barge_in_allowed_after_delay(self):
        """Barge-in succeeds after delay threshold passes."""
        controller = TurnController()
        controller.barge_in_delay_ms = 50  # 50ms delay for fast test

        controller.start_speaking()

        # Wait for delay to pass
        time.sleep(0.06)

        assert controller.request_barge_in() is True
        assert controller.state == TurnState.LISTENING

    def test_zero_delay_allows_immediate_barge_in(self):
        """Zero delay allows immediate barge-in."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        controller.start_speaking()
        assert controller.request_barge_in() is True


class TestBargeInCallback:
    """Tests for barge-in callback (plan-human.md Phase B.1)."""

    def test_callback_invoked_on_barge_in(self):
        """Registered callback is called on barge-in."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        callback = MagicMock()
        controller.on_barge_in(callback)

        controller.start_speaking()
        controller.request_barge_in()

        callback.assert_called_once()

    def test_callback_not_invoked_on_failed_barge_in(self):
        """Callback is not called when barge-in fails."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        callback = MagicMock()
        controller.on_barge_in(callback)

        # Barge-in while not speaking should fail
        controller.request_barge_in()

        callback.assert_not_called()

    def test_callback_exception_is_handled(self):
        """Callback exceptions don't crash the controller."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        def bad_callback():
            raise RuntimeError("Callback failed")

        controller.on_barge_in(bad_callback)
        controller.start_speaking()

        # Should not raise
        result = controller.request_barge_in()
        assert result is True
        assert controller.state == TurnState.LISTENING

    def test_no_callback_registered(self):
        """Barge-in works without a registered callback."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        controller.start_speaking()
        result = controller.request_barge_in()

        assert result is True
        assert controller.state == TurnState.LISTENING


class TestThreadSafety:
    """Tests for thread safety of TurnController."""

    def test_concurrent_state_changes(self):
        """Multiple threads can safely change state."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0
        errors = []

        def worker(operation: str):
            try:
                for _ in range(100):
                    if operation == "listen":
                        controller.start_listening("user")
                    elif operation == "process":
                        controller.start_processing()
                    elif operation == "speak":
                        controller.start_speaking()
                    elif operation == "reset":
                        controller.reset()
                    elif operation == "barge":
                        controller.request_barge_in()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("listen",)),
            threading.Thread(target=worker, args=("process",)),
            threading.Thread(target=worker, args=("speak",)),
            threading.Thread(target=worker, args=("reset",)),
            threading.Thread(target=worker, args=("barge",)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Controller should be in some valid state
        assert controller.state in TurnState

    def test_cancel_flag_visibility(self):
        """Cancel flag is visible across threads."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0
        cancel_seen = threading.Event()

        def checker():
            for _ in range(1000):
                if controller.is_cancelled():
                    cancel_seen.set()
                    return
                time.sleep(0.001)

        controller.start_speaking()

        checker_thread = threading.Thread(target=checker)
        checker_thread.start()

        # Give checker time to start
        time.sleep(0.01)

        # Trigger barge-in
        controller.request_barge_in()

        checker_thread.join(timeout=2.0)

        assert cancel_seen.is_set(), "Cancel flag not seen by other thread"


class TestTTSIntegrationScenarios:
    """Tests simulating real TTS integration scenarios."""

    def test_tts_generation_loop_checks_cancellation(self):
        """Simulate TTS loop that checks for cancellation."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        controller.start_speaking()

        chunks_generated = 0
        for i in range(10):
            if controller.is_cancelled():
                break
            # Simulate chunk generation
            chunks_generated += 1
            time.sleep(0.01)

            # Simulate user interruption at chunk 5
            if i == 4:
                controller.request_barge_in()

        # Should have stopped after 5 chunks
        assert chunks_generated == 5

    def test_multiple_barge_in_requests(self):
        """Multiple barge-in requests during same turn only count once."""
        controller = TurnController()
        controller.barge_in_delay_ms = 0

        controller.start_speaking()
        initial_count = controller.barge_in_count

        # First barge-in succeeds
        assert controller.request_barge_in() is True
        assert controller.barge_in_count == initial_count + 1

        # Subsequent requests fail (no longer SPEAKING)
        assert controller.request_barge_in() is False
        assert controller.barge_in_count == initial_count + 1

    def test_quick_user_response_scenario(self):
        """Test scenario: user responds quickly after bot speaks."""
        controller = TurnController()
        controller.barge_in_delay_ms = 100  # 100ms minimum

        # Bot speaks
        controller.start_speaking()

        # User tries to interrupt immediately (should fail)
        assert controller.request_barge_in() is False

        # Wait for delay
        time.sleep(0.15)

        # User interrupts successfully
        assert controller.request_barge_in() is True

        # Bot detects cancellation
        assert controller.is_cancelled() is True

        # New user turn begins
        controller.start_listening("user")
        assert controller.is_cancelled() is False


class TestGenerationTracker:
    """Tests for generation ID tracking (pipecat-inspired audio discard)."""

    def test_initial_state(self):
        """Tracker starts at generation 0, confirmed."""
        tracker = GenerationTracker()
        assert tracker.generation == 0
        assert tracker.confirmed_generation == 0
        assert tracker.is_valid() is True

    def test_increment_invalidates_audio(self):
        """Incrementing generation invalidates pending audio."""
        tracker = GenerationTracker()
        tracker.confirm()  # Start valid

        assert tracker.is_valid() is True

        tracker.increment()  # Interrupt

        assert tracker.is_valid() is False
        assert tracker.generation == 1
        assert tracker.confirmed_generation == 0

    def test_confirm_validates_new_generation(self):
        """Confirming generation validates new audio."""
        tracker = GenerationTracker()
        tracker.increment()  # Interrupt
        assert tracker.is_valid() is False

        tracker.confirm()  # Start new generation
        assert tracker.is_valid() is True
        assert tracker.generation == 1
        assert tracker.confirmed_generation == 1

    def test_multiple_increments(self):
        """Multiple interruptions increment correctly."""
        tracker = GenerationTracker()

        for i in range(5):
            tracker.increment()
            assert tracker.generation == i + 1
            assert tracker.is_valid() is False

            tracker.confirm()
            assert tracker.is_valid() is True

    def test_reset(self):
        """Reset clears generation tracking."""
        tracker = GenerationTracker()
        tracker.increment()
        tracker.increment()

        tracker.reset()

        assert tracker.generation == 0
        assert tracker.confirmed_generation == 0
        assert tracker.is_valid() is True

    def test_thread_safety(self):
        """Generation tracker is thread-safe."""
        tracker = GenerationTracker()
        errors = []

        def incrementer():
            try:
                for _ in range(100):
                    tracker.increment()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def confirmer():
            try:
                for _ in range(100):
                    tracker.confirm()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def checker():
            try:
                for _ in range(100):
                    tracker.is_valid()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=incrementer),
            threading.Thread(target=confirmer),
            threading.Thread(target=checker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestInterruptionClassifier:
    """Tests for interruption classification."""

    def test_very_short_is_no_interruption(self):
        """Speech under min threshold is not an interruption."""
        classifier = InterruptionClassifier(min_speech_ms=100)
        result = classifier.classify(50)
        assert result == InterruptionType.NONE

    def test_short_speech_is_interjection(self):
        """Speech under short threshold is interjection."""
        classifier = InterruptionClassifier(short_threshold_ms=500, min_speech_ms=100)
        result = classifier.classify(300)
        assert result == InterruptionType.SHORT_INTERJECTION

    def test_long_speech_is_full_interruption(self):
        """Speech over short threshold is full interruption."""
        classifier = InterruptionClassifier(short_threshold_ms=500, min_speech_ms=100)
        result = classifier.classify(700)
        assert result == InterruptionType.FULL_INTERRUPTION

    def test_known_interjection_phrases(self):
        """Known interjection phrases are classified as short."""
        classifier = InterruptionClassifier()
        phrases = ["yeah", "yep", "uh-huh", "ok", "sure", "yes", "no", "wait"]

        for phrase in phrases:
            # Even long duration, known interjections are SHORT
            result = classifier.classify(800, transcript=phrase)
            assert result == InterruptionType.SHORT_INTERJECTION, f"'{phrase}' should be interjection"

    def test_interjection_with_punctuation(self):
        """Interjections with punctuation are recognized."""
        classifier = InterruptionClassifier()

        assert classifier.classify(800, "Yeah!") == InterruptionType.SHORT_INTERJECTION
        assert classifier.classify(800, "Okay.") == InterruptionType.SHORT_INTERJECTION

    def test_real_speech_uses_duration(self):
        """Non-interjection uses duration-based classification."""
        classifier = InterruptionClassifier(short_threshold_ms=500)

        result = classifier.classify(800, "Actually I have a question")
        assert result == InterruptionType.FULL_INTERRUPTION


class TestInterruptionMetrics:
    """Tests for interruption metrics tracking."""

    def test_record_interruption(self):
        """Record and retrieve interruption latency."""
        metrics = InterruptionMetrics()

        metrics.record_interruption(1000, 1100, InterruptionType.SHORT_INTERJECTION)
        metrics.record_interruption(2000, 2150, InterruptionType.FULL_INTERRUPTION)

        assert metrics.average_interruption_latency() == 125.0
        stats = metrics.get_stats()
        assert stats["total_interruptions"] == 2
        assert stats["short_interjections"] == 1
        assert stats["full_interruptions"] == 1

    def test_record_response_latency(self):
        """Record and retrieve response latency."""
        metrics = InterruptionMetrics()

        metrics.record_response_latency(1000, 1400)  # 400ms
        metrics.record_response_latency(2000, 2600)  # 600ms

        assert metrics.average_response_latency() == 500.0

    def test_interruption_target(self):
        """Test interruption latency target check."""
        metrics = InterruptionMetrics()

        # Good latency (100ms < 150ms target)
        metrics.record_interruption(0, 100, InterruptionType.SHORT_INTERJECTION)
        assert metrics.interruption_target_met() is True

        # Add bad latency (average now 200ms)
        metrics.record_interruption(0, 300, InterruptionType.FULL_INTERRUPTION)
        assert metrics.interruption_target_met() is False

    def test_response_target(self):
        """Test response latency target check."""
        metrics = InterruptionMetrics()

        # Good latency (400ms < 500ms target)
        metrics.record_response_latency(0, 400)
        assert metrics.response_target_met() is True

        # Add bad latency (average now 600ms)
        metrics.record_response_latency(0, 800)
        assert metrics.response_target_met() is False

    def test_reset(self):
        """Reset clears all metrics."""
        metrics = InterruptionMetrics()
        metrics.record_interruption(0, 100, InterruptionType.SHORT_INTERJECTION)
        metrics.record_response_latency(0, 400)

        metrics.reset()

        stats = metrics.get_stats()
        assert stats["total_interruptions"] == 0
        assert stats["avg_interruption_latency_ms"] == 0.0


class TestInterruptionHandler:
    """Tests for LLM cancel/resume strategy (plan-human.md Section 3.C2)."""

    def test_initial_state(self):
        """Handler starts with no interruption."""
        handler = InterruptionHandler()
        assert handler.should_resume() is False
        assert handler.should_restart() is False

    def test_short_interjection_triggers_resume(self):
        """Short interjection with partial response enables resume."""
        handler = InterruptionHandler()
        handler.on_interruption(
            InterruptionType.SHORT_INTERJECTION,
            partial_response="I was saying that"
        )

        assert handler.should_resume() is True
        assert handler.should_restart() is False

    def test_full_interruption_triggers_restart(self):
        """Full interruption triggers restart, not resume."""
        handler = InterruptionHandler()
        handler.on_interruption(InterruptionType.FULL_INTERRUPTION)

        assert handler.should_resume() is False
        assert handler.should_restart() is True

    def test_get_resumed_response(self):
        """Get and clear resumed response."""
        handler = InterruptionHandler()
        handler.on_interruption(
            InterruptionType.SHORT_INTERJECTION,
            partial_response="The answer is"
        )

        response = handler.get_resumed_response()
        assert response == "The answer is"

        # Second call returns None (consumed)
        assert handler.get_resumed_response() is None

    def test_acknowledgment_after_full_interruption(self):
        """Acknowledgment phrase added after full interruption."""
        handler = InterruptionHandler()
        handler.on_interruption(InterruptionType.FULL_INTERRUPTION)

        assert handler.should_add_acknowledgment() is True
        ack = handler.get_acknowledgment()
        assert ack in handler.ACKNOWLEDGMENT_PHRASES

    def test_no_acknowledgment_after_interjection(self):
        """No acknowledgment for short interjection."""
        handler = InterruptionHandler()
        handler.on_interruption(InterruptionType.SHORT_INTERJECTION)

        assert handler.should_add_acknowledgment() is False

    def test_shorter_tokens_after_interruption(self):
        """Max tokens reduced after interruption."""
        handler = InterruptionHandler()

        # Normal tokens before interruption
        assert handler.get_max_tokens() == handler.normal_max_tokens

        # Shorter after interruption
        handler.on_interruption(InterruptionType.FULL_INTERRUPTION)
        assert handler.get_max_tokens() == handler.post_interrupt_max_tokens

    def test_response_style_hint(self):
        """Response style hint added after interruption."""
        handler = InterruptionHandler()

        # No hint before interruption
        assert handler.get_response_style_hint() == ""

        # Hint after interruption
        handler.on_interruption(InterruptionType.FULL_INTERRUPTION)
        hint = handler.get_response_style_hint()
        assert "brief" in hint.lower()
        assert "interrupted" in hint.lower()

    def test_reset_turn(self):
        """Reset clears all interruption state."""
        handler = InterruptionHandler()
        handler.on_interruption(
            InterruptionType.FULL_INTERRUPTION,
            partial_response="Something"
        )

        handler.reset_turn()

        assert handler.should_resume() is False
        assert handler.should_restart() is False
        assert handler.get_max_tokens() == handler.normal_max_tokens

    def test_disable_resume(self):
        """Resume can be disabled via configuration."""
        handler = InterruptionHandler(resume_on_interjection=False)
        handler.on_interruption(
            InterruptionType.SHORT_INTERJECTION,
            partial_response="Something"
        )

        # Even with interjection and partial response, resume is disabled
        assert handler.should_resume() is False

    def test_multiple_interruptions_in_turn(self):
        """Track multiple interruptions within a turn."""
        handler = InterruptionHandler()

        handler.on_interruption(InterruptionType.SHORT_INTERJECTION)
        handler.on_interruption(InterruptionType.FULL_INTERRUPTION)

        # Should use short tokens (interrupted multiple times)
        assert handler.get_max_tokens() == handler.post_interrupt_max_tokens

        # Last interruption type determines restart
        assert handler.should_restart() is True
