"""Tests for conversation state machine."""

import asyncio
import time

import pytest

from mumble_voice_bot.conversation_state import (
    ConversationState,
    ConversationStateMachine,
    ContinuousASRBuffer,
    StateTransition,
)


class TestConversationState:
    """Tests for ConversationState enum."""

    def test_all_states_exist(self):
        """Test that all expected states exist."""
        assert ConversationState.IDLE
        assert ConversationState.LISTENING
        assert ConversationState.THINKING
        assert ConversationState.SPEAKING
        assert ConversationState.INTERRUPTED
        assert ConversationState.COOLDOWN


class TestConversationStateMachine:
    """Tests for ConversationStateMachine class."""

    @pytest.fixture
    def sm(self):
        """Create a fresh state machine for each test."""
        return ConversationStateMachine()

    def test_initial_state_is_idle(self, sm):
        """Test that initial state is IDLE."""
        assert sm.state == ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_valid_transition(self, sm):
        """Test valid state transitions."""
        result = await sm.transition(ConversationState.LISTENING, "user started speaking")
        assert result is True
        assert sm.state == ConversationState.LISTENING

    @pytest.mark.asyncio
    async def test_invalid_transition_rejected(self, sm):
        """Test that invalid transitions are rejected."""
        # Can't go directly from IDLE to SPEAKING
        result = await sm.transition(ConversationState.SPEAKING, "invalid")
        assert result is False
        assert sm.state == ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, sm):
        """Test complete conversation flow."""
        # IDLE -> LISTENING -> THINKING -> SPEAKING -> COOLDOWN -> IDLE
        await sm.transition(ConversationState.LISTENING)
        assert sm.state == ConversationState.LISTENING

        await sm.transition(ConversationState.THINKING)
        assert sm.state == ConversationState.THINKING

        await sm.transition(ConversationState.SPEAKING)
        assert sm.state == ConversationState.SPEAKING

        await sm.transition(ConversationState.COOLDOWN)
        assert sm.state == ConversationState.COOLDOWN

        await sm.transition(ConversationState.IDLE)
        assert sm.state == ConversationState.IDLE

        assert len(sm.transitions) == 5

    @pytest.mark.asyncio
    async def test_interruption_flow(self, sm):
        """Test interruption flow."""
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.THINKING)
        await sm.transition(ConversationState.SPEAKING)
        await sm.transition(ConversationState.INTERRUPTED, "user interrupted")
        await sm.transition(ConversationState.LISTENING)
        assert sm.state == ConversationState.LISTENING
        assert sm.was_interrupted is False  # Last transition was to LISTENING

    def test_sync_transition(self, sm):
        """Test synchronous state transitions."""
        result = sm.transition_sync(ConversationState.LISTENING, "user started speaking")
        assert result is True
        assert sm.state == ConversationState.LISTENING

    def test_sync_invalid_transition(self, sm):
        """Test synchronous invalid transition rejection."""
        result = sm.transition_sync(ConversationState.SPEAKING, "invalid")
        assert result is False
        assert sm.state == ConversationState.IDLE

    def test_time_in_state(self, sm):
        """Test time_in_state tracking."""
        initial_time = sm.time_in_state()
        assert initial_time >= 0
        time.sleep(0.1)
        assert sm.time_in_state() > initial_time

    def test_can_respond_property(self, sm):
        """Test can_respond property."""
        assert sm.can_respond is False  # IDLE state
        sm.transition_sync(ConversationState.LISTENING)
        assert sm.can_respond is True  # LISTENING state
        sm.transition_sync(ConversationState.THINKING)
        assert sm.can_respond is False  # THINKING state

    def test_is_speaking_property(self, sm):
        """Test is_speaking property."""
        assert sm.is_speaking is False
        sm.transition_sync(ConversationState.LISTENING)
        sm.transition_sync(ConversationState.THINKING)
        sm.transition_sync(ConversationState.SPEAKING)
        assert sm.is_speaking is True

    def test_is_listening_property(self, sm):
        """Test is_listening property."""
        assert sm.is_listening is False
        sm.transition_sync(ConversationState.LISTENING)
        assert sm.is_listening is True

    def test_is_idle_property(self, sm):
        """Test is_idle property."""
        assert sm.is_idle is True
        sm.transition_sync(ConversationState.LISTENING)
        assert sm.is_idle is False

    def test_is_thinking_property(self, sm):
        """Test is_thinking property."""
        assert sm.is_thinking is False
        sm.transition_sync(ConversationState.LISTENING)
        sm.transition_sync(ConversationState.THINKING)
        assert sm.is_thinking is True

    def test_reset(self, sm):
        """Test reset method."""
        sm.transition_sync(ConversationState.LISTENING)
        sm.reset()
        assert sm.state == ConversationState.IDLE

    def test_get_state_summary(self, sm):
        """Test get_state_summary method."""
        summary = sm.get_state_summary()
        assert "state" in summary
        assert "time_in_state" in summary
        assert "transition_count" in summary
        assert "can_respond" in summary
        assert "is_speaking" in summary
        assert summary["state"] == "IDLE"

    def test_transition_history_limit(self, sm):
        """Test that transition history is limited."""
        # Make many transitions
        for _ in range(50):
            sm.transition_sync(ConversationState.LISTENING)
            sm.transition_sync(ConversationState.THINKING)
            sm.transition_sync(ConversationState.SPEAKING)
            sm.transition_sync(ConversationState.COOLDOWN)
            sm.transition_sync(ConversationState.IDLE)

        # History should be limited to max_transitions (100)
        assert len(sm.transitions) <= 100

    def test_on_state_change_callback(self):
        """Test state change callback."""
        changes = []

        def callback(old, new, reason):
            changes.append((old, new, reason))

        sm = ConversationStateMachine(on_state_change=callback)
        sm.transition_sync(ConversationState.LISTENING, "test reason")

        assert len(changes) == 1
        assert changes[0][0] == ConversationState.IDLE
        assert changes[0][1] == ConversationState.LISTENING
        assert changes[0][2] == "test reason"


class TestContinuousASRBuffer:
    """Tests for ContinuousASRBuffer class."""

    def test_init(self):
        """Test initialization."""
        buffer = ContinuousASRBuffer()
        assert buffer.buffer == []
        assert buffer.is_buffering is False

    def test_start_stop_buffering(self):
        """Test start and stop buffering."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        assert buffer.is_buffering is True

        results = buffer.stop_buffering()
        assert buffer.is_buffering is False
        assert results == []

    def test_add_transcript_while_buffering(self):
        """Test adding transcripts while buffering."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("hello", 0.9)
        buffer.add_transcript("world", 0.8)

        assert len(buffer.buffer) == 2

    def test_add_transcript_not_buffering(self):
        """Test that transcripts are ignored when not buffering."""
        buffer = ContinuousASRBuffer()
        buffer.add_transcript("hello", 0.9)
        assert len(buffer.buffer) == 0

    def test_stop_buffering_returns_transcripts(self):
        """Test that stop_buffering returns collected transcripts."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("hello", 0.9)
        buffer.add_transcript("world", 0.8)

        results = buffer.stop_buffering()
        assert len(results) == 2
        assert results[0] == ("hello", 0.9)
        assert results[1] == ("world", 0.8)

    def test_stop_buffering_clears_buffer(self):
        """Test that stop_buffering clears the buffer."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("hello", 0.9)
        buffer.stop_buffering()

        assert len(buffer.buffer) == 0

    def test_check_for_interruption_empty(self):
        """Test interruption check with empty buffer."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        assert buffer.check_for_interruption() is False

    def test_check_for_interruption_low_confidence(self):
        """Test interruption check with low confidence."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("hello world", 0.5)
        assert buffer.check_for_interruption(confidence_threshold=0.7) is False

    def test_check_for_interruption_success(self):
        """Test interruption check with sufficient speech."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("hello world", 0.9)
        assert buffer.check_for_interruption(confidence_threshold=0.7, min_words=2) is True

    def test_check_for_interruption_insufficient_words(self):
        """Test interruption check with insufficient words."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("hello", 0.9)
        assert buffer.check_for_interruption(min_words=2) is False

    def test_get_combined_transcript(self):
        """Test combined transcript retrieval."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("hello", 0.9)
        buffer.add_transcript("world", 0.8)

        combined = buffer.get_combined_transcript()
        assert combined == "hello world"

    def test_get_combined_transcript_empty(self):
        """Test combined transcript with empty buffer."""
        buffer = ContinuousASRBuffer()
        assert buffer.get_combined_transcript() == ""

    def test_clear(self):
        """Test clear method."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("hello", 0.9)
        buffer.clear()

        assert buffer.buffer == []
        assert buffer.is_buffering is False

    def test_empty_transcript_ignored(self):
        """Test that empty transcripts are ignored."""
        buffer = ContinuousASRBuffer()
        buffer.start_buffering()
        buffer.add_transcript("", 0.9)
        buffer.add_transcript("   ", 0.9)
        assert len(buffer.buffer) == 0
