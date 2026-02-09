"""Conversation state machine for turn handling.

This module provides a finite state machine for managing conversation states,
enabling proper turn-taking, interruption handling, and state transitions.

States:
- IDLE: Not in active conversation
- LISTENING: Actively listening to user speech
- THINKING: Processing/generating LLM response
- SPEAKING: TTS is playing
- INTERRUPTED: User interrupted bot speech
- COOLDOWN: Brief pause after speaking (prevents immediate re-triggering)
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from mumble_voice_bot.logging_config import get_logger

logger = get_logger(__name__)


class ConversationState(Enum):
    """States in the conversation state machine."""

    IDLE = auto()        # Not in active conversation
    LISTENING = auto()   # Actively listening to user speech
    THINKING = auto()    # Processing/generating LLM response
    SPEAKING = auto()    # TTS is playing
    INTERRUPTED = auto() # User interrupted bot speech
    COOLDOWN = auto()    # Brief pause after speaking


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: ConversationState
    to_state: ConversationState
    timestamp: float
    reason: str


class ConversationStateMachine:
    """Finite state machine for conversation turn management.

    Manages transitions between conversation states (listening, thinking,
    speaking, etc.) and enforces valid state transitions. This provides
    a foundation for proper turn-taking and interruption handling.

    Example usage:
        sm = ConversationStateMachine()

        # Normal conversation flow
        await sm.transition(ConversationState.LISTENING)  # User starts talking
        await sm.transition(ConversationState.THINKING)   # Process speech
        await sm.transition(ConversationState.SPEAKING)   # Start TTS
        await sm.transition(ConversationState.COOLDOWN)   # TTS finished
        await sm.transition(ConversationState.LISTENING)  # Ready for next turn

        # Interruption flow
        await sm.transition(ConversationState.INTERRUPTED)  # User interrupts
        await sm.transition(ConversationState.LISTENING)    # Listen to user
    """

    # Valid state transitions: from_state -> set of valid to_states
    VALID_TRANSITIONS: dict[ConversationState, set[ConversationState]] = {
        ConversationState.IDLE: {
            ConversationState.LISTENING,
        },
        ConversationState.LISTENING: {
            ConversationState.THINKING,
            ConversationState.IDLE,
        },
        ConversationState.THINKING: {
            ConversationState.SPEAKING,
            ConversationState.LISTENING,  # User spoke again, abort response
            ConversationState.IDLE,       # Error or timeout
        },
        ConversationState.SPEAKING: {
            ConversationState.COOLDOWN,
            ConversationState.INTERRUPTED,
        },
        ConversationState.INTERRUPTED: {
            ConversationState.LISTENING,
        },
        ConversationState.COOLDOWN: {
            ConversationState.LISTENING,
            ConversationState.IDLE,
        },
    }

    def __init__(
        self,
        cooldown_duration: float = 0.5,
        on_state_change: Callable[[ConversationState, ConversationState, str], None] | None = None,
    ):
        """Initialize the state machine.

        Args:
            cooldown_duration: How long to stay in COOLDOWN state (seconds).
            on_state_change: Optional callback for state changes.
                            Receives (old_state, new_state, reason).
        """
        self._state = ConversationState.IDLE
        self._state_entered_at: float = time.time()
        self._transitions: list[StateTransition] = []
        self._lock = asyncio.Lock()
        self._cooldown_duration = cooldown_duration
        self._on_state_change = on_state_change

        # Keep limited history
        self._max_transitions = 100

    @property
    def state(self) -> ConversationState:
        """Current conversation state."""
        return self._state

    @property
    def transitions(self) -> list[StateTransition]:
        """History of state transitions."""
        return self._transitions.copy()

    async def transition(
        self,
        new_state: ConversationState,
        reason: str = "",
    ) -> bool:
        """Attempt a state transition.

        Args:
            new_state: The state to transition to.
            reason: Optional reason for the transition (for logging).

        Returns:
            True if transition was successful, False if invalid.
        """
        async with self._lock:
            # Check if transition is valid
            valid_next_states = self.VALID_TRANSITIONS.get(self._state, set())
            if new_state not in valid_next_states:
                logger.debug(
                    f"Invalid state transition: {self._state.name} -> {new_state.name}",
                    extra={"reason": reason}
                )
                return False

            # Record transition
            old_state = self._state
            transition = StateTransition(
                from_state=old_state,
                to_state=new_state,
                timestamp=time.time(),
                reason=reason,
            )
            self._transitions.append(transition)

            # Trim history if needed
            if len(self._transitions) > self._max_transitions:
                self._transitions = self._transitions[-self._max_transitions:]

            # Update state
            self._state = new_state
            self._state_entered_at = time.time()

            logger.debug(
                f"State transition: {old_state.name} -> {new_state.name}",
                extra={"reason": reason}
            )

            # Call callback if registered
            if self._on_state_change:
                try:
                    self._on_state_change(old_state, new_state, reason)
                except Exception as e:
                    logger.warning(f"State change callback failed: {e}")

            return True

    def transition_sync(
        self,
        new_state: ConversationState,
        reason: str = "",
    ) -> bool:
        """Synchronous state transition (for non-async contexts).

        Args:
            new_state: The state to transition to.
            reason: Optional reason for the transition.

        Returns:
            True if transition was successful, False if invalid.
        """
        # Check if transition is valid
        valid_next_states = self.VALID_TRANSITIONS.get(self._state, set())
        if new_state not in valid_next_states:
            logger.debug(
                f"Invalid state transition: {self._state.name} -> {new_state.name}",
                extra={"reason": reason}
            )
            return False

        # Record transition
        old_state = self._state
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=time.time(),
            reason=reason,
        )
        self._transitions.append(transition)

        # Trim history if needed
        if len(self._transitions) > self._max_transitions:
            self._transitions = self._transitions[-self._max_transitions:]

        # Update state
        self._state = new_state
        self._state_entered_at = time.time()

        logger.debug(
            f"State transition: {old_state.name} -> {new_state.name}",
            extra={"reason": reason}
        )

        # Call callback if registered
        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state, reason)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

        return True

    def time_in_state(self) -> float:
        """Get time spent in current state (seconds)."""
        return time.time() - self._state_entered_at

    @property
    def can_respond(self) -> bool:
        """Check if bot is allowed to start a response.

        Returns:
            True if in a state where responding is appropriate.
        """
        return self._state == ConversationState.LISTENING

    @property
    def is_speaking(self) -> bool:
        """Check if bot is currently speaking."""
        return self._state == ConversationState.SPEAKING

    @property
    def is_listening(self) -> bool:
        """Check if bot is currently listening."""
        return self._state == ConversationState.LISTENING

    @property
    def is_idle(self) -> bool:
        """Check if bot is idle."""
        return self._state == ConversationState.IDLE

    @property
    def is_thinking(self) -> bool:
        """Check if bot is thinking/processing."""
        return self._state == ConversationState.THINKING

    @property
    def was_interrupted(self) -> bool:
        """Check if the last transition was an interruption."""
        if not self._transitions:
            return False
        last = self._transitions[-1]
        return last.to_state == ConversationState.INTERRUPTED

    def reset(self) -> None:
        """Reset state machine to IDLE."""
        self._state = ConversationState.IDLE
        self._state_entered_at = time.time()

    def get_state_summary(self) -> dict:
        """Get a summary of current state for debugging.

        Returns:
            Dict with state info.
        """
        return {
            "state": self._state.name,
            "time_in_state": self.time_in_state(),
            "transition_count": len(self._transitions),
            "can_respond": self.can_respond,
            "is_speaking": self.is_speaking,
        }


@dataclass
class ContinuousASRBuffer:
    """Buffer ASR results while bot is speaking.

    This allows continuous listening even while TTS is playing,
    enabling detection of interruptions and capturing speech that
    occurs during bot output.
    """

    buffer: list[tuple[str, float]] = field(default_factory=list)  # (transcript, confidence)
    is_buffering: bool = False
    _started_at: float = 0.0

    def start_buffering(self) -> None:
        """Start buffering ASR results (call when TTS starts)."""
        self.is_buffering = True
        self.buffer.clear()
        self._started_at = time.time()

    def add_transcript(self, text: str, confidence: float = 1.0) -> None:
        """Add ASR result to buffer.

        Args:
            text: Transcribed text.
            confidence: Confidence score (0-1).
        """
        if self.is_buffering and text and text.strip():
            self.buffer.append((text.strip(), confidence))

    def stop_buffering(self) -> list[tuple[str, float]]:
        """Stop buffering and return collected transcripts.

        Returns:
            List of (transcript, confidence) tuples.
        """
        self.is_buffering = False
        results = self.buffer.copy()
        self.buffer.clear()
        return results

    def check_for_interruption(
        self,
        confidence_threshold: float = 0.7,
        min_words: int = 2,
    ) -> bool:
        """Check if buffered speech indicates interruption.

        Args:
            confidence_threshold: Minimum confidence to consider.
            min_words: Minimum word count across all buffers.

        Returns:
            True if buffered content suggests user is interrupting.
        """
        if not self.buffer:
            return False

        # Check if we have high-confidence speech
        max_confidence = max(conf for _, conf in self.buffer)
        if max_confidence < confidence_threshold:
            return False

        # Check total word count
        total_words = sum(len(text.split()) for text, _ in self.buffer)
        return total_words >= min_words

    def get_combined_transcript(self) -> str:
        """Get all buffered transcripts combined.

        Returns:
            Combined transcript text.
        """
        return " ".join(text for text, _ in self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.is_buffering = False
