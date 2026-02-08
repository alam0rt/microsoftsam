"""Turn-taking and barge-in control with interruption handling.

Implements plan-human.md requirements:
- Barge-in detection while TTS active
- Interruption classification (short interjection vs full interruption)
- Generation versioning for audio discard (inspired by pipecat magpie_websocket_tts.py)
- Interruption metrics tracking
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class TurnState(Enum):
    """States for turn-taking state machine."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class InterruptionType(Enum):
    """Types of interruption detected (plan-human.md Section 2.1)."""
    NONE = "none"
    SHORT_INTERJECTION = "short_interjection"  # <= 500ms, e.g., "yeah", "uh-huh"
    FULL_INTERRUPTION = "full_interruption"    # > 500ms sustained speech


@dataclass
class TurnController:
    """Manages turn-taking with barge-in support.

    This controller tracks the state of the conversation turn and enables
    barge-in detection, allowing users to interrupt the bot while it's speaking.

    State transitions:
        IDLE -> LISTENING: User starts speaking
        LISTENING -> PROCESSING: User stops speaking, processing begins
        PROCESSING -> SPEAKING: Bot starts speaking
        SPEAKING -> LISTENING: User interrupts (barge-in)
        SPEAKING -> IDLE: Bot finishes speaking
        Any -> IDLE: reset()

    Usage:
        controller = TurnController()

        # When user starts speaking
        controller.start_listening(user_id)

        # When processing user's speech
        controller.start_processing()

        # When bot starts speaking
        controller.start_speaking()

        # Check if user is interrupting
        if controller.state == TurnState.SPEAKING and user_is_speaking:
            if controller.request_barge_in():
                # Stop TTS, clear audio buffer
                pass

        # During TTS generation, check if cancelled
        for chunk in tts_stream:
            if controller.is_cancelled():
                break
            play(chunk)
    """

    state: TurnState = TurnState.IDLE
    _cancel_event: threading.Event = field(default_factory=threading.Event)
    _current_user: Optional[str] = None
    _state_lock: threading.Lock = field(default_factory=threading.Lock)
    _barge_in_callback: Optional[Callable[[], None]] = None
    _barge_in_count: int = 0
    _last_state_change: float = field(default_factory=time.time)

    # Configuration
    barge_in_delay_ms: int = 200  # Minimum ms before barge-in is allowed

    def start_listening(self, user_id: str):
        """User started speaking.

        Args:
            user_id: Identifier for the speaking user.
        """
        with self._state_lock:
            self._current_user = user_id
            self.state = TurnState.LISTENING
            self._cancel_event.clear()
            self._last_state_change = time.time()

    def start_processing(self):
        """Processing user's speech."""
        with self._state_lock:
            self.state = TurnState.PROCESSING
            self._last_state_change = time.time()

    def start_speaking(self):
        """Bot is now speaking."""
        with self._state_lock:
            self.state = TurnState.SPEAKING
            self._cancel_event.clear()
            self._last_state_change = time.time()

    def request_barge_in(self) -> bool:
        """Called when user speaks while bot is outputting.

        Returns:
            True if barge-in was triggered, False otherwise.
        """
        with self._state_lock:
            if self.state != TurnState.SPEAKING:
                return False

            # Prevent very quick barge-ins (accidental)
            elapsed_ms = (time.time() - self._last_state_change) * 1000
            if elapsed_ms < self.barge_in_delay_ms:
                return False

            self._cancel_event.set()
            self.state = TurnState.LISTENING
            self._barge_in_count += 1
            self._last_state_change = time.time()

            # Call barge-in callback if registered
            if self._barge_in_callback:
                try:
                    self._barge_in_callback()
                except Exception:
                    pass

            return True

    def is_cancelled(self) -> bool:
        """Check if current generation should be cancelled.

        Returns:
            True if cancelled (barge-in occurred), False otherwise.
        """
        return self._cancel_event.is_set()

    def reset(self):
        """Reset to idle state."""
        with self._state_lock:
            self._cancel_event.clear()
            self._current_user = None
            self.state = TurnState.IDLE
            self._last_state_change = time.time()

    def on_barge_in(self, callback: Callable[[], None]):
        """Register a callback for barge-in events.

        The callback is called when a barge-in is successfully triggered.
        Use this to clear audio buffers, stop TTS, etc.

        Args:
            callback: Function to call on barge-in.
        """
        self._barge_in_callback = callback

    @property
    def current_user(self) -> Optional[str]:
        """Get the current speaking user."""
        return self._current_user

    @property
    def barge_in_count(self) -> int:
        """Get total barge-in count (for stats)."""
        return self._barge_in_count

    def is_speaking(self) -> bool:
        """Check if bot is currently speaking."""
        return self.state == TurnState.SPEAKING

    def is_listening(self) -> bool:
        """Check if currently listening to a user."""
        return self.state == TurnState.LISTENING

    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self.state == TurnState.PROCESSING

    def is_idle(self) -> bool:
        """Check if idle (not in a conversation turn)."""
        return self.state == TurnState.IDLE


@dataclass
class GenerationTracker:
    """Track generation IDs to discard stale audio after interruption.

    Inspired by pipecat's magpie_websocket_tts.py:
    - _gen is incremented on interruption to invalidate old audio
    - _confirmed_gen is set to _gen when new generation starts
    - Audio is only accepted when confirmed_gen == gen

    Usage:
        tracker = GenerationTracker()

        # On interruption
        tracker.increment()  # Invalidates all pending audio

        # When starting new TTS generation
        tracker.confirm()  # Audio after this is valid

        # When receiving audio chunk
        if tracker.is_valid():
            play(chunk)
        else:
            discard(chunk)  # Stale audio from before interruption
    """

    _gen: int = 0
    _confirmed_gen: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self):
        """Increment generation ID on interruption. Invalidates pending audio."""
        with self._lock:
            self._gen += 1

    def confirm(self):
        """Confirm current generation. Audio after this point is valid."""
        with self._lock:
            self._confirmed_gen = self._gen

    def is_valid(self) -> bool:
        """Check if audio from current generation is valid.

        Returns:
            True if audio should be accepted, False if stale (discard).
        """
        with self._lock:
            return self._confirmed_gen == self._gen

    @property
    def generation(self) -> int:
        """Get current generation ID."""
        return self._gen

    @property
    def confirmed_generation(self) -> int:
        """Get confirmed generation ID."""
        return self._confirmed_gen

    def reset(self):
        """Reset generation tracking."""
        with self._lock:
            self._gen = 0
            self._confirmed_gen = 0


@dataclass
class InterruptionClassifier:
    """Classify interruptions based on duration and content.

    Implements plan-human.md Section 3.A2:
    - Short interjection: speech <= short_threshold_ms
    - Full interruption: speech > short_threshold_ms

    Also checks for known interjection phrases like "yeah", "uh-huh", etc.
    """

    short_threshold_ms: int = 500  # Max duration for short interjection
    min_speech_ms: int = 100       # Minimum speech to consider as interruption

    # Known interjection phrases
    INTERJECTIONS = frozenset({
        "yeah", "yep", "uh-huh", "mm-hmm", "ok", "okay",
        "right", "sure", "got it", "i see", "hmm", "mhm",
        "uh huh", "mm hmm", "yes", "no", "wait", "hold on"
    })

    def classify(
        self,
        speech_duration_ms: float,
        transcript: Optional[str] = None
    ) -> InterruptionType:
        """Classify an interruption based on duration and optional transcript.

        Args:
            speech_duration_ms: Duration of user speech in milliseconds.
            transcript: Optional transcript text for content-based classification.

        Returns:
            InterruptionType indicating the type of interruption.
        """
        if speech_duration_ms < self.min_speech_ms:
            return InterruptionType.NONE

        # Check for known interjection phrases if transcript available
        if transcript:
            cleaned = transcript.lower().strip().rstrip(".,!?")
            if cleaned in self.INTERJECTIONS:
                return InterruptionType.SHORT_INTERJECTION

        # Duration-based classification
        if speech_duration_ms <= self.short_threshold_ms:
            return InterruptionType.SHORT_INTERJECTION

        return InterruptionType.FULL_INTERRUPTION


@dataclass
class InterruptionMetrics:
    """Track interruption latency metrics.

    From plan-human.md Section 3.E1:
    - Time from user speech start -> TTS stop (target < 150ms)
    - Time from user finish -> bot response start (target < 500ms)
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _interruption_latencies: list = field(default_factory=list)
    _response_latencies: list = field(default_factory=list)
    _interruption_count: int = 0
    _short_interjection_count: int = 0
    _full_interruption_count: int = 0

    def record_interruption(
        self,
        speech_start_ms: float,
        tts_stop_ms: float,
        interruption_type: InterruptionType
    ):
        """Record an interruption event with latency.

        Args:
            speech_start_ms: Timestamp when user started speaking.
            tts_stop_ms: Timestamp when TTS was stopped.
            interruption_type: Type of interruption detected.
        """
        latency = tts_stop_ms - speech_start_ms
        with self._lock:
            self._interruption_latencies.append(latency)
            self._interruption_count += 1
            if interruption_type == InterruptionType.SHORT_INTERJECTION:
                self._short_interjection_count += 1
            elif interruption_type == InterruptionType.FULL_INTERRUPTION:
                self._full_interruption_count += 1

    def record_response_latency(self, user_finish_ms: float, bot_response_ms: float):
        """Record latency from user finish to bot response.

        Args:
            user_finish_ms: Timestamp when user stopped speaking.
            bot_response_ms: Timestamp when bot started responding.
        """
        latency = bot_response_ms - user_finish_ms
        with self._lock:
            self._response_latencies.append(latency)

    def average_interruption_latency(self) -> float:
        """Get average interruption latency in ms."""
        with self._lock:
            return self._avg_interruption_latency_unlocked()

    def _avg_interruption_latency_unlocked(self) -> float:
        """Get average interruption latency (caller must hold lock)."""
        if not self._interruption_latencies:
            return 0.0
        return sum(self._interruption_latencies) / len(self._interruption_latencies)

    def average_response_latency(self) -> float:
        """Get average response latency in ms."""
        with self._lock:
            return self._avg_response_latency_unlocked()

    def _avg_response_latency_unlocked(self) -> float:
        """Get average response latency (caller must hold lock)."""
        if not self._response_latencies:
            return 0.0
        return sum(self._response_latencies) / len(self._response_latencies)

    def interruption_target_met(self, target_ms: float = 150.0) -> bool:
        """Check if interruption latency target is met (< 150ms)."""
        return self.average_interruption_latency() < target_ms

    def response_target_met(self, target_ms: float = 500.0) -> bool:
        """Check if response latency target is met (< 500ms)."""
        return self.average_response_latency() < target_ms

    def get_stats(self) -> dict:
        """Get all metrics as a dictionary."""
        with self._lock:
            avg_int = self._avg_interruption_latency_unlocked()
            avg_resp = self._avg_response_latency_unlocked()
            return {
                "total_interruptions": self._interruption_count,
                "short_interjections": self._short_interjection_count,
                "full_interruptions": self._full_interruption_count,
                "avg_interruption_latency_ms": avg_int,
                "avg_response_latency_ms": avg_resp,
                "interruption_target_met": avg_int < 150.0,
                "response_target_met": avg_resp < 500.0,
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._interruption_latencies.clear()
            self._response_latencies.clear()
            self._interruption_count = 0
            self._short_interjection_count = 0
            self._full_interruption_count = 0


@dataclass
class InterruptionHandler:
    """Handle interruptions with resume/restart strategy.

    Implements plan-human.md Section 3.C2:
    - Short interjection: RESUME previous response
    - Full interruption: RESTART with new context

    Also implements plan-human.md Section 3.D (Human-Like Response Mode):
    - After interruption, switch to shorter, faster responses
    - Add acknowledgment phrases ("Sure —", "Okay —")
    """

    # Resume strategy configuration
    resume_on_interjection: bool = True

    # Response style after interruption (plan-human.md Section 3.D)
    post_interrupt_max_tokens: int = 100  # Shorter responses after interrupt
    normal_max_tokens: int = 300

    # Acknowledgment phrases to add after interruption
    ACKNOWLEDGMENT_PHRASES = (
        "Sure — ",
        "Okay — ",
        "Right — ",
        "Got it — ",
        "Alright — ",
    )

    # State tracking
    _last_interruption_type: InterruptionType = InterruptionType.NONE
    _interrupted_response: Optional[str] = None
    _interrupt_count_this_turn: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def on_interruption(
        self,
        interruption_type: InterruptionType,
        partial_response: Optional[str] = None
    ):
        """Record an interruption event.

        Args:
            interruption_type: Type of interruption detected.
            partial_response: The partial LLM response that was interrupted (for resume).
        """
        with self._lock:
            self._last_interruption_type = interruption_type
            self._interrupt_count_this_turn += 1
            if partial_response:
                self._interrupted_response = partial_response

    def should_resume(self) -> bool:
        """Determine if bot should resume previous response.

        Returns:
            True if previous response should be resumed (short interjection).
        """
        with self._lock:
            return (
                self.resume_on_interjection and
                self._last_interruption_type == InterruptionType.SHORT_INTERJECTION and
                self._interrupted_response is not None
            )

    def should_restart(self) -> bool:
        """Determine if bot should restart with new context.

        Returns:
            True if new response should be generated (full interruption).
        """
        with self._lock:
            return self._last_interruption_type == InterruptionType.FULL_INTERRUPTION

    def get_resumed_response(self) -> Optional[str]:
        """Get the interrupted response to resume.

        Returns:
            The partial response to resume, or None if not available.
        """
        with self._lock:
            response = self._interrupted_response
            self._interrupted_response = None
            return response

    def get_acknowledgment(self) -> str:
        """Get an acknowledgment phrase for post-interrupt response.

        Returns:
            A random acknowledgment phrase.
        """
        import random
        return random.choice(self.ACKNOWLEDGMENT_PHRASES)

    def get_max_tokens(self) -> int:
        """Get max tokens based on interruption state.

        After interruption, use shorter responses for faster recovery.

        Returns:
            Max tokens to use for LLM generation.
        """
        with self._lock:
            if self._interrupt_count_this_turn > 0:
                return self.post_interrupt_max_tokens
            return self.normal_max_tokens

    def should_add_acknowledgment(self) -> bool:
        """Check if acknowledgment phrase should be added.

        Returns:
            True if response should start with acknowledgment.
        """
        with self._lock:
            # Add acknowledgment after full interruption
            return self._last_interruption_type == InterruptionType.FULL_INTERRUPTION

    def reset_turn(self):
        """Reset state for new conversation turn."""
        with self._lock:
            self._last_interruption_type = InterruptionType.NONE
            self._interrupted_response = None
            self._interrupt_count_this_turn = 0

    def get_response_style_hint(self) -> str:
        """Get a hint to append to system prompt for response style.

        After interruption, instruct LLM to be more concise.

        Returns:
            Style hint string to append to prompt, or empty string.
        """
        with self._lock:
            if self._interrupt_count_this_turn > 0:
                return (
                    " The user just interrupted, so keep your response very brief "
                    "and to the point. One or two sentences maximum."
                )
            return ""
