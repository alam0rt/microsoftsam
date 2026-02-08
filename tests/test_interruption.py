"""Tests for interruption detection and classification.

Tests requirements from plan-human.md Section 2.1:
- Detect user speaking while bot is speaking (barge-in)
- Detect implicit interruption (user starts before bot finishes)
- Identify short interjections ("yeah", "uh-huh", "wait") for different handling

And Section 3.A2:
- Short interjection (<= 500ms speech)
- Full interruption (sustained speech)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InterruptionType(Enum):
    """Types of interruption detected."""
    NONE = "none"
    SHORT_INTERJECTION = "short_interjection"  # <= 500ms, e.g., "yeah", "uh-huh"
    FULL_INTERRUPTION = "full_interruption"    # > 500ms sustained speech


@dataclass
class InterruptionClassifier:
    """Classifies interruptions based on speech duration and content.

    Implements plan-human.md Section 3.A2:
    - Short interjection: speech <= short_threshold_ms
    - Full interruption: speech > short_threshold_ms
    """

    short_threshold_ms: int = 500  # Max duration for short interjection
    min_speech_ms: int = 100       # Minimum speech to consider as interruption

    def classify(self, speech_duration_ms: float, transcript: Optional[str] = None) -> InterruptionType:
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
            interjections = {"yeah", "yep", "uh-huh", "mm-hmm", "ok", "okay",
                            "right", "sure", "got it", "i see", "hmm", "mhm",
                            "uh huh", "mm hmm", "yes", "no", "wait"}
            cleaned = transcript.lower().strip().rstrip(".,!?")
            if cleaned in interjections:
                return InterruptionType.SHORT_INTERJECTION

        # Duration-based classification
        if speech_duration_ms <= self.short_threshold_ms:
            return InterruptionType.SHORT_INTERJECTION

        return InterruptionType.FULL_INTERRUPTION


class TestInterruptionClassifier:
    """Tests for interruption classification logic."""

    def test_very_short_speech_is_no_interruption(self):
        """Speech under minimum threshold is not an interruption."""
        classifier = InterruptionClassifier(min_speech_ms=100)

        result = classifier.classify(50)  # 50ms speech
        assert result == InterruptionType.NONE

    def test_short_speech_is_interjection(self):
        """Speech under short threshold is classified as interjection."""
        classifier = InterruptionClassifier(short_threshold_ms=500, min_speech_ms=100)

        result = classifier.classify(300)  # 300ms speech
        assert result == InterruptionType.SHORT_INTERJECTION

    def test_long_speech_is_full_interruption(self):
        """Speech over short threshold is classified as full interruption."""
        classifier = InterruptionClassifier(short_threshold_ms=500, min_speech_ms=100)

        result = classifier.classify(700)  # 700ms speech
        assert result == InterruptionType.FULL_INTERRUPTION

    def test_boundary_at_threshold(self):
        """Speech exactly at threshold is interjection."""
        classifier = InterruptionClassifier(short_threshold_ms=500, min_speech_ms=100)

        result = classifier.classify(500)  # Exactly 500ms
        assert result == InterruptionType.SHORT_INTERJECTION

    def test_boundary_just_over_threshold(self):
        """Speech just over threshold is full interruption."""
        classifier = InterruptionClassifier(short_threshold_ms=500, min_speech_ms=100)

        result = classifier.classify(501)  # Just over 500ms
        assert result == InterruptionType.FULL_INTERRUPTION


class TestInterruptionByTranscript:
    """Tests for content-based interruption classification."""

    def test_interjection_phrases(self):
        """Known interjection phrases are classified as short interjection."""
        classifier = InterruptionClassifier()
        interjections = ["yeah", "yep", "uh-huh", "mm-hmm", "ok", "okay",
                        "right", "sure", "got it", "yes", "no", "wait"]

        for phrase in interjections:
            # Even with long duration, known interjections stay as SHORT
            result = classifier.classify(800, transcript=phrase)
            assert result == InterruptionType.SHORT_INTERJECTION, f"'{phrase}' should be interjection"

    def test_interjection_with_punctuation(self):
        """Interjections with punctuation are still recognized."""
        classifier = InterruptionClassifier()

        assert classifier.classify(800, "Yeah!") == InterruptionType.SHORT_INTERJECTION
        assert classifier.classify(800, "Okay.") == InterruptionType.SHORT_INTERJECTION
        assert classifier.classify(800, "Wait,") == InterruptionType.SHORT_INTERJECTION

    def test_interjection_case_insensitive(self):
        """Interjection matching is case insensitive."""
        classifier = InterruptionClassifier()

        assert classifier.classify(800, "YEAH") == InterruptionType.SHORT_INTERJECTION
        assert classifier.classify(800, "Okay") == InterruptionType.SHORT_INTERJECTION
        assert classifier.classify(800, "OK") == InterruptionType.SHORT_INTERJECTION

    def test_non_interjection_transcript(self):
        """Non-interjection transcripts use duration-based classification."""
        classifier = InterruptionClassifier(short_threshold_ms=500)

        # Long duration with real speech -> full interruption
        result = classifier.classify(800, "Actually I wanted to ask about something else")
        assert result == InterruptionType.FULL_INTERRUPTION

        # Short duration with real speech -> still interjection by duration
        result = classifier.classify(300, "stop")
        assert result == InterruptionType.SHORT_INTERJECTION

    def test_empty_transcript_uses_duration(self):
        """Empty or None transcript falls back to duration classification."""
        classifier = InterruptionClassifier(short_threshold_ms=500, min_speech_ms=100)

        assert classifier.classify(300, None) == InterruptionType.SHORT_INTERJECTION
        assert classifier.classify(300, "") == InterruptionType.SHORT_INTERJECTION
        assert classifier.classify(700, None) == InterruptionType.FULL_INTERRUPTION


class TestInterruptionHandling:
    """Tests for how interruptions should be handled per plan-human.md Section 3.C2."""

    @dataclass
    class InterruptionHandler:
        """Handles interruptions based on classification.

        Resume strategy (from plan-human.md Section 3.C2):
        - Short interjection: RESUME previous response
        - Full interruption: RESTART with new context
        """

        def should_resume(self, interruption_type: InterruptionType) -> bool:
            """Determine if bot should resume previous response.

            Returns:
                True if previous response should be resumed, False if restart needed.
            """
            return interruption_type == InterruptionType.SHORT_INTERJECTION

        def should_restart(self, interruption_type: InterruptionType) -> bool:
            """Determine if bot should restart with new context.

            Returns:
                True if new response should be generated.
            """
            return interruption_type == InterruptionType.FULL_INTERRUPTION

    def test_short_interjection_triggers_resume(self):
        """Short interjections should resume previous response."""
        handler = self.InterruptionHandler()

        assert handler.should_resume(InterruptionType.SHORT_INTERJECTION) is True
        assert handler.should_restart(InterruptionType.SHORT_INTERJECTION) is False

    def test_full_interruption_triggers_restart(self):
        """Full interruptions should restart with new context."""
        handler = self.InterruptionHandler()

        assert handler.should_resume(InterruptionType.FULL_INTERRUPTION) is False
        assert handler.should_restart(InterruptionType.FULL_INTERRUPTION) is True

    def test_no_interruption_neither(self):
        """No interruption means neither resume nor restart."""
        handler = self.InterruptionHandler()

        assert handler.should_resume(InterruptionType.NONE) is False
        assert handler.should_restart(InterruptionType.NONE) is False


class TestVADInterruptionDetection:
    """Tests for VAD-based interruption detection (plan-human.md Section 3.A1)."""

    @dataclass
    class VADInterruptionDetector:
        """Detects interruptions using VAD while TTS is active.

        From plan-human.md:
        - Monitor VAD while bot is speaking
        - Threshold: VAD triggered for >100-200ms while TTS active
        """

        min_speech_duration_ms: int = 150  # VAD must be active for this long
        rms_threshold: float = 500.0       # Minimum RMS to consider speech

        def __init__(self, min_speech_duration_ms: int = 150, rms_threshold: float = 500.0):
            self.min_speech_duration_ms = min_speech_duration_ms
            self.rms_threshold = rms_threshold
            self._speech_start_time: Optional[float] = None
            self._is_tts_active: bool = False

        def set_tts_active(self, active: bool):
            """Set whether TTS is currently outputting."""
            self._is_tts_active = active
            if not active:
                self._speech_start_time = None

        def process_audio_rms(self, rms: float, timestamp_ms: float) -> bool:
            """Process audio RMS value and detect interruption.

            Args:
                rms: RMS value of audio chunk.
                timestamp_ms: Timestamp in milliseconds.

            Returns:
                True if interruption detected.
            """
            if not self._is_tts_active:
                return False

            if rms > self.rms_threshold:
                if self._speech_start_time is None:
                    self._speech_start_time = timestamp_ms
                elif (timestamp_ms - self._speech_start_time) >= self.min_speech_duration_ms:
                    return True
            else:
                # Reset on silence
                self._speech_start_time = None

            return False

    def test_no_detection_when_tts_inactive(self):
        """No interruption detected when TTS is not active."""
        detector = self.VADInterruptionDetector()
        detector.set_tts_active(False)

        # High RMS but TTS not active
        result = detector.process_audio_rms(1000.0, 0)
        assert result is False

    def test_no_detection_for_short_speech(self):
        """Speech under threshold duration is not detected."""
        detector = self.VADInterruptionDetector(min_speech_duration_ms=150)
        detector.set_tts_active(True)

        # 100ms of speech (under 150ms threshold)
        assert detector.process_audio_rms(1000.0, 0) is False
        assert detector.process_audio_rms(1000.0, 50) is False
        assert detector.process_audio_rms(1000.0, 100) is False

    def test_detection_for_sustained_speech(self):
        """Sustained speech over threshold is detected."""
        detector = self.VADInterruptionDetector(min_speech_duration_ms=150)
        detector.set_tts_active(True)

        # Build up speech duration
        assert detector.process_audio_rms(1000.0, 0) is False
        assert detector.process_audio_rms(1000.0, 50) is False
        assert detector.process_audio_rms(1000.0, 100) is False
        # At 150ms, should trigger
        assert detector.process_audio_rms(1000.0, 150) is True

    def test_silence_resets_detection(self):
        """Silence gap resets speech detection timer."""
        detector = self.VADInterruptionDetector(min_speech_duration_ms=150)
        detector.set_tts_active(True)

        # Start speech
        detector.process_audio_rms(1000.0, 0)
        detector.process_audio_rms(1000.0, 50)

        # Silence gap
        detector.process_audio_rms(100.0, 100)

        # Speech again - should restart from here
        assert detector.process_audio_rms(1000.0, 200) is False  # Restart at 200
        assert detector.process_audio_rms(1000.0, 300) is False  # Only 100ms in
        assert detector.process_audio_rms(1000.0, 350) is True   # Now 150ms

    def test_low_rms_not_considered_speech(self):
        """Low RMS values are not considered speech."""
        detector = self.VADInterruptionDetector(rms_threshold=500.0)
        detector.set_tts_active(True)

        # Low RMS over long duration
        for t in range(0, 1000, 50):
            result = detector.process_audio_rms(200.0, t)
            assert result is False


class TestInterruptionMetrics:
    """Tests for interruption latency metrics (plan-human.md Section 3.E1)."""

    @dataclass
    class InterruptionMetrics:
        """Track interruption latency metrics.

        From plan-human.md Section 3.E1:
        - Time from user speech start -> TTS stop (target < 150ms)
        - Time from user finish -> bot response start (target < 500ms)
        """

        user_speech_start_to_tts_stop_ms: list = None
        user_finish_to_bot_response_ms: list = None

        def __post_init__(self):
            self.user_speech_start_to_tts_stop_ms = []
            self.user_finish_to_bot_response_ms = []

        def record_interruption_latency(self, speech_start_ms: float, tts_stop_ms: float):
            """Record latency from user speech to TTS stop."""
            latency = tts_stop_ms - speech_start_ms
            self.user_speech_start_to_tts_stop_ms.append(latency)

        def record_response_latency(self, user_finish_ms: float, bot_response_ms: float):
            """Record latency from user finish to bot response."""
            latency = bot_response_ms - user_finish_ms
            self.user_finish_to_bot_response_ms.append(latency)

        def average_interruption_latency(self) -> float:
            """Get average interruption latency."""
            if not self.user_speech_start_to_tts_stop_ms:
                return 0.0
            return sum(self.user_speech_start_to_tts_stop_ms) / len(self.user_speech_start_to_tts_stop_ms)

        def average_response_latency(self) -> float:
            """Get average response latency."""
            if not self.user_finish_to_bot_response_ms:
                return 0.0
            return sum(self.user_finish_to_bot_response_ms) / len(self.user_finish_to_bot_response_ms)

        def interruption_target_met(self, target_ms: float = 150.0) -> bool:
            """Check if interruption latency target is met."""
            return self.average_interruption_latency() < target_ms

        def response_target_met(self, target_ms: float = 500.0) -> bool:
            """Check if response latency target is met."""
            return self.average_response_latency() < target_ms

    def test_interruption_latency_recording(self):
        """Test recording interruption latency."""
        metrics = self.InterruptionMetrics()

        metrics.record_interruption_latency(1000, 1100)  # 100ms
        metrics.record_interruption_latency(2000, 2150)  # 150ms

        assert metrics.average_interruption_latency() == 125.0

    def test_response_latency_recording(self):
        """Test recording response latency."""
        metrics = self.InterruptionMetrics()

        metrics.record_response_latency(1000, 1400)  # 400ms
        metrics.record_response_latency(2000, 2600)  # 600ms

        assert metrics.average_response_latency() == 500.0

    def test_interruption_target_met(self):
        """Test interruption latency target check."""
        metrics = self.InterruptionMetrics()

        # Good latency
        metrics.record_interruption_latency(0, 100)
        assert metrics.interruption_target_met(150.0) is True

        # Add bad latency
        metrics.record_interruption_latency(0, 300)
        # Average is now 200ms
        assert metrics.interruption_target_met(150.0) is False

    def test_response_target_met(self):
        """Test response latency target check."""
        metrics = self.InterruptionMetrics()

        # Good latency
        metrics.record_response_latency(0, 400)
        assert metrics.response_target_met(500.0) is True

        # Add bad latency
        metrics.record_response_latency(0, 800)
        # Average is now 600ms
        assert metrics.response_target_met(500.0) is False
