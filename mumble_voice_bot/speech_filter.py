"""Speech filtering for echo detection and utterance classification.

This module provides filters to prevent the bot from responding to:
1. Echoes of its own speech (picked up by STT from TTS output)
2. Non-meaningful utterances (fillers, very short fragments)
"""

import time
from dataclasses import dataclass, field


@dataclass
class EchoFilter:
    """Filter to detect when STT picks up the bot's own TTS output.

    The bot's TTS output can be picked up by STT (either from the user's
    microphone or other audio feedback). This filter tracks recent bot
    outputs and checks if new transcriptions are likely echoes.

    Attributes:
        decay_time: How long to remember bot outputs (seconds).
        recent_outputs: List of (text, timestamp) tuples.
    """

    decay_time: float = 3.0
    recent_outputs: list[tuple[str, float]] = field(default_factory=list)

    def add_output(self, text: str) -> None:
        """Record what the bot just said.

        Args:
            text: The text that was spoken by the bot.
        """
        if not text or not text.strip():
            return
        self.recent_outputs.append((text.lower().strip(), time.time()))
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove old entries that have decayed."""
        cutoff = time.time() - self.decay_time
        self.recent_outputs = [(t, ts) for t, ts in self.recent_outputs if ts > cutoff]

    def is_echo(self, transcript: str, threshold: float = 0.8) -> bool:
        """Check if transcript matches recent bot output.

        Args:
            transcript: The transcribed text to check.
            threshold: Unused, kept for API compatibility.

        Returns:
            True if the transcript appears to be an echo of bot output.
        """
        self._cleanup()
        if not transcript or not transcript.strip():
            return False

        transcript_lower = transcript.lower().strip()
        transcript_words = set(transcript_lower.split())

        if not transcript_words:
            return False

        for output, _ in self.recent_outputs:
            # Check for exact substring match
            if transcript_lower in output:
                return True

            # Check if all transcript words appear in the output
            output_words = set(output.split())
            if transcript_words.issubset(output_words):
                return True

            # Check for significant word overlap (>80% of transcript words in output)
            if output_words:
                overlap = len(transcript_words & output_words)
                if overlap > 0 and overlap / len(transcript_words) >= 0.8:
                    return True

        return False

    def clear(self) -> None:
        """Clear all recorded outputs."""
        self.recent_outputs.clear()


@dataclass
class UtteranceClassifier:
    """Classify whether an utterance is meaningful enough to respond to.

    Very short transcriptions like "okay", "um", "you" shouldn't trigger
    full LLM responses. This classifier filters out:
    - Filler words (um, uh, hmm, etc.)
    - Very short utterances
    - Single words that aren't questions or commands
    """

    # Filler words that shouldn't trigger responses alone
    FILLERS: set[str] = field(default_factory=lambda: {
        "um", "uh", "hmm", "okay", "ok", "yeah", "yep", "nope",
        "mhm", "ah", "oh", "huh", "eh", "erm", "er", "like",
        "well", "so", "right", "sure", "yes", "no", "yea", "nah",
        "mm", "mmm", "ugh", "ooh", "aah", "whoa",
    })

    # Minimum word count for meaningful utterance
    min_words: int = 2

    # Minimum character count
    min_chars: int = 5

    # Single words that ARE meaningful (questions/commands)
    MEANINGFUL_SINGLE_WORDS: set[str] = field(default_factory=lambda: {
        "what", "why", "how", "when", "where", "who", "which",
        "help", "stop", "start", "go", "wait", "pause", "play",
        "hello", "hi", "hey", "bye", "goodbye",
    })

    def is_meaningful(self, text: str) -> bool:
        """Determine if utterance warrants a response.

        Args:
            text: The transcribed text to classify.

        Returns:
            True if the utterance is meaningful and should get a response.
        """
        if not text:
            return False

        text = text.strip().lower()
        words = text.split()

        if not words:
            return False

        # Check meaningful single words FIRST (before length checks)
        # This allows short but important words like "hi", "why", "help"
        if len(words) == 1:
            word = text.rstrip("?!.")
            if word in self.MEANINGFUL_SINGLE_WORDS:
                return True
            if text.endswith("?"):
                return True

        # Empty or very short (after single word check)
        if len(text) < self.min_chars:
            return False

        # Just filler words
        if all(w in self.FILLERS for w in words):
            return False

        # Multi-word check: need at least min_words
        if len(words) < self.min_words:
            return False

        return True


@dataclass
class TurnPredictor:
    """Predict when user has finished their turn.

    Uses heuristics to estimate turn completion probability based on:
    - Punctuation at end of utterance
    - Continuation words that suggest more is coming
    - Duration of silence

    This helps the bot respond at natural turn-taking points without
    waiting for long silences or interrupting mid-sentence.
    """

    # Base delay before responding (seconds)
    base_delay: float = 0.3

    # Maximum delay to wait (seconds)
    max_delay: float = 1.5

    # Indicators that user is done speaking
    TURN_END_MARKERS: dict[str, float] = field(default_factory=lambda: {
        "?": 0.9,   # Questions strongly indicate turn end
        "!": 0.8,   # Exclamations indicate turn end
        ".": 0.7,   # Statements somewhat indicate turn end
    })

    # Phrases that indicate more coming (trailing words)
    CONTINUATION_PHRASES: set[str] = field(default_factory=lambda: {
        "and", "but", "so", "because", "however", "also",
        "first", "second", "then", "next", "finally",
        "although", "though", "unless", "while", "since",
        "like", "that", "which", "who", "where", "when",
    })

    def predict_turn_complete(self, transcript: str, silence_duration: float) -> float:
        """Predict confidence (0-1) that user is done speaking.

        Args:
            transcript: The current transcription.
            silence_duration: How long since last speech (seconds).

        Returns:
            Confidence score from 0 to 1 that user is done speaking.
        """
        if not transcript:
            return 0.0

        text = transcript.strip()
        confidence = 0.5  # Base confidence

        # Check punctuation at end
        if text:
            last_char = text[-1]
            confidence += self.TURN_END_MARKERS.get(last_char, 0)

        # Check for continuation words at end
        words = text.lower().split()
        if words and words[-1] in self.CONTINUATION_PHRASES:
            confidence -= 0.4  # Likely more coming

        # Factor in silence duration
        # Longer silence = more likely turn is complete
        silence_factor = min(silence_duration / self.max_delay, 1.0)
        confidence = confidence * 0.6 + silence_factor * 0.4

        return min(max(confidence, 0.0), 1.0)

    def get_response_delay(self, transcript: str) -> float:
        """Calculate how long to wait before responding.

        Args:
            transcript: The current transcription.

        Returns:
            Recommended delay in seconds before responding.
        """
        if not transcript:
            return self.max_delay

        text = transcript.strip()

        # Questions get fast responses
        if text.endswith("?"):
            return self.base_delay

        # Check for continuation indicators
        words = text.lower().split()
        if words and words[-1] in self.CONTINUATION_PHRASES:
            return self.max_delay

        # Default: moderate delay
        return self.base_delay + 0.2

    def should_respond(self, transcript: str, silence_duration: float, threshold: float = 0.7) -> bool:
        """Determine if bot should respond now.

        Args:
            transcript: The current transcription.
            silence_duration: How long since last speech (seconds).
            threshold: Confidence threshold for responding.

        Returns:
            True if bot should respond now.
        """
        confidence = self.predict_turn_complete(transcript, silence_duration)
        return confidence >= threshold
