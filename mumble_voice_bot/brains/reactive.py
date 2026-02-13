"""ReactiveBrain - LLM-free responses using fillers, echoes, and deflections.

This brain provides natural-sounding responses without using an LLM.
It's used by AdaptiveBrain when brain_power decides not to think,
and as a graceful degradation when the LLM is unavailable.

Response types:
- Echo fragment: "[key phrase]... huh." / "wait, [noun]?"
- Stalling echo: "[repeats question]... uh, good question."
- Filler: "mmhm" / "yeah" / "heh" / "right"
- Deflection: "hmm, I dunno" / "hah, yeah"
- Thinking stall: "umm... one second" / "let me think..."
- Silence: (returns None)
"""

from __future__ import annotations

import logging
import random

from mumble_voice_bot.interfaces.brain import BotResponse, Utterance

logger = logging.getLogger(__name__)

# Default filler pools — diverse selection for natural conversational presence
DEFAULT_FILLERS = [
    # Short acknowledgments
    "mmhm",
    "yeah",
    "right",
    "mm",
    "uh huh",
    "sure",
    "yep",
    "okay",
    "mhm",
    # Slightly longer acknowledgments
    "yeah, totally",
    "right, right",
    "oh yeah",
    "for sure",
    "makes sense",
    # Casual reactions
    "heh",
    "hah",
    "oh",
    "ah",
    "wow",
    "nice",
    "cool",
]

DEFAULT_DEFLECTIONS = [
    "hmm, I dunno",
    "hah, yeah",
    "interesting",
    "huh",
    "oh really",
    "that's wild",
    "fair enough",
    "yeah, I guess so",
    "I mean... sure",
    "oh, okay",
    "huh, weird",
    "that's something",
]

DEFAULT_THINKING_STALLS = [
    "umm... one second",
    "let me think...",
    "hmm...",
    "give me a sec",
    "hold on...",
    "uh... good question",
    "let me see...",
    "one sec...",
]

DEFAULT_BARGE_IN_ACKS = [
    "oh, go ahead",
    "sorry, what?",
    "yeah?",
    "oh, my bad",
    "go on",
    "sorry, you were saying?",
]

ECHO_TEMPLATES = [
    "{fragment}... huh.",
    "wait, {fragment}?",
    "{fragment}... yeah.",
    "sorry, what about {fragment}?",
    "{fragment}... hmm.",
    "{fragment}... interesting.",
]


class ReactiveBrain:
    """Brain that responds with LLM-free reactive behaviors.

    Uses fillers, echo fragments, and deflections to maintain presence
    in conversation without requiring an LLM call.

    Features:
    - Context-weighted filler selection
    - Filler rotation (avoids repeating the same one)
    - Occasional silence for naturalness
    - Echo fragment extraction from ASR transcript
    - Barge-in acknowledgment phrases

    Attributes:
        fillers: Pool of filler phrases.
        deflections: Pool of deflection phrases.
        thinking_stalls: Pool of "I'm thinking" phrases.
        barge_in_acks: Pool of barge-in acknowledgment phrases.
        silence_weight: Probability of choosing silence (0.0-1.0).
        echo_weight: Probability of attempting echo fragment (0.0-1.0).
    """

    def __init__(
        self,
        fillers: list[str] | None = None,
        deflections: list[str] | None = None,
        thinking_stalls: list[str] | None = None,
        barge_in_acks: list[str] | None = None,
        silence_weight: float = 0.3,
        echo_weight: float = 0.3,
        soul_config: object | None = None,
    ):
        self.fillers = fillers or list(DEFAULT_FILLERS)
        self.deflections = deflections or list(DEFAULT_DEFLECTIONS)
        self.thinking_stalls = thinking_stalls or list(DEFAULT_THINKING_STALLS)
        self.barge_in_acks = barge_in_acks or list(DEFAULT_BARGE_IN_ACKS)
        self.silence_weight = silence_weight
        self.echo_weight = echo_weight

        # Filler rotation: track recently used fillers to avoid repetition
        self._recent_fillers: list[str] = []
        self._recent_max = 5  # Remember last 5 fillers used

        # Override pools from soul config if available
        if soul_config and hasattr(soul_config, 'fallbacks') and soul_config.fallbacks:
            fb = soul_config.fallbacks
            if hasattr(fb, 'thinking') and fb.thinking:
                self.fillers = list(fb.thinking)
            if hasattr(fb, 'still_thinking') and fb.still_thinking:
                self.thinking_stalls = list(fb.still_thinking)
            if hasattr(fb, 'interrupted') and fb.interrupted:
                self.barge_in_acks = list(fb.interrupted)

    def process(self, utterance: Utterance) -> BotResponse | None:
        """Generate a reactive response (no LLM).

        Decision tree:
        1. Roll for silence
        2. If question detected, try stalling echo or deflection
        3. Try echo fragment
        4. Fall back to filler

        Args:
            utterance: Complete utterance from ASR.

        Returns:
            BotResponse with reactive text, or None for silence.
        """
        # Roll for silence
        if random.random() < self.silence_weight:
            logger.debug("ReactiveBrain: chose silence")
            return None

        text = utterance.text.strip()

        # Questions get special treatment
        if utterance.is_question:
            if random.random() < 0.5:
                # Stalling echo
                stall = self._stalling_echo(text)
                if stall:
                    return BotResponse(text=stall, is_filler=True, skip_broadcast=True)
            # Deflection
            return BotResponse(
                text=random.choice(self.deflections),
                is_filler=True,
                skip_broadcast=True,
            )

        # Try echo fragment
        if random.random() < self.echo_weight:
            fragment = self._echo_fragment(text)
            if fragment:
                return BotResponse(text=fragment, is_filler=True, skip_broadcast=True)

        # Filler (with rotation to avoid repetition)
        filler = self._pick_filler()
        return BotResponse(
            text=filler,
            is_filler=True,
            skip_broadcast=True,
        )

    def on_bot_utterance(self, speaker_name: str, text: str) -> BotResponse | None:
        """ReactiveBrain ignores other bots."""
        return None

    def on_text_message(self, sender: str, text: str) -> BotResponse | None:
        """ReactiveBrain ignores text messages."""
        return None

    def get_thinking_stall(self) -> str:
        """Get a random thinking stall phrase (for LLM fallback mode)."""
        return random.choice(self.thinking_stalls)

    def get_barge_in_ack(self) -> str:
        """Get a random barge-in acknowledgment phrase."""
        return random.choice(self.barge_in_acks)

    def _pick_filler(self) -> str:
        """Pick a filler avoiding recent repeats.

        Uses rotation: recently used fillers are deprioritized.
        Falls back to any filler if all have been used recently.
        """
        # Try to find one we haven't used recently
        available = [f for f in self.fillers if f not in self._recent_fillers]
        if not available:
            # All used recently — reset and pick from full pool
            self._recent_fillers.clear()
            available = self.fillers

        choice = random.choice(available)

        # Track recent usage
        self._recent_fillers.append(choice)
        if len(self._recent_fillers) > self._recent_max:
            self._recent_fillers.pop(0)

        return choice

    def _echo_fragment(self, text: str) -> str | None:
        """Extract a key phrase and wrap it in an echo template.

        Takes the last clause or key noun phrase and reflects it back.

        Args:
            text: The user's utterance.

        Returns:
            Echo fragment string, or None if extraction fails.
        """
        words = text.split()
        if len(words) < 2:
            return None

        # Extract last 1-3 words as the fragment
        fragment_len = min(3, len(words))
        fragment = " ".join(words[-fragment_len:]).rstrip(".,!?")

        if not fragment or len(fragment) < 2:
            return None

        template = random.choice(ECHO_TEMPLATES)
        return template.format(fragment=fragment)

    def _stalling_echo(self, text: str) -> str | None:
        """Repeat part of a question with a stalling suffix.

        Args:
            text: The user's question.

        Returns:
            Stalling echo like "how does that work... uh, good question."
        """
        # Take first ~5 words of the question
        words = text.split()
        if len(words) < 3:
            return None

        fragment = " ".join(words[:min(5, len(words))]).rstrip(".,!?")

        suffixes = [
            "... uh, good question.",
            "... hmm, let me think.",
            "... that's a good one.",
            "... huh, interesting question.",
        ]

        return f"{fragment}{random.choice(suffixes)}"
