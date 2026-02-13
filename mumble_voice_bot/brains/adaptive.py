"""AdaptiveBrain - Routes between LLMBrain and ReactiveBrain based on brain_power.

The brain_power parameter controls how often the bot "thinks" (uses the LLM)
vs. "reacts" (uses fillers, echoes, deflections without the LLM).

brain_power values:
- 0.0: Pure reactive (never uses LLM)
- 0.5: Think about half the time
- 1.0: Always use LLM (current default behavior)

Utterance scoring determines urgency:
- Directed at bot (name mentioned): +0.4
- Is a question: +0.2
- Volume/emphasis (high RMS): +0.1
- New speaker: +0.1
- Engagement debt (time since last response): +0.2

Decision: should_think = (urgency >= 1.0 - brain_power)

Some utterances always trigger thinking regardless of brain_power:
- Bot's name mentioned + question
- Tool-calling keywords detected
- Direct text message to the bot
"""

from __future__ import annotations

import logging
import random
import time

from mumble_voice_bot.interfaces.brain import BotResponse, Brain, Utterance

logger = logging.getLogger(__name__)


class AdaptiveBrain:
    """Brain that routes between two sub-brains based on brain_power.

    Wraps any two brains (typically LLMBrain + ReactiveBrain) and uses
    utterance scoring to decide which one handles each utterance.

    Attributes:
        llm_brain: Brain to use for "thinking" (typically LLMBrain).
        reactive_brain: Brain to use for "reacting" (typically ReactiveBrain).
        brain_power: How often to use the LLM brain (0.0 to 1.0).
        force_think_keywords: Keywords that force LLM usage.
    """

    def __init__(
        self,
        llm_brain: Brain,
        reactive_brain: Brain,
        brain_power: float = 0.7,
        bot_name: str = "",
        force_think_keywords: list[str] | None = None,
    ):
        self.llm_brain = llm_brain
        self.reactive_brain = reactive_brain
        self.brain_power = max(0.0, min(1.0, brain_power))
        self.bot_name = bot_name.lower()

        self.force_think_keywords = force_think_keywords or [
            "search for", "look up", "play the sound", "play sound",
            "switch to", "change to", "list souls",
        ]

        # Engagement debt tracking
        self._last_response_time = time.time()
        self._engagement_window = 300.0  # 5 minutes normalizes to 1.0

        # Override tracking
        self._brain_power_override: float | None = None

    @property
    def effective_brain_power(self) -> float:
        """Get the effective brain_power (accounting for overrides)."""
        if self._brain_power_override is not None:
            return self._brain_power_override
        return self.brain_power

    def set_override(self, brain_power: float | None) -> None:
        """Set or clear a brain_power override (e.g., for LLM failure fallback).

        Args:
            brain_power: Override value, or None to clear.
        """
        self._brain_power_override = brain_power
        if brain_power is not None:
            logger.info(f"AdaptiveBrain: override set to {brain_power}")
        else:
            logger.info(f"AdaptiveBrain: override cleared, using configured {self.brain_power}")

    def process(self, utterance: Utterance) -> BotResponse | None:
        """Score utterance and route to appropriate brain.

        Args:
            utterance: Complete utterance from ASR.

        Returns:
            BotResponse from whichever brain handles the utterance.
        """
        bp = self.effective_brain_power

        # Always think at brain_power 1.0
        if bp >= 1.0:
            return self._think(utterance)

        # Never think at brain_power 0.0 (but still react sometimes)
        if bp <= 0.0:
            return self._react(utterance)

        # Check for forced thinking
        if self._should_force_think(utterance):
            logger.debug("AdaptiveBrain: forced think (keyword/direct address)")
            return self._think(utterance)

        # Score the utterance
        urgency = self._score_utterance(utterance)

        # Decision function
        should_think = urgency >= (1.0 - bp)

        if should_think:
            logger.debug(f"AdaptiveBrain: thinking (urgency={urgency:.2f}, bp={bp})")
            return self._think(utterance)
        else:
            # Even when not thinking, we may react
            response_rate = self._response_rate(bp)
            if random.random() < response_rate:
                logger.debug(f"AdaptiveBrain: reacting (urgency={urgency:.2f}, bp={bp})")
                return self._react(utterance)
            else:
                logger.debug(f"AdaptiveBrain: silent (urgency={urgency:.2f}, bp={bp})")
                return None

    def on_bot_utterance(self, speaker_name: str, text: str) -> BotResponse | None:
        """Route bot utterances to the LLM brain (if brain_power allows)."""
        if self.effective_brain_power >= 0.5:
            return self.llm_brain.on_bot_utterance(speaker_name, text)
        return None

    def on_text_message(self, sender: str, text: str) -> BotResponse | None:
        """Text messages always go to the LLM brain."""
        return self.llm_brain.on_text_message(sender, text)

    def _think(self, utterance: Utterance) -> BotResponse | None:
        """Route to the LLM brain."""
        result = self.llm_brain.process(utterance)
        if result:
            self._last_response_time = time.time()
        return result

    def _react(self, utterance: Utterance) -> BotResponse | None:
        """Route to the reactive brain."""
        result = self.reactive_brain.process(utterance)
        if result:
            self._last_response_time = time.time()
        return result

    def _score_utterance(self, utterance: Utterance) -> float:
        """Score an utterance's urgency from 0.0 to 1.0.

        Signals and weights:
        - Directed at bot (name mentioned): 0.4
        - Is a question: 0.2
        - Volume/emphasis: 0.1
        - New speaker: 0.1
        - Engagement debt: 0.2
        """
        score = 0.0

        # Directed at bot
        if utterance.is_directed:
            score += 0.4

        # Is a question
        if utterance.is_question:
            score += 0.2

        # Volume/emphasis (normalize RMS -- high energy = emphasis)
        if utterance.rms > 5000:  # Well above typical threshold
            score += 0.1

        # New speaker
        if utterance.is_first_speech:
            score += 0.1

        # Engagement debt
        time_since_response = time.time() - self._last_response_time
        debt = min(1.0, time_since_response / self._engagement_window)
        score += 0.2 * debt

        return min(1.0, score)

    def _should_force_think(self, utterance: Utterance) -> bool:
        """Check if this utterance should always trigger LLM thinking."""
        text_lower = utterance.text.lower()

        # Bot name mentioned + question
        if self.bot_name and self.bot_name in text_lower and utterance.is_question:
            return True

        # Tool-calling keywords
        for keyword in self.force_think_keywords:
            if keyword in text_lower:
                return True

        # Explicitly directed at bot
        if utterance.is_directed and utterance.is_question:
            return True

        return False

    def _response_rate(self, brain_power: float) -> float:
        """Calculate response rate based on brain_power.

        Returns probability of responding reactively when not thinking.

        brain_power -> response_rate:
        0.0 -> 0.20
        0.2 -> 0.35
        0.5 -> 0.60
        0.8 -> 0.90
        1.0 -> 1.00
        """
        # Quadratic curve that starts at 0.2 and reaches 1.0
        return 0.2 + 0.8 * (brain_power ** 0.7)
