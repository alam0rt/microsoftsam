"""LLMBrain - Unified brain with LLM intelligence and reactive fallbacks.

This is the single brain for all intelligent bot behaviors. It combines:
- LLM-powered response generation with tool calling
- Reactive responses (fillers, echo fragments, deflections) for low brain_power
- Utterance scoring and brain_power routing (formerly AdaptiveBrain)
- Graceful degradation when the LLM is unavailable or rate-limited

brain_power controls how often the brain uses the LLM vs. reactive responses:
- 0.0: Pure reactive (never uses LLM, fillers/echoes/deflections only)
- 0.5: Think about half the time (score-based routing)
- 1.0: Always use LLM (default)

Even at brain_power=1.0, the reactive pool is available for:
- Thinking stalls while waiting for slow LLM responses
- Barge-in acknowledgments
- Rate-limit / error fallbacks
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from typing import Any

from mumble_voice_bot.interfaces.brain import BotResponse, Utterance

logger = logging.getLogger(__name__)

# ============================================================================
# Reactive response pools
# ============================================================================

DEFAULT_FILLERS = [
    "mmhm", "yeah", "right", "mm", "uh huh", "sure", "yep", "okay", "mhm",
    "yeah, totally", "right, right", "oh yeah", "for sure", "makes sense",
    "heh", "hah", "oh", "ah", "wow", "nice", "cool",
]

DEFAULT_DEFLECTIONS = [
    "hmm, I dunno", "hah, yeah", "interesting", "huh", "oh really",
    "that's wild", "fair enough", "yeah, I guess so", "I mean... sure",
    "oh, okay", "huh, weird", "that's something",
]

DEFAULT_THINKING_STALLS = [
    "umm... one second", "let me think...", "hmm...", "give me a sec",
    "hold on...", "uh... good question", "let me see...", "one sec...",
]

DEFAULT_BARGE_IN_ACKS = [
    "oh, go ahead", "sorry, what?", "yeah?", "oh, my bad",
    "go on", "sorry, you were saying?",
]

ECHO_TEMPLATES = [
    "{fragment}... huh.",
    "wait, {fragment}?",
    "{fragment}... yeah.",
    "sorry, what about {fragment}?",
    "{fragment}... hmm.",
    "{fragment}... interesting.",
]


class LLMBrain:
    """Unified brain with LLM intelligence and built-in reactive fallbacks.

    At brain_power=1.0, every utterance goes to the LLM. At lower values,
    utterances are scored for urgency and routed to either the LLM or the
    reactive pool. The reactive pool is always available for stalls, acks,
    and error fallbacks regardless of brain_power.

    Attributes:
        llm: LLM provider (OpenAIChatLLM or compatible).
        tools: Tool registry for function calling (optional).
        shared_services: SharedBotServices for journal and coordination.
        bot_name: Name of this bot (for journal context).
        soul_config: Soul configuration for personality and events.
        brain_power: How often to use the LLM (0.0 to 1.0).
        echo_filter: Echo filter to detect bot's own speech in ASR.
        utterance_classifier: Classifier to filter non-meaningful speech.
    """

    def __init__(
        self,
        llm: Any = None,
        bot_name: str = "",
        shared_services: Any = None,
        tools: Any = None,
        soul_config: Any = None,
        echo_filter: Any = None,
        utterance_classifier: Any = None,
        system_prompt: str = "",
        channel_history_max: int = 20,
        conversation_timeout: float = 300.0,
        talks_to_bots: bool = False,
        brain_power: float = 1.0,
        # Reactive pool overrides
        fillers: list[str] | None = None,
        deflections: list[str] | None = None,
        thinking_stalls: list[str] | None = None,
        barge_in_acks: list[str] | None = None,
        silence_weight: float = 0.3,
        echo_weight: float = 0.3,
    ):
        # LLM
        self.llm = llm
        self.bot_name = bot_name
        self._shared_services = shared_services
        self.tools = tools
        self.soul_config = soul_config
        self.echo_filter = echo_filter
        self.utterance_classifier = utterance_classifier
        self._bot_system_prompt = system_prompt
        self.channel_history_max = channel_history_max
        self.conversation_timeout = conversation_timeout
        self.talks_to_bots = talks_to_bots

        # brain_power (0.0 = pure reactive, 1.0 = always LLM)
        self.brain_power = max(0.0, min(1.0, brain_power))
        self._brain_power_override: float | None = None

        # Reactive pool
        self.fillers = fillers or list(DEFAULT_FILLERS)
        self.deflections = deflections or list(DEFAULT_DEFLECTIONS)
        self.thinking_stalls = thinking_stalls or list(DEFAULT_THINKING_STALLS)
        self.barge_in_acks = barge_in_acks or list(DEFAULT_BARGE_IN_ACKS)
        self.silence_weight = silence_weight
        self.echo_weight = echo_weight

        # Override reactive pools from soul config
        if soul_config and hasattr(soul_config, 'fallbacks') and soul_config.fallbacks:
            fb = soul_config.fallbacks
            if hasattr(fb, 'thinking') and fb.thinking:
                self.fillers = list(fb.thinking)
            if hasattr(fb, 'still_thinking') and fb.still_thinking:
                self.thinking_stalls = list(fb.still_thinking)
            if hasattr(fb, 'interrupted') and fb.interrupted:
                self.barge_in_acks = list(fb.interrupted)

        # Filler rotation: track recently used fillers to avoid repetition
        self._recent_fillers: list[str] = []
        self._recent_max = 5

        # Engagement debt tracking (for brain_power scoring)
        self._last_response_time = time.time()
        self._engagement_window = 300.0  # 5 minutes normalizes to 1.0

        # Force-think keywords
        self.force_think_keywords = [
            "search for", "look up", "play the sound", "play sound",
            "switch to", "change to", "list souls",
        ]

        # Channel history (fallback when shared_services is not available)
        self.channel_history: list[dict] = []
        self.last_activity_time = time.time()

    # =========================================================================
    # brain_power
    # =========================================================================

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
            logger.info(f"LLMBrain: brain_power override set to {brain_power}")
        else:
            logger.info(f"LLMBrain: override cleared, using configured {self.brain_power}")

    # =========================================================================
    # Brain protocol
    # =========================================================================

    def process(self, utterance: Utterance) -> BotResponse | None:
        """Process an utterance — route to LLM or reactive based on brain_power.

        Args:
            utterance: Complete utterance from ASR.

        Returns:
            BotResponse with text, or None if filtered out / choosing silence.
        """
        text = utterance.text

        # Echo filter: check if this is the bot's own speech being picked up
        if self.echo_filter and self.echo_filter.is_echo(text):
            logger.debug(f"Echo filter: ignoring '{text}'")
            return None

        # Utterance classifier: check if this is meaningful speech
        if self.utterance_classifier and not self.utterance_classifier.is_meaningful(text):
            logger.debug(f"Utterance filter: ignoring '{text}'")
            return None

        bp = self.effective_brain_power

        # Always LLM at brain_power 1.0 (with LLM available)
        if bp >= 1.0 and self.llm:
            return self._think(utterance)

        # Never LLM at brain_power 0.0 or no LLM
        if bp <= 0.0 or not self.llm:
            return self._react(utterance)

        # Mixed mode: score and route
        if self._should_force_think(utterance):
            logger.debug("LLMBrain: forced think (keyword/direct address)")
            return self._think(utterance)

        urgency = self._score_utterance(utterance)
        should_think = urgency >= (1.0 - bp)

        if should_think:
            logger.debug(f"LLMBrain: thinking (urgency={urgency:.2f}, bp={bp})")
            return self._think(utterance)
        else:
            # Even when not thinking, we may still react
            response_rate = self._response_rate(bp)
            if random.random() < response_rate:
                logger.debug(f"LLMBrain: reacting (urgency={urgency:.2f}, bp={bp})")
                return self._react(utterance)
            else:
                logger.debug(f"LLMBrain: silent (urgency={urgency:.2f}, bp={bp})")
                return None

    def on_bot_utterance(self, speaker_name: str, text: str) -> BotResponse | None:
        """Handle an utterance from another bot.

        Only responds if talks_to_bots is enabled and brain_power is sufficient.
        """
        if not self.talks_to_bots:
            return None
        if self.effective_brain_power < 0.5:
            return None
        if not self.llm:
            return None

        # Ignore very short utterances
        clean_text = text.strip().rstrip('.')
        if len(clean_text.split()) < 3:
            return None

        try:
            if hasattr(self.llm, 'system_prompt'):
                self.llm.system_prompt = self._bot_system_prompt

            messages = self._build_llm_messages()
            response = self._run_coro_sync(
                self.llm.chat(messages, bot_name=self.bot_name)
            )

            if response.content:
                return BotResponse(text=response.content)
        except Exception as e:
            logger.error(f"Bot-to-bot response failed: {e}", exc_info=True)

        return None

    def on_text_message(self, sender: str, text: str) -> BotResponse | None:
        """Handle a text chat message. Always uses LLM if available."""
        if not self.llm:
            return None

        try:
            response_text = self._generate_response_sync(0, text, sender)
            if response_text:
                return BotResponse(text=response_text)
        except Exception as e:
            logger.error(f"Text response generation failed: {e}", exc_info=True)

        return None

    # =========================================================================
    # LLM path ("think")
    # =========================================================================

    def _think(self, utterance: Utterance) -> BotResponse | None:
        """Generate an LLM response for the utterance."""
        try:
            response_text = self._generate_response_sync(
                utterance.user_id, utterance.text, utterance.user_name
            )
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}", exc_info=True)
            fallback = self._get_event_response('rate_limited')
            if fallback:
                return BotResponse(text=fallback, is_filler=True)
            return BotResponse(text=self.get_thinking_stall(), is_filler=True)

        if not response_text:
            return None

        self._last_response_time = time.time()
        return BotResponse(text=response_text)

    def _generate_response_sync(self, user_id: int, text: str, user_name: str = None) -> str:
        """Generate LLM response synchronously."""
        return self._run_coro_sync(self._generate_response(user_id, text, user_name))

    async def _generate_response(self, user_id: int, text: str, user_name: str = None) -> str:
        """Generate LLM response with tool execution loop."""
        # Check for keyword-based tool triggers
        keyword_result = await self._check_keyword_tools(text)
        if keyword_result:
            self._log_user_message(user_name, text)
            return keyword_result

        self._log_user_message(user_name, text)

        # Build messages for LLM
        messages = self._build_llm_messages()

        # Get tool definitions
        tools = self.tools.get_definitions() if self.tools else None

        # Set system prompt
        if hasattr(self.llm, 'system_prompt') and self._bot_system_prompt:
            self.llm.system_prompt = self._bot_system_prompt

        # Tool execution loop
        max_iterations = 5
        for iteration in range(max_iterations):
            try:
                response = await self.llm.chat(messages, tools=tools, bot_name=self.bot_name)
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str or "too many" in error_str:
                    logger.warning(f"Rate limited: {e}")
                    fallback = self._get_event_response('rate_limited')
                    return fallback or self.get_thinking_stall()
                raise

            if response.has_tool_calls:
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })

                # Execute tools
                for tool_call in response.tool_calls:
                    logger.info(f"Executing tool: {tool_call.name}({tool_call.arguments})")
                    if self.tools:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    else:
                        result = f"Error: Tool '{tool_call.name}' not available"

                    if hasattr(self.llm, 'tool_formatter'):
                        tool_msg = self.llm.tool_formatter.format_tool_result(
                            tool_call.id, tool_call.name, result
                        )
                        messages.append(tool_msg)
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                continue

            if response.content:
                return response.content

            logger.warning("LLM returned empty response")
            return ""

        logger.warning(f"Tool loop hit max iterations ({max_iterations})")
        return "Sorry, I got stuck in a loop trying to look that up."

    async def _check_keyword_tools(self, text: str) -> str | None:
        """Check for keyword-based tool triggers (fallback for models without tool support)."""
        if not self.tools:
            return None
        return None

    # =========================================================================
    # Reactive path ("react")
    # =========================================================================

    def _react(self, utterance: Utterance) -> BotResponse | None:
        """Generate a reactive response (no LLM).

        Decision tree:
        1. Roll for silence
        2. If question, try stalling echo or deflection
        3. Try echo fragment
        4. Fall back to filler
        """
        # Roll for silence
        if random.random() < self.silence_weight:
            logger.debug("LLMBrain: reactive chose silence")
            return None

        text = utterance.text.strip()

        # Questions get special treatment
        if utterance.is_question:
            if random.random() < 0.5:
                stall = self._stalling_echo(text)
                if stall:
                    result = BotResponse(text=stall, is_filler=True, skip_broadcast=True)
                    self._last_response_time = time.time()
                    return result
            result = BotResponse(
                text=random.choice(self.deflections),
                is_filler=True,
                skip_broadcast=True,
            )
            self._last_response_time = time.time()
            return result

        # Try echo fragment
        if random.random() < self.echo_weight:
            fragment = self._echo_fragment(text)
            if fragment:
                result = BotResponse(text=fragment, is_filler=True, skip_broadcast=True)
                self._last_response_time = time.time()
                return result

        # Filler
        filler = self._pick_filler()
        self._last_response_time = time.time()
        return BotResponse(text=filler, is_filler=True, skip_broadcast=True)

    # =========================================================================
    # Utterance scoring (brain_power routing)
    # =========================================================================

    def _score_utterance(self, utterance: Utterance) -> float:
        """Score an utterance's urgency from 0.0 to 1.0.

        Signals and weights:
        - Directed at bot (name mentioned): 0.4
        - Is a question: 0.2
        - Volume/emphasis (high RMS): 0.1
        - New speaker: 0.1
        - Engagement debt: 0.2
        """
        score = 0.0

        if utterance.is_directed:
            score += 0.4
        if utterance.is_question:
            score += 0.2
        if utterance.rms > 5000:
            score += 0.1
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
        bot_lower = self.bot_name.lower()

        # Bot name mentioned + question
        if bot_lower and bot_lower in text_lower and utterance.is_question:
            return True

        # Tool-calling keywords
        for keyword in self.force_think_keywords:
            if keyword in text_lower:
                return True

        # Explicitly directed at bot + question
        if utterance.is_directed and utterance.is_question:
            return True

        return False

    def _response_rate(self, brain_power: float) -> float:
        """Calculate reactive response rate based on brain_power.

        brain_power -> response_rate:
        0.0 -> 0.20
        0.5 -> 0.60
        1.0 -> 1.00
        """
        return 0.2 + 0.8 * (brain_power ** 0.7)

    # =========================================================================
    # Reactive pool helpers
    # =========================================================================

    def _pick_filler(self) -> str:
        """Pick a filler avoiding recent repeats."""
        available = [f for f in self.fillers if f not in self._recent_fillers]
        if not available:
            self._recent_fillers.clear()
            available = self.fillers

        choice = random.choice(available)
        self._recent_fillers.append(choice)
        if len(self._recent_fillers) > self._recent_max:
            self._recent_fillers.pop(0)

        return choice

    def _echo_fragment(self, text: str) -> str | None:
        """Extract a key phrase and wrap it in an echo template."""
        words = text.split()
        if len(words) < 2:
            return None

        fragment_len = min(3, len(words))
        fragment = " ".join(words[-fragment_len:]).rstrip(".,!?")

        if not fragment or len(fragment) < 2:
            return None

        template = random.choice(ECHO_TEMPLATES)
        return template.format(fragment=fragment)

    def _stalling_echo(self, text: str) -> str | None:
        """Repeat part of a question with a stalling suffix."""
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

    def get_thinking_stall(self) -> str:
        """Get a random thinking stall phrase (for external callers)."""
        return random.choice(self.thinking_stalls)

    def get_barge_in_ack(self) -> str:
        """Get a random barge-in acknowledgment phrase."""
        return random.choice(self.barge_in_acks)

    # =========================================================================
    # Context Building
    # =========================================================================

    def _build_llm_messages(self) -> list[dict]:
        """Build LLM message list from shared journal or local history."""
        messages = []

        if self._shared_services:
            journal = self._shared_services.get_journal_for_llm(max_events=50)

            context_parts = [f"Current time: {self._get_time_context()}"]

            recent_events = [
                e for e in journal
                if e.get("seconds_ago", 999) <= 60
                and e.get("event") in ("user_joined", "user_left")
            ]
            for e in recent_events[-3:]:
                speaker = e.get("speaker", "someone")
                if e["event"] == "user_joined":
                    context_parts.append(f"{speaker} just joined")
                elif e["event"] == "user_left":
                    context_parts.append(f"{speaker} just left")

            if context_parts:
                messages.append({"role": "system", "content": " | ".join(context_parts)})

            history = self._shared_services.get_recent_messages_for_llm(
                max_messages=self.channel_history_max, bot_name=self.bot_name
            )
            messages.extend(history)
        else:
            messages.extend(self.channel_history[-self.channel_history_max:])

        return messages

    def _log_user_message(self, user_name: str, text: str):
        """Log a user message to the shared journal."""
        if self._shared_services:
            self._shared_services.log_event("user_message", user_name, text)
        else:
            self.channel_history.append({
                "role": "user",
                "content": f"{user_name}: {text}" if user_name else text,
            })

    def _get_time_context(self) -> str:
        """Get current time context for the LLM."""
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        day_name = now.strftime("%A")
        time_str = now.strftime("%I:%M %p").lstrip("0")
        return f"It's {time_of_day}, {day_name} at {time_str}."

    def _get_event_response(self, event_type: str, user: str = None) -> str | None:
        """Get a themed event response from soul config."""
        if not self.soul_config:
            return None

        responses = None
        if self.soul_config.events:
            responses = getattr(self.soul_config.events, event_type, None)

        if not responses and self.soul_config.fallbacks:
            fallback_map = {
                'rate_limited': 'thinking',
                'thinking': 'thinking',
                'still_thinking': 'still_thinking',
                'interrupted': 'interrupted',
            }
            fallback_key = fallback_map.get(event_type)
            if fallback_key:
                responses = getattr(self.soul_config.fallbacks, fallback_key, None)

        if not responses:
            return None

        response = random.choice(responses)
        if user:
            response = response.replace("{user}", user)
        return response

    def _run_coro_sync(self, coroutine):
        """Run an async coroutine from sync code."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, coroutine)
                    return future.result(timeout=35.0)
            return loop.run_until_complete(coroutine)
        except RuntimeError:
            return asyncio.run(coroutine)
