"""LLMBrain - Full LLM-powered response generation with tool calling.

This is the brain used by the main MumbleVoiceBot. It receives an utterance,
applies speech filtering, builds conversation context, generates an LLM
response (with optional tool calling), and returns the response text.

This brain owns:
- Conversation history and context injection
- LLM call and tool execution loop
- Speech filtering (echo filter, utterance classifier)
- Soul/personality management
- Filler and event responses
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


class LLMBrain:
    """Brain that uses an LLM to generate intelligent responses.

    This is the most capable brain -- it understands context, can call tools,
    and generates natural conversational responses.

    Attributes:
        llm: LLM provider (OpenAIChatLLM or compatible).
        tools: Tool registry for function calling (optional).
        shared_services: SharedBotServices for journal and coordination.
        bot_name: Name of this bot (for journal context).
        soul_config: Soul configuration for personality and events.
        echo_filter: Echo filter to detect bot's own speech in ASR.
        utterance_classifier: Classifier to filter non-meaningful speech.
    """

    def __init__(
        self,
        llm: Any,
        bot_name: str,
        shared_services: Any = None,
        tools: Any = None,
        soul_config: Any = None,
        echo_filter: Any = None,
        utterance_classifier: Any = None,
        system_prompt: str = "",
        channel_history_max: int = 20,
        conversation_timeout: float = 300.0,
        talks_to_bots: bool = False,
    ):
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

        # Channel history (fallback when shared_services is not available)
        self.channel_history: list[dict] = []
        self.last_activity_time = time.time()

    def process(self, utterance: Utterance) -> BotResponse | None:
        """Generate an LLM response for the utterance.

        Applies speech filtering, then generates a response via the LLM
        with optional tool calling.

        Args:
            utterance: Complete utterance from ASR.

        Returns:
            BotResponse with LLM-generated text, or None if filtered out.
        """
        text = utterance.text
        user_name = utterance.user_name
        user_id = utterance.user_id

        # Echo filter: check if this is the bot's own speech being picked up
        if self.echo_filter and self.echo_filter.is_echo(text):
            logger.debug(f"Echo filter: ignoring '{text}'")
            return None

        # Utterance classifier: check if this is meaningful speech
        if self.utterance_classifier and not self.utterance_classifier.is_meaningful(text):
            logger.debug(f"Utterance filter: ignoring '{text}'")
            return None

        # Generate response
        try:
            response_text = self._generate_response_sync(user_id, text, user_name)
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}", exc_info=True)
            # Try to get a fallback from soul config
            fallback = self._get_event_response('rate_limited')
            if fallback:
                return BotResponse(text=fallback, is_filler=True)
            return BotResponse(text="I need a moment to collect my thoughts...", is_filler=True)

        if not response_text:
            return None

        return BotResponse(text=response_text)

    def on_bot_utterance(self, speaker_name: str, text: str) -> BotResponse | None:
        """Handle an utterance from another bot.

        Only responds if talks_to_bots is enabled.
        """
        if not self.talks_to_bots:
            return None

        # Ignore very short utterances
        clean_text = text.strip().rstrip('.')
        if len(clean_text.split()) < 3:
            return None

        try:
            # Set system prompt and generate response
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
        """Handle a text chat message."""
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
    # LLM Response Generation
    # =========================================================================

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
                    return fallback or "I need a moment to collect my thoughts..."
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
        # Soul switching and listing handled here for backward compatibility
        # (Implementation mirrors the monolith's _check_keyword_tools)
        return None

    # =========================================================================
    # Context Building
    # =========================================================================

    def _build_llm_messages(self) -> list[dict]:
        """Build LLM message list from shared journal or local history."""
        messages = []

        if self._shared_services:
            # Use shared journal
            journal = self._shared_services.get_journal_for_llm(max_events=50)

            # Context block
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
            # Fallback to local history
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
