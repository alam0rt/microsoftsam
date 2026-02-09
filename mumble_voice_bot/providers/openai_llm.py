"""OpenAI-compatible LLM provider.

This provider works with any service that implements the OpenAI Chat Completions API:
- OpenAI API
- vLLM
- Ollama (with OpenAI compatibility)
- llama.cpp server
- LocalAI
- LiteLLM
"""

import json
import time
from typing import AsyncIterator

import httpx

from mumble_voice_bot.interfaces.llm import LLMProvider, LLMResponse, ToolCall
from mumble_voice_bot.interfaces.tool_formatter import (
    OpenAIToolFormatter,
    ToolFormatter,
    get_tool_formatter,
)
from mumble_voice_bot.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIChatLLM(LLMProvider):
    """LLM provider for OpenAI-compatible chat completion APIs.

    This provider sends requests to any endpoint implementing the
    OpenAI Chat Completions API format:

        POST /v1/chat/completions
        {
            "model": "...",
            "messages": [{"role": "...", "content": "..."}]
        }

    Attributes:
        endpoint: The full URL to the chat completions endpoint.
        model: The model identifier to use.
        api_key: Optional API key for authentication.
        system_prompt: Optional system prompt to prepend to all conversations.
        timeout: Request timeout in seconds.
        max_tokens: Maximum tokens in the response (optional).
        temperature: Sampling temperature (optional).
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        timeout: float = 30.0,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ):
        """Initialize the OpenAI-compatible LLM provider.

        Args:
            endpoint: Full URL to the chat completions endpoint.
                     e.g., "http://localhost:11434/v1/chat/completions" for Ollama
                     e.g., "https://api.openai.com/v1/chat/completions" for OpenAI
            model: Model identifier to use (e.g., "llama3.2:3b", "gpt-4o-mini").
            api_key: Optional API key for authenticated endpoints.
            system_prompt: Optional system prompt prepended to all conversations.
            timeout: HTTP request timeout in seconds (default: 30).
            max_tokens: Maximum tokens in the response (optional).
            temperature: Sampling temperature (optional).
            top_p: Nucleus sampling parameter (optional).
            top_k: Top-k sampling parameter (optional).
            repetition_penalty: Penalty for repetition (optional).
        """
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # Get the appropriate tool formatter for this model
        self._tool_formatter = get_tool_formatter(model)
        self._is_openai_tools = isinstance(self._tool_formatter, OpenAIToolFormatter)
        formatter_name = "OpenAI" if self._is_openai_tools else "LFM2.5"
        logger.info(f"Using {formatter_name} tool formatter for model: {model}")

    @property
    def tool_formatter(self) -> ToolFormatter:
        """Get the tool formatter for this LLM."""
        return self._tool_formatter

    def _build_headers(self) -> dict:
        """Build HTTP headers for the request."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_messages(self, messages: list[dict], tools: list[dict] | None = None) -> list[dict]:
        """Build the full message list including system prompt and tool definitions."""
        full_messages = []

        # Build system prompt, potentially with tool definitions appended
        system_content = self.system_prompt or ""

        # For text-based tool formatters, add tool definitions to system prompt
        if tools and not self._is_openai_tools:
            formatted = self._tool_formatter.format_tools(tools)
            if formatted.system_prompt_addition:
                system_content = system_content + formatted.system_prompt_addition
                logger.debug(f"Added {len(tools)} tool(s) to system prompt for text-based formatter")

        # Prepend system prompt if we have any content
        if system_content:
            full_messages.append({
                "role": "system",
                "content": system_content
            })
            logger.debug(f"System prompt length: {len(system_content)} chars")

        full_messages.extend(messages)
        return full_messages

    def _build_request_body(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Build the request body for the API call."""
        body = {
            "model": self.model,
            "messages": self._build_messages(messages, tools),
        }

        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            body["temperature"] = self.temperature

        if self.top_p is not None:
            body["top_p"] = self.top_p

        if self.top_k is not None:
            body["top_k"] = self.top_k

        if self.repetition_penalty is not None:
            body["repetition_penalty"] = self.repetition_penalty

        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty

        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty

        # Add tools as API parameter only for OpenAI-compatible models
        if tools and self._is_openai_tools:
            formatted = self._tool_formatter.format_tools(tools)
            if formatted.tools_parameter:
                body["tools"] = formatted.tools_parameter
                body["tool_choice"] = "auto"

        return body

    async def chat(
        self,
        messages: list[dict],
        context: dict | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Generate a chat completion response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            context: Optional context dict (currently unused, for future extensions).
            tools: Optional list of tool definitions in OpenAI format.

        Returns:
            LLMResponse with the generated text, tool calls, and metadata.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.RequestError: If the request fails (network error, timeout, etc.).
        """
        headers = self._build_headers()
        body = self._build_request_body(messages, tools)

        # Log the request with context size
        user_message = messages[-1].get("content", "") if messages else ""
        context_count = len(messages) - 1  # Exclude current message
        logger.info(f'LLM request ({context_count} ctx msgs): "{user_message[:100]}..."' if len(user_message) > 100 else f'LLM request ({context_count} ctx msgs): "{user_message}"')
        if tools:
            logger.info(f"LLM request includes {len(tools)} tool(s): {[t.get('function', {}).get('name', '?') for t in tools]}")

        start_time = time.time()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                headers=headers,
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.time() - start_time) * 1000

        # Extract the response message
        message = data["choices"][0]["message"]
        content = message.get("content")
        finish_reason = data["choices"][0].get("finish_reason", "stop")

        # Debug: Log when content is empty but we got other fields
        if not content:
            logger.warning(f"LLM returned empty content. Message keys: {list(message.keys())}")
            # Some models (e.g., gpt-oss) return reasoning but empty content
            if message.get("reasoning"):
                logger.warning(f"Model returned reasoning but no content: {message.get('reasoning')[:200]}")

        # Parse tool calls - either from structured response or from text
        tool_calls = []

        # First check for OpenAI-style structured tool calls
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                try:
                    # Parse arguments from JSON string
                    args = tc["function"].get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)

                    tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args,
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse tool call: {e}")

        # For text-based tool formats (e.g., LFM2.5), parse from content
        if not tool_calls and content and tools:
            # Log raw content for debugging tool call detection
            if "<|tool_call" in content or "tool_call" in content.lower():
                logger.info(f"Potential tool call in response: {content[:200]}")
            text_tool_calls = self._tool_formatter.parse_tool_calls(content)
            if text_tool_calls:
                tool_calls = text_tool_calls
                # Strip tool call markup from content for cleaner output
                content = self._tool_formatter.strip_tool_calls(content)

        # Handle models that include <think>...</think> tags (e.g., Qwen3)
        # Strip out thinking content for cleaner TTS output
        if content and "<think>" in content and "</think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Log the response
        if tool_calls:
            logger.info(f'LLM response ({latency_ms:.0f}ms): {len(tool_calls)} tool call(s): {[tc.name for tc in tool_calls]}')
        elif content:
            logger.info(f'LLM response ({latency_ms:.0f}ms): "{content[:100]}..."' if len(content) > 100 else f'LLM response ({latency_ms:.0f}ms): "{content}"')
        else:
            logger.warning(f'LLM response ({latency_ms:.0f}ms): EMPTY (no content, no tool calls). finish_reason={finish_reason}')

        if latency_ms > 2000:
            logger.warning(f"LLM slow response: {latency_ms:.0f}ms (>2s)")

        return LLMResponse(
            content=content,
            model=data.get("model"),
            usage=data.get("usage"),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def chat_stream(
        self,
        messages: list[dict],
        context: dict | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        Note: Tool calling is not fully supported in streaming mode.
        If tools are provided, they will be passed to the API but
        tool calls may not be properly handled. Use non-streaming
        chat() for reliable tool execution.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            context: Optional context dict (unused).
            tools: Optional tools (limited support in streaming).

        Yields:
            Text chunks as they arrive from the API.
        """
        headers = self._build_headers()
        body = self._build_request_body(messages, tools)
        body["stream"] = True

        # Track if we're inside a <think> block (for models like Qwen3)
        in_think_block = False

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.endpoint,
                headers=headers,
                json=body,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Strip "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")

                        if not content:
                            continue

                        # Filter out <think>...</think> blocks for models like Qwen3
                        # Handle block start
                        if "<think>" in content:
                            in_think_block = True
                            # Emit any content before the think tag
                            pre_think = content.split("<think>")[0]
                            if pre_think:
                                yield pre_think
                            continue

                        # Handle block end
                        if "</think>" in content:
                            in_think_block = False
                            # Emit any content after the think tag
                            post_think = content.split("</think>")[-1]
                            if post_think:
                                yield post_think
                            continue

                        # Skip content inside think block
                        if in_think_block:
                            continue

                        yield content

                    except json.JSONDecodeError:
                        continue

    async def is_available(self) -> bool:
        """Check if the LLM service is available.

        Attempts a simple request to verify connectivity.

        Returns:
            True if the service responds successfully.
        """
        try:
            # Send a minimal request to check availability
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.endpoint,
                    headers=self._build_headers(),
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1,
                    },
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"OpenAIChatLLM(endpoint={self.endpoint!r}, model={self.model!r})"
