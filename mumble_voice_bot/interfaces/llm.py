"""Abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ToolCall:
    """A tool call request from the LLM.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool to execute.
        arguments: Arguments to pass to the tool.
    """
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM chat completion.

    Attributes:
        content: The generated text response (may be None if tool_calls present).
        model: The model that generated the response (if available).
        usage: Token usage statistics (if available).
        tool_calls: List of tool calls requested by the LLM.
        finish_reason: Why the response ended ("stop", "tool_calls", etc).
    """
    content: str | None
    model: str | None = None
    usage: dict | None = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM implementations (OpenAI, Ollama, vLLM, etc.) should inherit
    from this class and implement the chat() method.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        context: dict | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Generate a response from a conversation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles are typically 'system', 'user', 'assistant', or 'tool'.
            context: Optional context dict for provider-specific options.
            tools: Optional list of tool definitions in OpenAI format.
                   When provided, the LLM may return tool_calls instead of content.

        Returns:
            LLMResponse containing the generated text, tool calls, and metadata.

        Raises:
            Exception: If the LLM request fails.
        """
        pass

    async def chat_stream(
        self,
        messages: list[dict],
        context: dict | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        Default implementation falls back to non-streaming.
        Subclasses should override this for true streaming support.

        Note: Tool calls are not supported in streaming mode. If tools
        are provided and the LLM wants to call them, you'll get incomplete
        results. Use non-streaming chat() for tool-enabled conversations.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            context: Optional context dict for provider-specific options.
            tools: Optional list of tool definitions (may not work with streaming).

        Yields:
            Text chunks as they arrive from the API.
        """
        response = await self.chat(messages, context, tools)
        if response.content:
            yield response.content

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the LLM service is available.

        Returns:
            True if the service is reachable and responding.
        """
        pass
