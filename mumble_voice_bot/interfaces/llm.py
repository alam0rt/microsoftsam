"""Abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class LLMResponse:
    """Response from an LLM chat completion.
    
    Attributes:
        content: The generated text response.
        model: The model that generated the response (if available).
        usage: Token usage statistics (if available).
    """
    content: str
    model: str | None = None
    usage: dict | None = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM implementations (OpenAI, Ollama, vLLM, etc.) should inherit
    from this class and implement the chat() method.
    """
    
    @abstractmethod
    async def chat(
        self, 
        messages: list[dict], 
        context: dict | None = None
    ) -> LLMResponse:
        """Generate a response from a conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles are typically 'system', 'user', or 'assistant'.
            context: Optional context dict for provider-specific options.
        
        Returns:
            LLMResponse containing the generated text and metadata.
        
        Raises:
            Exception: If the LLM request fails.
        """
        pass
    
    async def chat_stream(
        self,
        messages: list[dict],
        context: dict | None = None
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.
        
        Default implementation falls back to non-streaming.
        Subclasses should override this for true streaming support.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            context: Optional context dict for provider-specific options.
        
        Yields:
            Text chunks as they arrive from the API.
        """
        response = await self.chat(messages, context)
        yield response.content
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the LLM service is available.
        
        Returns:
            True if the service is reachable and responding.
        """
        pass
