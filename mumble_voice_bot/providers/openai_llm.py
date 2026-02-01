"""OpenAI-compatible LLM provider.

This provider works with any service that implements the OpenAI Chat Completions API:
- OpenAI API
- vLLM
- Ollama (with OpenAI compatibility)
- llama.cpp server
- LocalAI
- LiteLLM
"""

import httpx

from mumble_voice_bot.interfaces.llm import LLMProvider, LLMResponse


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
        """
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def _build_headers(self) -> dict:
        """Build HTTP headers for the request."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _build_messages(self, messages: list[dict]) -> list[dict]:
        """Build the full message list including system prompt."""
        full_messages = []
        
        # Prepend system prompt if configured
        if self.system_prompt:
            full_messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        full_messages.extend(messages)
        return full_messages
    
    def _build_request_body(self, messages: list[dict]) -> dict:
        """Build the request body for the API call."""
        body = {
            "model": self.model,
            "messages": self._build_messages(messages),
        }
        
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        
        if self.temperature is not None:
            body["temperature"] = self.temperature
        
        return body
    
    async def chat(
        self,
        messages: list[dict],
        context: dict | None = None
    ) -> LLMResponse:
        """Generate a chat completion response.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            context: Optional context dict (currently unused, for future extensions).
        
        Returns:
            LLMResponse with the generated text and metadata.
        
        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.RequestError: If the request fails (network error, timeout, etc.).
        """
        headers = self._build_headers()
        body = self._build_request_body(messages)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                headers=headers,
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        
        # Extract the response content
        content = data["choices"][0]["message"]["content"]
        
        # Handle models that include <think>...</think> tags (e.g., Qwen3)
        # Strip out thinking content for cleaner TTS output
        if "<think>" in content and "</think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        
        return LLMResponse(
            content=content,
            model=data.get("model"),
            usage=data.get("usage"),
        )
    
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
