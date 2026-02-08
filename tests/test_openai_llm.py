"""Tests for OpenAI-compatible LLM provider.

Tests cover:
- Constructor and configuration
- Header and request building
- Chat completion (non-streaming)
- Streaming
- Tool call handling for both OpenAI and LFM2.5 formats
- Error handling
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mumble_voice_bot.interfaces.llm import LLMResponse, ToolCall
from mumble_voice_bot.interfaces.tool_formatter import (
    LFM25ToolFormatter,
    OpenAIToolFormatter,
)
from mumble_voice_bot.providers.openai_llm import OpenAIChatLLM


# --- Fixtures ---


@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        },
    ]


@pytest.fixture
def mock_openai_response():
    """Mock response from OpenAI-style API."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


@pytest.fixture
def mock_openai_tool_call_response():
    """Mock response with tool calls from OpenAI-style API."""
    return {
        "id": "chatcmpl-456",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query": "weather in London"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
    }


@pytest.fixture
def mock_lfm25_tool_call_response():
    """Mock response with LFM2.5-style tool calls in text."""
    return {
        "id": "chatcmpl-789",
        "model": "liquid/lfm-2.5-1.2b-instruct:free",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '<|tool_call_start|>[web_search(query="current news")]<|tool_call_end|>',
                },
                "finish_reason": "stop",
            }
        ],
    }


# --- Test Classes ---


class TestOpenAIChatLLMInit:
    """Test OpenAIChatLLM constructor and configuration."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        llm = OpenAIChatLLM(
            endpoint="https://api.openai.com/v1/chat/completions",
            model="gpt-4o",
            api_key="sk-test-key",
            system_prompt="You are a helpful assistant.",
            timeout=60.0,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
        )

        assert llm.endpoint == "https://api.openai.com/v1/chat/completions"
        assert llm.model == "gpt-4o"
        assert llm.api_key == "sk-test-key"
        assert llm.system_prompt == "You are a helpful assistant."
        assert llm.timeout == 60.0
        assert llm.max_tokens == 1000
        assert llm.temperature == 0.7
        assert llm.top_p == 0.9
        assert llm.top_k == 40
        assert llm.repetition_penalty == 1.1

    def test_init_minimal_parameters(self):
        """Test initialization with minimal required parameters."""
        llm = OpenAIChatLLM(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="llama3.2:3b",
        )

        assert llm.endpoint == "http://localhost:11434/v1/chat/completions"
        assert llm.model == "llama3.2:3b"
        assert llm.api_key is None
        assert llm.system_prompt is None
        assert llm.timeout == 30.0
        assert llm.max_tokens is None
        assert llm.temperature is None

    def test_tool_formatter_auto_selection_openai(self):
        """Test that OpenAI models get OpenAIToolFormatter."""
        llm = OpenAIChatLLM(endpoint="http://test", model="gpt-4o")
        assert isinstance(llm.tool_formatter, OpenAIToolFormatter)
        assert llm._is_openai_tools is True

    def test_tool_formatter_auto_selection_lfm25(self):
        """Test that LFM2.5 models get LFM25ToolFormatter."""
        llm = OpenAIChatLLM(
            endpoint="http://test",
            model="liquid/lfm-2.5-1.2b-instruct:free",
        )
        assert isinstance(llm.tool_formatter, LFM25ToolFormatter)
        assert llm._is_openai_tools is False

    def test_repr(self):
        """Test string representation."""
        llm = OpenAIChatLLM(endpoint="http://test", model="gpt-4")
        assert "OpenAIChatLLM" in repr(llm)
        assert "http://test" in repr(llm)
        assert "gpt-4" in repr(llm)


class TestOpenAIChatLLMHeaders:
    """Test header building."""

    def test_build_headers_with_api_key(self):
        """Test headers include authorization when API key is set."""
        llm = OpenAIChatLLM(
            endpoint="http://test",
            model="gpt-4",
            api_key="sk-test-key",
        )
        headers = llm._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer sk-test-key"

    def test_build_headers_without_api_key(self):
        """Test headers without authorization when no API key."""
        llm = OpenAIChatLLM(endpoint="http://test", model="gpt-4")
        headers = llm._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers


class TestOpenAIChatLLMMessages:
    """Test message building."""

    def test_build_messages_with_system_prompt(self):
        """Test messages include system prompt when set."""
        llm = OpenAIChatLLM(
            endpoint="http://test",
            model="gpt-4",
            system_prompt="You are a helpful assistant.",
        )
        messages = [{"role": "user", "content": "Hello"}]
        built = llm._build_messages(messages)

        assert len(built) == 2
        assert built[0]["role"] == "system"
        assert built[0]["content"] == "You are a helpful assistant."
        assert built[1]["role"] == "user"

    def test_build_messages_without_system_prompt(self):
        """Test messages without system prompt."""
        llm = OpenAIChatLLM(endpoint="http://test", model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]
        built = llm._build_messages(messages)

        assert len(built) == 1
        assert built[0]["role"] == "user"

    def test_build_messages_with_tools_openai_style(self, sample_tools):
        """Test OpenAI-style tools don't modify system prompt."""
        llm = OpenAIChatLLM(
            endpoint="http://test",
            model="gpt-4",
            system_prompt="Base prompt.",
        )
        messages = [{"role": "user", "content": "Search for news"}]
        built = llm._build_messages(messages, tools=sample_tools)

        # OpenAI style: tools go to API parameter, not system prompt
        assert len(built) == 2
        assert built[0]["role"] == "system"
        assert built[0]["content"] == "Base prompt."

    def test_build_messages_with_tools_lfm25_style(self, sample_tools):
        """Test LFM2.5-style tools are added to system prompt."""
        llm = OpenAIChatLLM(
            endpoint="http://test",
            model="liquid/lfm-2.5-1.2b-instruct:free",
            system_prompt="Base prompt.",
        )
        messages = [{"role": "user", "content": "Search for news"}]
        built = llm._build_messages(messages, tools=sample_tools)

        # LFM2.5 style: tools added to system prompt
        assert len(built) == 2
        assert built[0]["role"] == "system"
        assert "web_search" in built[0]["content"]
        assert "Base prompt." in built[0]["content"]


class TestOpenAIChatLLMRequestBody:
    """Test request body building."""

    def test_build_request_body_basic(self):
        """Test basic request body."""
        llm = OpenAIChatLLM(endpoint="http://test", model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]
        body = llm._build_request_body(messages)

        assert body["model"] == "gpt-4"
        assert "messages" in body
        assert "tools" not in body

    def test_build_request_body_with_parameters(self):
        """Test request body with optional parameters."""
        llm = OpenAIChatLLM(
            endpoint="http://test",
            model="gpt-4",
            max_tokens=500,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
        )
        messages = [{"role": "user", "content": "Hello"}]
        body = llm._build_request_body(messages)

        assert body["max_tokens"] == 500
        assert body["temperature"] == 0.8
        assert body["top_p"] == 0.95
        assert body["top_k"] == 50
        assert body["repetition_penalty"] == 1.2

    def test_build_request_body_with_tools_openai(self, sample_tools):
        """Test request body with OpenAI-style tools."""
        llm = OpenAIChatLLM(endpoint="http://test", model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]
        body = llm._build_request_body(messages, tools=sample_tools)

        assert "tools" in body
        assert len(body["tools"]) == 2
        assert body["tool_choice"] == "auto"

    def test_build_request_body_with_tools_lfm25(self, sample_tools):
        """Test request body with LFM2.5 - tools not in body."""
        llm = OpenAIChatLLM(
            endpoint="http://test",
            model="liquid/lfm-2.5-1.2b-instruct:free",
        )
        messages = [{"role": "user", "content": "Hello"}]
        body = llm._build_request_body(messages, tools=sample_tools)

        # LFM2.5 doesn't use tools parameter
        assert "tools" not in body


class TestOpenAIChatLLMChat:
    """Test chat completion."""

    @pytest.mark.asyncio
    async def test_chat_simple_response(self, mock_openai_response):
        """Test simple chat completion."""
        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "Hello"}]
            response = await llm.chat(messages)

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! How can I help you today?"
            assert response.model == "gpt-4"
            assert response.finish_reason == "stop"
            assert not response.has_tool_calls

    @pytest.mark.asyncio
    async def test_chat_with_tools_no_tool_call(self, mock_openai_response, sample_tools):
        """Test chat with tools available but no tool call made."""
        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "Hello"}]
            response = await llm.chat(messages, tools=sample_tools)

            assert response.content == "Hello! How can I help you today?"
            assert not response.has_tool_calls

    @pytest.mark.asyncio
    async def test_chat_with_tool_call_openai_format(
        self, mock_openai_tool_call_response, sample_tools
    ):
        """Test chat with OpenAI-format tool calls."""
        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_tool_call_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "What's the weather in London?"}]
            response = await llm.chat(messages, tools=sample_tools)

            assert response.has_tool_calls
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "web_search"
            assert response.tool_calls[0].arguments == {"query": "weather in London"}
            assert response.tool_calls[0].id == "call_abc123"
            assert response.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_chat_with_tool_call_lfm25_format(
        self, mock_lfm25_tool_call_response, sample_tools
    ):
        """Test chat with LFM2.5-format tool calls in text."""
        llm = OpenAIChatLLM(
            endpoint="http://test/v1/chat/completions",
            model="liquid/lfm-2.5-1.2b-instruct:free",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_lfm25_tool_call_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "Search for news"}]
            response = await llm.chat(messages, tools=sample_tools)

            assert response.has_tool_calls
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "web_search"
            assert response.tool_calls[0].arguments == {"query": "current news"}

    @pytest.mark.asyncio
    async def test_chat_multiple_tool_calls(self, sample_tools):
        """Test chat with multiple tool calls."""
        multi_tool_response = {
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "news"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = multi_tool_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "Get news and weather"}]
            response = await llm.chat(messages, tools=sample_tools)

            assert response.has_tool_calls
            assert len(response.tool_calls) == 2
            assert response.tool_calls[0].name == "web_search"
            assert response.tool_calls[1].name == "get_weather"

    @pytest.mark.asyncio
    async def test_chat_strips_think_tags(self):
        """Test that <think>...</think> tags are stripped from response."""
        response_with_think = {
            "model": "qwen3",
            "choices": [
                {
                    "message": {
                        "content": "<think>Let me think about this...</think>The answer is 42.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="qwen3")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = response_with_think
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "What is the meaning of life?"}]
            response = await llm.chat(messages)

            assert response.content == "The answer is 42."
            assert "<think>" not in response.content

    @pytest.mark.asyncio
    async def test_chat_http_error_handling(self):
        """Test HTTP error is propagated correctly."""
        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "Hello"}]
            with pytest.raises(httpx.HTTPStatusError):
                await llm.chat(messages)

    @pytest.mark.asyncio
    async def test_chat_timeout_handling(self):
        """Test timeout error is propagated correctly."""
        llm = OpenAIChatLLM(
            endpoint="http://test/v1/chat/completions", model="gpt-4", timeout=1.0
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "Hello"}]
            with pytest.raises(httpx.TimeoutException):
                await llm.chat(messages)

    @pytest.mark.asyncio
    async def test_chat_invalid_json_tool_call(self, sample_tools):
        """Test graceful handling of invalid JSON in tool call arguments."""
        invalid_tool_response = {
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": "not valid json",
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = invalid_tool_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "Search"}]
            # Should not raise, but tool call should be skipped
            response = await llm.chat(messages, tools=sample_tools)
            assert len(response.tool_calls) == 0


class TestOpenAIChatLLMStreaming:
    """Test streaming chat completion."""

    @pytest.mark.asyncio
    async def test_chat_stream_basic(self):
        """Test basic streaming response."""
        from contextlib import asynccontextmanager
        
        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        # Simulate SSE stream
        async def mock_aiter_lines():
            chunks = [
                'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                'data: {"choices": [{"delta": {"content": " world"}}]}',
                'data: {"choices": [{"delta": {"content": "!"}}]}',
                "data: [DONE]",
            ]
            for chunk in chunks:
                yield chunk

        mock_stream_response = MagicMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_lines = mock_aiter_lines

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield mock_stream_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.stream = mock_stream

            @asynccontextmanager
            async def mock_client_cm():
                yield mock_client

            mock_client_class.return_value = mock_client_cm()

            messages = [{"role": "user", "content": "Hello"}]
            chunks = []
            async for chunk in llm.chat_stream(messages):
                chunks.append(chunk)

            assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_chat_stream_filters_think_blocks(self):
        """Test streaming filters out <think>...</think> blocks."""
        from contextlib import asynccontextmanager
        
        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="qwen3")

        async def mock_aiter_lines():
            chunks = [
                'data: {"choices": [{"delta": {"content": "<think>"}}]}',
                'data: {"choices": [{"delta": {"content": "thinking..."}}]}',
                'data: {"choices": [{"delta": {"content": "</think>"}}]}',
                'data: {"choices": [{"delta": {"content": "Hello!"}}]}',
                "data: [DONE]",
            ]
            for chunk in chunks:
                yield chunk

        mock_stream_response = MagicMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_lines = mock_aiter_lines

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield mock_stream_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.stream = mock_stream

            @asynccontextmanager
            async def mock_client_cm():
                yield mock_client

            mock_client_class.return_value = mock_client_cm()

            messages = [{"role": "user", "content": "Hello"}]
            chunks = []
            async for chunk in llm.chat_stream(messages):
                chunks.append(chunk)

            # Should only have content after think block
            assert "Hello!" in chunks
            assert "thinking..." not in chunks


class TestOpenAIChatLLMAvailability:
    """Test availability checking."""

    @pytest.mark.asyncio
    async def test_is_available_success(self):
        """Test availability check returns True on success."""
        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await llm.is_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_failure(self):
        """Test availability check returns False on failure."""
        llm = OpenAIChatLLM(endpoint="http://test/v1/chat/completions", model="gpt-4")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await llm.is_available()
            assert result is False


class TestOpenAILLMIntegration:
    """Integration tests with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_full_conversation_with_tool_use(self, sample_tools):
        """Test a full conversation flow with tool execution."""
        llm = OpenAIChatLLM(
            endpoint="http://test/v1/chat/completions",
            model="gpt-4",
            system_prompt="You are a helpful assistant.",
        )

        # First call: LLM requests tool
        tool_request_response = {
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "latest news"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        # Second call: LLM responds with tool result
        final_response = {
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "content": "Based on the search results, here's what I found...",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        call_count = [0]

        def mock_json():
            call_count[0] += 1
            if call_count[0] == 1:
                return tool_request_response
            return final_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json = mock_json
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First turn: user asks, LLM requests tool
            messages = [{"role": "user", "content": "What's in the news?"}]
            response = await llm.chat(messages, tools=sample_tools)

            assert response.has_tool_calls
            assert response.tool_calls[0].name == "web_search"

            # Simulate tool execution and add result
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": []}
            )
            tool_result = llm.tool_formatter.format_tool_result(
                "call_1", "web_search", "News headline 1, News headline 2"
            )
            messages.append(tool_result)

            # Second turn: LLM responds with tool result incorporated
            response = await llm.chat(messages, tools=sample_tools)
            assert response.content is not None
            assert "search results" in response.content

    @pytest.mark.asyncio
    async def test_conversation_history_maintained(self, mock_openai_response):
        """Test that conversation history is correctly formatted."""
        llm = OpenAIChatLLM(
            endpoint="http://test/v1/chat/completions",
            model="gpt-4",
            system_prompt="You are helpful.",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Multi-turn conversation
            messages = [
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What's my name?"},
            ]

            await llm.chat(messages)

            # Verify the call included all messages plus system
            call_args = mock_client.post.call_args
            body = call_args.kwargs["json"]
            assert len(body["messages"]) == 4  # system + 3 conversation
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["content"] == "My name is Alice"
