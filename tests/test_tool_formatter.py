"""Tests for tool call formatters."""

import pytest

from mumble_voice_bot.interfaces.llm import ToolCall
from mumble_voice_bot.interfaces.tool_formatter import (
    ToolFormatter,
    OpenAIToolFormatter,
    LFM25ToolFormatter,
    FormattedTools,
    get_tool_formatter,
)


# Sample tools in OpenAI format for testing
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return",
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["location"]
            }
        }
    }
]


class TestOpenAIToolFormatter:
    """Test the OpenAI-compatible tool formatter."""

    def test_format_tools_returns_parameter(self):
        """Test that tools are returned as API parameter."""
        formatter = OpenAIToolFormatter()
        result = formatter.format_tools(SAMPLE_TOOLS)

        assert result.tools_parameter == SAMPLE_TOOLS
        assert result.system_prompt_addition is None

    def test_parse_tool_calls_returns_empty(self):
        """Test that text parsing returns empty (OpenAI uses structured response)."""
        formatter = OpenAIToolFormatter()
        result = formatter.parse_tool_calls("Some response text with no tool calls")

        assert result == []

    def test_format_tool_result(self):
        """Test formatting tool result for OpenAI."""
        formatter = OpenAIToolFormatter()
        result = formatter.format_tool_result(
            tool_call_id="call_123",
            tool_name="web_search",
            result='{"results": ["result1", "result2"]}'
        )

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "web_search"
        assert "results" in result["content"]

    def test_strip_tool_calls_returns_unchanged(self):
        """Test that strip_tool_calls returns text unchanged for OpenAI."""
        formatter = OpenAIToolFormatter()
        text = "This is a response without tool markup."
        result = formatter.strip_tool_calls(text)

        assert result == text


class TestLFM25ToolFormatter:
    """Test the LFM2.5 tool formatter."""

    def test_format_tools_returns_system_prompt(self):
        """Test that tools are formatted into system prompt."""
        formatter = LFM25ToolFormatter()
        result = formatter.format_tools(SAMPLE_TOOLS)

        assert result.tools_parameter is None
        assert result.system_prompt_addition is not None
        assert "List of tools:" in result.system_prompt_addition
        assert "web_search" in result.system_prompt_addition
        assert "get_weather" in result.system_prompt_addition

    def test_parse_tool_calls_single(self):
        """Test parsing a single tool call."""
        formatter = LFM25ToolFormatter()
        response = '<|tool_call_start|>[web_search(query="python tutorials")]<|tool_call_end|>Let me search for that.'

        calls = formatter.parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"query": "python tutorials"}

    def test_parse_tool_calls_multiple_args(self):
        """Test parsing tool call with multiple arguments."""
        formatter = LFM25ToolFormatter()
        response = '<|tool_call_start|>[web_search(query="AI news", max_results=5)]<|tool_call_end|>'

        calls = formatter.parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].arguments["query"] == "AI news"
        assert calls[0].arguments["max_results"] == 5

    def test_parse_tool_calls_multiple_calls(self):
        """Test parsing multiple tool calls in one response."""
        formatter = LFM25ToolFormatter()
        response = '''<|tool_call_start|>[web_search(query="weather API")]<|tool_call_end|>
        <|tool_call_start|>[get_weather(location="London")]<|tool_call_end|>'''

        calls = formatter.parse_tool_calls(response)

        assert len(calls) == 2
        assert calls[0].name == "web_search"
        assert calls[1].name == "get_weather"
        assert calls[1].arguments["location"] == "London"

    def test_parse_tool_calls_with_single_quotes(self):
        """Test parsing tool call with single-quoted strings."""
        formatter = LFM25ToolFormatter()
        response = "<|tool_call_start|>[web_search(query='single quotes')]<|tool_call_end|>"

        calls = formatter.parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].arguments["query"] == "single quotes"

    def test_parse_tool_calls_with_numbers(self):
        """Test parsing tool call with numeric arguments."""
        formatter = LFM25ToolFormatter()
        response = '<|tool_call_start|>[some_func(count=42, ratio=3.14)]<|tool_call_end|>'

        calls = formatter.parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].arguments["count"] == 42
        assert calls[0].arguments["ratio"] == 3.14

    def test_parse_tool_calls_with_booleans(self):
        """Test parsing tool call with boolean arguments."""
        formatter = LFM25ToolFormatter()
        response = '<|tool_call_start|>[toggle(enabled=True, verbose=False)]<|tool_call_end|>'

        calls = formatter.parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].arguments["enabled"] is True
        assert calls[0].arguments["verbose"] is False

    def test_parse_tool_calls_no_args(self):
        """Test parsing tool call with no arguments."""
        formatter = LFM25ToolFormatter()
        response = '<|tool_call_start|>[list_tools()]<|tool_call_end|>'

        calls = formatter.parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].name == "list_tools"
        assert calls[0].arguments == {}

    def test_parse_tool_calls_empty_response(self):
        """Test parsing response with no tool calls."""
        formatter = LFM25ToolFormatter()
        response = "This is just a regular response without any tool calls."

        calls = formatter.parse_tool_calls(response)

        assert calls == []

    def test_parse_tool_calls_malformed_graceful(self):
        """Test that malformed tool calls are handled gracefully."""
        formatter = LFM25ToolFormatter()
        # Malformed - missing closing bracket
        response = '<|tool_call_start|>[broken_call(query="test"<|tool_call_end|>'

        calls = formatter.parse_tool_calls(response)

        # Should not crash, may return empty or partial results
        assert isinstance(calls, list)

    def test_format_tool_result(self):
        """Test formatting tool result for LFM2.5."""
        formatter = LFM25ToolFormatter()
        result = formatter.format_tool_result(
            tool_call_id="lfm_call_0",
            tool_name="web_search",
            result='[{"title": "Result 1", "url": "http://example.com"}]'
        )

        # LFM2.5 via OpenRouter uses 'user' role with context to ensure proper handling
        assert result["role"] == "user"
        assert "Result 1" in result["content"]
        assert "web_search" in result["content"]

    def test_strip_tool_calls(self):
        """Test stripping tool call markup from response."""
        formatter = LFM25ToolFormatter()
        response = '<|tool_call_start|>[web_search(query="test")]<|tool_call_end|>Let me search for that information.'

        stripped = formatter.strip_tool_calls(response)

        assert '<|tool_call_start|>' not in stripped
        assert '<|tool_call_end|>' not in stripped
        assert 'Let me search for that information.' in stripped

    def test_strip_tool_calls_multiple(self):
        """Test stripping multiple tool calls from response."""
        formatter = LFM25ToolFormatter()
        response = '''<|tool_call_start|>[func1()]<|tool_call_end|>Some text.
        <|tool_call_start|>[func2()]<|tool_call_end|>More text.'''

        stripped = formatter.strip_tool_calls(response)

        assert '<|tool_call_start|>' not in stripped
        assert 'Some text.' in stripped
        assert 'More text.' in stripped

    def test_strip_tool_calls_preserves_non_tool_content(self):
        """Test that non-tool content is preserved when stripping."""
        formatter = LFM25ToolFormatter()
        response = "Before call. <|tool_call_start|>[search(q=\"test\")]<|tool_call_end|> After call."

        stripped = formatter.strip_tool_calls(response)

        assert "Before call." in stripped
        assert "After call." in stripped


class TestGetToolFormatter:
    """Test the get_tool_formatter factory function."""

    def test_lfm25_model_detection(self):
        """Test that LFM2.5 models get the correct formatter."""
        formatter = get_tool_formatter("liquid/lfm-2.5-1.2b-instruct:free")
        assert isinstance(formatter, LFM25ToolFormatter)

    def test_lfm25_model_detection_variants(self):
        """Test LFM2.5 detection with various model name formats."""
        models = [
            "liquid/lfm-2.5-1.2b-instruct",
            "LiquidAI/LFM2.5-1.2B-Instruct",
            "lfm-2.5-thinking",
            "lfm2.5-base",
        ]
        for model in models:
            formatter = get_tool_formatter(model)
            assert isinstance(formatter, LFM25ToolFormatter), f"Failed for model: {model}"

    def test_openai_model_default(self):
        """Test that non-LFM models get OpenAI formatter."""
        formatter = get_tool_formatter("gpt-4o")
        assert isinstance(formatter, OpenAIToolFormatter)

    def test_openai_model_variants(self):
        """Test that various OpenAI-compatible models get correct formatter."""
        models = [
            "gpt-4o-mini",
            "claude-3-opus",
            "qwen/qwen3-32b",
            "meta-llama/llama-3.3-70b-instruct",
        ]
        for model in models:
            formatter = get_tool_formatter(model)
            assert isinstance(formatter, OpenAIToolFormatter), f"Failed for model: {model}"


class TestToolCallDataclass:
    """Test the ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        call = ToolCall(
            id="call_123",
            name="web_search",
            arguments={"query": "test"}
        )

        assert call.id == "call_123"
        assert call.name == "web_search"
        assert call.arguments == {"query": "test"}

    def test_tool_call_default_arguments(self):
        """Test ToolCall with empty arguments dict."""
        call = ToolCall(id="call_456", name="no_args", arguments={})

        assert call.arguments == {}


class TestFormattedToolsDataclass:
    """Test the FormattedTools dataclass."""

    def test_default_values(self):
        """Test FormattedTools with default values."""
        result = FormattedTools()

        assert result.system_prompt_addition is None
        assert result.tools_parameter is None

    def test_with_system_prompt(self):
        """Test FormattedTools with system prompt addition."""
        result = FormattedTools(system_prompt_addition="Tool list here")

        assert result.system_prompt_addition == "Tool list here"
        assert result.tools_parameter is None

    def test_with_tools_parameter(self):
        """Test FormattedTools with tools parameter."""
        tools = [{"type": "function", "function": {"name": "test"}}]
        result = FormattedTools(tools_parameter=tools)

        assert result.system_prompt_addition is None
        assert result.tools_parameter == tools
