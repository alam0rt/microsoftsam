"""Tests for the web search tool.

Tests cover:
- Tool definition and schema
- Search execution
- Result parsing
- Error handling (timeouts, rate limits, etc.)
- DuckDuckGo HTML parsing
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mumble_voice_bot.tools.web_search import WebSearchTool


# --- Fixtures ---


@pytest.fixture
def web_search_tool():
    """Create a WebSearchTool instance."""
    return WebSearchTool(max_results=5, timeout=10.0)


@pytest.fixture
def mock_duckduckgo_html():
    """Sample DuckDuckGo HTML response."""
    return """
    <html>
    <body>
    <div class="result">
        <a class="result__a" href="https://example.com/page1">Example Page 1</a>
        <a class="result__snippet">This is the first search result snippet.</a>
    </div>
    <div class="result">
        <a class="result__a" href="https://example.com/page2">Example Page 2</a>
        <a class="result__snippet">Second result with more information.</a>
    </div>
    <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage3">Example Page 3</a>
        <a class="result__snippet">Third result from redirect URL.</a>
    </div>
    </body>
    </html>
    """


# --- Test Classes ---


class TestWebSearchToolDefinition:
    """Test tool definition and schema."""

    def test_tool_name(self, web_search_tool):
        """Test tool name property."""
        assert web_search_tool.name == "web_search"

    def test_tool_description(self, web_search_tool):
        """Test tool description."""
        description = web_search_tool.description
        assert "search" in description.lower()
        assert "web" in description.lower()

    def test_tool_parameters(self, web_search_tool):
        """Test tool parameters schema."""
        params = web_search_tool.parameters
        
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "count" in params["properties"]
        assert "query" in params["required"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"

    def test_to_schema(self, web_search_tool):
        """Test full schema generation."""
        schema = web_search_tool.to_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "web_search"
        assert "parameters" in schema["function"]


class TestWebSearchToolExecution:
    """Test search execution."""

    @pytest.mark.asyncio
    async def test_execute_returns_results(self, web_search_tool, mock_duckduckgo_html):
        """Test successful search execution."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = mock_duckduckgo_html
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="test query")

            assert "Search results for: test query" in result
            assert "Example Page" in result
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_count(self, web_search_tool, mock_duckduckgo_html):
        """Test search with specific count."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = mock_duckduckgo_html
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="test", count=2)

            # Should have at most 2 numbered results
            lines = result.split("\n")
            numbered_lines = [l for l in lines if l.strip().startswith(("1.", "2.", "3."))]
            assert len(numbered_lines) <= 2

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, web_search_tool):
        """Test search with empty query still works (API will handle it)."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = "<html></html>"  # No results
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="")

            assert "No results found" in result

    @pytest.mark.asyncio
    async def test_execute_count_clamped_minimum(self, web_search_tool, mock_duckduckgo_html):
        """Test that count is clamped to minimum of 1."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = mock_duckduckgo_html
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="test", count=0)

            # Should still get at least 1 result
            assert "1." in result

    @pytest.mark.asyncio
    async def test_execute_count_clamped_maximum(self, web_search_tool, mock_duckduckgo_html):
        """Test that count is clamped to maximum of 10."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = mock_duckduckgo_html
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Request 100, should be clamped to 10
            result = await web_search_tool.execute(query="test", count=100)

            # Tool should not fail
            assert "Search results" in result


class TestWebSearchErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_execute_timeout(self, web_search_tool):
        """Test timeout error handling."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Request timed out")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="test query")

            assert "Error" in result
            assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_http_error(self, web_search_tool):
        """Test HTTP error handling."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="test")

            assert "Error" in result

    @pytest.mark.asyncio
    async def test_execute_connection_error(self, web_search_tool):
        """Test connection error handling."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="test")

            assert "Error" in result

    @pytest.mark.asyncio
    async def test_execute_no_results(self, web_search_tool):
        """Test handling when no results are found."""
        empty_html = "<html><body></body></html>"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = empty_html
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="xyzzy12345nonexistent")

            assert "No results found" in result


class TestDuckDuckGoHTMLParsing:
    """Test DuckDuckGo HTML parsing."""

    def test_parse_basic_results(self, web_search_tool, mock_duckduckgo_html):
        """Test parsing basic HTML results."""
        results = web_search_tool._parse_duckduckgo_html(mock_duckduckgo_html, 10)

        assert len(results) >= 2
        assert results[0]["title"] == "Example Page 1"
        assert results[0]["url"] == "https://example.com/page1"
        assert "first search result" in results[0]["snippet"]

    def test_parse_redirect_urls(self, web_search_tool):
        """Test parsing of DuckDuckGo redirect URLs."""
        html = """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftest">Test Page</a>
        <a class="result__snippet">Test snippet</a>
        """
        results = web_search_tool._parse_duckduckgo_html(html, 10)

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/test"

    def test_parse_limits_results(self, web_search_tool, mock_duckduckgo_html):
        """Test that parsing respects num_results limit."""
        results = web_search_tool._parse_duckduckgo_html(mock_duckduckgo_html, 1)

        assert len(results) == 1

    def test_parse_empty_html(self, web_search_tool):
        """Test parsing empty HTML."""
        results = web_search_tool._parse_duckduckgo_html("", 10)
        assert results == []

    def test_parse_strips_html_from_snippet(self, web_search_tool):
        """Test that HTML tags are stripped from snippets."""
        html = """
        <a class="result__a" href="https://example.com">Test</a>
        <a class="result__snippet">Text with <b>bold</b> and <i>italic</i></a>
        """
        results = web_search_tool._parse_duckduckgo_html(html, 10)

        assert len(results) == 1
        assert "<b>" not in results[0]["snippet"]
        assert "bold" in results[0]["snippet"]


class TestResultFormatting:
    """Test result formatting for LLM consumption."""

    @pytest.mark.asyncio
    async def test_result_format(self, web_search_tool, mock_duckduckgo_html):
        """Test that results are formatted properly for LLM."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = mock_duckduckgo_html
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await web_search_tool.execute(query="test query")

            # Check format includes header
            assert "Search results for: test query" in result

            # Check numbered results
            assert "1." in result

            # Check URLs are present
            assert "http" in result


class TestToolConfiguration:
    """Test tool configuration options."""

    def test_custom_max_results(self):
        """Test custom max_results setting."""
        tool = WebSearchTool(max_results=3)
        assert tool.max_results == 3

    def test_custom_timeout(self):
        """Test custom timeout setting."""
        tool = WebSearchTool(timeout=5.0)
        assert tool.timeout == 5.0

    def test_default_settings(self):
        """Test default settings."""
        tool = WebSearchTool()
        assert tool.max_results == 5
        assert tool.timeout == 10.0


class TestWebSearchIntegration:
    """Integration tests with tool registry."""

    def test_tool_can_be_registered(self, web_search_tool):
        """Test that tool can be registered in registry."""
        from mumble_voice_bot.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(web_search_tool)

        assert registry.has("web_search")
        assert registry.get("web_search") is web_search_tool

    def test_tool_definition_in_registry(self, web_search_tool):
        """Test tool definition through registry."""
        from mumble_voice_bot.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(web_search_tool)

        definitions = registry.get_definitions()
        assert len(definitions) == 1
        assert definitions[0]["function"]["name"] == "web_search"

    @pytest.mark.asyncio
    async def test_tool_execution_through_registry(
        self, web_search_tool, mock_duckduckgo_html
    ):
        """Test executing tool through registry."""
        from mumble_voice_bot.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(web_search_tool)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = mock_duckduckgo_html
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await registry.execute("web_search", {"query": "test"})

            assert "Search results" in result
