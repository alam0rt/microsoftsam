"""Web search tool using DuckDuckGo.

This tool provides web search capabilities without requiring an API key,
using DuckDuckGo's instant answer API and HTML search.
"""

from typing import Any
from urllib.parse import quote_plus

import httpx

from mumble_voice_bot.logging_config import get_logger
from mumble_voice_bot.tools.base import Tool

logger = get_logger(__name__)

# User agent to avoid blocks
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo.

    This tool uses DuckDuckGo's HTML search to find web results.
    No API key required, but may be rate-limited on heavy use.

    Args:
        max_results: Maximum number of results to return (default: 5).
        timeout: HTTP request timeout in seconds (default: 10).
    """

    def __init__(self, max_results: int = 5, timeout: float = 10.0):
        self.max_results = max_results
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Use this when you need to look up "
            "current events, facts, news, or any information you don't know. "
            "Returns titles, URLs, and snippets from search results."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific for better results.",
                    "minLength": 1,
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                }
            },
            "required": ["query"]
        }

    @property
    def example_call(self) -> str:
        return 'web_search(query="latest news about topic")'

    @property
    def usage_hint(self) -> str:
        return "Use when asked to search, look up, or find information online"

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        """Execute web search.

        Args:
            query: Search query.
            count: Number of results (defaults to max_results).

        Returns:
            Formatted search results or error message.
        """
        n = min(max(count or self.max_results, 1), 10)

        try:
            results = await self._search_duckduckgo(query, n)

            if not results:
                return f"No results found for: {query}"

            # Format results for LLM
            lines = [f"Search results for: {query}\n"]
            for i, result in enumerate(results[:n], 1):
                lines.append(f"{i}. {result['title']}")
                lines.append(f"   {result['url']}")
                if result.get('snippet'):
                    lines.append(f"   {result['snippet']}")
                lines.append("")

            return "\n".join(lines)

        except httpx.TimeoutException:
            return f"Error: Search timed out for: {query}"
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Error: Search failed - {str(e)}"

    async def _search_duckduckgo(self, query: str, num_results: int) -> list[dict]:
        """Search DuckDuckGo and parse results.

        Uses DuckDuckGo HTML search which doesn't require an API key.
        """
        # Use DuckDuckGo HTML endpoint
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            html = response.text

        return self._parse_duckduckgo_html(html, num_results)

    def _parse_duckduckgo_html(self, html: str, num_results: int) -> list[dict]:
        """Parse DuckDuckGo HTML search results.

        This is a simple parser that extracts result links and snippets
        without requiring a full HTML parsing library.
        """
        results = []

        # Find result blocks - they're in <div class="result ...">
        # Each result has <a class="result__a" href="...">title</a>
        # and <a class="result__snippet">snippet</a>

        import re

        # Pattern for result links
        link_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>',
            re.IGNORECASE
        )

        # Pattern for snippets
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>([^<]*(?:<[^>]*>[^<]*)*)</a>',
            re.IGNORECASE
        )

        # Find all links
        links = link_pattern.findall(html)

        # Find all snippets
        snippets = snippet_pattern.findall(html)

        for i, (url, title) in enumerate(links[:num_results]):
            # Clean up URL (DuckDuckGo uses redirect URLs)
            if "uddg=" in url:
                # Extract actual URL from redirect
                match = re.search(r'uddg=([^&]*)', url)
                if match:
                    from urllib.parse import unquote
                    url = unquote(match.group(1))

            # Get corresponding snippet
            snippet = ""
            if i < len(snippets):
                # Clean HTML tags from snippet
                snippet = re.sub(r'<[^>]+>', '', snippets[i])
                snippet = snippet.strip()

            # Clean up title
            title = title.strip()

            if title and url:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })

        return results
