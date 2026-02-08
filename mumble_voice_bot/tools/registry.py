"""Tool registry for dynamic tool management."""

from typing import Any

from mumble_voice_bot.logging_config import get_logger
from mumble_voice_bot.tools.base import Tool

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for agent tools.

    Allows dynamic registration, lookup, and execution of tools.
    Tools are identified by their unique name property.

    Example:
        registry = ToolRegistry()
        registry.register(WebSearchTool(api_key="..."))

        # Get tool definitions for LLM
        tools = registry.get_definitions()

        # Execute a tool
        result = await registry.execute("web_search", {"query": "latest news"})
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register.
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Name of the tool to remove.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")

    def get(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Name of the tool.

        Returns:
            The tool instance, or None if not found.
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Name of the tool.

        Returns:
            True if the tool exists in the registry.
        """
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format.

        Returns:
            List of tool schemas suitable for the OpenAI tools parameter.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters.

        Args:
            name: Tool name.
            params: Tool parameters.

        Returns:
            Tool execution result as string.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        try:
            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)

            # Execute the tool
            logger.info(f"Executing tool: {name}")
            result = await tool.execute(**params)
            logger.debug(f"Tool {name} result: {result[:200]}..." if len(result) > 200 else f"Tool {name} result: {result}")
            return result

        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return f"Error executing {name}: {str(e)}"

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"
