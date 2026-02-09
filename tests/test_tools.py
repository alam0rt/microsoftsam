"""Tests for the tool system."""

from typing import Any

import pytest

from mumble_voice_bot.tools import Tool, ToolRegistry


class MockTool(Tool):
    """A simple mock tool for testing."""

    def __init__(self, name: str = "mock_tool", return_value: str = "mock result"):
        self._name = name
        self._return_value = return_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A mock tool for testing."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Test query",
                    "minLength": 1,
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, count: int = 5, **kwargs: Any) -> str:
        return f"{self._return_value}: {query} (count={count})"


class TestTool:
    """Test the Tool base class."""

    def test_to_schema(self):
        """Test tool schema generation."""
        tool = MockTool()
        schema = tool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mock_tool"
        assert schema["function"]["description"] == "A mock tool for testing."
        assert "properties" in schema["function"]["parameters"]

    def test_validate_params_valid(self):
        """Test parameter validation with valid params."""
        tool = MockTool()
        errors = tool.validate_params({"query": "test", "count": 5})
        assert errors == []

    def test_validate_params_missing_required(self):
        """Test parameter validation with missing required field."""
        tool = MockTool()
        errors = tool.validate_params({"count": 5})
        assert len(errors) == 1
        assert "required" in errors[0]

    def test_validate_params_type_error(self):
        """Test parameter validation with wrong type."""
        tool = MockTool()
        errors = tool.validate_params({"query": "test", "count": "not a number"})
        assert len(errors) == 1
        assert "integer" in errors[0]

    def test_validate_params_range_error(self):
        """Test parameter validation with out-of-range value."""
        tool = MockTool()
        errors = tool.validate_params({"query": "test", "count": 100})
        assert len(errors) == 1
        assert "maximum" in errors[0] or "<=" in errors[0]

    def test_validate_params_min_length(self):
        """Test parameter validation with too-short string."""
        tool = MockTool()
        errors = tool.validate_params({"query": ""})
        assert len(errors) == 1
        assert "minLength" in errors[0] or "at least" in errors[0]


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()
        tool = MockTool("test_tool")
        registry.register(tool)

        assert registry.has("test_tool")
        assert registry.get("test_tool") is tool
        assert "test_tool" in registry

    def test_unregister(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = MockTool("test_tool")
        registry.register(tool)
        registry.unregister("test_tool")

        assert not registry.has("test_tool")
        assert registry.get("test_tool") is None

    def test_get_definitions(self):
        """Test getting all tool definitions."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))

        definitions = registry.get_definitions()
        assert len(definitions) == 2
        names = [d["function"]["name"] for d in definitions]
        assert "tool1" in names
        assert "tool2" in names

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        registry.register(MockTool("search", "Found"))

        result = await registry.execute("search", {"query": "test query"})
        assert "Found" in result
        assert "test query" in result

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        """Test executing non-existent tool."""
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {"query": "test"})
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_execute_invalid_params(self):
        """Test executing tool with invalid parameters."""
        registry = ToolRegistry()
        registry.register(MockTool("search"))

        result = await registry.execute("search", {"count": 5})  # Missing required 'query'
        assert "Error" in result
        assert "Invalid parameters" in result

    def test_tool_names(self):
        """Test getting list of tool names."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))

        names = registry.tool_names
        assert "tool1" in names
        assert "tool2" in names

    def test_len(self):
        """Test registry length."""
        registry = ToolRegistry()
        assert len(registry) == 0

        registry.register(MockTool("tool1"))
        assert len(registry) == 1

        registry.register(MockTool("tool2"))
        assert len(registry) == 2
