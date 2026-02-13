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


# =============================================================================
# Phase 5/6: Rate limiting, allowlisting, audit logging
# =============================================================================


class TestToolAllowlisting:
    """Tests for per-soul tool allowlisting."""

    def test_allowlist_filters_definitions(self):
        """Test that get_definitions respects allowlist."""
        registry = ToolRegistry(allowed_tools=["tool1"])
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))

        defs = registry.get_definitions()
        names = [d["function"]["name"] for d in defs]
        assert "tool1" in names
        assert "tool2" not in names

    def test_no_allowlist_returns_all(self):
        """Test that None allowlist returns all tools."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))

        defs = registry.get_definitions()
        assert len(defs) == 2

    @pytest.mark.asyncio
    async def test_blocked_tool_returns_error(self):
        """Test that executing a non-allowed tool returns an error."""
        registry = ToolRegistry(allowed_tools=["tool1"])
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))

        result = await registry.execute("tool2", {"query": "test"})
        assert "not allowed" in result

    def test_set_allowed_tools(self):
        """Test dynamic allowlist update."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))

        assert len(registry.get_definitions()) == 2

        registry.set_allowed_tools(["tool1"])
        assert len(registry.get_definitions()) == 1

        registry.set_allowed_tools(None)
        assert len(registry.get_definitions()) == 2


class TestToolRateLimiting:
    """Tests for per-user rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_after_max(self):
        """Test that rate limit blocks after max calls."""
        registry = ToolRegistry(rate_limit_per_minute=3)
        registry.register(MockTool("search"))

        # First 3 should succeed
        for i in range(3):
            result = await registry.execute("search", {"query": f"test{i}"}, user_id="sam")
            assert "mock result" in result

        # 4th should be rate limited
        result = await registry.execute("search", {"query": "test4"}, user_id="sam")
        assert "Rate limit" in result

    @pytest.mark.asyncio
    async def test_rate_limit_per_user(self):
        """Test that rate limits are per-user."""
        registry = ToolRegistry(rate_limit_per_minute=2)
        registry.register(MockTool("search"))

        # User A uses 2 calls
        await registry.execute("search", {"query": "a1"}, user_id="alice")
        await registry.execute("search", {"query": "a2"}, user_id="alice")

        # User A is rate limited
        result = await registry.execute("search", {"query": "a3"}, user_id="alice")
        assert "Rate limit" in result

        # User B is not affected
        result = await registry.execute("search", {"query": "b1"}, user_id="bob")
        assert "mock result" in result

    @pytest.mark.asyncio
    async def test_no_rate_limit_without_user_id(self):
        """Test that rate limiting doesn't apply without user_id."""
        registry = ToolRegistry(rate_limit_per_minute=1)
        registry.register(MockTool("search"))

        # Without user_id, no rate limiting
        for i in range(5):
            result = await registry.execute("search", {"query": f"test{i}"})
            assert "mock result" in result


class TestToolAuditLogging:
    """Tests for tool invocation audit logging."""

    @pytest.mark.asyncio
    async def test_audit_log_records_calls(self):
        """Test that audit log records tool invocations."""
        registry = ToolRegistry()
        registry.register(MockTool("search"))

        await registry.execute("search", {"query": "test"}, user_id="sam")

        log = registry.get_audit_log()
        assert len(log) == 1
        assert log[0]["tool"] == "search"
        assert log[0]["user"] == "sam"
        assert not log[0]["blocked"]
        assert not log[0]["error"]

    @pytest.mark.asyncio
    async def test_audit_log_records_blocked(self):
        """Test that audit log records blocked calls."""
        registry = ToolRegistry(allowed_tools=["other"])
        registry.register(MockTool("search"))

        await registry.execute("search", {"query": "test"}, user_id="sam")

        log = registry.get_audit_log()
        assert len(log) == 1
        assert log[0]["blocked"]

    @pytest.mark.asyncio
    async def test_audit_log_limit(self):
        """Test audit log size limit."""
        registry = ToolRegistry()
        registry._audit_max_entries = 5
        registry.register(MockTool("search"))

        for i in range(10):
            await registry.execute("search", {"query": f"test{i}"})

        log = registry.get_audit_log()
        assert len(log) == 5
