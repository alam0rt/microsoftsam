"""Tests for the souls tool.

Tests cover:
- Tool definition and schema
- Listing available souls
- Switching souls
- Getting current soul
- Error handling
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mumble_voice_bot.tools.souls import SoulsTool


# --- Fixtures ---


@pytest.fixture
def temp_souls_dir(tmp_path):
    """Create a temporary souls directory with test souls."""
    souls_dir = tmp_path / "souls"
    souls_dir.mkdir()

    # Create first soul
    soul1_dir = souls_dir / "test_soul1"
    soul1_dir.mkdir()
    (soul1_dir / "soul.yaml").write_text(
        """
name: "Test Soul 1"
description: "First test personality"
author: "Test Author"
version: "1.0.0"
voice:
  ref_audio: "audio/"
  num_steps: 4
"""
    )
    (soul1_dir / "personality.md").write_text("You are a helpful assistant.")
    (soul1_dir / "audio").mkdir()

    # Create second soul
    soul2_dir = souls_dir / "test_soul2"
    soul2_dir.mkdir()
    (soul2_dir / "soul.yaml").write_text(
        """
name: "Test Soul 2"
description: "Second test personality"
"""
    )
    (soul2_dir / "personality.md").write_text("You are a pirate.")

    # Create soul with minimal config
    soul3_dir = souls_dir / "minimal_soul"
    soul3_dir.mkdir()
    (soul3_dir / "soul.yaml").write_text(
        """
name: "Minimal"
"""
    )

    # Create invalid soul (no soul.yaml)
    invalid_dir = souls_dir / "invalid_soul"
    invalid_dir.mkdir()
    # No soul.yaml file

    return souls_dir


@pytest.fixture
def souls_tool(temp_souls_dir):
    """Create a SoulsTool with temp directory."""
    return SoulsTool(souls_dir=temp_souls_dir)


@pytest.fixture
def switch_callback():
    """Create a mock switch callback."""
    callback = AsyncMock(return_value="Switched successfully!")
    return callback


@pytest.fixture
def get_current_callback():
    """Create a mock get_current callback."""
    callback = MagicMock(return_value="test_soul1")
    return callback


# --- Test Classes ---


class TestSoulsToolDefinition:
    """Test tool definition and schema."""

    def test_tool_name(self, souls_tool):
        """Test tool name property."""
        assert souls_tool.name == "souls"

    def test_tool_description(self, souls_tool):
        """Test tool description."""
        description = souls_tool.description
        assert "personalities" in description.lower() or "souls" in description.lower()
        assert "list" in description.lower()
        assert "switch" in description.lower()

    def test_tool_parameters(self, souls_tool):
        """Test tool parameters schema."""
        params = souls_tool.parameters

        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "soul_name" in params["properties"]
        assert "action" in params["required"]

        # Check action enum
        action_enum = params["properties"]["action"]["enum"]
        assert "list" in action_enum
        assert "switch" in action_enum
        assert "current" in action_enum

    def test_to_schema(self, souls_tool):
        """Test full schema generation."""
        schema = souls_tool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "souls"
        assert "parameters" in schema["function"]


class TestListSouls:
    """Test listing available souls."""

    def test_list_souls_returns_available(self, souls_tool):
        """Test that list_souls returns available souls."""
        souls = souls_tool.list_souls()

        # Should find 3 valid souls (test_soul1, test_soul2, minimal_soul)
        # invalid_soul has no soul.yaml so should be excluded
        assert len(souls) == 3

        names = [s["name"] for s in souls]
        assert "test_soul1" in names
        assert "test_soul2" in names
        assert "minimal_soul" in names
        assert "invalid_soul" not in names

    def test_list_souls_includes_metadata(self, souls_tool):
        """Test that list_souls includes display name and description."""
        souls = souls_tool.list_souls()

        soul1 = next(s for s in souls if s["name"] == "test_soul1")
        assert soul1["display_name"] == "Test Soul 1"
        assert soul1["description"] == "First test personality"

    def test_list_souls_handles_missing_description(self, souls_tool):
        """Test that souls with no description work."""
        souls = souls_tool.list_souls()

        minimal = next(s for s in souls if s["name"] == "minimal_soul")
        assert minimal["display_name"] == "Minimal"
        assert minimal["description"] == ""

    def test_list_souls_empty_directory(self, tmp_path):
        """Test list_souls with empty directory."""
        empty_dir = tmp_path / "empty_souls"
        empty_dir.mkdir()
        tool = SoulsTool(souls_dir=empty_dir)

        souls = tool.list_souls()
        assert souls == []

    def test_list_souls_nonexistent_directory(self, tmp_path):
        """Test list_souls with non-existent directory."""
        tool = SoulsTool(souls_dir=tmp_path / "nonexistent")

        souls = tool.list_souls()
        assert souls == []


class TestExecuteListAction:
    """Test execute with action='list'."""

    @pytest.mark.asyncio
    async def test_execute_list_action(self, souls_tool):
        """Test listing souls via execute."""
        result = await souls_tool.execute(action="list")

        assert "Available personalities" in result
        assert "test_soul1" in result
        assert "test_soul2" in result
        assert "First test personality" in result

    @pytest.mark.asyncio
    async def test_execute_list_shows_current(
        self, temp_souls_dir, get_current_callback
    ):
        """Test that list shows current soul marker."""
        tool = SoulsTool(
            souls_dir=temp_souls_dir, get_current_callback=get_current_callback
        )

        result = await tool.execute(action="list")

        assert "(active)" in result
        assert "test_soul1" in result

    @pytest.mark.asyncio
    async def test_execute_list_empty_directory(self, tmp_path):
        """Test listing when no souls exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        tool = SoulsTool(souls_dir=empty_dir)

        result = await tool.execute(action="list")

        assert "No souls found" in result


class TestExecuteCurrentAction:
    """Test execute with action='current'."""

    @pytest.mark.asyncio
    async def test_execute_current_action(self, temp_souls_dir, get_current_callback):
        """Test getting current soul."""
        tool = SoulsTool(
            souls_dir=temp_souls_dir, get_current_callback=get_current_callback
        )

        result = await tool.execute(action="current")

        assert "Current personality" in result
        assert "test_soul1" in result

    @pytest.mark.asyncio
    async def test_execute_current_no_active(self, temp_souls_dir):
        """Test current when no soul is active."""
        callback = MagicMock(return_value=None)
        tool = SoulsTool(souls_dir=temp_souls_dir, get_current_callback=callback)

        result = await tool.execute(action="current")

        assert "No soul is currently active" in result or "default" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_current_no_callback(self, souls_tool):
        """Test current when no callback configured."""
        result = await souls_tool.execute(action="current")

        assert "Unable to determine" in result


class TestExecuteSwitchAction:
    """Test execute with action='switch'."""

    @pytest.mark.asyncio
    async def test_execute_switch_success(
        self, temp_souls_dir, switch_callback, get_current_callback
    ):
        """Test successful soul switch."""
        tool = SoulsTool(
            souls_dir=temp_souls_dir,
            switch_callback=switch_callback,
            get_current_callback=get_current_callback,
        )

        result = await tool.execute(action="switch", soul_name="test_soul2")

        assert "Switched successfully" in result
        switch_callback.assert_called_once_with("test_soul2")

    @pytest.mark.asyncio
    async def test_execute_switch_missing_soul_name(self, souls_tool):
        """Test switch without soul_name."""
        result = await souls_tool.execute(action="switch")

        assert "Error" in result
        assert "soul_name is required" in result

    @pytest.mark.asyncio
    async def test_execute_switch_nonexistent_soul(self, souls_tool, switch_callback):
        """Test switching to non-existent soul."""
        tool = SoulsTool(
            souls_dir=souls_tool.souls_dir, switch_callback=switch_callback
        )

        result = await tool.execute(action="switch", soul_name="nonexistent")

        assert "Error" in result
        assert "not found" in result
        # Should list available souls
        assert "test_soul1" in result or "Available" in result

    @pytest.mark.asyncio
    async def test_execute_switch_no_callback(self, souls_tool):
        """Test switch when no callback configured."""
        result = await souls_tool.execute(action="switch", soul_name="test_soul1")

        assert "Error" in result
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_execute_switch_callback_error(self, temp_souls_dir):
        """Test switch when callback raises error."""
        async def failing_callback(soul_name):
            raise RuntimeError("TTS model load failed")

        tool = SoulsTool(souls_dir=temp_souls_dir, switch_callback=failing_callback)

        result = await tool.execute(action="switch", soul_name="test_soul1")

        assert "Error" in result
        assert "test_soul1" in result


class TestUnknownAction:
    """Test handling of unknown actions."""

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, souls_tool):
        """Test unknown action returns error."""
        result = await souls_tool.execute(action="delete")

        assert "Unknown action" in result
        assert "list" in result
        assert "switch" in result
        assert "current" in result


class TestSoulsToolConfiguration:
    """Test tool configuration."""

    def test_default_souls_dir(self):
        """Test default souls directory."""
        tool = SoulsTool()
        assert tool.souls_dir == Path("souls")

    def test_custom_souls_dir_string(self, tmp_path):
        """Test custom souls directory as string."""
        tool = SoulsTool(souls_dir=str(tmp_path / "custom_souls"))
        assert tool.souls_dir == tmp_path / "custom_souls"

    def test_custom_souls_dir_path(self, tmp_path):
        """Test custom souls directory as Path."""
        custom_path = tmp_path / "custom_souls"
        tool = SoulsTool(souls_dir=custom_path)
        assert tool.souls_dir == custom_path


class TestSoulsToolIntegration:
    """Integration tests with tool registry."""

    def test_tool_can_be_registered(self, souls_tool):
        """Test that tool can be registered in registry."""
        from mumble_voice_bot.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(souls_tool)

        assert registry.has("souls")
        assert registry.get("souls") is souls_tool

    def test_tool_definition_in_registry(self, souls_tool):
        """Test tool definition through registry."""
        from mumble_voice_bot.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(souls_tool)

        definitions = registry.get_definitions()
        assert len(definitions) == 1
        assert definitions[0]["function"]["name"] == "souls"

    @pytest.mark.asyncio
    async def test_tool_execution_through_registry(self, souls_tool):
        """Test executing tool through registry."""
        from mumble_voice_bot.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(souls_tool)

        result = await registry.execute("souls", {"action": "list"})

        assert "Available personalities" in result

    @pytest.mark.asyncio
    async def test_tool_validation_in_registry(self, souls_tool):
        """Test parameter validation through registry."""
        from mumble_voice_bot.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(souls_tool)

        # Missing required 'action' parameter
        result = await registry.execute("souls", {})

        assert "Error" in result


class TestYAMLParsing:
    """Test YAML configuration parsing edge cases."""

    def test_list_souls_handles_malformed_yaml(self, tmp_path):
        """Test that malformed YAML doesn't crash the tool."""
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()

        # Create soul with invalid YAML
        bad_soul = souls_dir / "bad_soul"
        bad_soul.mkdir()
        (bad_soul / "soul.yaml").write_text("invalid: yaml: content: [[[")

        tool = SoulsTool(souls_dir=souls_dir)
        souls = tool.list_souls()

        # Should still return the soul, but with error marker
        assert len(souls) == 1
        assert souls[0]["name"] == "bad_soul"
        assert "error" in souls[0]["description"].lower()

    def test_list_souls_handles_empty_yaml(self, tmp_path):
        """Test handling of empty YAML file."""
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()

        empty_soul = souls_dir / "empty_soul"
        empty_soul.mkdir()
        (empty_soul / "soul.yaml").write_text("")

        tool = SoulsTool(souls_dir=souls_dir)
        souls = tool.list_souls()

        # Should handle gracefully
        assert len(souls) == 1
        assert souls[0]["name"] == "empty_soul"


class TestSoulPathValidation:
    """Test soul path validation for switch action."""

    @pytest.mark.asyncio
    async def test_switch_validates_soul_yaml_exists(self, tmp_path, switch_callback):
        """Test that switch validates soul.yaml exists."""
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()

        # Create directory without soul.yaml
        incomplete = souls_dir / "incomplete"
        incomplete.mkdir()

        tool = SoulsTool(souls_dir=souls_dir, switch_callback=switch_callback)

        result = await tool.execute(action="switch", soul_name="incomplete")

        assert "Error" in result
        assert "not found" in result
        switch_callback.assert_not_called()
