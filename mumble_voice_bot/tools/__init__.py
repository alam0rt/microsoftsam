"""Tool system for agent capabilities.

This module provides a generic framework for extending the bot with
tools that can be invoked by the LLM via function calling.
"""

from mumble_voice_bot.tools.base import Tool
from mumble_voice_bot.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolRegistry"]
