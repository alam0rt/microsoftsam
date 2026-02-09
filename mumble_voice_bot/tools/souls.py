"""Souls tool for listing and switching bot personalities.

This tool allows users to discover available souls and switch
the bot's personality at runtime.
"""

from pathlib import Path
from typing import Any, Awaitable, Callable

from mumble_voice_bot.logging_config import get_logger
from mumble_voice_bot.tools.base import Tool

logger = get_logger(__name__)


class SoulsTool(Tool):
    """Tool to list available souls and switch personalities.

    This tool allows the LLM to:
    - List all available souls in the souls directory
    - Switch to a different soul/personality

    The actual switching is delegated to a callback provided by the bot,
    since it requires updating TTS voice and LLM prompts.
    """

    def __init__(
        self,
        souls_dir: str | Path = "souls",
        switch_callback: Callable[[str], Awaitable[str]] | None = None,
        get_current_callback: Callable[[], str | None] | None = None,
    ):
        """Initialize the souls tool.

        Args:
            souls_dir: Directory containing soul configurations.
            switch_callback: Async callback to switch souls. Takes soul name,
                           returns success/error message.
            get_current_callback: Callback to get current soul name.
        """
        self.souls_dir = Path(souls_dir)
        self._switch_callback = switch_callback
        self._get_current_callback = get_current_callback

    @property
    def name(self) -> str:
        return "souls"

    @property
    def description(self) -> str:
        return (
            "Manage bot personalities (souls). Use action='list' to see available "
            "personalities, or action='switch' with a soul_name to change personality. "
            "Switching changes the bot's voice and behavior."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "switch", "current"],
                    "description": "Action to perform: 'list' available souls, 'switch' to a new soul, or 'current' to see active soul.",
                },
                "soul_name": {
                    "type": "string",
                    "description": "Name of the soul to switch to (required for 'switch' action).",
                }
            },
            "required": ["action"]
        }

    @property
    def example_call(self) -> str:
        return 'souls(action="switch", soul_name="raf")'

    @property
    def usage_hint(self) -> str:
        return "Use when asked to switch personality, voice, or character"

    def list_souls(self) -> list[dict]:
        """List all available souls with their metadata.

        Returns:
            List of dicts with soul name and description.
        """
        souls = []

        if not self.souls_dir.exists():
            return souls

        for soul_path in sorted(self.souls_dir.iterdir()):
            if not soul_path.is_dir():
                continue

            soul_yaml = soul_path / "soul.yaml"
            if not soul_yaml.exists():
                continue

            # Parse basic info from soul.yaml
            try:
                import yaml
                with open(soul_yaml) as f:
                    config = yaml.safe_load(f) or {}

                souls.append({
                    "name": soul_path.name,
                    "display_name": config.get("name", soul_path.name),
                    "description": config.get("description", ""),
                })
            except Exception as e:
                logger.warning(f"Failed to read soul {soul_path.name}: {e}")
                souls.append({
                    "name": soul_path.name,
                    "display_name": soul_path.name,
                    "description": "(config error)",
                })

        return souls

    async def execute(
        self,
        action: str,
        soul_name: str | None = None,
        **kwargs: Any
    ) -> str:
        """Execute the souls action.

        Args:
            action: 'list', 'switch', or 'current'
            soul_name: Soul to switch to (for 'switch' action)

        Returns:
            Result message.
        """
        if action == "list":
            souls = self.list_souls()
            if not souls:
                return "No souls found in the souls directory."

            # Get current soul for marking
            current = None
            if self._get_current_callback:
                current = self._get_current_callback()

            lines = ["Available personalities:"]
            for soul in souls:
                marker = " (active)" if soul["name"] == current else ""
                if soul["description"]:
                    lines.append(f"- {soul['name']}: {soul['description']}{marker}")
                else:
                    lines.append(f"- {soul['name']}{marker}")

            return "\n".join(lines)

        elif action == "current":
            if self._get_current_callback:
                current = self._get_current_callback()
                if current:
                    return f"Current personality: {current}"
                return "No soul is currently active (using default personality)."
            return "Unable to determine current personality."

        elif action == "switch":
            if not soul_name:
                return "Error: soul_name is required for 'switch' action."

            # Check if soul exists
            soul_path = self.souls_dir / soul_name
            if not soul_path.exists() or not (soul_path / "soul.yaml").exists():
                available = [s["name"] for s in self.list_souls()]
                return f"Error: Soul '{soul_name}' not found. Available: {', '.join(available)}"

            # Call the switch callback
            if self._switch_callback:
                try:
                    result = await self._switch_callback(soul_name)
                    return result
                except Exception as e:
                    logger.error(f"Failed to switch soul: {e}")
                    return f"Error switching to '{soul_name}': {str(e)}"
            else:
                return "Error: Soul switching is not configured."

        else:
            return f"Unknown action: {action}. Use 'list', 'switch', or 'current'."
