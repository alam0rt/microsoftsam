"""Tool registry for dynamic tool management.

Supports:
- Dynamic registration and execution of tools
- Per-user rate limiting on tool calls
- Per-soul tool allowlisting
- Audit logging of all tool invocations
"""

import time
from typing import Any

from mumble_voice_bot.logging_config import get_logger
from mumble_voice_bot.tools.base import Tool

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for agent tools with rate limiting and audit logging.

    Allows dynamic registration, lookup, and execution of tools.
    Tools are identified by their unique name property.

    Features:
    - Per-user rate limiting (configurable max calls per minute)
    - Per-soul allowlisting (restrict which tools each persona can use)
    - Audit logging of all tool invocations with user, tool, args, result

    Example:
        registry = ToolRegistry(rate_limit_per_minute=10)
        registry.register(WebSearchTool(api_key="..."))
        registry.set_allowed_tools(["web_search"])  # Only allow web search

        # Get tool definitions for LLM
        tools = registry.get_definitions()

        # Execute a tool
        result = await registry.execute("web_search", {"query": "latest news"}, user_id="sam")
    """

    def __init__(self, rate_limit_per_minute: int = 20, allowed_tools: list[str] | None = None):
        """Initialize the tool registry.

        Args:
            rate_limit_per_minute: Maximum tool calls per user per minute (0=unlimited).
            allowed_tools: List of allowed tool names (None=all allowed).
        """
        self._tools: dict[str, Tool] = {}
        self._rate_limit = rate_limit_per_minute
        self._allowed_tools: set[str] | None = set(allowed_tools) if allowed_tools else None

        # Rate limiting: user_id -> list of timestamps
        self._user_calls: dict[str, list[float]] = {}

        # Audit log: list of (timestamp, user_id, tool_name, args_summary, result_summary)
        self._audit_log: list[dict] = []
        self._audit_max_entries = 1000

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

    def set_allowed_tools(self, tool_names: list[str] | None) -> None:
        """Set the allowlist of tools this registry will execute.

        Args:
            tool_names: List of allowed tool names, or None to allow all.
        """
        self._allowed_tools = set(tool_names) if tool_names else None
        if self._allowed_tools:
            logger.info(f"Tool allowlist set: {sorted(self._allowed_tools)}")
        else:
            logger.info("Tool allowlist cleared (all tools allowed)")

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI format, filtered by allowlist.

        Returns:
            List of tool schemas suitable for the OpenAI tools parameter.
        """
        definitions = []
        for tool in self._tools.values():
            if self._allowed_tools is not None and tool.name not in self._allowed_tools:
                continue
            definitions.append(tool.to_schema())
        return definitions

    async def execute(self, name: str, params: dict[str, Any], user_id: str | None = None) -> str:
        """Execute a tool by name with given parameters.

        Args:
            name: Tool name.
            params: Tool parameters.
            user_id: Optional user ID for rate limiting and audit logging.

        Returns:
            Tool execution result as string.
        """
        # Allowlist check
        if self._allowed_tools is not None and name not in self._allowed_tools:
            msg = f"Error: Tool '{name}' is not allowed for this persona"
            logger.warning(f"Tool blocked by allowlist: {name} (user={user_id})")
            self._audit(user_id, name, params, msg, blocked=True)
            return msg

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        # Rate limiting check
        if user_id and self._rate_limit > 0:
            if not self._check_rate_limit(user_id):
                msg = f"Error: Rate limit exceeded ({self._rate_limit} calls/min). Please wait."
                logger.warning(f"Rate limited: user={user_id}, tool={name}")
                self._audit(user_id, name, params, msg, blocked=True)
                return msg

        try:
            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                msg = f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
                self._audit(user_id, name, params, msg, blocked=True)
                return msg

            # Execute the tool
            logger.info(f"Executing tool: {name} (user={user_id or 'unknown'})")
            result = await tool.execute(**params)
            logger.debug(f"Tool {name} result: {result[:200]}..." if len(result) > 200 else f"Tool {name} result: {result}")

            # Audit log
            self._audit(user_id, name, params, result)

            return result

        except Exception as e:
            msg = f"Error executing {name}: {str(e)}"
            logger.error(f"Tool {name} failed: {e}")
            self._audit(user_id, name, params, msg, error=True)
            return msg

    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if a user is within rate limits.

        Args:
            user_id: User identifier.

        Returns:
            True if the call is allowed, False if rate limited.
        """
        now = time.time()
        window = 60.0  # 1 minute window

        if user_id not in self._user_calls:
            self._user_calls[user_id] = []

        # Prune old entries
        self._user_calls[user_id] = [t for t in self._user_calls[user_id] if now - t < window]

        if len(self._user_calls[user_id]) >= self._rate_limit:
            return False

        self._user_calls[user_id].append(now)
        return True

    def _audit(
        self,
        user_id: str | None,
        tool_name: str,
        params: dict,
        result: str,
        blocked: bool = False,
        error: bool = False,
    ) -> None:
        """Record a tool invocation in the audit log.

        Args:
            user_id: User who triggered the call.
            tool_name: Name of the tool.
            params: Parameters passed to the tool.
            result: Result or error message.
            blocked: Whether the call was blocked (rate limit, allowlist).
            error: Whether the call resulted in an error.
        """
        entry = {
            "time": time.time(),
            "user": user_id or "unknown",
            "tool": tool_name,
            "args": {k: str(v)[:100] for k, v in params.items()},  # Truncate long args
            "result_preview": result[:200] if result else "",
            "blocked": blocked,
            "error": error,
        }
        self._audit_log.append(entry)

        # Trim audit log
        if len(self._audit_log) > self._audit_max_entries:
            self._audit_log = self._audit_log[-self._audit_max_entries:]

        # Structured log for monitoring
        status = "BLOCKED" if blocked else ("ERROR" if error else "OK")
        logger.info(
            f"[AUDIT] tool={tool_name} user={user_id or 'unknown'} status={status} "
            f"args={list(params.keys())}"
        )

    def get_audit_log(self, limit: int = 50) -> list[dict]:
        """Get recent audit log entries.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of audit log entries (most recent last).
        """
        return self._audit_log[-limit:]

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
