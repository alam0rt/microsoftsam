"""Base class for agent tools."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base class for agent tools.

    Tools are capabilities that the agent can use to interact with
    the environment, such as searching the web, looking up information, etc.

    To create a new tool, subclass this and implement:
    - name: Unique identifier for the tool
    - description: What the tool does (shown to the LLM)
    - parameters: JSON Schema for the tool's parameters
    - execute(): The actual tool implementation

    Example:
        class TimeTool(Tool):
            @property
            def name(self) -> str:
                return "get_time"

            @property
            def description(self) -> str:
                return "Get the current time and date."

            @property
            def parameters(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

            async def execute(self, **kwargs: Any) -> str:
                from datetime import datetime
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    """

    # Type mapping for JSON Schema validation
    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls.

        Should be a valid identifier (lowercase, underscores ok).
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does.

        This is shown to the LLM to help it decide when to use the tool.
        Be clear and specific about capabilities and limitations.
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters.

        Must be an object-type schema with properties and required fields.
        Example:
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["query"]
            }
        """
        pass

    @property
    def example_call(self) -> str | None:
        """Example of how to call this tool (for LLM prompting).

        Override to provide a specific example for the LLM.
        Example: 'sound_effects(action="play", query="airhorn")'

        Returns:
            Example call string, or None to auto-generate from parameters.
        """
        return None

    @property
    def usage_hint(self) -> str | None:
        """Short hint about when to use this tool.

        Example: "Use when asked to play sounds or sound effects"

        Returns:
            Usage hint string, or None if not needed.
        """
        return None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters matching the JSON schema.

        Returns:
            String result of the tool execution. This will be fed back
            to the LLM as the tool result.
        """
        pass

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate tool parameters against JSON schema.

        Args:
            params: Parameters to validate.

        Returns:
            List of error messages (empty if valid).
        """
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        """Recursively validate a value against a JSON schema."""
        t = schema.get("type")
        label = path or "parameter"

        # Type check
        if t in self._TYPE_MAP and not isinstance(val, self._TYPE_MAP[t]):
            return [f"{label} should be {t}"]

        errors = []

        # Enum check
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")

        # Number range checks
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")

        # String length checks
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")

        # Object validation (check required fields and nested properties)
        if t == "object" and isinstance(val, dict):
            props = schema.get("properties", {})
            required = schema.get("required", [])

            for req in required:
                if req not in val:
                    errors.append(f"{label}.{req} is required")

            for key, prop_schema in props.items():
                if key in val:
                    errors.extend(self._validate(val[key], prop_schema, f"{label}.{key}"))

        # Array validation
        if t == "array" and isinstance(val, list):
            items_schema = schema.get("items", {})
            for i, item in enumerate(val):
                errors.extend(self._validate(item, items_schema, f"{label}[{i}]"))

        return errors

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema format.

        Returns:
            Tool definition in OpenAI function calling format.
            Includes extra fields for LLM prompting (example_call, usage_hint).
        """
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
        # Add optional prompting hints (used by some formatters)
        if self.example_call:
            schema["function"]["example_call"] = self.example_call
        if self.usage_hint:
            schema["function"]["usage_hint"] = self.usage_hint
        return schema

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
