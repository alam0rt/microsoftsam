"""Tool call formatting interfaces for different LLM providers.

Different LLMs have different ways of handling tool/function calls:
- OpenAI/Anthropic: Use structured `tools` parameter and `tool_calls` in response
- LFM2.5: Uses special tokens and Pythonic function calls in text
- Others: May use XML, JSON in text, etc.

This module provides an abstraction layer for formatting tool definitions
and parsing tool calls from responses.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from mumble_voice_bot.interfaces.llm import ToolCall


@dataclass
class FormattedTools:
    """Result of formatting tools for an LLM.
    
    Attributes:
        system_prompt_addition: Text to add to system prompt (for text-based tool defs).
        tools_parameter: Structured tools list for API parameter (for OpenAI-style).
    """
    system_prompt_addition: str | None = None
    tools_parameter: list[dict] | None = None


class ToolFormatter(ABC):
    """Abstract base class for tool call formatting.
    
    Implementations handle the specific format for different LLM providers.
    """
    
    @abstractmethod
    def format_tools(self, tools: list[dict]) -> FormattedTools:
        """Format tool definitions for the LLM.
        
        Args:
            tools: List of tool definitions in OpenAI format.
            
        Returns:
            FormattedTools with either system prompt addition or tools parameter.
        """
        pass
    
    @abstractmethod
    def parse_tool_calls(self, response_text: str) -> list[ToolCall]:
        """Parse tool calls from the LLM response.
        
        Args:
            response_text: The raw response text from the LLM.
            
        Returns:
            List of ToolCall objects, empty if no tool calls found.
        """
        pass
    
    @abstractmethod
    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format a tool result for sending back to the LLM.
        
        Args:
            tool_call_id: The ID of the tool call.
            tool_name: The name of the tool.
            result: The string result from executing the tool.
            
        Returns:
            Message dict to add to the conversation.
        """
        pass
    
    def strip_tool_calls(self, response_text: str) -> str:
        """Remove tool call markup from response text.
        
        Args:
            response_text: The raw response text.
            
        Returns:
            Clean text with tool calls removed.
        """
        # Default: return as-is (OpenAI format has tool calls separate from text)
        return response_text


class OpenAIToolFormatter(ToolFormatter):
    """Tool formatter for OpenAI-compatible APIs.
    
    Uses structured tools parameter and parses tool_calls from response.
    """
    
    def format_tools(self, tools: list[dict]) -> FormattedTools:
        """Pass tools directly as API parameter."""
        return FormattedTools(tools_parameter=tools)
    
    def parse_tool_calls(self, response_text: str) -> list[ToolCall]:
        """OpenAI tool calls come from response object, not text.
        
        Returns empty list - actual parsing happens in the LLM provider.
        """
        return []
    
    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format tool result as OpenAI tool message."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }


class LFM25ToolFormatter(ToolFormatter):
    """Tool formatter for Liquid LFM2.5 models.
    
    LFM2.5 uses:
    - Tools defined as JSON list in system prompt
    - Tool calls as: <|tool_call_start|>[func(arg="val")]<|tool_call_end|>
    - Tool results as "tool" role messages with JSON
    
    See: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
    """
    
    # Regex to match LFM2.5 tool calls
    TOOL_CALL_PATTERN = re.compile(
        r'<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>',
        re.DOTALL
    )
    
    # Regex to parse individual function calls like: func_name(arg1="val1", arg2=123)
    FUNC_CALL_PATTERN = re.compile(
        r'(\w+)\((.*?)\)(?:,\s*)?',
        re.DOTALL
    )
    
    def format_tools(self, tools: list[dict]) -> FormattedTools:
        """Format tools as JSON list for system prompt."""
        # Convert OpenAI format to simpler format for LFM2.5
        lfm_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                lfm_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
        
        tool_json = json.dumps(lfm_tools, indent=2)
        
        # Add explicit instructions for tool use per LFM2.5 format
        prompt_addition = f"""

List of tools: {tool_json}

When you need to use a tool, output your tool call in this exact format:
<|tool_call_start|>[function_name(param="value")]<|tool_call_end|>

For example, to switch personality: <|tool_call_start|>[souls(action="switch", soul_name="raf")]<|tool_call_end|>
For example, to search the web: <|tool_call_start|>[web_search(query="latest news")]<|tool_call_end|>

Always use the tool when the user asks to switch personality/soul or search for information."""
        
        return FormattedTools(system_prompt_addition=prompt_addition)
    
    def parse_tool_calls(self, response_text: str) -> list[ToolCall]:
        """Parse LFM2.5 style tool calls from response text."""
        tool_calls = []
        
        # Find all tool call blocks
        matches = self.TOOL_CALL_PATTERN.findall(response_text)
        
        for i, match in enumerate(matches):
            # Parse individual function calls within the block
            func_matches = self.FUNC_CALL_PATTERN.findall(match)
            
            for func_name, args_str in func_matches:
                try:
                    arguments = self._parse_pythonic_args(args_str)
                    tool_calls.append(ToolCall(
                        id=f"lfm_call_{i}_{func_name}",
                        name=func_name,
                        arguments=arguments,
                    ))
                except Exception as e:
                    # Log but continue on parse errors
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Failed to parse tool call {func_name}({args_str}): {e}"
                    )
        
        return tool_calls
    
    def _parse_pythonic_args(self, args_str: str) -> dict:
        """Parse Pythonic function arguments into a dict.
        
        Handles: arg="value", arg=123, arg=True, arg='value'
        """
        if not args_str.strip():
            return {}
        
        result = {}
        # Match key=value pairs
        # This regex handles: key="value", key='value', key=123, key=True/False
        arg_pattern = re.compile(
            r'(\w+)\s*=\s*(?:'
            r'"([^"]*)"'  # Double-quoted string
            r"|'([^']*)'"  # Single-quoted string
            r'|(\d+(?:\.\d+)?)'  # Number
            r'|(True|False|None)'  # Boolean/None
            r')',
            re.IGNORECASE
        )
        
        for match in arg_pattern.finditer(args_str):
            key = match.group(1)
            if match.group(2) is not None:  # Double-quoted
                result[key] = match.group(2)
            elif match.group(3) is not None:  # Single-quoted
                result[key] = match.group(3)
            elif match.group(4) is not None:  # Number
                num_str = match.group(4)
                result[key] = float(num_str) if '.' in num_str else int(num_str)
            elif match.group(5) is not None:  # Boolean/None
                val = match.group(5).lower()
                if val == 'true':
                    result[key] = True
                elif val == 'false':
                    result[key] = False
                else:
                    result[key] = None
        
        return result
    
    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format tool result for LFM2.5.
        
        OpenRouter may not support 'tool' role properly for LFM2.5,
        so we format the result as a user message with clear context.
        """
        return {
            "role": "user",
            "content": f"[Tool Result from {tool_name}]: {result}\n\nPlease summarize the above results for the user in a conversational way.",
        }
    
    def strip_tool_calls(self, response_text: str) -> str:
        """Remove tool call markup from response, keeping any surrounding text."""
        # Remove the tool call blocks but keep text after them
        cleaned = self.TOOL_CALL_PATTERN.sub('', response_text)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()


def get_tool_formatter(model_name: str) -> ToolFormatter:
    """Get the appropriate tool formatter for a model.
    
    Args:
        model_name: The model identifier (e.g., "liquid/lfm-2.5-1.2b-instruct:free").
        
    Returns:
        Appropriate ToolFormatter implementation.
    """
    model_lower = model_name.lower()
    
    # LFM2.5 models use text-based tool calling
    if 'lfm' in model_lower and ('2.5' in model_lower or '2-5' in model_lower):
        return LFM25ToolFormatter()
    
    # Default to OpenAI format
    return OpenAIToolFormatter()
