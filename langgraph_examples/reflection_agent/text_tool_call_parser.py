"""
DEPRECATED: This module is no longer needed in the main flow.

The reflection agent now uses `with_structured_output()` which returns
Pydantic objects directly, eliminating the need for text-based tool call parsing.

This file is kept for:
1. Backward compatibility with any code that imports from it
2. Reference for handling edge cases with models that don't support structured output
3. Potential fallback mechanism if needed

---
Original purpose:
Text-Based Tool Call Parser for LangGraph

This module handles the case where LLMs (especially Ollama models) output tool calls
as text in the content field instead of using the proper tool_calls mechanism.

Supported formats:
- <function-call>{"name": "...", "arguments": {...}}</function-call>
- <function_call>...</function_call>
- <tool-call>...</tool-call>
- <tool_call>...</tool_call>
- JSON code blocks with {"name": "...", "arguments": {...}}
"""

import re
import json
import uuid
from typing import List, Optional, Union
from langchain_core.messages import AIMessage, BaseMessage


# Regex patterns for different text-based tool call formats
TOOL_CALL_PATTERNS = [
    # <function-call>...</function-call>
    r'<function-call>\s*(\{[\s\S]*?\})\s*</function-call>',
    # <function_call>...</function_call>
    r'<function_call>\s*(\{[\s\S]*?\})\s*</function_call>',
    # <tool-call>...</tool-call>
    r'<tool-call>\s*(\{[\s\S]*?\})\s*</tool-call>',
    # <tool_call>...</tool_call>
    r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>',
    # ```json ... ``` with tool call structure
    r'```(?:json)?\s*(\{[^`]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^`]*\}[^`]*\})\s*```',
]


def detect_text_tool_calls(content: str) -> bool:
    """
    Detect if content contains text-based tool calls.

    Args:
        content: The text content to check

    Returns:
        True if text-based tool calls are detected
    """
    if not content:
        return False

    for pattern in TOOL_CALL_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True
    return False


def extract_tool_calls_from_text(content: str) -> List[dict]:
    """
    Extract all tool calls from text content.

    Args:
        content: The text content containing tool calls

    Returns:
        List of tool call dictionaries with 'name', 'args', 'id', 'type'
    """
    tool_calls = []

    for pattern in TOOL_CALL_PATTERNS:
        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                json_str = match.group(1)
                # Clean up the JSON string
                json_str = json_str.strip()
                data = json.loads(json_str)

                # Extract name and arguments
                name = data.get('name')
                args = data.get('arguments', data.get('args', {}))

                if name:
                    tool_call = {
                        'name': name,
                        'args': args if isinstance(args, dict) else {},
                        'id': f"call_{uuid.uuid4().hex[:8]}",
                        'type': 'tool_call'
                    }
                    tool_calls.append(tool_call)
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                # Log but continue trying other patterns
                print(f"Warning: Failed to parse tool call: {e}")
                continue

    return tool_calls


def parse_text_tool_calls(
    message: AIMessage,
    preserve_content: bool = True,
    override_existing: bool = False
) -> AIMessage:
    """
    Parse text-based tool calls and convert to proper tool_calls format.

    Args:
        message: The AIMessage to process
        preserve_content: If True, keep the original content; if False, clean it
        override_existing: If True, override existing tool_calls; if False, only add if empty

    Returns:
        New AIMessage with populated tool_calls
    """
    # Skip if already has tool_calls and we shouldn't override
    if message.tool_calls and not override_existing:
        return message

    content = message.content or ""

    # Try to extract tool calls from text
    extracted_calls = extract_tool_calls_from_text(content)

    if not extracted_calls:
        return message

    # Create new message with tool_calls
    new_content = content
    if not preserve_content:
        # Remove the tool call tags from content
        for pattern in TOOL_CALL_PATTERNS:
            new_content = re.sub(pattern, '', new_content, flags=re.IGNORECASE | re.DOTALL)
        new_content = new_content.strip()

    return AIMessage(
        content=new_content,
        tool_calls=extracted_calls,
        additional_kwargs=message.additional_kwargs,
        response_metadata=message.response_metadata,
        id=message.id,
    )


def ensure_tool_calls(message: AIMessage) -> AIMessage:
    """
    Convenience function to ensure an AIMessage has tool_calls populated.
    If tool_calls is empty but text contains tool calls, parse them.

    Args:
        message: The AIMessage to process

    Returns:
        AIMessage with tool_calls populated (if found in text)
    """
    if message.tool_calls:
        return message
    return parse_text_tool_calls(message, preserve_content=True, override_existing=False)


def create_tool_call_parser_node(
    preserve_content: bool = True,
    override_existing: bool = False
):
    """
    Create a LangGraph node function that parses text-based tool calls.

    Usage:
        builder.add_node("parse_tools", create_tool_call_parser_node())

    Args:
        preserve_content: If True, keep original content
        override_existing: If True, override existing tool_calls

    Returns:
        A function suitable for use as a LangGraph node
    """
    def parse_node(state: List[BaseMessage]) -> AIMessage:
        """Parse text-based tool calls from the last AI message."""
        # Find the last AI message
        last_ai_message = None
        for msg in reversed(state):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break

        if last_ai_message is None:
            raise ValueError("No AI message found in state")

        # Parse and return
        return parse_text_tool_calls(
            last_ai_message,
            preserve_content=preserve_content,
            override_existing=override_existing
        )

    return parse_node


class TextToolCallParser:
    """
    Class-based parser for text-based tool calls with custom pattern support.

    Usage:
        parser = TextToolCallParser(
            custom_patterns=[r"<my-tag>(.*?)</my-tag>"],
            preserve_content=True
        )
        parsed_msg = parser.parse(message)
    """

    def __init__(
        self,
        custom_patterns: Optional[List[str]] = None,
        preserve_content: bool = True,
        override_existing: bool = False
    ):
        self.patterns = list(TOOL_CALL_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        self.preserve_content = preserve_content
        self.override_existing = override_existing

    def parse(self, message: AIMessage) -> AIMessage:
        """Parse a single message."""
        if message.tool_calls and not self.override_existing:
            return message

        content = message.content or ""
        tool_calls = []

        for pattern in self.patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    json_str = match.group(1).strip()
                    data = json.loads(json_str)
                    name = data.get('name')
                    args = data.get('arguments', data.get('args', {}))

                    if name:
                        tool_calls.append({
                            'name': name,
                            'args': args if isinstance(args, dict) else {},
                            'id': f"call_{uuid.uuid4().hex[:8]}",
                            'type': 'tool_call'
                        })
                except (json.JSONDecodeError, AttributeError, KeyError):
                    continue

        if not tool_calls:
            return message

        new_content = content
        if not self.preserve_content:
            for pattern in self.patterns:
                new_content = re.sub(pattern, '', new_content, flags=re.IGNORECASE | re.DOTALL)
            new_content = new_content.strip()

        return AIMessage(
            content=new_content,
            tool_calls=tool_calls,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            id=message.id,
        )

    def as_node(self):
        """Return a function suitable for use as a LangGraph node."""
        def node_func(state: List[BaseMessage]) -> AIMessage:
            for msg in reversed(state):
                if isinstance(msg, AIMessage):
                    return self.parse(msg)
            raise ValueError("No AI message found in state")
        return node_func


if __name__ == '__main__':
    # Test the parser
    test_content = '''
    <function-call>
    {"name": "AnswerQuestion", "arguments": {"answer": "Test answer", "reflection": "Test reflection", "search_queries": ["query1", "query2"]}}
    </function-call>
    '''

    test_msg = AIMessage(content=test_content, tool_calls=[])
    parsed = parse_text_tool_calls(test_msg)

    print("Original tool_calls:", test_msg.tool_calls)
    print("Parsed tool_calls:", parsed.tool_calls)
    print("Success!" if parsed.tool_calls else "Failed!")
