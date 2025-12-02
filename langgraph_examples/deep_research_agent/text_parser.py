"""
Deep Research Agent - Text Tool Call Parser

This module handles parsing text-based tool calls from LLM outputs.
Adapted from the reflection_agent module.
"""

import re
import json
import uuid
from typing import List, Optional
from langchain_core.messages import AIMessage


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


def extract_tool_calls_from_text(content: str) -> List[dict]:
    """
    Extract all tool calls from text content.
    """
    tool_calls = []

    for pattern in TOOL_CALL_PATTERNS:
        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                json_str = match.group(1).strip()
                data = json.loads(json_str)

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
    """
    if message.tool_calls and not override_existing:
        return message

    content = message.content or ""
    extracted_calls = extract_tool_calls_from_text(content)

    if not extracted_calls:
        return message

    new_content = content
    if not preserve_content:
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
