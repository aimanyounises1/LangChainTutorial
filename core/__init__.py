"""
Core utilities and shared components.

This module contains:
- schemas: Pydantic models and prompt templates
- tools: LangChain tools for agents
"""

from core.schemas import AgentResponse, Source, REACT_PROMPT_TEMPLATE
from core.tools import search_tool

__all__ = ["AgentResponse", "Source", "REACT_PROMPT_TEMPLATE", "search_tool"]