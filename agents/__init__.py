"""
Agent implementations using LangChain.

This module contains various agent patterns:
- ReAct agents with Tavily search
- Search agents with structured outputs
"""

from agents.react_agent import main as run_react_agent
from agents.search_agent import main as run_search_agent

__all__ = ["run_react_agent", "run_search_agent"]