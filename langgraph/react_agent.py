"""
LangGraph ReAct Agent using prebuilt create_react_agent.

This demonstrates the modern way to create ReAct agents using LangGraph's
prebuilt components. LangGraph provides stateful, graph-based agent workflows
with built-in persistence and streaming support.

References:
- https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/
- https://blog.langchain.com/langgraph-0-3-release-prebuilt-agents/
"""

from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search_web(query: str) -> dict[str, Any]:
    """
    Search the web for information using Tavily.

    Args:
        query: The search query string

    Returns:
        Dictionary containing search results with titles, URLs, and content
    """
    return tavily.search(query=query)


@tool
def get_current_time() -> str:
    """
    Get the current date and time.

    Returns:
        Current datetime as a formatted string
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_agent():
    """
    Create a LangGraph ReAct agent with web search capability.

    Returns:
        A compiled LangGraph agent ready for invocation
    """
    llm = ChatOllama(
        model="qwen3:30b-a3b",
        temperature=0.1,
    )

    # Create the ReAct agent using LangGraph's prebuilt function
    agent = create_react_agent(
        model=llm,
        tools=[search_web, get_current_time],
    )

    return agent


def main():
    """Run the LangGraph ReAct agent with a sample query."""
    agent = create_agent()

    # Invoke the agent with a query
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "What is the current price of Bitcoin?"}
        ]
    })

    # Print the final response
    print("\n" + "=" * 50)
    print("Agent Response:")
    print("=" * 50)

    # Get the last message (the agent's final response)
    if result.get("messages"):
        final_message = result["messages"][-1]
        print(final_message.content)


if __name__ == "__main__":
    main()