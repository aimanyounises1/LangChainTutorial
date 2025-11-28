"""
Custom LangGraph Agent with State Management.

This demonstrates how to build a custom graph-based agent workflow
using LangGraph's low-level API. This gives you full control over:
- State management
- Node definitions
- Edge routing (conditional branching)
- Graph compilation

References:
- https://github.com/langchain-ai/langgraph
- https://www.datacamp.com/tutorial/langgraph-agents
"""

from typing import Annotated, TypedDict, Literal
from operator import add

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


# Define the state schema
class AgentState(TypedDict):
    """State that is passed between nodes in the graph."""
    messages: Annotated[list[BaseMessage], add]


# Define tools
@tool
def search_web(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: The search query

    Returns:
        Search results as formatted string
    """
    results = tavily.search(query=query)
    formatted = []
    for r in results.get("results", [])[:3]:
        formatted.append(f"- {r['title']}: {r['content'][:200]}...")
    return "\n".join(formatted) if formatted else "No results found."


tools = [search_web]


def create_model():
    """Create the LLM with tool binding."""
    llm = ChatOllama(
        model="qwen3:30b-a3b",
        temperature=0.1,
    )
    return llm.bind_tools(tools)


# Define graph nodes
def agent_node(state: AgentState) -> dict:
    """
    The main agent node that decides what to do.

    This node:
    1. Takes the current messages
    2. Calls the LLM to decide next action
    3. Returns the LLM's response (may include tool calls)
    """
    model = create_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Determine whether to continue to tools or end.

    This is the routing function that checks if the last message
    has tool calls. If yes, route to tools node. If no, end.
    """
    last_message = state["messages"][-1]

    # Check if the AI wants to use tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def build_graph():
    """
    Build the LangGraph workflow.

    Graph structure:
        START -> agent -> (tools -> agent)* -> END

    The agent can loop through tools multiple times before ending.
    """
    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # Tools always go back to agent
    workflow.add_edge("tools", "agent")

    # Compile the graph
    return workflow.compile()


def main():
    """Run the custom LangGraph agent."""
    # Build the graph
    graph = build_graph()

    # Create initial state with user query
    initial_state = {
        "messages": [
            HumanMessage(content="What are the latest news about AI?")
        ]
    }

    # Run the graph
    print("\n" + "=" * 50)
    print("Running Custom LangGraph Agent")
    print("=" * 50)

    result = graph.invoke(initial_state)

    # Print all messages
    print("\nConversation:")
    print("-" * 50)
    for msg in result["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if hasattr(msg, "content") else str(msg)
        print(f"\n{role}: {content[:500]}...")


if __name__ == "__main__":
    main()