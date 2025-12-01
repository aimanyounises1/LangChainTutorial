"""
Reflection Agent with StateGraph and SQLite Checkpointing

This module implements a reflection agent that:
1. Drafts an initial response
2. Executes search tools to gather information
3. Revises the response based on search results
4. Repeats until done or max iterations reached

State is persisted to SQLite for debugging and inspection.
"""
from pathlib import Path
from typing import List
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, BaseMessage, AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, END

from langgraph_examples.reflection_agent.chains import first_responder, reviser
from langgraph_examples.reflection_agent.text_tool_call_parser import (
    parse_text_tool_calls,
)
from langgraph_examples.reflection_agent.tools_executor import execute_tools

load_dotenv(verbose=True)

MAX_ITERATIONS = 3

# Create checkpoints directory
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DB = str(CHECKPOINTS_DIR / "agent_state.db")


def _parse_last_ai_message_tool_calls(messages: List[BaseMessage]) -> AIMessage:
    """
    Helper function to parse text-based tool calls from the last AI message.
    Handles case where model outputs <function-call> as text instead of proper tool_calls.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                return msg
            return parse_text_tool_calls(msg, preserve_content=True)
    raise ValueError("No AI message found in state")


# ============================================================================
# NODE FUNCTIONS - Each returns dict update for StateGraph
# ============================================================================

def draft_node(state: MessagesState) -> dict:
    """Draft node - invokes first_responder chain."""
    result = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [result]}


def parse_draft_tool_calls(state: MessagesState) -> dict:
    """Parse text-based tool calls from draft node output."""
    parsed = _parse_last_ai_message_tool_calls(state["messages"])
    return {"messages": [parsed]}


def execute_tools_node(state: MessagesState) -> dict:
    """Execute tools node - wraps ToolNode for StateGraph compatibility."""
    result = execute_tools.invoke(state["messages"])
    return {"messages": result}


def reviser_node(state: MessagesState) -> dict:
    """Reviser node - invokes reviser chain."""
    result = reviser.invoke({"messages": state["messages"]})
    return {"messages": [result]}


def parse_reviser_tool_calls(state: MessagesState) -> dict:
    """Parse text-based tool calls from reviser node output."""
    parsed = _parse_last_ai_message_tool_calls(state["messages"])
    return {"messages": [parsed]}


# ============================================================================
# CONDITIONAL EDGE FUNCTION
# ============================================================================

def event_loop(state: MessagesState) -> str:
    """
    Determine whether to continue the loop or end.
    Checks the LAST message in state (which is now from parse_reviser).
    """
    messages = state["messages"]
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in messages)
    if count_tool_visits > MAX_ITERATIONS:
        return END

    # Get the last AI message (should be from parse_reviser with populated tool_calls)
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    # If reviser didn't request tools (empty search_queries), we're done
    if last_ai_message is None:
        return END

    # Check tool_calls - if empty or search_queries is empty, we're done
    if not last_ai_message.tool_calls:
        return END

    # Check if search_queries in the tool call args is empty (signals completion)
    for tool_call in last_ai_message.tool_calls:
        args = tool_call.get('args', {})
        search_queries = args.get('search_queries', [])
        if not search_queries:
            return END

    return "execute_tools"


# ============================================================================
# BUILD GRAPH
# ============================================================================

builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("draft", draft_node)
builder.add_node("parse_draft", parse_draft_tool_calls)
builder.add_node("execute_tools", execute_tools_node)
builder.add_node("reviser", reviser_node)
builder.add_node("parse_reviser", parse_reviser_tool_calls)

# Add edges
builder.add_edge("draft", "parse_draft")
builder.add_edge("parse_draft", "execute_tools")
builder.add_edge("execute_tools", "reviser")
builder.add_edge("reviser", "parse_reviser")

# Add conditional edges
builder.add_conditional_edges("parse_reviser", event_loop, {
    END: END,
    "execute_tools": "execute_tools",
})

# Set entry point
builder.set_entry_point("draft")

# Compile with SQLite checkpointer for state persistence
checkpointer = SqliteSaver.from_conn_string(CHECKPOINT_DB)
graph = builder.compile(checkpointer=checkpointer)

print(graph.get_graph().draw_ascii())
graph.get_graph().draw_mermaid_png(output_file_path=str(Path(__file__).parent / "graph.png"))


# ============================================================================
# STATE INSPECTION UTILITIES
# ============================================================================

def inspect_latest_state(thread_id: str) -> None:
    """Inspect the latest state for a given thread without loading full messages."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    if state.values:
        print(f"\n{'='*60}")
        print(f"Thread: {thread_id}")
        print(f"Next nodes: {state.next}")
        print(f"Message count: {len(state.values.get('messages', []))}")

        # Show last message summary without full content
        messages = state.values.get('messages', [])
        if messages:
            last_msg = messages[-1]
            print(f"Last message type: {type(last_msg).__name__}")
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"Tool calls: {[tc['name'] for tc in last_msg.tool_calls]}")
        print(f"{'='*60}\n")
    else:
        print(f"No state found for thread: {thread_id}")


def list_checkpoint_history(thread_id: str, limit: int = 10) -> None:
    """List checkpoint history for debugging without loading full state."""
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))

    print(f"\n{'='*60}")
    print(f"Checkpoint History for Thread: {thread_id}")
    print(f"Total checkpoints: {len(history)}")
    print(f"{'='*60}")

    for i, snapshot in enumerate(history[:limit]):
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        metadata = snapshot.metadata
        msg_count = len(snapshot.values.get('messages', []))

        print(f"\n[{i}] Checkpoint: {checkpoint_id[:20]}...")
        print(f"    Step: {metadata.step if hasattr(metadata, 'step') else 'N/A'}")
        print(f"    Source: {metadata.source if hasattr(metadata, 'source') else 'N/A'}")
        print(f"    Messages: {msg_count}")

        # Show tool message count
        tool_msgs = sum(1 for m in snapshot.values.get('messages', [])
                       if isinstance(m, ToolMessage))
        if tool_msgs:
            print(f"    Tool messages: {tool_msgs}")


def stream_with_debug(question: str, thread_id: str = None) -> dict:
    """
    Run the graph with debug streaming to see state changes in real-time.
    """
    if thread_id is None:
        thread_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": [HumanMessage(content=question)]}

    print(f"\n{'='*60}")
    print(f"Starting debug stream - Thread: {thread_id}")
    print(f"{'='*60}\n")

    final_state = None
    for i, chunk in enumerate(graph.stream(input_state, config, stream_mode="updates")):
        node_name = list(chunk.keys())[0]
        update = chunk[node_name]

        print(f"[Step {i}] Node: {node_name}")
        if 'messages' in update:
            for msg in update['messages']:
                print(f"    → {type(msg).__name__}: {str(msg.content)[:100]}...")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"      Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
        print()
        final_state = chunk

    print(f"\n{'='*60}")
    print(f"Stream complete. Thread ID: {thread_id}")
    print(f"Use inspect_latest_state('{thread_id}') to inspect state")
    print(f"Use list_checkpoint_history('{thread_id}') to see all checkpoints")
    print(f"{'='*60}\n")

    return final_state


if __name__ == '__main__':
    print("Reflexion Agent with SQLite Checkpointing")
    print(f"Checkpoints stored in: {CHECKPOINT_DB}")

    # Create unique thread ID for this run
    thread_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run with standard invoke
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [
            HumanMessage(content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.")
        ]
    }

    print(f"\nStarting run with thread_id: {thread_id}")
    res = graph.invoke(initial_state, config)

    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(res)

    # Show how to inspect state
    print("\n" + "="*60)
    print("STATE INSPECTION:")
    print("="*60)
    inspect_latest_state(thread_id)
    list_checkpoint_history(thread_id, limit=5)
