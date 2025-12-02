"""
Reflection Agent with StateGraph and SQLite Checkpointing

REFACTORED VERSION using with_structured_output for direct Pydantic returns.

This module implements a reflection agent that:
1. Drafts an initial response (returns AnswerQuestion Pydantic object)
2. Executes search tools to gather information
3. Revises the response (returns ReviseAnswer Pydantic object)
4. Repeats until done or max iterations reached

Key Improvements:
- Uses with_structured_output() for direct Pydantic object returns
- No manual parsing or text extraction needed
- Built-in retry via .with_retry() on chains
- RetryPolicy on nodes for additional resilience
- Clean separation between LLM output (Pydantic) and message state (AIMessage)
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, BaseMessage, AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.types import RetryPolicy
from pydantic import ValidationError

from langgraph_examples.reflection_agent.chains import first_responder, reviser
from langgraph_examples.reflection_agent.schemas import AnswerQuestion, ReviseAnswer
from langgraph_examples.reflection_agent.tools_executor import execute_tools

load_dotenv(verbose=True)

MAX_ITERATIONS = 3

# Create checkpoints directory
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DB = str(CHECKPOINTS_DIR / "agent_state.db")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pydantic_to_ai_message(obj: AnswerQuestion | ReviseAnswer, tool_name: str) -> AIMessage:
    """
    Convert a Pydantic object to an AIMessage with proper tool_calls structure.
    
    This bridges the gap between:
    - with_structured_output() which returns Pydantic objects
    - ToolNode which expects AIMessage with tool_calls
    
    Args:
        obj: The Pydantic object (AnswerQuestion or ReviseAnswer)
        tool_name: The tool name to use in tool_calls
    
    Returns:
        AIMessage with tool_calls containing the Pydantic object's data
    """
    import uuid

    return AIMessage(
        content="",  # Content is in the tool call
        tool_calls=[{
            "name": tool_name,
            "args": obj.model_dump(),
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call"
        }]
    )


def get_last_structured_response(messages: List[BaseMessage]) -> Optional[dict]:
    """
    Extract the last structured response arguments from messages.
    
    Returns the 'args' dict from the last AIMessage's tool_calls, or None.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[0].get("args", {})
    return None


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def draft_node(state: MessagesState) -> dict:
    """
    Draft node - invokes first_responder chain.
    
    Returns AnswerQuestion Pydantic object directly (via with_structured_output),
    then converts to AIMessage for the graph state.
    """
    print("[draft_node] Generating initial answer...")

    # Chain returns AnswerQuestion Pydantic object directly
    result: AnswerQuestion = first_responder.invoke({"messages": state["messages"]})

    # Log the structured output
    print(f"[draft_node] ✅ Got AnswerQuestion:")
    print(f"  - Answer: {len(result.answer)} chars")
    print(f"  - Reflection: {result.reflection[:100]}...")
    print(f"  - Search queries: {result.search_queries}")

    # Convert to AIMessage with tool_calls for ToolNode compatibility
    ai_message = pydantic_to_ai_message(result, AnswerQuestion.__name__)

    return {"messages": [ai_message]}


def execute_tools_node(state: MessagesState) -> dict:
    """
    Execute tools node - runs search queries from the last AI message.
    """
    print("[execute_tools_node] Running search queries...")

    result = execute_tools.invoke(state["messages"])

    print(f"[execute_tools_node] ✅ Got {len(result)} tool results")

    return {"messages": result}


def reviser_node(state: MessagesState) -> dict:
    """
    Reviser node - invokes reviser chain.
    
    Returns ReviseAnswer Pydantic object directly (via with_structured_output),
    then converts to AIMessage for the graph state.
    """
    print("[reviser_node] Revising answer with search results...")

    # Chain returns ReviseAnswer Pydantic object directly
    result: ReviseAnswer = reviser.invoke({"messages": state["messages"]})

    # Log the structured output
    print(f"[reviser_node] ✅ Got ReviseAnswer:")
    print(f"  - Answer: {len(result.answer)} chars")
    print(f"  - References: {len(result.references)} refs")
    print(f"  - Search queries: {result.search_queries}")

    # Convert to AIMessage with tool_calls for ToolNode compatibility
    ai_message = pydantic_to_ai_message(result, ReviseAnswer.__name__)

    return {"messages": [ai_message]}


# ============================================================================
# CONDITIONAL EDGE FUNCTION
# ============================================================================

def should_continue(state: MessagesState) -> str:
    """
    Determine whether to continue the loop or end.
    
    Ends if:
    - Max iterations reached
    - search_queries is empty (answer is complete)
    - No valid tool_calls in last message
    """
    messages = state["messages"]

    # Count tool message visits (each search iteration adds ToolMessages)
    tool_message_count = sum(isinstance(msg, ToolMessage) for msg in messages)

    if tool_message_count >= MAX_ITERATIONS:
        print(f"[should_continue] Max iterations ({MAX_ITERATIONS}) reached → END")
        return END

    # Get the last AI message's structured response
    args = get_last_structured_response(messages)

    if args is None:
        print("[should_continue] No structured response found → END")
        return END

    # Check if search_queries is empty (signals completion)
    search_queries = args.get("search_queries", [])

    if not search_queries:
        print("[should_continue] Empty search_queries → END (answer complete)")
        return END

    print(f"[should_continue] {len(search_queries)} queries pending → execute_tools")
    return "execute_tools"


# ============================================================================
# BUILD GRAPH
# ============================================================================

builder = StateGraph(MessagesState)

# Add nodes with RetryPolicy for resilience
# RetryPolicy handles transient errors (network, validation) automatically
default_retry = RetryPolicy(
    max_attempts=3,
    retry_on=(ValidationError, ValueError, TypeError, ConnectionError)
)

builder.add_node("draft", draft_node, retry_policy=default_retry)
builder.add_node("execute_tools", execute_tools_node, retry_policy=default_retry)
builder.add_node("reviser", reviser_node, retry_policy=default_retry)

# Add edges - simplified flow (no separate parse nodes needed!)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "reviser")

# Conditional edge from reviser
builder.add_conditional_edges("reviser", should_continue, {
    END: END,
    "execute_tools": "execute_tools",
})

# Set entry point
builder.set_entry_point("draft")

# Compile with SQLite checkpointer for state persistence
checkpointer = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=checkpointer)

# Visualize the graph
print("\n" + "=" * 60)
print("GRAPH STRUCTURE:")
print("=" * 60)
print(graph.get_graph().draw_ascii())

try:
    graph.get_graph().draw_mermaid_png(output_file_path=str(Path(__file__).parent / "graph.png"))
    print("Graph visualization saved to graph.png")
except Exception as e:
    print(f"Could not save graph visualization: {e}")


# ============================================================================
# STATE INSPECTION UTILITIES
# ============================================================================

def inspect_latest_state(thread_id: str) -> None:
    """Inspect the latest state for a given thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    if state.values:
        print(f"\n{'=' * 60}")
        print(f"Thread: {thread_id}")
        print(f"Next nodes: {state.next}")
        print(f"Message count: {len(state.values.get('messages', []))}")

        messages = state.values.get('messages', [])
        if messages:
            last_msg = messages[-1]
            print(f"Last message type: {type(last_msg).__name__}")
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"Tool calls: {[tc['name'] for tc in last_msg.tool_calls]}")
        print(f"{'=' * 60}\n")
    else:
        print(f"No state found for thread: {thread_id}")


def list_checkpoint_history(thread_id: str, limit: int = 10) -> None:
    """List checkpoint history for debugging."""
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))

    print(f"\n{'=' * 60}")
    print(f"Checkpoint History for Thread: {thread_id}")
    print(f"Total checkpoints: {len(history)}")
    print(f"{'=' * 60}")

    for i, snapshot in enumerate(history[:limit]):
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        metadata = snapshot.metadata
        msg_count = len(snapshot.values.get('messages', []))

        print(f"\n[{i}] Checkpoint: {checkpoint_id[:20]}...")
        print(f"    Step: {metadata.step if hasattr(metadata, 'step') else 'N/A'}")
        print(f"    Source: {metadata.source if hasattr(metadata, 'source') else 'N/A'}")
        print(f"    Messages: {msg_count}")

        tool_msgs = sum(1 for m in snapshot.values.get('messages', [])
                        if isinstance(m, ToolMessage))
        if tool_msgs:
            print(f"    Tool messages: {tool_msgs}")


def get_final_answer(thread_id: str) -> Optional[str]:
    """Extract the final answer from a completed run."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    if not state.values:
        return None

    args = get_last_structured_response(state.values.get("messages", []))
    return args.get("answer") if args else None


def stream_with_debug(question: str, thread_id: str = None) -> dict:
    """Run the graph with debug streaming."""
    if thread_id is None:
        thread_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": [HumanMessage(content=question)]}

    print(f"\n{'=' * 60}")
    print(f"Starting debug stream - Thread: {thread_id}")
    print(f"Question: {question}")
    print(f"{'=' * 60}\n")

    final_state = None
    for i, chunk in enumerate(graph.stream(input_state, config, stream_mode="updates")):
        node_name = list(chunk.keys())[0]
        update = chunk[node_name]

        print(f"[Step {i}] Node: {node_name}")
        if 'messages' in update:
            for msg in update['messages']:
                msg_preview = str(msg.content)[:100] if msg.content else "(empty content)"
                print(f"    → {type(msg).__name__}: {msg_preview}...")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"      Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
        print()
        final_state = chunk

    print(f"\n{'=' * 60}")
    print(f"Stream complete. Thread ID: {thread_id}")
    print(f"{'=' * 60}\n")

    return final_state


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("REFLECTION AGENT - Refactored with Structured Output")
    print("=" * 60)
    print(f"Checkpoints stored in: {CHECKPOINT_DB}")
    print(f"Max iterations: {MAX_ITERATIONS}")

    # Create unique thread ID for this run
    thread_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run with standard invoke
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [
            HumanMessage(
                content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.")
        ]
    }

    print(f"\nStarting run with thread_id: {thread_id}")
    print(f"Question: {initial_state['messages'][0].content}\n")

    try:
        res = graph.invoke(initial_state, config)

        print("\n" + "=" * 60)
        print("FINAL RESULT:")
        print("=" * 60)

        # Extract the final answer
        final_answer = get_final_answer(thread_id)
        if final_answer:
            print(f"\n{final_answer}")
        else:
            print("Could not extract final answer")
            print(res)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # Show state inspection
    print("\n" + "=" * 60)
    print("STATE INSPECTION:")
    print("=" * 60)
    inspect_latest_state(thread_id)
    list_checkpoint_history(thread_id, limit=5)
