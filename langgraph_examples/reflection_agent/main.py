from typing import List

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, BaseMessage, AIMessage
from langgraph.graph import MessageGraph, END

from langgraph_examples.reflection_agent.chains import first_responder, reviser
from langgraph_examples.reflection_agent.tools_executor import execute_tools
from langgraph_examples.reflection_agent.text_tool_call_parser import (
    parse_text_tool_calls,
    detect_text_tool_calls,
)

load_dotenv(verbose=True)

MAX_ITERATIONS = 3

# Note: When running with LangGraph API (langgraph dev), checkpointing is handled automatically
# For standalone execution, you can add MemorySaver when compiling the graph


def parse_draft_tool_calls(state: List[BaseMessage]) -> AIMessage:
    """
    Parse text-based tool calls from draft node output.
    Handles case where model outputs <function-call> as text instead of proper tool_calls.
    """
    # Find the last AI message
    for msg in reversed(state):
        if isinstance(msg, AIMessage):
            # If already has tool_calls, return as-is
            if msg.tool_calls:
                return msg
            # Try to parse from text content
            return parse_text_tool_calls(msg, preserve_content=True)
    raise ValueError("No AI message found in state")


def parse_reviser_tool_calls(state: List[BaseMessage]) -> AIMessage:
    """
    Parse text-based tool calls from reviser node output.
    Handles case where model outputs <function-call> as text instead of proper tool_calls.
    """
    # Find the last AI message
    for msg in reversed(state):
        if isinstance(msg, AIMessage):
            # If already has tool_calls, return as-is
            if msg.tool_calls:
                return msg
            # Try to parse from text content
            return parse_text_tool_calls(msg, preserve_content=True)
    raise ValueError("No AI message found in state")


builder = MessageGraph()

# Nodes - with text tool call parsing as fallback
builder.add_node("draft", first_responder)
builder.add_node("parse_draft", parse_draft_tool_calls)
builder.add_node("execute_tools", execute_tools)
builder.add_node("reviser", reviser)
builder.add_node("parse_reviser", parse_reviser_tool_calls)

# Edges - flow through parser nodes to handle text-based tool calls
builder.add_edge("draft", "parse_draft")
builder.add_edge("parse_draft", "execute_tools")
builder.add_edge("execute_tools", "reviser")
builder.add_edge("reviser", "parse_reviser")


def event_loop(state: List[BaseMessage]) -> str:
    """
    Determine whether to continue the loop or end.
    Checks the LAST message in state (which is now from parse_reviser).
    """
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END

    # Get the last AI message (should be from parse_reviser with populated tool_calls)
    last_ai_message = None
    for msg in reversed(state):
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


builder.add_conditional_edges("parse_reviser", event_loop, {
    END: END,
    "execute_tools": "execute_tools",
})
builder.set_entry_point("draft")
graph = builder.compile()
print(graph.get_graph().draw_ascii())
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == '__main__':
    print("Reflexion Agent")

    res = graph.invoke(
        "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
    )

    print(res)
