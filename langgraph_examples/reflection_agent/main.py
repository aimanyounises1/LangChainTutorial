from typing import List

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.graph import MessageGraph, END

from langgraph_examples.reflection_agent.chains import first_responder, reviser
from langgraph_examples.reflection_agent.tools_executor import execute_tools

load_dotenv(verbose=True)

MAX_ITERATIONS = 3

# Note: When running with LangGraph API (langgraph dev), checkpointing is handled automatically
# For standalone execution, you can add MemorySaver when compiling the graph

builder = MessageGraph()

# Nodes
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("reviser", reviser)

# Edges
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "reviser")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END

    # Get the last AI message (from reviser)
    last_ai_message = None
    for msg in reversed(state):
        if hasattr(msg, 'tool_calls'):
            last_ai_message = msg
            break

    # If reviser didn't request tools, we're done
    if last_ai_message is None or not last_ai_message.tool_calls:
        return END

    return "execute_tools"


builder.add_conditional_edges("reviser", event_loop, {
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
