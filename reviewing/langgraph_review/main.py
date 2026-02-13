from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState

from reviewing.langgraph_review.nodes import run_agent_reasoning, tool_node

AGENT_REASON = "Agent Reason"
ACT = "Act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if state['messages'][LAST].tool_calls:
        return ACT
    return END


load_dotenv(verbose=True)
# Create the graph nodes
flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)
flow.set_entry_point(AGENT_REASON)

# Now we're defining the edges of the graph
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    ACT: ACT,
    END: END
})
flow.add_edge(ACT, AGENT_REASON)

graph = flow.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == '__main__':
    print("First Graph Review")
    res= graph.invoke({
        "messages": [HumanMessage(content="What is OpenClaw, why ppl going crazy about it?")]
    })
    print(res['messages'][LAST].content)