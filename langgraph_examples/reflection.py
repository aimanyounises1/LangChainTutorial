from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.constants import END
from langgraph.graph import StateGraph, add_messages

from langgraph_examples.chain import generate_chain, reflection_chain

load_dotenv(verbose=True)


class MessageGraph(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


REFLECT = 'reflect'
GENERATE = 'generate'


def generate_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflect_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATE, generate_node)
builder.add_node(REFLECT, reflect_node)

builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
if __name__ == '__main__':
    res = graph.invoke({"messages": [HumanMessage(content="")]})
    print("Hello LangGraph")
