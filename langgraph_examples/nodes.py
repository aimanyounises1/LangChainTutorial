from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from langgraph_examples.react import llm, tools

load_dotenv(verbose=True)

SYSTEM_MESSAGE = """"
You are helpful  assistant that can use tools to answer questions. 
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node
    """
    response = llm.invoke([{'role': 'system', 'content': SYSTEM_MESSAGE}, *state['messages']])

    return {'messages': [response]}


tool_node = ToolNode(tools)
