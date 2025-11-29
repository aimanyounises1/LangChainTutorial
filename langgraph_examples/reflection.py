from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.constants import END
from langgraph.graph import StateGraph, add_messages

load_dotenv(verbose=True)




if __name__ == '__main__':
    print("Hello LangGraph")