from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv(verbose=True)


@tool()
def triple(num: float) -> float:
    """
    param num: float
    return: float the triple of the given number
    """
    return num * 3


tools = [TavilySearch(max_results=1), triple]
llm = ChatOllama(model='qwen3:30b-a3b',
                 validate_model_on_init=True,
                 temperature=0.8,
                 reasoning=True,
                 ).bind_tools(tools)
