from typing import List, Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from pydantic import BaseModel
from pydantic import Field
from tavily import TavilyClient
from dotenv import load_dotenv
from schemas import REACT_PROMPT_TEMPLATE
import os
api_key_check = os.getenv("TAVILY_API_KEY")
print(api_key_check)
load_dotenv()

class Source(BaseModel):
    url: str = Field(description="the url of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="The agent answer to the query.")
    sources: List[Source] = Field(default_factory=list, description="The list of sources.")


@tool
def search_tool(search_query: str) -> dict[str, Any]:
    ''''
    This tool is used to search web using Tavily
    '''
    search_results = TavilyClient.search(query=search_query, time_range='d')
    return search_results


def run_llm():
    llm = ChatOllama(
        model="qwen3:30b-a3b",
        validate_model_on_init=True,
        temperature=0.8,
        reasoning=True
    )
    agent = create_agent(model=llm, tools=[TavilySearch()], response_format=AgentResponse)
    response  = agent.invoke({
        "messages": HumanMessage(content="What are the latest AI trends?")
    }
    )
    print(response)

if __name__ == '__main__':
    run_llm()

