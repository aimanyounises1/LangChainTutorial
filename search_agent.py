from typing import Any, List

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient

tavily = TavilyClient()


class Source(BaseModel):
    """Schema for the url sources used by the agent this must be a url to the source."""
    url: str = Field(description="the url of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources."""
    answer: str = Field(description="The agent answer to the query.")
    source: List[Source] = Field(default_factory=list ,description="The list of the url sources used to generate the answer to the query.")


@tool
def search(query: str) -> dict[str, Any]:
    """
    Tool to search over the internet and return the result

    :param query: Query to search over
    :return: string with the result
    """
    return tavily.search(query=query)


def main():
    llm = ChatOllama(model="qwen3:30b-a3b",
                     temperature=0.1,
                     reasoning=True)
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the Bitcoin price now?")})

    print(result)
    structured_output = result['structured_response']

    # You can now access fields directly on the structured object:
    print(f"Agent Answer: {structured_output.answer}")
    print("Sources Used:")
    for source in structured_output.source:
        print(f" - {source.url}")


if __name__ == "__main__":
    main()
