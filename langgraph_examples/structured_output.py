"""
LangGraph Agent with Structured Output.

This demonstrates how to get structured (typed) outputs from a LangGraph agent
using Pydantic models. Useful when you need guaranteed response schemas.

References:
- https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/
"""

from typing import List, Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


# Define structured output schema
class Source(BaseModel):
    """A source URL with title."""
    url: str = Field(description="The URL of the source")
    title: str = Field(description="The title of the source")


class ResearchResponse(BaseModel):
    """Structured response from the research agent."""
    answer: str = Field(description="The comprehensive answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )
    confidence: float = Field(
        description="Confidence score from 0 to 1",
        ge=0,
        le=1
    )


@tool
def search_web(query: str) -> dict[str, Any]:
    """
    Search the web for information.

    Args:
        query: The search query

    Returns:
        Search results with URLs and content
    """
    return tavily.search(query=query)


def create_structured_agent():
    """
    Create a LangGraph agent that returns structured output.

    Returns:
        A compiled agent with structured response format
    """
    llm = ChatOllama(
        model="qwen3:30b-a3b",
        temperature=0.1,
    )

    # Create agent with structured output
    agent = create_react_agent(
        model=llm,
        tools=[search_web],
        response_format=ResearchResponse,
    )

    return agent


def main():
    """Run the structured output agent."""
    agent = create_structured_agent()

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "What is quantum computing and its applications?"}
        ]
    })

    print("\n" + "=" * 50)
    print("Structured Agent Response")
    print("=" * 50)

    # Access the structured response
    if "structured_response" in result:
        response: ResearchResponse = result["structured_response"]
        print(f"\nAnswer: {response.answer}")
        print(f"\nConfidence: {response.confidence:.0%}")
        print(f"\nSources ({len(response.sources)}):")
        for source in response.sources:
            print(f"  - {source.title}: {source.url}")
    else:
        # Fallback to regular message output
        if result.get("messages"):
            print(result["messages"][-1].content)


if __name__ == "__main__":
    main()