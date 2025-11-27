from typing import Any

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langsmith import Client
from tavily import TavilyClient

from schemas import REACT_PROMPT_TEMPLATE
from search_agent import AgentResponse

load_dotenv()

# Initialize LangSmith client for pulling prompts from hub
hub_client = Client()

tavily = TavilyClient()


@tool
def search_tool(query: str) -> dict[str, Any]:
    """
    Search the internet for information.

    Args:
        query: The search query string

    Returns:
        Dictionary containing search results
    """
    return tavily.search(query=query)

def main():
    """Run the ReAct agent with a sample query."""
    llm = ChatOllama(model="qwen3:30b-a3b",
                     temperature=0.1,
                     reasoning=True)
    output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
    react_prompt_with_format_instructions = PromptTemplate(
        template=REACT_PROMPT_TEMPLATE,
        input_variables=["input", "agent_scratchpad", "tool_names"],
    ).partial(format_instructions=output_parser.get_format_instructions())
    # Run the agent with a query
    agent = create_react_agent(
        llm=llm,
        tools=[search_tool],
        prompt=react_prompt_with_format_instructions,
    )
    agent_executor = AgentExecutor(agent=agent,
                                   tools=[search_tool],
                                   verbose=True,
                                   handle_parsing_errors=True)
    chain = agent_executor
    result = chain.invoke(
        input= {
        "input": "What is the current price of Bitcoin?"
    })

    print(result)


if __name__ == "__main__":
    main()


