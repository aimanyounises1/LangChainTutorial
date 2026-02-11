from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv(verbose=True)

tavily_search = TavilySearch()
tools = [tavily_search]

llm = init_chat_model(
    "ollama:llama3.3:70b",
    temperature=0.1,
    profile={
        "tool_calling": True,
        "structured_output": True,
    },
)

agent = create_agent(
    llm,
    tools=tools,
    system_prompt="You are a helpful research assistant. Use the tavily_search tool to find information. "
                  "After getting search results, you MUST use the AgentResponse tool to return your final answer "
                  "with the source URLs from the search results.",
    response_format=ToolStrategy(
        AgentResponse,
        handle_errors="Wrong field names. The AgentResponse tool requires: 'answer' (str) and 'sources' (list of objects each with a 'url' field). Please retry with correct field names.",
    ),
)


def main():
    print("Hello first Agent ")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Search for LinkedIn jobs opened in Israel"}]}
    )

    if "structured_response" in result:
        response: AgentResponse = result["structured_response"]
        print(response.answer)
        print("\nSources:")
        for source in response.sources:
            print(f"  - {source.url}")
    else:
        print("Result keys:", result.keys())
        for msg in result["messages"][-3:]:
            print(f"\n[{type(msg).__name__}] tool_calls={getattr(msg, 'tool_calls', None)}")
            print(f"  content: {str(msg.content)[:200]}")


if __name__ == '__main__':
    main()
