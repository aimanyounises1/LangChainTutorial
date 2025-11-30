from dotenv import load_dotenv

from langgraph_examples.reflection_agent.schemas import AnswerQuestion, ReviseAnswer

load_dotenv(verbose=True)
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

tavily_search = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """
    Run the generated queries.
    """
    return tavily_search.batch([{"query": query for query in search_queries}])


execute_tool = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
