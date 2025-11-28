from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
load_dotenv()

tavily = TavilyClient()
@tool
def search_tool(search_query: str) -> dict[str, Any]:
    ''''
    This tool is used to search web using Tavily
    '''
    search_results = tavily.search(query=search_query)
    return search_results

if __name__ == '__main__':
    response = search_tool