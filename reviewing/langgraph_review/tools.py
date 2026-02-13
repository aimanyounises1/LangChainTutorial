from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv(verbose=True)
tavily_client = TavilySearch(max_results=20, search_depth='advanced')


@tool
def tavily_search(query: str):
    """
    tavily search tool to search the query in WEB and returns the results, within Documents format
    """
    res = tavily_client.invoke({'query': query})
    all_doc = [[Document(page_content=result['content'], metadata={'source': result['url']}) for result in
               res['results']]]
    return all_doc



