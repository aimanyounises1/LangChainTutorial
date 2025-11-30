from typing import Tuple, List

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

# Import the new Object
from langgraph_examples.reflection_agent.schemas import AnswerQuestion, ReviseAnswer, SearchResult, SearchQueriesInput

load_dotenv(verbose=True)

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs) -> Tuple[str, List[SearchResult]]:
    """
    Run queries and return Typed Objects.
    Returns:
        content (str): The clean string for the LLM.
        artifact (List[SearchResult]): The STRICTLY TYPED list of objects for the system.
    """
    # 1. Deduplicate
    unique_queries = list(set(search_queries))

    # 2. Batch Run
    batch_input = [{"query": q} for q in unique_queries]
    results = tavily_tool.batch(batch_input)

    # 3. Create the Artifacts using the NEW OBJECT
    # We convert raw dicts into strict Pydantic Models
    # This is the "List of Annotated Objects" you wanted.
    structured_artifact: List[SearchResult] = []

    formatted_chunks = []

    for i, res_list in enumerate(results):
        query = unique_queries[i]
        formatted_chunks.append(f"### Search: {query}")

        if not res_list:
            formatted_chunks.append("No results found.")
            continue

        for item in res_list['results']:
            # Create the Pydantic Object
            # This ensures every item matches your Schema structure strictly
            result_object = SearchResult(
                url=item.get("url", "Unknown URL"),
                content=item.get("content", "")
            )
            structured_artifact.append(result_object)

            # Format the clean string for the LLM
            formatted_chunks.append(f"- Source: {result_object.url}\n  Content: {result_object.content}\n")

        formatted_chunks.append("---")

    content_str = "\n".join(formatted_chunks)

    # Return the Tuple: (String for LLM, List[SearchResult] for System)
    return content_str, structured_artifact


# 4. Tool Definition
# Using SearchQueriesInput as args_schema since run_queries only needs search_queries
# Tool names must match what the LLM calls (AnswerQuestion and ReviseAnswer)
execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            func=run_queries,
            name=AnswerQuestion.__name__,
            description="Search the web for information to improve the initial answer",
            args_schema=SearchQueriesInput,
            response_format="content_and_artifact"
        ),
        StructuredTool.from_function(
            func=run_queries,
            name=ReviseAnswer.__name__,
            description="Search the web for information to revise the answer",
            args_schema=SearchQueriesInput,
            response_format="content_and_artifact"
        ),
    ]
)
if __name__ == '__main__':
    test_queries = [
        "Best practices for using salicylic acid without causing dryness or irritation",
        "How to differentiate between acne purging and irritation from new skincare products",
        "Effectiveness of non-comedogenic moisturizers for acne-prone skin in clinical studies"
    ]

    # Manually invoking the function to test output
    content, artifact = run_queries(test_queries)

    print("--- CONTENT (Seen by LLM) ---")
    print(content[:500] + "...\n(truncated for view)\n")

    print("--- ARTIFACT (Hidden State) ---")
    print(f"Number of queries run: {len(artifact)}")

