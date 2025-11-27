"""
Alternative Implementation: Using LCEL and with_structured_output()

This shows the MODERN way to get structured outputs using LCEL chains.
Compare this to react_agent.py which uses the CLASSIC AgentExecutor pattern.
"""

from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


# Define the structured output schema
class Source(BaseModel):
    """A source URL used to answer the query."""
    url: str = Field(description="The URL of the source")


class ResearchResponse(BaseModel):
    """Structured response with answer and sources."""
    answer: str = Field(description="The answer to the query")
    sources: List[Source] = Field(description="List of sources used")


def search_tavily(query: str) -> dict:
    """Search using Tavily API."""
    return tavily.search(query=query)


def format_search_results(search_response: dict) -> str:
    """Format Tavily search results into a readable string."""
    results = search_response.get("results", [])

    formatted = "Search Results:\n\n"
    for i, result in enumerate(results[:5], 1):
        formatted += f"{i}. {result['title']}\n"
        formatted += f"   URL: {result['url']}\n"
        formatted += f"   Content: {result['content']}\n\n"

    return formatted


# ============================================================================
# APPROACH 1: Using with_structured_output() - RECOMMENDED FOR MODERN LANGCHAIN
# ============================================================================

def create_lcel_chain_with_structured_output():
    """
    Creates an LCEL chain that returns structured output.

    This demonstrates:
    1. Building a chain BEFORE execution (not after)
    2. Using with_structured_output() for guaranteed structure
    3. Proper use of the pipe operator |
    """

    # Initialize LLM
    llm = ChatOllama(
        model="qwen3:30b-a3b",
        temperature=0.1
    )

    # Add structured output capability to the LLM
    structured_llm = llm.with_structured_output(ResearchResponse)

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant.
Based on the search results provided, answer the user's query and list the sources you used.

Search Results:
{search_results}
"""),
        ("human", "{query}")
    ])

    # Build the LCEL chain using pipe operator
    # This is a RUNNABLE CHAIN - not executed until .invoke() is called
    chain = (
        {
            "query": RunnablePassthrough(),
            "search_results": RunnableLambda(search_tavily) | RunnableLambda(format_search_results)
        }
        | prompt
        | structured_llm
    )

    return chain


# ============================================================================
# APPROACH 2: Manual Extraction (like react_agent.py)
# ============================================================================

def create_simple_chain_with_manual_extraction():
    """
    Creates a chain without structured output, requiring manual extraction.

    This is similar to what we did in react_agent.py - simpler but requires
    post-processing to extract structured data.
    """

    llm = ChatOllama(
        model="qwen3:30b-a3b",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant.
Based on the search results provided, answer the user's query.
Make sure to mention which sources you used.

Search Results:
{search_results}
"""),
        ("human", "{query}")
    ])

    # Simple chain without structured output
    chain = (
        {
            "query": RunnablePassthrough(),
            "search_results": RunnableLambda(search_tavily) | RunnableLambda(format_search_results)
        }
        | prompt
        | llm
    )

    return chain


def extract_sources_manually(search_results: dict) -> List[Source]:
    """
    Manually extract sources from search results.
    This is POST-PROCESSING, not part of the LCEL chain.
    """
    sources = []
    for result in search_results.get("results", [])[:5]:
        sources.append(Source(url=result["url"]))
    return sources


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_structured_output():
    """
    Demo: Using with_structured_output() for automatic structure.
    """
    print("\n" + "="*70)
    print("DEMO 1: LCEL Chain with with_structured_output()")
    print("="*70)

    # Build the chain (NOT executed yet - this is the key!)
    chain = create_lcel_chain_with_structured_output()

    # Now execute it
    result = chain.invoke("What is the current price of Bitcoin?")

    # result is automatically a ResearchResponse object!
    print(f"\nType of result: {type(result)}")
    print(f"\nAnswer: {result.answer}")
    print(f"\nSources ({len(result.sources)}):")
    for i, source in enumerate(result.sources, 1):
        print(f"  {i}. {source.url}")


def demo_manual_extraction():
    """
    Demo: Manual extraction (similar to react_agent.py approach).
    """
    print("\n" + "="*70)
    print("DEMO 2: LCEL Chain with Manual Extraction")
    print("="*70)

    # Build the chain
    chain = create_simple_chain_with_manual_extraction()

    # Execute it
    result = chain.invoke("What is the current price of Bitcoin?")

    # result is a string, need to manually extract sources
    print(f"\nType of result: {type(result)}")
    print(f"\nAnswer: {result.content}")

    # Manually get sources (this is POST-PROCESSING)
    search_results = search_tavily("What is the current price of Bitcoin?")
    sources = extract_sources_manually(search_results)

    print(f"\nSources ({len(sources)}):")
    for i, source in enumerate(sources, 1):
        print(f"  {i}. {source.url}")


# ============================================================================
# UNDERSTANDING THE DIFFERENCE
# ============================================================================

def explain_the_difference():
    """
    Explains the key differences between the approaches.
    """
    print("\n" + "="*70)
    print("KEY DIFFERENCES EXPLAINED")
    print("="*70)

    print("""

Approach 1: with_structured_output()
-------------------------------------
✅ LLM automatically returns structured data
✅ Type-safe (Pydantic models)
✅ No manual parsing needed
✅ Clean and modern
❌ Only works with LLMs that support structured output
❌ Requires schema definition upfront

Usage:
    chain = prompt | llm.with_structured_output(Schema)
    result = chain.invoke(input)  # result is Schema object


Approach 2: Manual Extraction
------------------------------
✅ Works with any LLM
✅ More control over extraction logic
✅ Can extract from intermediate steps (like react_agent.py)
❌ Requires manual processing
❌ More code
❌ Less type safety

Usage:
    chain = prompt | llm
    result = chain.invoke(input)  # result is string/dict
    structured = extract_manually(result)  # manual processing


Your react_agent.py Issue:
---------------------------
❌ WRONG:
    result = agent_executor.invoke(...)  # ← Executed (returns dict)
    chain = result | extract | format    # ← Can't pipe dict!

✅ CORRECT (Post-processing):
    result = agent_executor.invoke(...)
    formatted = extract_sources_from_result(result)

✅ CORRECT (LCEL chain):
    chain = prompt | llm | parser  # ← Build chain first
    result = chain.invoke(input)   # ← Then execute

    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demos."""

    # Explain the concepts
    explain_the_difference()

    # Demo 1: Structured output (modern approach)
    demo_structured_output()

    # Demo 2: Manual extraction (classic approach)
    demo_manual_extraction()

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print("""
1. LCEL chains must be built BEFORE execution
2. Use with_structured_output() for automatic structure (modern)
3. Use manual extraction for classic patterns (like AgentExecutor)
4. Never try to pipe already-executed data (dict/string)
    """)


if __name__ == "__main__":
    main()
