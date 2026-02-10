"""
Deep Research Agent - Researcher Agent

The Researcher agent is responsible for:
1. Generating effective search queries for each sub-question
2. Executing searches via Tavily
3. Processing and structuring search results
4. Managing the search strategy
"""

import datetime
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from langgraph_examples.deep_research_agent.schemas import (
    ResearcherOutput,
    Citation,
    SubQuestion,
)

load_dotenv(verbose=True)

# ============================================================================
# RESEARCHER PROMPT
# ============================================================================

RESEARCHER_SYSTEM_PROMPT = """You are an expert research analyst specializing in information retrieval and search strategy.

Your task is to generate optimal search queries to investigate a specific sub-question as part of a larger research effort.

## Context:
- Main Research Query: {main_query}
- Current Sub-Question: {sub_question}
- Sub-Question ID: {sub_question_id}
- Previous Findings (if any): {previous_findings}

## Your Responsibilities:

1. **Analyze the Sub-Question**: Understand exactly what information is needed.

2. **Generate Search Queries**: Create 2-4 specific, targeted search queries that will:
   - Find authoritative sources
   - Cover different aspects of the sub-question
   - Use appropriate search operators and keywords
   - Avoid redundancy with previous searches

3. **Explain Your Strategy**: Briefly describe why you chose these queries.

## Search Query Best Practices:
- Use specific keywords, not vague terms
- Include industry-specific terminology when relevant
- Use quotes for exact phrases when needed
- Consider different angles (definition, examples, comparisons, recent news)

Current time: {time}

You MUST use the ResearcherOutput tool to provide your search queries.
"""

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RESEARCHER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Generate optimal search queries for the sub-question. Use the ResearcherOutput tool."),
]).partial(time=lambda: datetime.datetime.now().isoformat())


# ============================================================================
# RESEARCHER LLM CONFIGURATION
# ============================================================================

def create_researcher_llm(model_name: str = "llama3.3:70b"):
    """Create the LLM configured for the researcher agent."""
    llm = ChatOllama(
        model=model_name,
        temperature=0.1,  # Slight creativity for query variation
        num_ctx=8192,
    )
    return llm.bind_tools(tools=[ResearcherOutput], tool_choice="ResearcherOutput")


# ============================================================================
# SEARCH EXECUTION
# ============================================================================

# Initialize Tavily search tool
tavily_search = TavilySearch(max_results=5)


def execute_search_queries(queries: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Execute search queries and return formatted results.
    
    Returns:
        content (str): Formatted content for LLM consumption
        raw_results (List[Dict]): Raw search results for citation extraction
    """
    unique_queries = list(set(queries))
    batch_input = [{"query": q} for q in unique_queries]

    try:
        results = tavily_search.batch(batch_input)
    except Exception as e:
        print(f"[Researcher] Search error: {e}")
        return f"Search failed: {str(e)}", []

    all_raw_results = []
    formatted_chunks = []

    for i, res_data in enumerate(results):
        query = unique_queries[i]
        formatted_chunks.append(f"\n### Search Query: \"{query}\"\n")

        # Handle case where res_data is a string (raw JSON or error)
        if isinstance(res_data, str):
            import json
            try:
                res_data = json.loads(res_data)
            except json.JSONDecodeError:
                # It's just a plain text result, treat it as content
                formatted_chunks.append(f"Result: {res_data[:500]}...\n")
                all_raw_results.append({
                    "query": query,
                    "url": "Unknown",
                    "title": "Search Result",
                    "content": res_data,
                    "rank": 1
                })
                continue

        if not res_data or not isinstance(res_data, dict) or 'results' not in res_data:
            formatted_chunks.append("No results found for this query.\n")
            continue

        for j, item in enumerate(res_data['results'], 1):
            url = item.get("url", "Unknown URL")
            title = item.get("title", "No title")
            content = item.get("content", "No content")

            # Store raw result for citation creation
            all_raw_results.append({
                "query": query,
                "url": url,
                "title": title,
                "content": content,
                "rank": j
            })

            # Format for LLM
            formatted_chunks.append(f"""
**Source {j}:** {title}
- URL: {url}
- Content: {content}
""")

        formatted_chunks.append("---")

    content_str = "\n".join(formatted_chunks)
    return content_str, all_raw_results


def create_citations_from_results(
        raw_results: List[Dict[str, Any]],
        sub_question_id: str,
        existing_citation_count: int = 0
) -> List[Citation]:
    """
    Create Citation objects from raw search results.
    
    Args:
        raw_results: Raw search results from Tavily
        sub_question_id: ID of the sub-question these citations support
        existing_citation_count: Number of existing citations (for ID numbering)
    
    Returns:
        List of Citation objects
    """
    citations = []

    for i, result in enumerate(raw_results):
        citation_num = existing_citation_count + i + 1
        citation = Citation(
            id=f"[{citation_num}]",
            url=result.get("url", ""),
            title=result.get("title"),
            snippet=result.get("content", "")[:500],  # Truncate long snippets
            accessed_for=sub_question_id
        )
        citations.append(citation)

    return citations


# ============================================================================
# RESEARCHER CHAIN
# ============================================================================

researcher_llm = create_researcher_llm()
researcher_chain = RESEARCHER_PROMPT | researcher_llm

# Parser for extracting structured output
researcher_parser = PydanticToolsParser(tools=[ResearcherOutput], first_tool_only=True)


# ============================================================================
# HIGH-LEVEL RESEARCH FUNCTION
# ============================================================================

def research_sub_question(
        sub_question: SubQuestion,
        main_query: str,
        previous_findings: str = "",
        existing_citations: int = 0
) -> Tuple[str, List[Citation], List[str]]:
    """
    Complete research pipeline for a single sub-question.
    
    Args:
        sub_question: The sub-question to research
        main_query: The main research query for context
        previous_findings: Any previous findings for context
        existing_citations: Count of existing citations
    
    Returns:
        (search_content, new_citations, queries_used)
    """
    from langchain_core.messages import HumanMessage

    # Generate search queries
    prompt_vars = {
        "main_query": main_query,
        "sub_question": sub_question.question,
        "sub_question_id": sub_question.id,
        "previous_findings": previous_findings or "None yet",
        "messages": [HumanMessage(content=f"Research this: {sub_question.question}")]
    }

    try:
        result = researcher_chain.invoke(prompt_vars)

        if result.tool_calls:
            parsed = researcher_parser.invoke(result)
            if parsed:
                queries = parsed.search_queries
            else:
                # Fallback queries
                queries = [sub_question.question, f"{sub_question.question} 2024"]
        else:
            queries = [sub_question.question,
                       f"{sub_question.question} Date : {datetime.datetime.today().strftime('%Y-%m-%d')}"]
    except Exception as e:
        print(f"[Researcher] Query generation error: {e}")
        queries = [sub_question.question]

    # Execute searches
    content, raw_results = execute_search_queries(queries)

    # Create citations
    citations = create_citations_from_results(
        raw_results,
        sub_question.id,
        existing_citations
    )

    return content, citations, queries


if __name__ == "__main__":
    # Test the researcher
    from langgraph_examples.deep_research_agent.schemas import SubQuestion

    test_sq = SubQuestion(
        id="sq_test123",
        question="What are the leading AI-powered SOC platforms and their key features?",
        priority=1,
        status="in_progress"
    )

    print("Testing Researcher Agent...")
    print(f"Sub-question: {test_sq.question}\n")

    content, citations, queries = research_sub_question(
        test_sq,
        main_query="AI-Powered SOC solutions",
        previous_findings=""
    )

    print(f"Queries used: {queries}")
    print(f"\nCitations found: {len(citations)}")
    for c in citations[:3]:
        print(f"  - {c.id}: {c.title}")
    print(f"\nContent preview:\n{content[:1000]}...")
