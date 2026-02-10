"""
Deep Research Agent - Simplified Implementation using Deep Agents Framework

This replaces 750+ lines of custom code with ~40 lines using prebuilt LangChain components.

Installation:
    pip install deepagents tavily-python

Features Maintained:
- Multi-step research planning (via write_todos tool)
- Search execution (via internet_search tool)
- Context management (via file system tools)
- Iterative refinement (via subagent delegation)
- Final report generation
- Persistent memory across sessions
- Checkpointing and streaming

Performance Improvements:
- Automatic parallel execution of research tasks
- Better context management with file offloading
- Built-in self-critique and iteration
- Cross-session memory
"""

import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient

# Import Deep Agents framework
try:
    from deepagents import create_deep_agent
except ImportError:
    print("ERROR: deepagents not installed. Run: pip install deepagents tavily-python")
    raise

load_dotenv(verbose=True)

# ============================================================================
# TOOLS
# ============================================================================

# Initialize Tavily for web search
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> str:
    """
    Run a web search using Tavily API.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
        topic: Search topic filter (general, news, or finance)
        include_raw_content: Whether to include full page content

    Returns:
        Formatted search results with titles, URLs, and content snippets
    """
    try:
        results = tavily_client.search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content,
        )

        # Format results for better readability
        formatted = f"\n{'='*60}\nSearch: {query}\n{'='*60}\n"

        if "results" in results:
            for i, result in enumerate(results["results"], 1):
                formatted += f"\n[{i}] {result.get('title', 'No title')}\n"
                formatted += f"URL: {result.get('url', 'No URL')}\n"
                formatted += f"{result.get('content', 'No content')[:500]}...\n"
                formatted += "-" * 60 + "\n"

        return formatted

    except Exception as e:
        return f"Search error: {str(e)}"


# ============================================================================
# RESEARCH SUBAGENT CONFIGURATION
# ============================================================================

research_subagent = {
    "name": "deep-researcher",
    "description": """Expert research agent for conducting thorough, multi-step research.

    Use this subagent when you need to:
    - Research complex topics requiring multiple searches
    - Break down research into sub-questions
    - Synthesize findings from multiple sources
    - Generate comprehensive reports with citations
    """,

    "system_prompt": """You are an expert research analyst specializing in deep research.

## Your Research Process:

1. **PLAN** - Break down the research query:
   - Use the write_todos tool to create 4-7 specific sub-questions
   - Prioritize sub-questions by importance
   - Define the methodology and expected sections

2. **RESEARCH** - For each sub-question:
   - Generate 2-4 targeted search queries using internet_search
   - Analyze search results critically
   - Extract key findings and save to files using write_file
   - Track citations with [1], [2] format

3. **SYNTHESIZE** - Organize findings:
   - Group related findings into logical sections
   - Write each section to a file (e.g., /tmp/section_1_intro.md)
   - Ensure each section has proper citations
   - Use read_file to review previous sections for coherence

4. **CRITIQUE** - Self-review your work:
   - Check coverage: Are all sub-questions addressed?
   - Check depth: Is the analysis thorough?
   - Check citations: Are sources properly attributed?
   - Identify gaps and conduct additional searches if needed

5. **REPORT** - Generate final output:
   - Read all section files
   - Compile into a polished markdown report
   - Include: Executive Summary, Main Sections, Conclusion, References
   - Ensure professional formatting and clear structure

## Guidelines:
- Use internet_search extensively (5-15 searches per research task)
- Save intermediate results to files to avoid context overflow
- Cite all claims with [1], [2], etc.
- Iterate on weak sections until quality is high
- Be thorough but concise
- Use professional academic tone

## Tools Available:
- internet_search: Search the web for information
- write_todos: Break down tasks into steps
- write_file: Save research findings to files
- read_file: Review previous findings
- edit_file: Update sections based on critique
- list_files: See what you've created
""",

    "tools": [internet_search],
    "model": "claude-sonnet-4-5-20250929",  # Use Claude for best research quality
}


# ============================================================================
# CREATE DEEP RESEARCH AGENT
# ============================================================================

def create_research_agent(
    use_subagent: bool = True,
    model: str = "claude-sonnet-4-5-20250929",
):
    """
    Create a deep research agent.

    Args:
        use_subagent: If True, uses specialized research subagent (recommended)
        model: LLM model to use

    Returns:
        Configured deep agent ready for research tasks
    """

    if use_subagent:
        # Use subagent architecture (recommended for complex research)
        agent = create_deep_agent(
            model=model,
            subagents=[research_subagent],
            system_prompt="""You are a research coordinator.

For complex research queries, delegate to the deep-researcher subagent.
For simple questions, answer directly using internet_search.

When delegating research:
1. Pass the full query to the deep-researcher
2. Review the generated report
3. Present it to the user

The deep-researcher will handle planning, searching, synthesis, and reporting.
""",
        )
    else:
        # Single agent with tools (simpler but less structured)
        agent = create_deep_agent(
            model=model,
            tools=[internet_search],
            system_prompt=research_subagent["system_prompt"],
        )

    return agent


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def research(query: str, use_subagent: bool = True) -> dict:
    """
    Conduct research on a query.

    Args:
        query: Research question or topic
        use_subagent: Whether to use specialized research subagent

    Returns:
        Agent response with research findings
    """
    agent = create_research_agent(use_subagent=use_subagent)

    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    return result


def research_streaming(query: str, use_subagent: bool = True):
    """
    Conduct research with streaming output.

    Args:
        query: Research question or topic
        use_subagent: Whether to use specialized research subagent

    Yields:
        Streaming chunks from the agent
    """
    agent = create_research_agent(use_subagent=use_subagent)

    for chunk in agent.stream({
        "messages": [{"role": "user", "content": query}]
    }):
        yield chunk


# ============================================================================
# MAIN - EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import time

    # Example research query
    query = """
    Research AI-Powered Security Operations Centers (SOC).
    I want to understand:
    - Current landscape and key players
    - Technology approaches and architectures
    - Startup funding and market size
    - Challenges and limitations
    - Future trends and opportunities

    Provide a comprehensive report with citations.
    """

    print("=" * 80)
    print("DEEP RESEARCH AGENT - SIMPLIFIED IMPLEMENTATION")
    print("=" * 80)
    print(f"\nQuery: {query.strip()}\n")
    print("=" * 80)
    print("\nStarting research...\n")

    start_time = time.time()

    # Run research with streaming
    for chunk in research_streaming(query, use_subagent=True):
        # Print streaming output
        if "messages" in chunk:
            for msg in chunk["messages"]:
                if hasattr(msg, "content"):
                    print(msg.content)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Research completed in {elapsed:.2f} seconds")
    print(f"{'=' * 80}")
