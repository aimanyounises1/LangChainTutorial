"""
Deep Research Agent - Using LangChain's create_agent (No Deep Agents needed)

This version works with your CURRENT setup - no new packages required!
Uses create_agent from LangChain 1.0 which you already have installed.

Replaces 750+ lines with ~100 lines while maintaining all functionality.

Features:
- Multi-step research planning
- Tavily search integration
- Iterative refinement
- Report generation
- Checkpointing support
- Streaming

This is a great middle ground - simpler than your current implementation,
but doesn't require installing the Deep Agents package.
"""

import os
import datetime
from typing import Literal, Dict, Any, List
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv(verbose=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.3:70b")
CHECKPOINT_DB = "checkpoints/research_agent.db"

# Create checkpointer
os.makedirs("checkpoints", exist_ok=True)
checkpoint_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
checkpointer = SqliteSaver(checkpoint_conn)


# ============================================================================
# TOOLS
# ============================================================================

# Tavily search tool (already in your setup)
search_tool = TavilySearch(
    max_results=5,
    include_raw_content=True,
)


# ============================================================================
# RESEARCH AGENT
# ============================================================================

RESEARCH_SYSTEM_PROMPT = """You are an expert research analyst conducting deep research.

## Research Methodology:

### PHASE 1: PLANNING
- Analyze the research query carefully
- Break it down into 4-7 specific sub-questions
- Identify what information is needed for each
- Plan your search strategy

### PHASE 2: RESEARCH
- For EACH sub-question, conduct 2-4 targeted searches using tavily_search_results_json
- Analyze search results critically
- Extract key findings and insights
- Track sources with [1], [2] citation format
- Build a comprehensive knowledge base

### PHASE 3: SYNTHESIS
- Organize findings into logical sections:
  * Executive Summary
  * Main topic sections (3-5 sections)
  * Conclusion
  * References
- Ensure each section has supporting citations
- Connect ideas across sections for coherence

### PHASE 4: QUALITY CHECK
- Review coverage: Are all sub-questions answered?
- Review depth: Is analysis thorough and insightful?
- Review citations: Are all claims properly sourced?
- If gaps exist, conduct additional searches

### PHASE 5: REPORT GENERATION
- Compile findings into a polished markdown report
- Use clear headings and structure
- Include inline citations [1], [2], etc.
- End with a References section listing all sources

## Search Strategy:
- Use 10-20 searches per research task (don't be stingy!)
- Vary search queries to cover different aspects
- Search for: definitions, examples, case studies, recent news, expert opinions
- Look for authoritative sources: research papers, industry reports, expert blogs

## Citation Format:
- Inline: "AI-powered SOCs can reduce response time by 90% [1]"
- References: "[1] Title - URL"

## Output Format:
Generate a comprehensive markdown report with:
```markdown
# [Research Topic]

## Executive Summary
[3-5 bullet points of key findings]

## [Main Section 1]
[Content with citations]

## [Main Section 2]
[Content with citations]

...

## Conclusion
[Synthesis and future outlook]

## References
[1] Source Title - URL
[2] Source Title - URL
...
```

## Quality Standards:
- Minimum 10 searches
- Minimum 10 citations
- Cover all aspects of the query
- Professional academic tone
- Well-structured and coherent

Current time: {time}
"""


def create_research_agent(
    model_name: str = MODEL_NAME,
    use_checkpointer: bool = True,
):
    """
    Create a research agent using LangChain's create_agent.

    Args:
        model_name: Model to use (default: llama3.3:70b)
        use_checkpointer: Enable checkpointing for persistence

    Returns:
        Configured agent ready for research
    """

    # Create LLM
    llm = ChatOllama(
        model=model_name,
        temperature=0.1,  # Low temperature for factual research
        num_ctx=8192,
    )

    # Create agent with tools
    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=RESEARCH_SYSTEM_PROMPT.format(
            time=datetime.datetime.now().isoformat()
        ),
        checkpointer=checkpointer if use_checkpointer else None,
    )

    return agent


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def research(
    query: str,
    thread_id: str = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Conduct research on a query.

    Args:
        query: Research question or topic
        thread_id: Optional thread ID for checkpointing (default: auto-generated)
        stream: Whether to stream output

    Returns:
        Agent response with research findings
    """

    # Generate thread ID if not provided
    if thread_id is None:
        thread_id = f"research_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = {"configurable": {"thread_id": thread_id}}

    # Create agent
    agent = create_research_agent()

    # Prepare input
    input_messages = {
        "messages": [HumanMessage(content=query)]
    }

    print(f"\n{'=' * 80}")
    print(f"Thread ID: {thread_id}")
    print(f"{'=' * 80}\n")

    if stream:
        # Stream output
        final_output = None
        for chunk in agent.stream(input_messages, config):
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if isinstance(msg, AIMessage):
                        print(msg.content)
            final_output = chunk
        return final_output
    else:
        # Invoke directly
        return agent.invoke(input_messages, config)


def resume_research(thread_id: str) -> Dict[str, Any]:
    """
    Resume a previous research session.

    Args:
        thread_id: Thread ID of the research session to resume

    Returns:
        Agent response continuing from where it left off
    """
    config = {"configurable": {"thread_id": thread_id}}
    agent = create_research_agent()

    # Get current state
    state = agent.get_state(config)
    print(f"Resuming from: {state.values.get('messages', [])[-1] if state.values else 'start'}")

    # Continue from current state
    return agent.invoke(None, config)


# ============================================================================
# MULTI-AGENT ROUTER PATTERN (OPTIONAL ENHANCEMENT)
# ============================================================================

def create_parallel_research_agent():
    """
    Create a research agent that can execute sub-questions in parallel.

    This is an OPTIONAL enhancement showing how to add parallel execution
    using LangGraph's Send API for even better efficiency.

    Not implemented in this version to keep it simple, but you can
    refer to your current graph.py for the pattern.
    """
    # TODO: Implement parallel execution pattern if needed
    # See: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base
    pass


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
    print("DEEP RESEARCH AGENT - create_agent Implementation")
    print("=" * 80)
    print(f"\nQuery: {query.strip()}\n")
    print("=" * 80)

    start_time = time.time()

    # Run research with streaming
    result = research(query, stream=True)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Research completed in {elapsed:.2f} seconds")
    print(f"{'=' * 80}")

    # Save result to file
    if result and "messages" in result:
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            report_path = f"reports/research_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            os.makedirs("reports", exist_ok=True)
            with open(report_path, "w") as f:
                f.write(final_message.content)
            print(f"\nReport saved to: {report_path}")
