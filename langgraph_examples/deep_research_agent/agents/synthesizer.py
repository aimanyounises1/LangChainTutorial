"""
Deep Research Agent - Synthesizer Agent

The Synthesizer agent is responsible for:
1. Analyzing new search results
2. Extracting key insights and findings
3. Integrating findings into the evolving draft
4. Ensuring proper citation usage
5. Maintaining coherent narrative flow
"""

import datetime
import uuid
from typing import List, Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticToolsParser
from langchain_ollama import ChatOllama

from langgraph_examples.deep_research_agent.schemas import (
    SynthesizerOutput,
    DraftSection,
    Citation,
    ResearchDraft,
    SubQuestion,
)


# ============================================================================
# SYNTHESIZER PROMPT
# ============================================================================

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert research synthesizer and technical writer specializing in creating comprehensive research documents.

Your task is to integrate new research findings into an evolving research draft, ensuring proper citations and coherent narrative.

## Context:
- Main Research Query: {main_query}
- Current Sub-Question Being Addressed: {sub_question}
- Sub-Question ID: {sub_question_id}
- Target Section: {target_section}

## Current Draft State:
{current_draft}

## Available Citations:
{available_citations}

## New Search Results to Integrate:
{search_results}

## Your Responsibilities:

1. **Analyze the New Information**: Identify key insights, facts, statistics, and conclusions from the search results.

2. **Synthesize with Existing Content**: 
   - If the section exists, enhance and expand it with new findings
   - If it's a new section, create comprehensive content
   - Ensure logical flow with other sections

3. **Apply Inline Citations**: 
   - Use citation format like [1], [2] immediately after the relevant claim
   - Every factual claim MUST have a citation
   - Use multiple citations when multiple sources support a claim [1][2]

4. **Maintain Quality**:
   - Write in clear, professional prose
   - Use specific data and examples when available
   - Avoid vague statements without supporting evidence
   - Keep paragraphs focused and well-structured

## Writing Guidelines:
- Start sections with a clear topic sentence
- Use transition phrases between paragraphs
- Include specific numbers, percentages, and examples when available
- End sections with implications or connections to the broader topic
- Minimum 200 words per section for substantive coverage

Current time: {time}

You MUST use the SynthesizerOutput tool to provide your updated section.
"""

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYNTHESIZER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Synthesize the new findings into the draft section. Use the SynthesizerOutput tool."),
]).partial(time=lambda: datetime.datetime.now().isoformat())


# ============================================================================
# SYNTHESIZER LLM CONFIGURATION
# ============================================================================

def create_synthesizer_llm(model_name: str = "qwen3:30b-a3b"):
    """Create the LLM configured for the synthesizer agent."""
    llm = ChatOllama(
        model=model_name,
        temperature=0.2,  # Some creativity for writing
        num_ctx=16384,  # Larger context for draft + results
    )
    return llm.bind_tools(tools=[SynthesizerOutput], tool_choice="SynthesizerOutput")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_draft_for_context(draft: Optional[ResearchDraft]) -> str:
    """Format the current draft for inclusion in the prompt."""
    if not draft:
        return "No draft exists yet. This will be the first content."
    
    parts = [f"# {draft.title}\n"]
    
    if draft.abstract:
        parts.append(f"## Abstract\n{draft.abstract}\n")
    
    for section in draft.sections:
        parts.append(f"## {section.title}\n{section.content}\n")
    
    if draft.conclusion:
        parts.append(f"## Conclusion\n{draft.conclusion}\n")
    
    return "\n".join(parts)


def format_citations_for_context(citations: List[Citation]) -> str:
    """Format citations for inclusion in the prompt."""
    if not citations:
        return "No citations available yet."
    
    parts = []
    for c in citations:
        parts.append(f"{c.id} - {c.title or 'No title'}\n  URL: {c.url}\n  Snippet: {c.snippet[:200]}...")
    
    return "\n".join(parts)


def determine_target_section(
    sub_question: SubQuestion,
    expected_sections: List[str],
    existing_sections: List[DraftSection]
) -> str:
    """Determine which section the sub-question findings should go into."""
    # Map sub-question priorities to sections
    # Priority 1 (foundational) -> Early sections (Introduction, Background)
    # Priority 2 (analytical) -> Middle sections (Analysis, Discussion)
    # Priority 3 (forward-looking) -> Later sections (Trends, Recommendations)
    
    sq_lower = sub_question.question.lower()
    
    # Keyword-based mapping
    if any(kw in sq_lower for kw in ["what is", "definition", "background", "overview", "introduction"]):
        return "Introduction & Background"
    elif any(kw in sq_lower for kw in ["how", "compare", "analysis", "evaluate", "key"]):
        return "Key Findings & Analysis"
    elif any(kw in sq_lower for kw in ["challenge", "problem", "issue", "limitation"]):
        return "Challenges & Considerations"
    elif any(kw in sq_lower for kw in ["trend", "future", "predict", "outlook"]):
        return "Future Outlook & Trends"
    elif any(kw in sq_lower for kw in ["example", "case", "startup", "company", "player"]):
        return "Key Players & Examples"
    else:
        # Default based on priority
        if sub_question.priority == 1:
            return "Introduction & Background"
        elif sub_question.priority == 2:
            return "Key Findings & Analysis"
        else:
            return "Discussion & Implications"


def create_section(
    title: str,
    content: str,
    citations_used: List[str]
) -> DraftSection:
    """Create a new draft section."""
    return DraftSection(
        id=f"sec_{uuid.uuid4().hex[:8]}",
        title=title,
        content=content,
        citations=citations_used,
        last_updated=datetime.datetime.now().isoformat(),
        version=1
    )


def update_section(
    existing: DraftSection,
    new_content: str,
    additional_citations: List[str]
) -> DraftSection:
    """Update an existing section with new content."""
    # Combine content intelligently
    combined_content = f"{existing.content}\n\n{new_content}"
    combined_citations = list(set(existing.citations + additional_citations))
    
    return DraftSection(
        id=existing.id,
        title=existing.title,
        content=combined_content,
        citations=combined_citations,
        last_updated=datetime.datetime.now().isoformat(),
        version=existing.version + 1
    )


def initialize_draft(main_query: str) -> ResearchDraft:
    """Initialize a new research draft."""
    return ResearchDraft(
        title=f"Research Report: {main_query}",
        abstract=None,
        sections=[],
        conclusion=None,
        version=1
    )


# ============================================================================
# SYNTHESIZER CHAIN
# ============================================================================

synthesizer_llm = create_synthesizer_llm()
synthesizer_chain = SYNTHESIZER_PROMPT | synthesizer_llm

# Parser for extracting structured output
synthesizer_parser = PydanticToolsParser(tools=[SynthesizerOutput], first_tool_only=True)


# ============================================================================
# HIGH-LEVEL SYNTHESIS FUNCTION
# ============================================================================

def synthesize_findings(
    sub_question: SubQuestion,
    search_results: str,
    main_query: str,
    current_draft: Optional[ResearchDraft],
    available_citations: List[Citation],
    expected_sections: List[str]
) -> tuple[DraftSection, List[Citation], str]:
    """
    Synthesize search results into the draft.
    
    Args:
        sub_question: The sub-question that was researched
        search_results: Formatted search results
        main_query: The main research query
        current_draft: The current state of the draft
        available_citations: All available citations
        expected_sections: Expected sections from the plan
    
    Returns:
        (updated_section, new_citations, synthesis_notes)
    """
    from langchain_core.messages import HumanMessage
    
    # Determine target section
    existing_sections = current_draft.sections if current_draft else []
    target_section = determine_target_section(sub_question, expected_sections, existing_sections)
    
    # Prepare context
    prompt_vars = {
        "main_query": main_query,
        "sub_question": sub_question.question,
        "sub_question_id": sub_question.id,
        "target_section": target_section,
        "current_draft": format_draft_for_context(current_draft),
        "available_citations": format_citations_for_context(available_citations),
        "search_results": search_results,
        "messages": [HumanMessage(content=f"Synthesize the research findings for: {sub_question.question}")]
    }
    
    try:
        result = synthesizer_chain.invoke(prompt_vars)
        
        if result.tool_calls:
            parsed = synthesizer_parser.invoke(result)
            if parsed:
                return parsed.updated_section, parsed.new_citations, parsed.synthesis_notes
    except Exception as e:
        print(f"[Synthesizer] Error: {e}")
    
    # Fallback: create a basic section
    fallback_section = create_section(
        title=target_section,
        content=f"## Findings for: {sub_question.question}\n\n{search_results[:2000]}",
        citations_used=[c.id for c in available_citations[-5:]]  # Use recent citations
    )
    
    return fallback_section, [], "Fallback synthesis due to error"


def update_draft_with_section(
    draft: Optional[ResearchDraft],
    new_section: DraftSection,
    main_query: str
) -> ResearchDraft:
    """
    Update the draft with a new or updated section.
    
    Args:
        draft: The current draft (or None)
        new_section: The section to add or update
        main_query: The main query (for initializing new draft)
    
    Returns:
        Updated ResearchDraft
    """
    if draft is None:
        draft = initialize_draft(main_query)
    
    # Check if section exists
    existing_idx = None
    for i, sec in enumerate(draft.sections):
        if sec.title == new_section.title:
            existing_idx = i
            break
    
    if existing_idx is not None:
        # Update existing section
        existing = draft.sections[existing_idx]
        updated = update_section(existing, new_section.content, new_section.citations)
        draft.sections[existing_idx] = updated
    else:
        # Add new section
        draft.sections.append(new_section)
    
    # Increment draft version
    draft.version += 1
    
    return draft


if __name__ == "__main__":
    # Test the synthesizer
    from langgraph_examples.deep_research_agent.schemas import SubQuestion, Citation
    
    test_sq = SubQuestion(
        id="sq_test123",
        question="What are the leading AI-powered SOC platforms?",
        priority=1,
        status="in_progress"
    )
    
    test_citations = [
        Citation(
            id="[1]",
            url="https://example.com/soc",
            title="AI SOC Platforms Review",
            snippet="Leading platforms include...",
            accessed_for="sq_test123"
        )
    ]
    
    test_results = """
    ### Search Query: "AI SOC platforms"
    
    **Source 1:** AI-Powered SOC Guide
    - URL: https://example.com/soc
    - Content: The leading AI-powered SOC platforms include Microsoft Sentinel, Google Chronicle, and Splunk SOAR...
    """
    
    print("Testing Synthesizer Agent...")
    
    section, citations, notes = synthesize_findings(
        sub_question=test_sq,
        search_results=test_results,
        main_query="AI-Powered SOC solutions",
        current_draft=None,
        available_citations=test_citations,
        expected_sections=["Introduction", "Key Findings", "Conclusion"]
    )
    
    print(f"Section Title: {section.title}")
    print(f"Content preview: {section.content[:500]}...")
    print(f"Synthesis notes: {notes}")
