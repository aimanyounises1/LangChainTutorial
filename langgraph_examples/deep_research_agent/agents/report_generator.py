"""
Deep Research Agent - Report Generator Agent

The Report Generator agent is responsible for:
1. Transforming the research draft into a polished final report
2. Creating proper citation formatting (academic style)
3. Generating executive summary
4. Ensuring professional structure and formatting
5. Creating a complete, publication-ready document
"""

import datetime
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticToolsParser
from langchain_ollama import ChatOllama

from langgraph_examples.deep_research_agent.schemas import (
    ReportGeneratorOutput,
    ResearchDraft,
    ResearchPlan,
    Citation,
    QualityMetrics,
)


# ============================================================================
# REPORT GENERATOR PROMPT
# ============================================================================

REPORT_GENERATOR_SYSTEM_PROMPT = """You are an expert technical writer and report editor specializing in creating professional research documents.

Your task is to transform the research draft into a polished, publication-ready final report with proper formatting and citations.

## Research Context:
- Original Query: {main_query}
- Research Objective: {objective}
- Target Audience: Professional/Technical readers

## Current Draft to Polish:
{current_draft}

## All Available Citations:
{all_citations}

## Quality Metrics Achieved:
{quality_summary}

## Your Responsibilities:

### 1. Create Executive Summary
Write a compelling 150-200 word executive summary that:
- States the research question
- Summarizes key findings
- Highlights main conclusions
- Notes any limitations

### 2. Polish and Restructure Content
- Ensure logical flow between sections
- Add transition sentences between paragraphs
- Remove redundancy
- Improve clarity and precision
- Maintain consistent tone throughout

### 3. Format Citations Properly
Use a consistent citation format throughout:
- Inline citations: [1], [2], [3] etc.
- References section at the end with full details
- Ensure every claim has appropriate citation

### 4. Add Professional Elements
- Clear section headings
- Bullet points where appropriate (but not excessive)
- Tables or lists for comparative information
- Conclusion with actionable insights

### 5. Final Report Structure
```
# [Report Title]

## Executive Summary
[150-200 words]

## 1. Introduction
[Background, context, research objectives]

## 2. Methodology
[Brief description of research approach]

## 3. Key Findings
[Main research findings with citations]

## 4. Analysis & Discussion
[Deeper analysis, comparisons, implications]

## 5. Challenges & Considerations
[Limitations, risks, considerations]

## 6. Future Outlook
[Trends, predictions, recommendations]

## 7. Conclusion
[Summary of key points, call to action]

## References
[Full citation list]
```

## Quality Standards:
- Minimum 1500 words for the main content
- Every factual claim must have a citation
- No speculation without clear caveats
- Professional, objective tone
- Clear, concise language

Current time: {time}

You MUST use the ReportGeneratorOutput tool to provide your final report.
Focus on QUALITY over LENGTH. A well-written focused report is better than a verbose unfocused one.
"""

REPORT_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REPORT_GENERATOR_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Generate the final polished research report. Use the ReportGeneratorOutput tool."),
]).partial(time=lambda: datetime.datetime.now().isoformat())


# ============================================================================
# REPORT GENERATOR LLM CONFIGURATION
# ============================================================================

def create_report_generator_llm(model_name: str = "qwen3:30b-a3b"):
    """Create the LLM configured for the report generator agent."""
    llm = ChatOllama(
        model=model_name,
        temperature=0.3,  # Some creativity for polished writing
        num_ctx=32768,  # Large context for full report generation
    )
    return llm.bind_tools(tools=[ReportGeneratorOutput], tool_choice="ReportGeneratorOutput")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_draft_for_report(draft: ResearchDraft) -> str:
    """Format the draft content for the report generator."""
    parts = []
    
    if draft.abstract:
        parts.append(f"**Current Abstract:**\n{draft.abstract}\n")
    
    for section in draft.sections:
        parts.append(f"**Section: {section.title}** (v{section.version})\n{section.content}\n")
    
    if draft.conclusion:
        parts.append(f"**Current Conclusion:**\n{draft.conclusion}\n")
    
    return "\n---\n".join(parts)


def format_all_citations(citations: List[Citation]) -> str:
    """Format all citations for the report generator."""
    if not citations:
        return "No citations available."
    
    lines = []
    for c in citations:
        title = c.title or "No title"
        lines.append(f"{c.id} {title}\n   URL: {c.url}\n   Excerpt: {c.snippet[:150]}...")
    
    return "\n\n".join(lines)


def format_quality_summary(metrics: QualityMetrics) -> str:
    """Format quality metrics summary."""
    return f"""
- Coverage Score: {metrics.coverage_score:.2f}/1.0
- Depth Score: {metrics.depth_score:.2f}/1.0
- Citation Density: {metrics.citation_density:.2f}/1.0
- Coherence Score: {metrics.coherence_score:.2f}/1.0
- Overall Completeness: {metrics.completeness_score:.2f}/1.0
"""


def create_references_section(citations: List[Citation]) -> str:
    """Create a formatted references section."""
    if not citations:
        return "## References\n\nNo references available."
    
    # Deduplicate citations by URL
    seen_urls = set()
    unique_citations = []
    for c in citations:
        if c.url not in seen_urls:
            seen_urls.add(c.url)
            unique_citations.append(c)
    
    lines = ["## References\n"]
    for c in unique_citations:
        title = c.title or "Untitled"
        lines.append(f"{c.id} {title}. Retrieved from {c.url}")
    
    return "\n\n".join(lines)


def create_fallback_report(
    draft: ResearchDraft,
    plan: ResearchPlan,
    citations: List[Citation]
) -> str:
    """Create a fallback report if LLM fails."""
    parts = [
        f"# Research Report: {plan.main_query}\n",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
        "## Executive Summary\n",
        f"{plan.objective}\n",
        f"This report investigates {plan.main_query} through systematic research and analysis.\n",
        "---\n",
    ]
    
    # Add all sections from draft
    for i, section in enumerate(draft.sections, 1):
        parts.append(f"## {i}. {section.title}\n")
        parts.append(f"{section.content}\n")
        parts.append("---\n")
    
    # Add conclusion if exists
    if draft.conclusion:
        parts.append("## Conclusion\n")
        parts.append(f"{draft.conclusion}\n")
        parts.append("---\n")
    
    # Add references
    parts.append(create_references_section(citations))
    
    return "\n".join(parts)


# ============================================================================
# REPORT GENERATOR CHAIN
# ============================================================================

report_generator_llm = create_report_generator_llm()
report_generator_chain = REPORT_GENERATOR_PROMPT | report_generator_llm

# Parser for extracting structured output
report_generator_parser = PydanticToolsParser(tools=[ReportGeneratorOutput], first_tool_only=True)


# ============================================================================
# HIGH-LEVEL REPORT GENERATION FUNCTION
# ============================================================================

def generate_final_report(
    draft: ResearchDraft,
    plan: ResearchPlan,
    citations: List[Citation],
    quality_metrics: QualityMetrics
) -> tuple[str, dict]:
    """
    Generate the final polished research report.
    
    Args:
        draft: The research draft to polish
        plan: The original research plan
        citations: All collected citations
        quality_metrics: Quality metrics from the last critique
    
    Returns:
        (final_report_text, report_metadata)
    """
    from langchain_core.messages import HumanMessage
    
    # Prepare context
    prompt_vars = {
        "main_query": plan.main_query,
        "objective": plan.objective,
        "current_draft": format_draft_for_report(draft),
        "all_citations": format_all_citations(citations),
        "quality_summary": format_quality_summary(quality_metrics),
        "messages": [HumanMessage(content="Generate the final polished research report.")]
    }
    
    try:
        result = report_generator_chain.invoke(prompt_vars)
        
        if result.tool_calls:
            parsed = report_generator_parser.invoke(result)
            if parsed:
                return parsed.final_report, parsed.report_metadata
    except Exception as e:
        print(f"[ReportGenerator] Error: {e}")
    
    # Fallback
    fallback_report = create_fallback_report(draft, plan, citations)
    fallback_metadata = {
        "generated_at": datetime.datetime.now().isoformat(),
        "word_count": len(fallback_report.split()),
        "citation_count": len(citations),
        "section_count": len(draft.sections),
        "fallback": True
    }
    
    return fallback_report, fallback_metadata


def format_report_as_markdown(report: str) -> str:
    """Ensure report is properly formatted as Markdown."""
    # Basic cleanup
    report = report.strip()
    
    # Ensure proper heading levels
    lines = report.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Ensure consistent heading format
        if line.startswith('# ') or line.startswith('## ') or line.startswith('### '):
            formatted_lines.append('')  # Add blank line before heading
            formatted_lines.append(line)
            formatted_lines.append('')  # Add blank line after heading
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def calculate_report_statistics(report: str, citations: List[Citation]) -> dict:
    """Calculate statistics about the final report."""
    words = report.split()
    
    # Count sections (lines starting with ##)
    section_count = sum(1 for line in report.split('\n') if line.startswith('## '))
    
    # Count citation references in text
    import re
    citation_refs = re.findall(r'\[\d+\]', report)
    
    return {
        "word_count": len(words),
        "character_count": len(report),
        "section_count": section_count,
        "citation_references": len(citation_refs),
        "unique_citations": len(set(citation_refs)),
        "total_citations_available": len(citations),
        "estimated_reading_time_minutes": max(1, len(words) // 200)
    }


if __name__ == "__main__":
    # Test the report generator
    from langgraph_examples.deep_research_agent.schemas import (
        ResearchPlan, SubQuestion, ResearchDraft, DraftSection, Citation, QualityMetrics
    )
    
    # Create test data
    test_plan = ResearchPlan(
        main_query="AI-Powered SOC solutions",
        objective="Analyze the AI SOC landscape",
        scope="Enterprise cybersecurity",
        sub_questions=[
            SubQuestion(id="sq1", question="What is AI SOC?", priority=1, status="completed"),
        ],
        methodology="Iterative research",
        expected_sections=["Intro", "Findings", "Conclusion"]
    )
    
    test_draft = ResearchDraft(
        title="AI SOC Research",
        abstract="This report examines AI-powered Security Operations Centers...",
        sections=[
            DraftSection(
                id="s1",
                title="Introduction",
                content="AI-powered SOC solutions represent a significant advancement in cybersecurity operations. These systems leverage machine learning and artificial intelligence to automate threat detection, incident response, and security monitoring [1]. The market for AI SOC solutions has grown substantially, with enterprises increasingly adopting these technologies to address the cybersecurity skills gap [2].",
                citations=["[1]", "[2]"],
                last_updated="2024-01-01",
                version=2
            ),
            DraftSection(
                id="s2",
                title="Key Players",
                content="Several companies lead the AI SOC market, including Microsoft with Sentinel, Google with Chronicle, and Splunk with SOAR capabilities [3]. Emerging startups like SentinelOne and Cybereason also offer innovative solutions [4].",
                citations=["[3]", "[4]"],
                last_updated="2024-01-01",
                version=1
            )
        ],
        conclusion="AI-powered SOC solutions are transforming enterprise security..."
    )
    
    test_citations = [
        Citation(id="[1]", url="http://example1.com", title="AI SOC Overview", snippet="Overview of AI SOC...", accessed_for="sq1"),
        Citation(id="[2]", url="http://example2.com", title="Market Analysis", snippet="Market growth...", accessed_for="sq1"),
        Citation(id="[3]", url="http://example3.com", title="Vendor Comparison", snippet="Leading vendors...", accessed_for="sq2"),
        Citation(id="[4]", url="http://example4.com", title="Startup Landscape", snippet="Emerging players...", accessed_for="sq2"),
    ]
    
    test_metrics = QualityMetrics(
        coverage_score=0.8,
        depth_score=0.7,
        citation_density=0.75,
        coherence_score=0.8,
        completeness_score=0.77
    )
    
    print("Testing Report Generator Agent...")
    
    report, metadata = generate_final_report(
        draft=test_draft,
        plan=test_plan,
        citations=test_citations,
        quality_metrics=test_metrics
    )
    
    print(f"\nReport Preview:\n{report[:1500]}...")
    print(f"\nMetadata: {metadata}")
    
    stats = calculate_report_statistics(report, test_citations)
    print(f"\nStatistics: {stats}")
