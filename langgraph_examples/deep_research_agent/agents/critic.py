import datetime
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from langgraph_examples.deep_research_agent.schemas import (
    CriticOutput,
    CritiqueResult,
    QualityMetrics,
    ResearchDraft,
    ResearchPlan,
    SubQuestion,
    Citation,
    StopConditionConfig,
)

CRITIC_SYSTEM_PROMPT = """You are an expert research quality evaluator and academic reviewer.

Your task is to critically evaluate the current research draft and determine if it meets the quality standards for a comprehensive deep research report.

## Research Context:
- Original Query: {main_query}
- Research Objective: {objective}
- Research Scope: {scope}

## Research Plan - Sub-Questions Status:
{sub_questions_status}

## Current Draft:
{current_draft}

## Citations Used:
{citations_summary}

## Iteration Info:
- Current Iteration: {iteration}
- Maximum Iterations: {max_iterations}
- Previous Critique Scores: {previous_scores}

## Your Evaluation Criteria:

### 1. Coverage Score (0-1):
How well does the draft address all sub-questions?
- 1.0: All sub-questions fully addressed with depth
- 0.7-0.9: Most questions addressed, minor gaps
- 0.4-0.6: Several questions partially addressed
- 0.0-0.3: Major questions unanswered

### 2. Depth Score (0-1):
How deep is the analysis?
- 1.0: Expert-level analysis with nuanced insights
- 0.7-0.9: Good analysis with supporting evidence
- 0.4-0.6: Surface-level analysis
- 0.0-0.3: Superficial or lacking analysis

### 3. Citation Density (0-1):
How well are claims supported?
- 1.0: Every claim properly cited
- 0.7-0.9: Most claims cited
- 0.4-0.6: Some claims lack citations
- 0.0-0.3: Poor citation coverage

### 4. Coherence Score (0-1):
How well does the draft flow?
- 1.0: Excellent narrative flow, clear structure
- 0.7-0.9: Good flow with minor issues
- 0.4-0.6: Some disjointed sections
- 0.0-0.3: Incoherent or poorly structured

### 5. Completeness Score (0-1):
Overall completeness assessment
- 1.0: Ready for publication
- 0.7-0.9: Nearly complete, minor additions needed
- 0.4-0.6: Significant work remaining
- 0.0-0.3: Early stages, much work needed

## Stop Condition Thresholds:
- Minimum coverage: {min_coverage}
- Minimum depth: {min_depth}
- Minimum citations: {min_citations}
- Minimum completeness: {min_completeness}

## Decision Guidelines:

**Recommend "finalize"** if:
- Completeness score >= {min_completeness}
- Coverage score >= {min_coverage}
- All high-priority sub-questions addressed
- No critical gaps identified

**Recommend "continue"** if:
- Scores below thresholds but improving
- Specific actionable improvements identified
- Iteration < max_iterations

**Recommend "stop"** if:
- Max iterations reached
- No improvement in last 2 iterations
- Diminishing returns observed

Current time: {time}

You MUST use the CriticOutput tool to provide your evaluation.
Be STRICT but FAIR in your assessment. The goal is quality research, not endless iteration.
"""

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CRITIC_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Evaluate the research draft and determine next action. Use the CriticOutput tool."),
]).partial(time=lambda: datetime.datetime.now().isoformat())

def create_critic_llm(model_name: str = "llama3.3:70b"):
    """Create the LLM configured for the critic agent with structured output."""
    llm = ChatOllama(
        model=model_name,
        temperature=0,  # Deterministic evaluation
        num_ctx=16384,
    )
    # Use with_structured_output for robust schema enforcement
    return llm.with_structured_output(CriticOutput, include_raw=True)


def format_sub_questions_status(sub_questions: List[SubQuestion]) -> str:
    """Format sub-questions with their status for the prompt."""
    lines = []
    for sq in sub_questions:
        status_emoji = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "skipped": "⏭️"
        }.get(sq.status, "❓")

        findings_preview = ""
        if sq.findings:
            findings_preview = f" | Findings: {sq.findings[:100]}..."

        lines.append(
            f"{status_emoji} [{sq.priority}] {sq.question} "
            f"(Status: {sq.status}, Citations: {len(sq.citations)}){findings_preview}"
        )

    return "\n".join(lines)


def format_draft_for_critic(draft: Optional[ResearchDraft]) -> str:
    """Format the draft for critic evaluation."""
    if not draft:
        return "No draft exists yet."

    parts = [f"# {draft.title} (Version {draft.version})\n"]

    if draft.abstract:
        parts.append(f"## Abstract\n{draft.abstract}\n")

    for section in draft.sections:
        word_count = len(section.content.split())
        citation_count = len(section.citations)
        parts.append(
            f"## {section.title} (v{section.version}, {word_count} words, {citation_count} citations)\n"
            f"{section.content}\n"
        )

    if draft.conclusion:
        parts.append(f"## Conclusion\n{draft.conclusion}\n")

    total_words = sum(len(s.content.split()) for s in draft.sections)
    total_citations = sum(len(s.citations) for s in draft.sections)
    parts.append(f"\n---\nTotal: {len(draft.sections)} sections, {total_words} words, {total_citations} citations")

    return "\n".join(parts)


def format_citations_summary(citations: List[Citation]) -> str:
    """Create a summary of citations for the critic."""
    if not citations:
        return "No citations collected yet."

    # Group by sub-question
    by_question = {}
    for c in citations:
        key = c.accessed_for
        if key not in by_question:
            by_question[key] = []
        by_question[key].append(c)

    lines = [f"Total Citations: {len(citations)}"]
    for sq_id, sq_citations in by_question.items():
        lines.append(f"  - {sq_id}: {len(sq_citations)} citations")

    return "\n".join(lines)


def format_previous_scores(critique_history: List[CritiqueResult]) -> str:
    """Format previous critique scores for trend analysis."""
    if not critique_history:
        return "No previous critiques."

    lines = []
    for i, critique in enumerate(critique_history[-3:], 1):  # Last 3
        metrics = critique.quality_metrics
        lines.append(
            f"Iteration {i}: Coverage={metrics.coverage_score:.2f}, "
            f"Depth={metrics.depth_score:.2f}, "
            f"Completeness={metrics.completeness_score:.2f}"
        )

    return "\n".join(lines)


def calculate_improvement(
        current: QualityMetrics,
        previous: Optional[QualityMetrics]
) -> float:
    """Calculate improvement from previous iteration."""
    if not previous:
        return 1.0  # First iteration, assume improvement

    current_avg = (
                          current.coverage_score +
                          current.depth_score +
                          current.citation_density +
                          current.coherence_score +
                          current.completeness_score
                  ) / 5

    previous_avg = (
                           previous.coverage_score +
                           previous.depth_score +
                           previous.citation_density +
                           previous.coherence_score +
                           previous.completeness_score
                   ) / 5

    return current_avg - previous_avg

critic_llm = create_critic_llm()
critic_chain = CRITIC_PROMPT | critic_llm

# Note: No separate parser needed - with_structured_output handles parsing internally

def critique_draft(
        draft: Optional[ResearchDraft],
        plan: ResearchPlan,
        citations: List[Citation],
        iteration: int,
        max_iterations: int,
        critique_history: List[CritiqueResult],
        stop_config: StopConditionConfig,
        max_retries: int = 2
) -> Tuple[CritiqueResult, str]:
    """
    Perform comprehensive critique of the current draft with retry logic.

    Args:
        draft: Current research draft
        plan: The research plan
        citations: All collected citations
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations
        critique_history: History of previous critiques
        stop_config: Configuration for stop conditions
        max_retries: Maximum number of retry attempts on validation failure

    Returns:
        (CritiqueResult, next_action)
    """
    from langchain_core.messages import HumanMessage

    # Prepare context
    prompt_vars = {
        "main_query": plan.main_query,
        "objective": plan.objective,
        "scope": plan.scope,
        "sub_questions_status": format_sub_questions_status(plan.sub_questions),
        "current_draft": format_draft_for_critic(draft),
        "citations_summary": format_citations_summary(citations),
        "iteration": iteration,
        "max_iterations": max_iterations,
        "previous_scores": format_previous_scores(critique_history),
        "min_coverage": stop_config.min_coverage_score,
        "min_depth": stop_config.min_depth_score,
        "min_citations": stop_config.min_citation_density,
        "min_completeness": stop_config.min_completeness_score,
        "messages": [HumanMessage(content="Evaluate the current research draft and recommend next action.")]
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = critic_chain.invoke(prompt_vars)

            # with_structured_output with include_raw=True returns dict with 'parsed' and 'raw'
            if isinstance(result, dict) and 'parsed' in result:
                parsed = result['parsed']
                if parsed and isinstance(parsed, CriticOutput):
                    return parsed.critique, parsed.next_action
            # Direct CriticOutput return (without include_raw)
            elif isinstance(result, CriticOutput):
                return result.critique, result.next_action

        except Exception as e:
            last_error = e
            print(f"[Critic] Attempt {attempt + 1}/{max_retries + 1} Error: {e}")

            # Add error context to prompt for retry
            if attempt < max_retries:
                error_feedback = f"\n\nPrevious attempt failed with error: {str(e)[:500]}\nPlease ensure you provide ALL required fields: critique (with is_complete, quality_metrics, reasoning) and next_action."
                prompt_vars["messages"] = [
                    HumanMessage(content=f"Evaluate the current research draft and recommend next action.{error_feedback}")
                ]

    print(f"[Critic] All attempts failed. Last error: {last_error}")

    # Fallback evaluation
    return create_fallback_critique(
        draft, plan, citations, iteration, max_iterations, critique_history, stop_config
    )


def create_fallback_critique(
        draft: Optional[ResearchDraft],
        plan: ResearchPlan,
        citations: List[Citation],
        iteration: int,
        max_iterations: int,
        critique_history: List[CritiqueResult],
        stop_config: StopConditionConfig
) -> Tuple[CritiqueResult, str]:
    """Create a fallback critique when LLM fails."""
    # Calculate basic metrics
    completed_sq = sum(1 for sq in plan.sub_questions if sq.status == "completed")
    total_sq = len(plan.sub_questions)
    coverage = completed_sq / total_sq if total_sq > 0 else 0

    section_count = len(draft.sections) if draft else 0
    total_words = sum(len(s.content.split()) for s in draft.sections) if draft else 0

    # Simple heuristics
    depth = min(1.0, total_words / 2000)  # Aim for 2000 words
    citation_density = min(1.0, len(citations) / (section_count * 3)) if section_count > 0 else 0
    coherence = 0.7 if section_count >= 3 else 0.4
    completeness = (coverage + depth + citation_density + coherence) / 4

    metrics = QualityMetrics(
        coverage_score=coverage,
        depth_score=depth,
        citation_density=citation_density,
        coherence_score=coherence,
        completeness_score=completeness,
        gaps_identified=[sq.question for sq in plan.sub_questions if sq.status != "completed"],
        recommendations=["Continue researching remaining sub-questions"]
    )

    # Determine action
    if iteration >= max_iterations:
        next_action = "stop"
        is_complete = True
        reason = "Maximum iterations reached"
    elif completeness >= stop_config.min_completeness_score:
        next_action = "finalize"
        is_complete = True
        reason = "Quality thresholds met"
    else:
        next_action = "continue"
        is_complete = False
        reason = f"Completeness {completeness:.2f} below threshold {stop_config.min_completeness_score}"

    critique = CritiqueResult(
        is_complete=is_complete,
        quality_metrics=metrics,
        additional_questions=[],
        suggested_improvements=[f"Address: {sq.question}" for sq in plan.sub_questions if sq.status == "pending"][:3],
        reasoning=reason
    )

    return critique, next_action


def should_stop(
        critique: CritiqueResult,
        iteration: int,
        stop_config: StopConditionConfig,
        critique_history: List[CritiqueResult]
) -> Tuple[bool, str]:
    """
    Determine if research should stop based on critique and history.
    
    Returns:
        (should_stop, reason)
    """
    metrics = critique.quality_metrics

    # Check max iterations
    if iteration >= stop_config.max_iterations:
        return True, "Maximum iterations reached"

    # Check quality thresholds
    if (
            metrics.coverage_score >= stop_config.min_coverage_score and
            metrics.depth_score >= stop_config.min_depth_score and
            metrics.completeness_score >= stop_config.min_completeness_score
    ):
        return True, "Quality thresholds met"

    # Check for improvement stagnation
    if len(critique_history) >= stop_config.max_consecutive_no_improvement:
        recent = critique_history[-stop_config.max_consecutive_no_improvement:]
        improvements = []
        for i in range(1, len(recent)):
            improvement = calculate_improvement(
                recent[i].quality_metrics,
                recent[i - 1].quality_metrics
            )
            improvements.append(improvement)

        if all(imp <= 0.01 for imp in improvements):  # No significant improvement
            return True, "No improvement in recent iterations"

    return False, ""


if __name__ == "__main__":
    # Test the critic
    from langgraph_examples.deep_research_agent.schemas import (
        ResearchPlan, SubQuestion, ResearchDraft, DraftSection, Citation
    )

    # Create test data
    test_plan = ResearchPlan(
        main_query="AI-Powered SOC solutions",
        objective="Research AI SOC landscape",
        scope="Enterprise solutions",
        sub_questions=[
            SubQuestion(id="sq1", question="What is AI SOC?", priority=1, status="completed"),
            SubQuestion(id="sq2", question="Key players?", priority=1, status="completed"),
            SubQuestion(id="sq3", question="Future trends?", priority=2, status="pending"),
        ],
        methodology="Iterative",
        expected_sections=["Intro", "Findings", "Conclusion"]
    )

    test_draft = ResearchDraft(
        title="AI SOC Research",
        sections=[
            DraftSection(
                id="s1",
                title="Introduction",
                content="AI-powered SOC solutions are transforming cybersecurity...",
                citations=["[1]", "[2]"],
                last_updated="2024-01-01",
                version=1
            )
        ]
    )

    test_citations = [
        Citation(id="[1]", url="http://example.com", title="Test", snippet="...", accessed_for="sq1")
    ]

    stop_config = StopConditionConfig()

    print("Testing Critic Agent...")

    critique, action = critique_draft(
        draft=test_draft,
        plan=test_plan,
        citations=test_citations,
        iteration=1,
        max_iterations=5,
        critique_history=[],
        stop_config=stop_config
    )

    print(f"Critique Result:")
    print(f"  - Is Complete: {critique.is_complete}")
    print(f"  - Coverage: {critique.quality_metrics.coverage_score}")
    print(f"  - Completeness: {critique.quality_metrics.completeness_score}")
    print(f"  - Next Action: {action}")
    print(f"  - Reasoning: {critique.reasoning}")
