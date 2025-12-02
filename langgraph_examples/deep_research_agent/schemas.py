"""
Deep Research Agent - Schemas and State Definitions

This module defines all the data models and state structures for the deep research system.
The state is designed to support iterative research with evolving drafts, citations,
and quality assessment.
"""

from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class ResearchPhase(str, Enum):
    """Current phase of the research process."""
    PLANNING = "planning"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    CRITIQUING = "critiquing"
    FINALIZING = "finalizing"
    COMPLETE = "complete"


class Citation(BaseModel):
    """A single citation with source information."""
    id: str = Field(description="Unique citation identifier (e.g., [1], [2])")
    url: str = Field(description="Source URL")
    title: Optional[str] = Field(default=None, description="Title of the source")
    snippet: str = Field(description="Relevant snippet from the source")
    accessed_for: str = Field(description="Which sub-question this citation supports")


class SubQuestion(BaseModel):
    """A sub-question derived from the main research query."""
    id: str = Field(description="Unique identifier for this sub-question")
    question: str = Field(description="The sub-question to investigate")
    priority: int = Field(default=1, description="Priority level (1=highest)")
    status: str = Field(default="pending", description="pending, in_progress, completed, skipped")
    findings: Optional[str] = Field(default=None, description="Key findings for this sub-question")
    search_queries: List[str] = Field(default_factory=list, description="Search queries used")
    citations: List[str] = Field(default_factory=list, description="Citation IDs used")


class ResearchPlan(BaseModel):
    """The research plan with all sub-questions and strategy."""
    main_query: str = Field(description="The original research query")
    objective: str = Field(description="Clear objective statement")
    scope: str = Field(description="Defined scope and boundaries")
    sub_questions: List[SubQuestion] = Field(description="List of sub-questions to investigate")
    methodology: str = Field(description="Research methodology description")
    expected_sections: List[str] = Field(description="Expected sections in final report")


class DraftSection(BaseModel):
    """A section in the evolving research draft."""
    id: str = Field(description="Section identifier")
    title: str = Field(description="Section title")
    content: str = Field(description="Section content with inline citations")
    citations: List[str] = Field(default_factory=list, description="Citation IDs used in this section")
    last_updated: str = Field(description="Timestamp of last update")
    version: int = Field(default=1, description="Version number of this section")


class ResearchDraft(BaseModel):
    """The evolving research draft document."""
    title: str = Field(description="Draft title")
    abstract: Optional[str] = Field(default=None, description="Executive summary/abstract")
    sections: List[DraftSection] = Field(default_factory=list, description="Draft sections")
    conclusion: Optional[str] = Field(default=None, description="Conclusion section")
    version: int = Field(default=1, description="Overall draft version")


class QualityMetrics(BaseModel):
    """Quality assessment metrics for the research."""
    coverage_score: float = Field(default=0.0, description="How well sub-questions are covered (0-1)")
    depth_score: float = Field(default=0.0, description="Depth of analysis (0-1)")
    citation_density: float = Field(default=0.0, description="Citations per section (0-1)")
    coherence_score: float = Field(default=0.0, description="Logical flow and coherence (0-1)")
    completeness_score: float = Field(default=0.0, description="Overall completeness (0-1)")
    gaps_identified: List[str] = Field(default_factory=list, description="Identified gaps in research")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")


class CritiqueResult(BaseModel):
    """Result of the critic's evaluation."""
    is_complete: bool = Field(description="Whether research meets completion criteria")
    quality_metrics: QualityMetrics = Field(description="Detailed quality metrics")
    additional_questions: List[str] = Field(default_factory=list, description="New questions to investigate")
    suggested_improvements: List[str] = Field(default_factory=list, description="Specific improvements needed")
    reasoning: str = Field(description="Explanation of the critique")


# ============================================================================
# MAIN RESEARCH STATE
# ============================================================================

class DeepResearchState(BaseModel):
    """
    The complete state for the deep research agent.
    Uses Pydantic for validation and type safety.
    """
    # Core research data
    original_query: str = Field(description="The original user query")
    research_plan: Optional[ResearchPlan] = Field(default=None, description="The research plan")
    draft: Optional[ResearchDraft] = Field(default=None, description="The evolving draft")
    citations: List[Citation] = Field(default_factory=list, description="All collected citations")

    # Progress tracking
    phase: ResearchPhase = Field(default=ResearchPhase.PLANNING, description="Current phase")
    current_sub_question_id: Optional[str] = Field(default=None, description="Currently investigating")
    iteration: int = Field(default=0, description="Current iteration number")
    max_iterations: int = Field(default=5, description="Maximum iterations allowed")

    # Quality tracking
    critique_history: List[CritiqueResult] = Field(default_factory=list, description="History of critiques")
    latest_critique: Optional[CritiqueResult] = Field(default=None, description="Most recent critique")

    # Search results cache
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Raw search results")

    # Completion tracking
    is_complete: bool = Field(default=False, description="Whether research is complete")
    completion_reason: Optional[str] = Field(default=None, description="Why research ended")

    # Final output
    final_report: Optional[str] = Field(default=None, description="The final formatted report")

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# AGENT-SPECIFIC SCHEMAS (for tool calling)
# ============================================================================

class PlannerOutput(BaseModel):
    """Output schema for the Planner agent."""
    research_plan: ResearchPlan = Field(description="The complete research plan")
    reasoning: str = Field(description="Explanation of the planning decisions")


class ResearcherOutput(BaseModel):
    """Output schema for the Researcher agent."""
    search_queries: List[str] = Field(description="Search queries to execute")
    sub_question_id: str = Field(description="Which sub-question this addresses")
    search_strategy: str = Field(description="Strategy explanation")


class SynthesizerOutput(BaseModel):
    """Output schema for the Synthesizer agent."""
    updated_section: DraftSection = Field(description="The updated or new section")
    new_citations: List[Citation] = Field(description="New citations to add")
    synthesis_notes: str = Field(description="Notes on what was synthesized")


class CriticOutput(BaseModel):
    """Output schema for the Critic agent."""
    critique: CritiqueResult = Field(description="The critique result")
    next_action: str = Field(description="Recommended next action: 'continue', 'finalize', or 'stop'")


class ReportGeneratorOutput(BaseModel):
    """Output schema for the Report Generator agent."""
    final_report: str = Field(description="The complete formatted report")
    report_metadata: Dict[str, Any] = Field(description="Metadata about the report")


# ============================================================================
# STOP CONDITION CONFIGURATION
# ============================================================================

class StopConditionConfig(BaseModel):
    """Configuration for determining when research is complete."""
    min_coverage_score: float = Field(default=0.7, description="Minimum coverage required")
    min_depth_score: float = Field(default=0.6, description="Minimum depth required")
    min_citation_density: float = Field(default=0.5, description="Minimum citations per section")
    min_completeness_score: float = Field(default=0.7, description="Minimum overall completeness")
    max_iterations: int = Field(default=5, description="Maximum research iterations")
    max_consecutive_no_improvement: int = Field(default=2, description="Stop after N iterations with no improvement")
    min_sub_questions_completed: float = Field(default=0.8, description="Min % of sub-questions completed")
