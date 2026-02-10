"""
Comprehensive tests for deep_research_agent agents.

These tests focus on helper functions, data validation, and schemas.
LLM calls are NOT mocked - only non-LLM functions are tested.
"""

import pytest
import uuid
from typing import List

# Import schemas
from langgraph_examples.deep_research_agent.schemas import (
    ResearchPlan,
    SubQuestion,
    PlannerOutput,
    ResearchPhase,
    Citation,
    DraftSection,
    ResearchDraft,
    QualityMetrics,
    CritiqueResult,
    DeepResearchState,
    StopConditionConfig,
    ResearcherOutput,
    SynthesizerOutput,
    CriticOutput,
    ReportGeneratorOutput,
)

# Import helper functions from planner
from langgraph_examples.deep_research_agent.agents.planner import (
    create_sub_question,
    create_default_plan,
    validate_research_plan,
)

# Import helper function from prompt_engineer
from langgraph_examples.deep_research_agent.agents.prompt_engineer import (
    create_fallback_output,
    PromptAnalysis,
    PromptEngineerOutput,
)


# ============================================================================
# TEST HELPER FUNCTIONS - PLANNER
# ============================================================================

class TestCreateSubQuestion:
    """Test the create_sub_question helper function."""

    def test_creates_sub_question_with_unique_id(self):
        """Test that sub-questions are created with unique IDs."""
        sq1 = create_sub_question("What is AI?")
        sq2 = create_sub_question("What is ML?")

        # Both should have IDs starting with "sq_"
        assert sq1.id.startswith("sq_")
        assert sq2.id.startswith("sq_")

        # IDs should be unique
        assert sq1.id != sq2.id

    def test_creates_sub_question_with_correct_fields(self):
        """Test that all fields are set correctly."""
        question_text = "What are the benefits of AI?"
        priority = 2
        status = "in_progress"

        sq = create_sub_question(question_text, priority=priority, status=status)

        assert sq.question == question_text
        assert sq.priority == priority
        assert sq.status == status
        assert sq.findings is None
        assert sq.search_queries == []
        assert sq.citations == []

    def test_creates_sub_question_with_default_values(self):
        """Test default values when only question is provided."""
        sq = create_sub_question("Test question?")

        assert sq.priority == 1  # default
        assert sq.status == "pending"  # default
        assert sq.findings is None
        assert sq.search_queries == []
        assert sq.citations == []

    def test_id_format_is_valid(self):
        """Test that generated IDs have the correct format."""
        sq = create_sub_question("Test?")

        # ID should be "sq_" followed by 8 hex characters
        parts = sq.id.split("_")
        assert len(parts) == 2
        assert parts[0] == "sq"
        assert len(parts[1]) == 8
        assert all(c in "0123456789abcdef" for c in parts[1])

    def test_multiple_sub_questions_have_unique_ids(self):
        """Test that multiple created sub-questions have unique IDs."""
        sub_questions = [create_sub_question(f"Question {i}?") for i in range(10)]
        ids = [sq.id for sq in sub_questions]

        # All IDs should be unique
        assert len(ids) == len(set(ids))


class TestCreateDefaultPlan:
    """Test the create_default_plan helper function."""

    def test_creates_plan_with_query(self):
        """Test that plan is created with the provided query."""
        query = "What is quantum computing?"
        plan = create_default_plan(query)

        assert plan.main_query == query
        assert query in plan.objective

    def test_creates_five_sub_questions(self):
        """Test that default plan creates exactly 5 sub-questions."""
        plan = create_default_plan("Test query")

        assert len(plan.sub_questions) == 5

    def test_sub_questions_have_correct_priorities(self):
        """Test that sub-questions have appropriate priority distribution."""
        plan = create_default_plan("Test query")

        priorities = [sq.priority for sq in plan.sub_questions]

        # Should have at least 2 high priority (1)
        assert priorities.count(1) >= 2
        # Should have at least 2 medium priority (2)
        assert priorities.count(2) >= 2
        # Should have at least 1 low priority (3)
        assert priorities.count(3) >= 1

    def test_sub_questions_reference_query(self):
        """Test that sub-questions incorporate the query topic."""
        query = "machine learning"
        plan = create_default_plan(query)

        # At least some sub-questions should reference the query
        references = sum(1 for sq in plan.sub_questions if query in sq.question)
        assert references >= 3

    def test_has_expected_sections(self):
        """Test that plan includes expected report sections."""
        plan = create_default_plan("Test query")

        assert len(plan.expected_sections) == 5
        assert "Executive Summary" in plan.expected_sections
        assert "Introduction & Background" in plan.expected_sections
        assert "Conclusions & Recommendations" in plan.expected_sections

    def test_has_scope_and_methodology(self):
        """Test that plan includes scope and methodology."""
        plan = create_default_plan("Test query")

        assert plan.scope is not None
        assert len(plan.scope) > 0
        assert plan.methodology is not None
        assert len(plan.methodology) > 0

    def test_all_sub_questions_have_unique_ids(self):
        """Test that all generated sub-questions have unique IDs."""
        plan = create_default_plan("Test query")

        ids = [sq.id for sq in plan.sub_questions]
        assert len(ids) == len(set(ids))  # All IDs are unique

    def test_sub_questions_have_correct_structure(self):
        """Test that all sub-questions have the correct structure."""
        plan = create_default_plan("artificial intelligence")

        for sq in plan.sub_questions:
            assert isinstance(sq, SubQuestion)
            assert sq.id.startswith("sq_")
            assert len(sq.question) > 0
            assert sq.priority in [1, 2, 3]
            assert sq.status == "pending"

    def test_priority_distribution(self):
        """Test that priorities follow expected distribution."""
        plan = create_default_plan("Test query")

        priorities = [sq.priority for sq in plan.sub_questions]
        # Expected: [1, 1, 2, 2, 3]
        assert priorities == [1, 1, 2, 2, 3]


class TestValidateResearchPlan:
    """Test the validate_research_plan helper function."""

    def test_valid_plan_passes_validation(self):
        """Test that a properly formed plan passes validation."""
        plan = create_default_plan("Valid test query")

        is_valid, issues = validate_research_plan(plan)

        assert is_valid is True
        assert len(issues) == 0

    def test_missing_main_query_fails(self):
        """Test that plan without main query fails validation."""
        plan = ResearchPlan(
            main_query="",
            objective="Test objective",
            scope="Test scope",
            sub_questions=[
                SubQuestion(id="sq_1", question="Q1?", priority=1),
                SubQuestion(id="sq_2", question="Q2?", priority=1),
                SubQuestion(id="sq_3", question="Q3?", priority=1),
            ],
            methodology="Test methodology",
            expected_sections=["S1", "S2", "S3"]
        )

        is_valid, issues = validate_research_plan(plan)

        assert is_valid is False
        assert any("main query" in issue.lower() for issue in issues)

    def test_too_few_sub_questions_fails(self):
        """Test that plan with fewer than 3 sub-questions fails."""
        plan = ResearchPlan(
            main_query="Test query",
            objective="Test objective",
            scope="Test scope",
            sub_questions=[
                SubQuestion(id="sq_1", question="Q1?", priority=1),
                SubQuestion(id="sq_2", question="Q2?", priority=1),
            ],
            methodology="Test methodology",
            expected_sections=["S1", "S2", "S3"]
        )

        is_valid, issues = validate_research_plan(plan)

        assert is_valid is False
        assert any("3 sub-questions" in issue for issue in issues)

    def test_too_many_sub_questions_fails(self):
        """Test that plan with more than 10 sub-questions fails."""
        plan = ResearchPlan(
            main_query="Test query",
            objective="Test objective",
            scope="Test scope",
            sub_questions=[
                SubQuestion(id=f"sq_{i}", question=f"Question {i}?", priority=1)
                for i in range(11)  # 11 questions
            ],
            methodology="Test methodology",
            expected_sections=["S1", "S2", "S3"]
        )

        is_valid, issues = validate_research_plan(plan)

        assert is_valid is False
        assert any("Too many" in issue for issue in issues)

    def test_too_few_expected_sections_fails(self):
        """Test that plan with fewer than 3 expected sections fails."""
        plan = ResearchPlan(
            main_query="Test query",
            objective="Test objective",
            scope="Test scope",
            sub_questions=[
                SubQuestion(id="sq_1", question="Q1?", priority=1),
                SubQuestion(id="sq_2", question="Q2?", priority=1),
                SubQuestion(id="sq_3", question="Q3?", priority=1),
            ],
            methodology="Test methodology",
            expected_sections=["S1", "S2"]  # Only 2 sections
        )

        is_valid, issues = validate_research_plan(plan)

        assert is_valid is False
        assert any("3 expected sections" in issue for issue in issues)

    def test_duplicate_sub_questions_fails(self):
        """Test that plan with duplicate sub-questions fails."""
        plan = ResearchPlan(
            main_query="Test query",
            objective="Test objective",
            scope="Test scope",
            sub_questions=[
                SubQuestion(id="sq_1", question="What is AI?", priority=1),
                SubQuestion(id="sq_2", question="What is ML?", priority=1),
                SubQuestion(id="sq_3", question="what is ai?", priority=2),  # Duplicate (case-insensitive)
            ],
            methodology="Test methodology",
            expected_sections=["S1", "S2", "S3"]
        )

        is_valid, issues = validate_research_plan(plan)

        assert is_valid is False
        assert any("Duplicate" in issue for issue in issues)

    def test_multiple_issues_all_reported(self):
        """Test that all validation issues are reported."""
        plan = ResearchPlan(
            main_query="",  # Missing
            objective="Test objective",
            scope="Test scope",
            sub_questions=[  # Too few
                SubQuestion(id="sq_1", question="Q1?", priority=1),
            ],
            methodology="Test methodology",
            expected_sections=["S1"]  # Too few
        )

        is_valid, issues = validate_research_plan(plan)

        assert is_valid is False
        assert len(issues) == 3  # All three issues should be detected

    def test_validation_with_exact_minimum_requirements(self):
        """Test validation with exactly minimum requirements."""
        plan = ResearchPlan(
            main_query="Test query",
            objective="Test objective",
            scope="Test scope",
            sub_questions=[
                SubQuestion(id="sq_1", question="Q1?", priority=1),
                SubQuestion(id="sq_2", question="Q2?", priority=1),
                SubQuestion(id="sq_3", question="Q3?", priority=1),
            ],
            methodology="Test methodology",
            expected_sections=["S1", "S2", "S3"]
        )

        is_valid, issues = validate_research_plan(plan)
        assert is_valid is True
        assert len(issues) == 0

    def test_validation_with_maximum_sub_questions(self):
        """Test validation with exactly 10 sub-questions (maximum)."""
        plan = ResearchPlan(
            main_query="Test query",
            objective="Test objective",
            scope="Test scope",
            sub_questions=[
                SubQuestion(id=f"sq_{i}", question=f"Question {i}?", priority=1)
                for i in range(10)
            ],
            methodology="Test methodology",
            expected_sections=["S1", "S2", "S3"]
        )

        is_valid, issues = validate_research_plan(plan)
        assert is_valid is True
        assert len(issues) == 0


# ============================================================================
# TEST HELPER FUNCTIONS - PROMPT ENGINEER
# ============================================================================

class TestCreateFallbackOutput:
    """Test the create_fallback_output helper function."""

    def test_creates_valid_output(self):
        """Test that fallback output is properly structured."""
        original = "write some code"
        output = create_fallback_output(original)

        assert isinstance(output, PromptEngineerOutput)
        assert isinstance(output.analysis, PromptAnalysis)
        assert isinstance(output.refined_prompt, str)
        assert isinstance(output.changes_made, list)
        assert isinstance(output.reasoning, str)

    def test_includes_original_prompt(self):
        """Test that original prompt is included in refined version."""
        original = "explain quantum physics"
        output = create_fallback_output(original)

        assert original in output.refined_prompt

    def test_analysis_scores_are_valid(self):
        """Test that all analysis scores are in valid range [0, 1]."""
        output = create_fallback_output("test prompt")

        assert 0 <= output.analysis.clarity_score <= 1
        assert 0 <= output.analysis.specificity_score <= 1
        assert 0 <= output.analysis.context_score <= 1
        assert 0 <= output.analysis.actionability_score <= 1

    def test_has_default_scores(self):
        """Test that fallback uses 0.5 as default score."""
        output = create_fallback_output("test")

        assert output.analysis.clarity_score == 0.5
        assert output.analysis.specificity_score == 0.5
        assert output.analysis.context_score == 0.5
        assert output.analysis.actionability_score == 0.5

    def test_has_issues_and_strengths(self):
        """Test that analysis includes issues and strengths."""
        output = create_fallback_output("test")

        assert len(output.analysis.issues_identified) > 0
        assert len(output.analysis.strengths) > 0

    def test_changes_made_is_populated(self):
        """Test that changes_made list is not empty."""
        output = create_fallback_output("test")

        assert len(output.changes_made) > 0

    def test_reasoning_mentions_fallback(self):
        """Test that reasoning indicates this is a fallback."""
        output = create_fallback_output("test")

        assert "fallback" in output.reasoning.lower()

    def test_alternative_prompts_empty(self):
        """Test that alternative prompts list is empty in fallback."""
        output = create_fallback_output("test")

        assert output.alternative_prompts == []

    def test_different_prompts_produce_different_refined_outputs(self):
        """Test that different original prompts produce different refined prompts."""
        output1 = create_fallback_output("First prompt")
        output2 = create_fallback_output("Second prompt")

        assert "First prompt" in output1.refined_prompt
        assert "Second prompt" in output2.refined_prompt
        assert output1.refined_prompt != output2.refined_prompt

    def test_fallback_output_structure_is_complete(self):
        """Test that fallback output has all required fields populated."""
        output = create_fallback_output("test prompt")

        # Analysis fields
        assert output.analysis.clarity_score is not None
        assert output.analysis.specificity_score is not None
        assert output.analysis.context_score is not None
        assert output.analysis.actionability_score is not None
        assert output.analysis.issues_identified is not None
        assert output.analysis.strengths is not None

        # Output fields
        assert output.refined_prompt is not None
        assert output.changes_made is not None
        assert output.reasoning is not None
        assert output.alternative_prompts is not None


# ============================================================================
# TEST PYDANTIC SCHEMAS
# ============================================================================

class TestSubQuestionSchema:
    """Test the SubQuestion Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating SubQuestion with required fields."""
        sq = SubQuestion(id="sq_001", question="What is AI?", priority=1)

        assert sq.id == "sq_001"
        assert sq.question == "What is AI?"
        assert sq.priority == 1

    def test_default_values_set_correctly(self):
        """Test that default values are set."""
        sq = SubQuestion(id="sq_001", question="Test?", priority=1)

        assert sq.status == "pending"
        assert sq.findings is None
        assert sq.search_queries == []
        assert sq.citations == []

    def test_can_update_fields(self):
        """Test that fields can be updated after creation."""
        sq = SubQuestion(id="sq_001", question="Test?", priority=1)

        sq.status = "completed"
        sq.findings = "Important findings"
        sq.search_queries = ["query1", "query2"]
        sq.citations = ["cite1", "cite2"]

        assert sq.status == "completed"
        assert sq.findings == "Important findings"
        assert len(sq.search_queries) == 2
        assert len(sq.citations) == 2

    def test_accepts_all_valid_priorities(self):
        """Test that all valid priority values work."""
        for priority in [1, 2, 3]:
            sq = SubQuestion(id=f"sq_{priority}", question="Test?", priority=priority)
            assert sq.priority == priority

    def test_accepts_all_valid_statuses(self):
        """Test common status values."""
        for status in ["pending", "in_progress", "completed", "skipped"]:
            sq = SubQuestion(id="sq_001", question="Test?", priority=1, status=status)
            assert sq.status == status


class TestResearchPlanSchema:
    """Test the ResearchPlan Pydantic model."""

    def test_creates_with_all_fields(self):
        """Test creating ResearchPlan with all required fields."""
        plan = ResearchPlan(
            main_query="Test query",
            objective="Test objective",
            scope="Test scope",
            sub_questions=[
                SubQuestion(id="sq_1", question="Q1?", priority=1),
            ],
            methodology="Test methodology",
            expected_sections=["S1", "S2"]
        )

        assert plan.main_query == "Test query"
        assert plan.objective == "Test objective"
        assert plan.scope == "Test scope"
        assert len(plan.sub_questions) == 1
        assert plan.methodology == "Test methodology"
        assert len(plan.expected_sections) == 2

    def test_accepts_multiple_sub_questions(self):
        """Test that plan can hold multiple sub-questions."""
        sub_questions = [
            SubQuestion(id=f"sq_{i}", question=f"Q{i}?", priority=1)
            for i in range(5)
        ]

        plan = ResearchPlan(
            main_query="Test",
            objective="Test",
            scope="Test",
            sub_questions=sub_questions,
            methodology="Test",
            expected_sections=["S1", "S2", "S3"]
        )

        assert len(plan.sub_questions) == 5

    def test_expected_sections_is_list_of_strings(self):
        """Test that expected_sections is a list of strings."""
        plan = ResearchPlan(
            main_query="Test",
            objective="Test",
            scope="Test",
            sub_questions=[SubQuestion(id="sq_1", question="Q1?", priority=1)],
            methodology="Test",
            expected_sections=["Intro", "Body", "Conclusion"]
        )

        assert isinstance(plan.expected_sections, list)
        assert all(isinstance(s, str) for s in plan.expected_sections)


class TestPlannerOutputSchema:
    """Test the PlannerOutput Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating PlannerOutput with required fields."""
        plan = create_default_plan("Test query")

        output = PlannerOutput(
            research_plan=plan,
            reasoning="This is the reasoning"
        )

        assert output.research_plan == plan
        assert output.reasoning == "This is the reasoning"

    def test_reasoning_can_be_long_text(self):
        """Test that reasoning field can hold long text."""
        plan = create_default_plan("Test")
        long_reasoning = "This is a very long reasoning. " * 100

        output = PlannerOutput(
            research_plan=plan,
            reasoning=long_reasoning
        )

        assert len(output.reasoning) > 1000


class TestPromptAnalysisSchema:
    """Test the PromptAnalysis Pydantic model."""

    def test_creates_with_all_scores(self):
        """Test creating PromptAnalysis with all required fields."""
        analysis = PromptAnalysis(
            clarity_score=0.8,
            specificity_score=0.7,
            context_score=0.6,
            actionability_score=0.9,
            issues_identified=["Issue 1", "Issue 2"],
            strengths=["Strength 1"]
        )

        assert analysis.clarity_score == 0.8
        assert analysis.specificity_score == 0.7
        assert analysis.context_score == 0.6
        assert analysis.actionability_score == 0.9
        assert len(analysis.issues_identified) == 2
        assert len(analysis.strengths) == 1

    def test_scores_can_be_zero_or_one(self):
        """Test that scores can be at extremes."""
        analysis = PromptAnalysis(
            clarity_score=0.0,
            specificity_score=1.0,
            context_score=0.0,
            actionability_score=1.0,
            issues_identified=[],
            strengths=[]
        )

        assert analysis.clarity_score == 0.0
        assert analysis.specificity_score == 1.0

    def test_can_have_empty_lists(self):
        """Test that issues and strengths can be empty."""
        analysis = PromptAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            context_score=0.5,
            actionability_score=0.5,
            issues_identified=[],
            strengths=[]
        )

        assert analysis.issues_identified == []
        assert analysis.strengths == []


class TestPromptEngineerOutputSchema:
    """Test the PromptEngineerOutput Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating PromptEngineerOutput with required fields."""
        analysis = PromptAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            context_score=0.5,
            actionability_score=0.5,
            issues_identified=["Issue"],
            strengths=["Strength"]
        )

        output = PromptEngineerOutput(
            analysis=analysis,
            refined_prompt="Refined prompt here",
            changes_made=["Change 1", "Change 2"],
            reasoning="Reasoning here"
        )

        assert output.analysis == analysis
        assert output.refined_prompt == "Refined prompt here"
        assert len(output.changes_made) == 2
        assert output.reasoning == "Reasoning here"

    def test_alternative_prompts_default_empty(self):
        """Test that alternative_prompts defaults to empty list."""
        analysis = PromptAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            context_score=0.5,
            actionability_score=0.5,
            issues_identified=[],
            strengths=[]
        )

        output = PromptEngineerOutput(
            analysis=analysis,
            refined_prompt="Test",
            changes_made=[],
            reasoning="Test"
        )

        assert output.alternative_prompts == []

    def test_can_add_alternative_prompts(self):
        """Test adding alternative prompts."""
        analysis = PromptAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            context_score=0.5,
            actionability_score=0.5,
            issues_identified=[],
            strengths=[]
        )

        output = PromptEngineerOutput(
            analysis=analysis,
            refined_prompt="Primary",
            changes_made=[],
            reasoning="Test",
            alternative_prompts=["Alt 1", "Alt 2", "Alt 3"]
        )

        assert len(output.alternative_prompts) == 3


class TestCitationSchema:
    """Test the Citation Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating Citation with required fields."""
        citation = Citation(
            id="[1]",
            url="https://example.com",
            snippet="Important information",
            accessed_for="sq_001"
        )

        assert citation.id == "[1]"
        assert citation.url == "https://example.com"
        assert citation.snippet == "Important information"
        assert citation.accessed_for == "sq_001"

    def test_title_is_optional(self):
        """Test that title field is optional."""
        citation = Citation(
            id="[1]",
            url="https://example.com",
            snippet="Info",
            accessed_for="sq_001"
        )

        assert citation.title is None

        citation.title = "Page Title"
        assert citation.title == "Page Title"

    def test_citation_id_formats(self):
        """Test various citation ID formats."""
        for cid in ["[1]", "[2]", "[10]", "[100]"]:
            citation = Citation(
                id=cid,
                url="https://example.com",
                snippet="Info",
                accessed_for="sq_001"
            )
            assert citation.id == cid


class TestDraftSectionSchema:
    """Test the DraftSection Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating DraftSection."""
        section = DraftSection(
            id="sec_001",
            title="Introduction",
            content="This is the introduction [1].",
            last_updated="2024-01-01T00:00:00"
        )

        assert section.id == "sec_001"
        assert section.title == "Introduction"
        assert "[1]" in section.content
        assert section.version == 1  # default

    def test_default_values(self):
        """Test default values for optional fields."""
        section = DraftSection(
            id="sec_001",
            title="Test",
            content="Content",
            last_updated="2024-01-01T00:00:00"
        )

        assert section.citations == []
        assert section.version == 1

    def test_can_increment_version(self):
        """Test that version can be incremented."""
        section = DraftSection(
            id="sec_001",
            title="Test",
            content="Content v1",
            last_updated="2024-01-01T00:00:00",
            version=1
        )

        section.version = 2
        section.content = "Content v2"
        assert section.version == 2


class TestResearchDraftSchema:
    """Test the ResearchDraft Pydantic model."""

    def test_creates_with_title(self):
        """Test creating ResearchDraft."""
        draft = ResearchDraft(title="Research Report")

        assert draft.title == "Research Report"
        assert draft.abstract is None
        assert draft.sections == []
        assert draft.conclusion is None
        assert draft.version == 1

    def test_can_add_sections(self):
        """Test adding sections to draft."""
        section = DraftSection(
            id="sec_001",
            title="Introduction",
            content="Content",
            last_updated="2024-01-01T00:00:00"
        )

        draft = ResearchDraft(
            title="Report",
            sections=[section]
        )

        assert len(draft.sections) == 1

    def test_can_add_abstract_and_conclusion(self):
        """Test adding abstract and conclusion."""
        draft = ResearchDraft(
            title="Report",
            abstract="This is the abstract",
            conclusion="This is the conclusion"
        )

        assert draft.abstract == "This is the abstract"
        assert draft.conclusion == "This is the conclusion"


class TestQualityMetricsSchema:
    """Test the QualityMetrics Pydantic model."""

    def test_creates_with_default_values(self):
        """Test that QualityMetrics has proper defaults."""
        metrics = QualityMetrics()

        assert metrics.coverage_score == 0.0
        assert metrics.depth_score == 0.0
        assert metrics.citation_density == 0.0
        assert metrics.coherence_score == 0.0
        assert metrics.completeness_score == 0.0
        assert metrics.gaps_identified == []
        assert metrics.recommendations == []

    def test_can_set_all_scores(self):
        """Test setting all metric scores."""
        metrics = QualityMetrics(
            coverage_score=0.8,
            depth_score=0.7,
            citation_density=0.6,
            coherence_score=0.9,
            completeness_score=0.75,
            gaps_identified=["Gap 1"],
            recommendations=["Rec 1", "Rec 2"]
        )

        assert metrics.coverage_score == 0.8
        assert metrics.depth_score == 0.7
        assert len(metrics.gaps_identified) == 1
        assert len(metrics.recommendations) == 2

    def test_scores_can_be_floats(self):
        """Test that scores accept float values."""
        metrics = QualityMetrics(
            coverage_score=0.85,
            depth_score=0.72,
            citation_density=0.63,
            coherence_score=0.91,
            completeness_score=0.78
        )

        assert isinstance(metrics.coverage_score, float)
        assert isinstance(metrics.depth_score, float)


class TestCritiqueResultSchema:
    """Test the CritiqueResult Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating CritiqueResult."""
        metrics = QualityMetrics()

        critique = CritiqueResult(
            is_complete=True,
            quality_metrics=metrics,
            reasoning="Analysis complete"
        )

        assert critique.is_complete is True
        assert critique.quality_metrics == metrics
        assert critique.reasoning == "Analysis complete"

    def test_default_lists_empty(self):
        """Test that optional lists default to empty."""
        metrics = QualityMetrics()

        critique = CritiqueResult(
            is_complete=False,
            quality_metrics=metrics,
            reasoning="Not complete"
        )

        assert critique.additional_questions == []
        assert critique.suggested_improvements == []

    def test_can_add_questions_and_improvements(self):
        """Test adding additional questions and improvements."""
        metrics = QualityMetrics()

        critique = CritiqueResult(
            is_complete=False,
            quality_metrics=metrics,
            reasoning="Needs work",
            additional_questions=["Question 1", "Question 2"],
            suggested_improvements=["Improvement 1", "Improvement 2", "Improvement 3"]
        )

        assert len(critique.additional_questions) == 2
        assert len(critique.suggested_improvements) == 3


class TestDeepResearchStateSchema:
    """Test the DeepResearchState Pydantic model."""

    def test_creates_with_query(self):
        """Test creating minimal state."""
        state = DeepResearchState(original_query="What is AI?")

        assert state.original_query == "What is AI?"
        assert state.research_plan is None
        assert state.draft is None
        assert state.phase == ResearchPhase.PLANNING
        assert state.iteration == 0
        assert state.max_iterations == 5
        assert state.is_complete is False

    def test_default_values_set(self):
        """Test that all defaults are properly set."""
        state = DeepResearchState(original_query="Test")

        assert state.citations == []
        assert state.critique_history == []
        assert state.latest_critique is None
        assert state.search_results == []
        assert state.completion_reason is None
        assert state.final_report is None

    def test_can_set_research_plan(self):
        """Test setting research plan."""
        plan = create_default_plan("Test query")
        state = DeepResearchState(
            original_query="Test query",
            research_plan=plan
        )

        assert state.research_plan == plan
        assert state.research_plan.main_query == "Test query"

    def test_can_track_progress(self):
        """Test tracking progress through state updates."""
        state = DeepResearchState(original_query="Test")

        assert state.phase == ResearchPhase.PLANNING
        assert state.iteration == 0

        state.phase = ResearchPhase.RESEARCHING
        state.iteration = 1

        assert state.phase == ResearchPhase.RESEARCHING
        assert state.iteration == 1


class TestResearchPhaseEnum:
    """Test the ResearchPhase enum."""

    def test_all_phases_exist(self):
        """Test that all expected phases exist."""
        assert ResearchPhase.PLANNING == "planning"
        assert ResearchPhase.RESEARCHING == "researching"
        assert ResearchPhase.SYNTHESIZING == "synthesizing"
        assert ResearchPhase.CRITIQUING == "critiquing"
        assert ResearchPhase.FINALIZING == "finalizing"
        assert ResearchPhase.COMPLETE == "complete"

    def test_can_compare_phases(self):
        """Test that phases can be compared."""
        phase1 = ResearchPhase.PLANNING
        phase2 = ResearchPhase.PLANNING
        phase3 = ResearchPhase.COMPLETE

        assert phase1 == phase2
        assert phase1 != phase3

    def test_phase_values_are_strings(self):
        """Test that phase values are strings."""
        for phase in ResearchPhase:
            assert isinstance(phase.value, str)


class TestStopConditionConfigSchema:
    """Test the StopConditionConfig Pydantic model."""

    def test_creates_with_defaults(self):
        """Test that config has sensible defaults."""
        config = StopConditionConfig()

        assert config.min_coverage_score == 0.7
        assert config.min_depth_score == 0.6
        assert config.min_citation_density == 0.5
        assert config.min_completeness_score == 0.7
        assert config.max_iterations == 5
        assert config.max_consecutive_no_improvement == 2
        assert config.min_sub_questions_completed == 0.8

    def test_can_customize_values(self):
        """Test that config values can be customized."""
        config = StopConditionConfig(
            min_coverage_score=0.9,
            max_iterations=10
        )

        assert config.min_coverage_score == 0.9
        assert config.max_iterations == 10

    def test_all_thresholds_are_floats(self):
        """Test that threshold values are floats."""
        config = StopConditionConfig()

        assert isinstance(config.min_coverage_score, float)
        assert isinstance(config.min_depth_score, float)
        assert isinstance(config.min_citation_density, float)


# ============================================================================
# TEST AGENT OUTPUT SCHEMAS
# ============================================================================

class TestResearcherOutputSchema:
    """Test the ResearcherOutput Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating ResearcherOutput."""
        output = ResearcherOutput(
            search_queries=["query1", "query2"],
            sub_question_id="sq_001",
            search_strategy="Broad then narrow"
        )

        assert len(output.search_queries) == 2
        assert output.sub_question_id == "sq_001"
        assert output.search_strategy == "Broad then narrow"

    def test_can_have_multiple_queries(self):
        """Test multiple search queries."""
        queries = [f"query_{i}" for i in range(5)]
        output = ResearcherOutput(
            search_queries=queries,
            sub_question_id="sq_001",
            search_strategy="Test"
        )

        assert len(output.search_queries) == 5


class TestSynthesizerOutputSchema:
    """Test the SynthesizerOutput Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating SynthesizerOutput."""
        section = DraftSection(
            id="sec_001",
            title="Test",
            content="Content",
            last_updated="2024-01-01T00:00:00"
        )
        citation = Citation(
            id="[1]",
            url="https://example.com",
            snippet="Info",
            accessed_for="sq_001"
        )

        output = SynthesizerOutput(
            updated_section=section,
            new_citations=[citation],
            synthesis_notes="Combined findings"
        )

        assert output.updated_section == section
        assert len(output.new_citations) == 1
        assert output.synthesis_notes == "Combined findings"

    def test_can_have_multiple_citations(self):
        """Test multiple new citations."""
        section = DraftSection(
            id="sec_001",
            title="Test",
            content="Content",
            last_updated="2024-01-01T00:00:00"
        )
        citations = [
            Citation(id=f"[{i}]", url=f"https://example{i}.com", snippet=f"Info {i}", accessed_for="sq_001")
            for i in range(1, 6)
        ]

        output = SynthesizerOutput(
            updated_section=section,
            new_citations=citations,
            synthesis_notes="Test"
        )

        assert len(output.new_citations) == 5


class TestCriticOutputSchema:
    """Test the CriticOutput Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating CriticOutput."""
        critique = CritiqueResult(
            is_complete=False,
            quality_metrics=QualityMetrics(),
            reasoning="Needs more work"
        )

        output = CriticOutput(
            critique=critique,
            next_action="continue"
        )

        assert output.critique == critique
        assert output.next_action == "continue"

    def test_valid_next_actions(self):
        """Test various valid next actions."""
        critique = CritiqueResult(
            is_complete=False,
            quality_metrics=QualityMetrics(),
            reasoning="Test"
        )

        for action in ["continue", "finalize", "stop"]:
            output = CriticOutput(critique=critique, next_action=action)
            assert output.next_action == action


class TestReportGeneratorOutputSchema:
    """Test the ReportGeneratorOutput Pydantic model."""

    def test_creates_with_required_fields(self):
        """Test creating ReportGeneratorOutput."""
        output = ReportGeneratorOutput(
            final_report="# Final Report\n\nContent here",
            report_metadata={"word_count": 1000, "sections": 5}
        )

        assert "Final Report" in output.final_report
        assert output.report_metadata["word_count"] == 1000
        assert output.report_metadata["sections"] == 5

    def test_metadata_can_contain_various_types(self):
        """Test that metadata can contain various data types."""
        metadata = {
            "word_count": 1000,
            "sections": 5,
            "title": "Report",
            "has_citations": True,
            "quality_score": 0.85,
            "tags": ["tag1", "tag2"]
        }

        output = ReportGeneratorOutput(
            final_report="Report content",
            report_metadata=metadata
        )

        assert output.report_metadata["word_count"] == 1000
        assert output.report_metadata["title"] == "Report"
        assert output.report_metadata["has_citations"] is True


# ============================================================================
# IMPORT TESTS
# ============================================================================

class TestImports:
    """Test that all imports work correctly."""

    def test_can_import_schemas(self):
        """Test that schemas module can be imported."""
        from langgraph_examples.deep_research_agent import schemas

        assert hasattr(schemas, 'ResearchPlan')
        assert hasattr(schemas, 'SubQuestion')
        assert hasattr(schemas, 'DeepResearchState')

    def test_can_import_planner_helpers(self):
        """Test that planner helpers can be imported."""
        from langgraph_examples.deep_research_agent.agents.planner import (
            create_sub_question,
            create_default_plan,
            validate_research_plan
        )

        # Functions should be callable
        assert callable(create_sub_question)
        assert callable(create_default_plan)
        assert callable(validate_research_plan)

    def test_can_import_prompt_engineer_helpers(self):
        """Test that prompt engineer helpers can be imported."""
        from langgraph_examples.deep_research_agent.agents.prompt_engineer import (
            create_fallback_output,
            PromptEngineerOutput,
            PromptAnalysis
        )

        assert callable(create_fallback_output)
        # Should be able to instantiate the classes
        analysis = PromptAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            context_score=0.5,
            actionability_score=0.5,
            issues_identified=[],
            strengths=[]
        )
        assert analysis is not None

    def test_can_import_all_schema_models(self):
        """Test importing all schema models."""
        from langgraph_examples.deep_research_agent.schemas import (
            ResearchPhase,
            Citation,
            SubQuestion,
            ResearchPlan,
            DraftSection,
            ResearchDraft,
            QualityMetrics,
            CritiqueResult,
            DeepResearchState,
            PlannerOutput,
            ResearcherOutput,
            SynthesizerOutput,
            CriticOutput,
            ReportGeneratorOutput,
            StopConditionConfig,
        )

        # All should be importable
        assert ResearchPhase is not None
        assert Citation is not None
        assert SubQuestion is not None


# ============================================================================
# INTEGRATION TESTS (combining multiple components)
# ============================================================================

class TestIntegration:
    """Integration tests that combine multiple components."""

    def test_create_and_validate_plan_workflow(self):
        """Test the workflow of creating and validating a plan."""
        # Create a plan
        query = "What is blockchain technology?"
        plan = create_default_plan(query)

        # Validate the plan
        is_valid, issues = validate_research_plan(plan)

        # The default plan should be valid
        assert is_valid is True
        assert len(issues) == 0

    def test_modify_plan_and_revalidate(self):
        """Test modifying a plan and revalidating."""
        plan = create_default_plan("Test query")

        # Initially valid
        is_valid, _ = validate_research_plan(plan)
        assert is_valid is True

        # Break the plan by removing sub-questions
        plan.sub_questions = plan.sub_questions[:2]

        # Should now be invalid
        is_valid, issues = validate_research_plan(plan)
        assert is_valid is False
        assert len(issues) > 0

    def test_sub_questions_can_be_created_and_added_to_plan(self):
        """Test creating individual sub-questions and adding to plan."""
        # Create sub-questions
        sq1 = create_sub_question("What is AI?", priority=1)
        sq2 = create_sub_question("What is ML?", priority=2)
        sq3 = create_sub_question("What is DL?", priority=3)

        # Create plan with these questions
        plan = ResearchPlan(
            main_query="AI technologies",
            objective="Learn about AI",
            scope="Basic concepts",
            sub_questions=[sq1, sq2, sq3],
            methodology="Literature review",
            expected_sections=["Intro", "Body", "Conclusion"]
        )

        # Validate
        is_valid, _ = validate_research_plan(plan)
        assert is_valid is True

    def test_fallback_output_matches_schema(self):
        """Test that fallback output matches PromptEngineerOutput schema."""
        original_prompt = "do something with code"
        fallback = create_fallback_output(original_prompt)

        # Should be a valid PromptEngineerOutput
        assert isinstance(fallback, PromptEngineerOutput)

        # Should have all required fields
        assert fallback.analysis is not None
        assert fallback.refined_prompt is not None
        assert fallback.changes_made is not None
        assert fallback.reasoning is not None

        # All scores should be valid
        assert 0 <= fallback.analysis.clarity_score <= 1
        assert 0 <= fallback.analysis.specificity_score <= 1
        assert 0 <= fallback.analysis.context_score <= 1
        assert 0 <= fallback.analysis.actionability_score <= 1

    def test_complete_research_state_workflow(self):
        """Test creating and updating a complete research state."""
        # Create initial state
        state = DeepResearchState(original_query="Test query")
        assert state.phase == ResearchPhase.PLANNING

        # Add research plan
        state.research_plan = create_default_plan("Test query")
        assert state.research_plan is not None

        # Move to researching phase
        state.phase = ResearchPhase.RESEARCHING
        state.iteration = 1
        assert state.phase == ResearchPhase.RESEARCHING

        # Add citations
        citation = Citation(
            id="[1]",
            url="https://example.com",
            snippet="Info",
            accessed_for="sq_001"
        )
        state.citations.append(citation)
        assert len(state.citations) == 1

    def test_planner_output_with_default_plan(self):
        """Test creating PlannerOutput with default plan."""
        plan = create_default_plan("Test query")
        output = PlannerOutput(
            research_plan=plan,
            reasoning="Using default plan structure"
        )

        # Validate the plan within the output
        is_valid, issues = validate_research_plan(output.research_plan)
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
