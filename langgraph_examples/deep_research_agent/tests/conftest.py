"""
Pytest configuration and shared fixtures for Deep Research Agent tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from langgraph_examples.deep_research_agent.schemas import (
    ResearchPlan,
    SubQuestion,
    PlannerOutput,
)
from langgraph_examples.deep_research_agent.agents.prompt_engineer import (
    PromptAnalysis,
    PromptEngineerOutput,
)


# ============================================================================
# FIXTURES - Sample Data
# ============================================================================

@pytest.fixture
def sample_sub_question():
    """Fixture providing a sample SubQuestion."""
    return SubQuestion(
        id="sq_test001",
        question="What is the definition of the topic?",
        priority=1,
        status="pending",
        findings=None,
        search_queries=[],
        citations=[]
    )


@pytest.fixture
def sample_sub_questions():
    """Fixture providing a list of sample SubQuestions."""
    return [
        SubQuestion(
            id="sq_001",
            question="What is the background and history?",
            priority=1,
            status="pending"
        ),
        SubQuestion(
            id="sq_002",
            question="What are the key components?",
            priority=1,
            status="pending"
        ),
        SubQuestion(
            id="sq_003",
            question="What are current trends?",
            priority=2,
            status="pending"
        ),
        SubQuestion(
            id="sq_004",
            question="What are the challenges?",
            priority=2,
            status="pending"
        ),
        SubQuestion(
            id="sq_005",
            question="What are future prospects?",
            priority=3,
            status="pending"
        ),
    ]


@pytest.fixture
def sample_research_plan(sample_sub_questions):
    """Fixture providing a valid ResearchPlan."""
    return ResearchPlan(
        main_query="What are the best practices for AI in healthcare?",
        objective="Identify and analyze best practices for implementing AI solutions in healthcare settings",
        scope="Focus on clinical applications, patient care, and regulatory compliance. Out of scope: hardware infrastructure.",
        sub_questions=sample_sub_questions,
        methodology="Systematic literature review combined with case study analysis",
        expected_sections=[
            "Executive Summary",
            "Introduction",
            "Current State of AI in Healthcare",
            "Best Practices Analysis",
            "Case Studies",
            "Regulatory Considerations",
            "Conclusions and Recommendations"
        ]
    )


@pytest.fixture
def sample_planner_output(sample_research_plan):
    """Fixture providing a sample PlannerOutput."""
    return PlannerOutput(
        research_plan=sample_research_plan,
        reasoning="The plan breaks down the complex topic into manageable sub-questions covering historical context, technical components, current state, challenges, and future directions."
    )


@pytest.fixture
def sample_prompt_analysis():
    """Fixture providing a sample PromptAnalysis."""
    return PromptAnalysis(
        clarity_score=0.75,
        specificity_score=0.65,
        context_score=0.70,
        actionability_score=0.80,
        issues_identified=[
            "Lacks specific context about domain",
            "Could be more explicit about desired output format"
        ],
        strengths=[
            "Clear main intent",
            "Actionable request"
        ]
    )


@pytest.fixture
def sample_prompt_engineer_output(sample_prompt_analysis):
    """Fixture providing a sample PromptEngineerOutput."""
    return PromptEngineerOutput(
        analysis=sample_prompt_analysis,
        refined_prompt="""As an expert technical writer specializing in AI documentation, your goal is to create comprehensive documentation for machine learning models.

🎯 OBJECTIVE

Develop detailed technical documentation that covers:
- Model architecture and design decisions
- Training methodology and datasets
- Performance metrics and evaluation
- Deployment guidelines

🧱 STYLE & FORMAT

- Use clear, professional technical language
- Include code examples where relevant
- Organize content with clear hierarchical sections
- Target audience: ML engineers and data scientists

📤 OUTPUT REQUIREMENTS

- Markdown formatted documentation
- Minimum 2000 words
- Include diagrams and visual aids
- Provide working code snippets
- Reference best practices and standards

Your mission: Create documentation that enables engineers to understand, reproduce, and deploy the model effectively.""",
        changes_made=[
            "Added expert role framing (technical writer specializing in AI)",
            "Structured with emoji section headers for clarity",
            "Specified target audience explicitly",
            "Defined concrete output requirements (format, length, components)",
            "Added actionable mission statement"
        ],
        reasoning="The original prompt was too vague. The refined version provides clear structure, expert framing, specific deliverables, and target audience context to guide the LLM toward producing comprehensive technical documentation.",
        alternative_prompts=[
            "Brief version: 'Document this ML model's architecture, training process, and deployment steps in Markdown format.'",
            "Developer-focused: 'As a senior ML engineer, create API documentation for this model including usage examples and performance benchmarks.'"
        ]
    )


# ============================================================================
# FIXTURES - Mock Objects
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Fixture providing a mock LLM response structure."""
    def _create_mock_response(parsed_output: Any) -> Dict[str, Any]:
        return {
            'parsed': parsed_output,
            'raw': Mock()
        }
    return _create_mock_response


@pytest.fixture
def mock_successful_planner_chain(sample_planner_output, mock_llm_response):
    """Fixture providing a mocked successful planner chain."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_llm_response(sample_planner_output)
    return mock_chain


@pytest.fixture
def mock_successful_prompt_engineer_chain(sample_prompt_engineer_output, mock_llm_response):
    """Fixture providing a mocked successful prompt engineer chain."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_llm_response(sample_prompt_engineer_output)
    return mock_chain


@pytest.fixture
def mock_failed_llm_chain():
    """Fixture providing a mocked failing LLM chain."""
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("LLM processing error")
    return mock_chain


# ============================================================================
# FIXTURES - Test Queries
# ============================================================================

@pytest.fixture
def test_queries():
    """Fixture providing various test queries."""
    return {
        'simple': "What is AI?",
        'complex': "What are the best practices for implementing AI-powered cybersecurity solutions in enterprise environments with legacy systems?",
        'vague': "tell me about stuff",
        'specific': "Compare the performance of BERT and GPT-3 on sentiment analysis tasks using the IMDB dataset",
        'unicode': "What is 人工智能 (Artificial Intelligence)?",
        'technical': "Explain the mathematical foundations of backpropagation in neural networks"
    }


@pytest.fixture
def test_prompts():
    """Fixture providing various test prompts for refinement."""
    return {
        'lazy': "write code",
        'vague': "help me with my project",
        'decent': "Can you explain how to use Python decorators?",
        'good': "As a Python expert, explain the use cases for decorators with three practical examples including code snippets",
    }


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interaction"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )


# ============================================================================
# AUTO-USE FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_mocks():
    """Automatically reset all mocks after each test."""
    yield
    # Cleanup happens automatically with unittest.mock
