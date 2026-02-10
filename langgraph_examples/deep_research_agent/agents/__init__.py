"""
Deep Research Agent - Sub-Agents Package

This package contains all the specialized sub-agents for the deep research system:

1. Planner: Creates research plans and breaks down queries into sub-questions
2. Researcher: Generates search queries and executes searches
3. Synthesizer: Integrates findings into the evolving draft
4. Critic: Evaluates draft quality and determines completion
5. ReportGenerator: Creates the final polished report
6. PromptEngineer: Analyzes and refines prompts for optimal LLM performance
"""

from langgraph_examples.deep_research_agent.agents.planner import (
    planner_chain,
    create_sub_question,
    create_default_plan,
    validate_research_plan,
    create_research_plan,
)

from langgraph_examples.deep_research_agent.agents.researcher import (
    researcher_chain,
    researcher_parser,
    execute_search_queries,
    create_citations_from_results,
    research_sub_question,
)

from langgraph_examples.deep_research_agent.agents.synthesizer import (
    synthesizer_chain,
    synthesize_findings,
    update_draft_with_section,
    initialize_draft,
    determine_target_section,
)

from langgraph_examples.deep_research_agent.agents.critic import (
    critic_chain,
    critique_draft,
    should_stop,
    calculate_improvement,
)

from langgraph_examples.deep_research_agent.agents.report_generator import (
    report_generator_chain,
    report_generator_parser,
    generate_final_report,
    format_report_as_markdown,
    calculate_report_statistics,
)

from langgraph_examples.deep_research_agent.agents.prompt_engineer import (
    prompt_engineer_chain,
    refine_prompt,
    quick_refine,
    PromptEngineerOutput,
    PromptAnalysis,
)

__all__ = [
    # Planner
    "planner_chain",
    "create_sub_question",
    "create_default_plan",
    "validate_research_plan",
    "create_research_plan",
    # Researcher
    "researcher_chain",
    "researcher_parser",
    "execute_search_queries",
    "create_citations_from_results",
    "research_sub_question",
    # Synthesizer
    "synthesizer_chain",
    "synthesize_findings",
    "update_draft_with_section",
    "initialize_draft",
    "determine_target_section",
    # Critic
    "critic_chain",
    "critique_draft",
    "should_stop",
    "calculate_improvement",
    # Report Generator
    "report_generator_chain",
    "report_generator_parser",
    "generate_final_report",
    "format_report_as_markdown",
    "calculate_report_statistics",
    # Prompt Engineer
    "prompt_engineer_chain",
    "refine_prompt",
    "quick_refine",
    "PromptEngineerOutput",
    "PromptAnalysis",
]
