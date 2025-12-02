"""
Deep Research Agent - Planner Agent

The Planner agent is responsible for:
1. Analyzing the original research query
2. Breaking it down into sub-questions
3. Creating a comprehensive research plan
4. Defining the methodology and expected output structure
"""

import datetime
import uuid
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticToolsParser
from langchain_ollama import ChatOllama

from langgraph_examples.deep_research_agent.schemas import (
    ResearchPlan,
    SubQuestion,
    PlannerOutput,
)


# ============================================================================
# PLANNER PROMPT
# ============================================================================

PLANNER_SYSTEM_PROMPT = """You are an expert research planner specializing in creating comprehensive research strategies.

Your task is to analyze the user's research query and create a detailed research plan that will guide a deep research process.

## Your Responsibilities:

1. **Understand the Query**: Deeply analyze what the user wants to learn or investigate.

2. **Define Clear Objectives**: Create a precise objective statement that captures the research goal.

3. **Scope the Research**: Define boundaries - what's in scope and what's out of scope.

4. **Break Down into Sub-Questions**: Decompose the main query into 4-7 specific, answerable sub-questions that together will comprehensively address the main query. Each sub-question should:
   - Be specific and focused
   - Be researchable (can find information online)
   - Have a clear priority (1=highest, 3=lowest)
   - Contribute to answering the main query

5. **Plan the Methodology**: Describe how the research will be conducted.

6. **Structure the Output**: Define what sections the final report should contain.

## Guidelines for Sub-Questions:
- Start with foundational questions (definitions, context) at priority 1
- Move to analytical questions (comparisons, evaluations) at priority 2
- End with forward-looking questions (trends, predictions) at priority 3
- Ensure sub-questions are mutually exclusive but collectively exhaustive (MECE)

Current time: {time}

You MUST use the PlannerOutput tool to provide your research plan.
"""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Create a comprehensive research plan for the query above. Use the PlannerOutput tool."),
]).partial(time=lambda: datetime.datetime.now().isoformat())


# ============================================================================
# PLANNER LLM CONFIGURATION
# ============================================================================

def create_planner_llm(model_name: str = "qwen3:30b-a3b"):
    """Create the LLM configured for the planner agent."""
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=8192,
    )
    return llm.bind_tools(tools=[PlannerOutput], tool_choice="PlannerOutput")


# ============================================================================
# PLANNER CHAIN
# ============================================================================

planner_llm = create_planner_llm()
planner_chain = PLANNER_PROMPT | planner_llm

# Parser for extracting structured output
planner_parser = PydanticToolsParser(tools=[PlannerOutput], first_tool_only=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_sub_question(
    question: str,
    priority: int = 1,
    status: str = "pending"
) -> SubQuestion:
    """Create a sub-question with a unique ID."""
    return SubQuestion(
        id=f"sq_{uuid.uuid4().hex[:8]}",
        question=question,
        priority=priority,
        status=status,
        findings=None,
        search_queries=[],
        citations=[]
    )


def create_default_plan(query: str) -> ResearchPlan:
    """Create a default research plan if LLM fails."""
    return ResearchPlan(
        main_query=query,
        objective=f"Conduct comprehensive research on: {query}",
        scope="General exploration of the topic with focus on key aspects",
        sub_questions=[
            create_sub_question(f"What is the definition and background of {query}?", priority=1),
            create_sub_question(f"What are the key components or aspects of {query}?", priority=1),
            create_sub_question(f"What are the current trends and developments in {query}?", priority=2),
            create_sub_question(f"What are the challenges and considerations in {query}?", priority=2),
            create_sub_question(f"What are the future prospects and predictions for {query}?", priority=3),
        ],
        methodology="Iterative search and synthesis approach with quality assessment",
        expected_sections=[
            "Executive Summary",
            "Introduction & Background",
            "Key Findings",
            "Analysis & Discussion",
            "Conclusions & Recommendations"
        ]
    )


def validate_research_plan(plan: ResearchPlan) -> tuple[bool, List[str]]:
    """
    Validate a research plan for completeness.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    if not plan.main_query:
        issues.append("Missing main query")
    
    if not plan.sub_questions or len(plan.sub_questions) < 3:
        issues.append("Need at least 3 sub-questions")
    
    if len(plan.sub_questions) > 10:
        issues.append("Too many sub-questions (max 10)")
    
    if not plan.expected_sections or len(plan.expected_sections) < 3:
        issues.append("Need at least 3 expected sections")
    
    # Check for duplicate sub-questions
    questions = [sq.question.lower().strip() for sq in plan.sub_questions]
    if len(questions) != len(set(questions)):
        issues.append("Duplicate sub-questions detected")
    
    return len(issues) == 0, issues


if __name__ == "__main__":
    # Test the planner
    from langchain_core.messages import HumanMessage
    
    test_query = "What are the best practices for implementing AI-powered cybersecurity solutions in enterprise environments?"
    
    print("Testing Planner Agent...")
    print(f"Query: {test_query}\n")
    
    result = planner_chain.invoke({"messages": [HumanMessage(content=test_query)]})
    print(f"Raw result: {result}")
    
    if result.tool_calls:
        parsed = planner_parser.invoke(result)
        print(f"\nParsed plan: {parsed}")
