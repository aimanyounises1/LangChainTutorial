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
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

3. **Scope the Research**: Define boundaries - what's in scope and what's out of scope AS A SINGLE STRING.

4. **Break Down into Sub-Questions**: Decompose the main query into 4-7 specific, answerable sub-questions that together will comprehensively address the main query. Each sub-question MUST have:
   - id: A unique string like "sq_001", "sq_002", etc.
   - question: The specific question text
   - priority: Integer 1, 2, or 3 (1=highest, 3=lowest)

5. **Plan the Methodology**: Describe how the research will be conducted AS A SINGLE STRING.

6. **Structure the Output**: Define what sections the final report should contain AS A LIST OF STRINGS.

## CRITICAL: Output Format Requirements

You MUST return a JSON object with EXACTLY this structure:

```json
{{
  "research_plan": {{
    "main_query": "The exact user query as a string",
    "objective": "Clear objective statement as a string",
    "scope": "Single string describing what is in and out of scope",
    "sub_questions": [
      {{"id": "sq_001", "question": "First question?", "priority": 1}},
      {{"id": "sq_002", "question": "Second question?", "priority": 1}},
      {{"id": "sq_003", "question": "Third question?", "priority": 2}},
      {{"id": "sq_004", "question": "Fourth question?", "priority": 2}},
      {{"id": "sq_005", "question": "Fifth question?", "priority": 3}}
    ],
    "methodology": "Single string describing the research approach",
    "expected_sections": ["Section 1", "Section 2", "Section 3", "Section 4", "Section 5"]
  }},
  "reasoning": "Explanation of why this plan was chosen as a string"
}}
```

## IMPORTANT RULES:
- "scope" MUST be a single string, NOT an object with in_scope/out_of_scope keys
- "methodology" MUST be a single string, NOT an object or list
- Each sub_question MUST have "id", "question", and "priority" fields
- "expected_sections" MUST be a list of strings
- "reasoning" MUST be a top-level field in the output

Current time: {time}
"""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Create a comprehensive research plan for the query above. Return ONLY the JSON object with research_plan and reasoning fields. Remember: scope and methodology must be STRINGS, not objects. Each sub_question must have id, question, and priority."),
]).partial(time=lambda: datetime.datetime.now().isoformat())


# ============================================================================
# PLANNER LLM CONFIGURATION
# ============================================================================

def create_planner_llm(model_name: str = "llama3.3:70b"):
    """Create the LLM configured for the planner agent with structured output."""
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=8192,
    )
    # Use with_structured_output for robust schema enforcement
    return llm.with_structured_output(PlannerOutput, include_raw=True)


# ============================================================================
# PLANNER CHAIN
# ============================================================================

planner_llm = create_planner_llm()
planner_chain = PLANNER_PROMPT | planner_llm

# Note: No separate parser needed - with_structured_output handles parsing internally


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


# ============================================================================
# HIGH-LEVEL PLANNING FUNCTION
# ============================================================================

def create_research_plan(
    query: str,
    max_retries: int = 2
) -> Tuple[ResearchPlan, str]:
    """
    Create a research plan for the given query with retry logic.

    Args:
        query: The main research query
        max_retries: Maximum number of retry attempts on validation failure

    Returns:
        (ResearchPlan, reasoning)
    """
    from langchain_core.messages import HumanMessage

    prompt_vars = {
        "messages": [HumanMessage(content=query)]
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = planner_chain.invoke(prompt_vars)

            # with_structured_output with include_raw=True returns dict with 'parsed' and 'raw'
            if isinstance(result, dict) and 'parsed' in result:
                parsed = result['parsed']
                if parsed and isinstance(parsed, PlannerOutput):
                    # Ensure sub-questions have proper IDs
                    plan = parsed.research_plan
                    for i, sq in enumerate(plan.sub_questions):
                        if not sq.id or sq.id == "":
                            sq.id = f"sq_{uuid.uuid4().hex[:8]}"
                    return plan, parsed.reasoning
            # Direct PlannerOutput return (without include_raw)
            elif isinstance(result, PlannerOutput):
                plan = result.research_plan
                for i, sq in enumerate(plan.sub_questions):
                    if not sq.id or sq.id == "":
                        sq.id = f"sq_{uuid.uuid4().hex[:8]}"
                return plan, result.reasoning

        except Exception as e:
            last_error = e
            print(f"[Planner] Attempt {attempt + 1}/{max_retries + 1} Error: {e}")

            # Add error context to prompt for retry
            if attempt < max_retries:
                error_feedback = f"\n\nPrevious attempt failed with error: {str(e)[:500]}\nPlease ensure you provide ALL required fields in PlannerOutput: research_plan (with main_query, objective, scope as string, sub_questions with id/question/priority, methodology, expected_sections) and reasoning."
                prompt_vars["messages"] = [
                    HumanMessage(content=f"{query}{error_feedback}")
                ]

    print(f"[Planner] All attempts failed. Last error: {last_error}")

    # Fallback: create a default plan
    return create_default_plan(query), "Fallback plan due to validation errors"


if __name__ == "__main__":
    # Test the planner
    from langchain_core.messages import HumanMessage

    test_query = "What are the best practices for implementing AI-powered cybersecurity solutions in enterprise environments?"

    print("Testing Planner Agent...")
    print(f"Query: {test_query}\n")

    plan, reasoning = create_research_plan(test_query)
    print(f"Research Plan:")
    print(f"  - Main Query: {plan.main_query}")
    print(f"  - Objective: {plan.objective}")
    print(f"  - Scope: {plan.scope}")
    print(f"  - Sub-questions: {len(plan.sub_questions)}")
    for sq in plan.sub_questions:
        print(f"    [{sq.priority}] {sq.question}")
    print(f"  - Expected Sections: {plan.expected_sections}")
    print(f"  - Reasoning: {reasoning}")
