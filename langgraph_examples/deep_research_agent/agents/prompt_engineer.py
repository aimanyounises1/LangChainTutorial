"""
Deep Research Agent - Prompt Engineering Specialist Agent

The Prompt Engineering Specialist agent is responsible for:
1. Analyzing user prompts for clarity, specificity, and effectiveness
2. Identifying ambiguities, missing context, and improvement opportunities
3. Refining prompts to maximize LLM output quality
4. Applying best practices in prompt engineering
"""

import datetime
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# ============================================================================
# PROMPT ENGINEER OUTPUT SCHEMA
# ============================================================================

class PromptAnalysis(BaseModel):
    """Analysis of the original prompt."""
    clarity_score: float = Field(description="How clear is the prompt (0-1)")
    specificity_score: float = Field(description="How specific is the prompt (0-1)")
    context_score: float = Field(description="How much context is provided (0-1)")
    actionability_score: float = Field(description="How actionable is the request (0-1)")
    issues_identified: List[str] = Field(description="List of issues found in the prompt")
    strengths: List[str] = Field(description="What the prompt does well")


class PromptEngineerOutput(BaseModel):
    """Output schema for the Prompt Engineering Specialist agent."""
    analysis: PromptAnalysis = Field(description="Analysis of the original prompt")
    refined_prompt: str = Field(description="The improved, refined prompt")
    changes_made: List[str] = Field(description="List of changes/improvements made")
    reasoning: str = Field(description="Explanation of the refinement decisions")
    alternative_prompts: List[str] = Field(
        default_factory=list,
        description="Alternative refined versions for different use cases"
    )


# ============================================================================
# PROMPT ENGINEER SYSTEM PROMPT
# ============================================================================

PROMPT_ENGINEER_SYSTEM_PROMPT = """""
Your goal is to improve the prompt given below for {task} :
--------------------

Prompt: {lazy_prompt}

--------------------

Here are several tips on writing great prompts:

-------

Start the prompt by stating that it is an expert in the subject.

Put instructions at the beginning of the prompt and use ### or to separate the instruction and context 

Be specific, descriptive and as detailed as possible about the desired context, outcome, length, format, style, etc 

---------

Here's an example of a great prompt:

As a master YouTube content creator, develop an engaging script that revolves around the theme of "Exploring Ancient Ruins."

Your script should encompass exciting discoveries, historical insights, and a sense of adventure.

Include a mix of on-screen narration, engaging visuals, and possibly interactions with co-hosts or experts.

The script should ideally result in a video of around 10-15 minutes, providing viewers with a captivating journey through the secrets of the past.

Example:

"Welcome back, fellow history enthusiasts, to our channel! Today, we embark on a thrilling expedition..."

-----
Current time: {time}

You MUST use the PromptEngineerOutput tool to provide the refined_prompt
Now, improve the prompt.

IMPROVED PROMPT:
"""

PROMPT_ENGINEER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PROMPT_ENGINEER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Analyze the user's prompt and provide a refined version. Use the PromptEngineerOutput tool."),
]).partial(time=lambda: datetime.datetime.now().isoformat())


# ============================================================================
# PROMPT ENGINEER LLM CONFIGURATION
# ============================================================================

def create_prompt_engineer_llm(model_name: str = "llama3.3:70b"):
    """Create the LLM configured for the prompt engineer agent with structured output."""
    llm = ChatOllama(
        model=model_name,
        temperature=0.3,  # Slight creativity for prompt variations
        num_ctx=8192,
    )
    # Use with_structured_output for robust schema enforcement
    return llm.with_structured_output(schema=PromptEngineerOutput, include_raw=True)


# ============================================================================
# PROMPT ENGINEER CHAIN
# ============================================================================

prompt_engineer_llm = create_prompt_engineer_llm()
prompt_engineer_chain = PROMPT_ENGINEER_PROMPT | prompt_engineer_llm


# Note: No separate parser needed - with_structured_output handles parsing internally


# ============================================================================
# HIGH-LEVEL PROMPT REFINEMENT FUNCTION
# ============================================================================

def refine_prompt(
        user_prompt: str,
        context: Optional[str] = None,
        target_model: Optional[str] = None,
        max_retries: int = 2
) -> Tuple[PromptEngineerOutput, bool]:
    """
    Analyze and refine a user prompt with retry logic.

    Args:
        user_prompt: The original prompt to refine
        context: Optional additional context about the use case
        target_model: Optional target model the prompt will be used with
        max_retries: Maximum number of retry attempts on validation failure

    Returns:
        (PromptEngineerOutput, success_flag)

    """

    # Build the request message

    request_parts = [f"Please analyze and refine this prompt:\n\n---\n{user_prompt}\n---"]

    if context:
        request_parts.append(f"\nAdditional context: {context}")

    if target_model:
        request_parts.append(f"\nTarget model: {target_model}")

    prompt_vars = {
        "messages": [HumanMessage(content="\n".join(request_parts))]
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = prompt_engineer_chain.invoke(prompt_vars)

            # with_structured_output with include_raw=True returns dict with 'parsed' and 'raw'
            if isinstance(result, dict) and 'parsed' in result:
                parsed = result['parsed']
                if parsed and isinstance(parsed, PromptEngineerOutput):
                    return parsed, True
            # Direct PromptEngineerOutput return (without include_raw)
            elif isinstance(result, PromptEngineerOutput):
                return result, True

        except Exception as e:
            last_error = e
            print(f"[PromptEngineer] Attempt {attempt + 1}/{max_retries + 1} Error: {e}")

            # Add error context to prompt for retry
            if attempt < max_retries:
                error_feedback = f"\n\nPrevious attempt failed with error: {str(e)[:500]}\nPlease ensure you provide ALL required fields in PromptEngineerOutput: analysis (with all scores and lists), refined_prompt, changes_made, and reasoning."
                # Extract newline join to avoid backslash in f-string
                request_text = '\n'.join(request_parts)
                prompt_vars["messages"] = [
                    HumanMessage(content=f"{request_text}{error_feedback}")
                ]

    print(f"[PromptEngineer] All attempts failed. Last error: {last_error}")

    # Fallback: return basic refinement
    return create_fallback_output(user_prompt), False


def create_fallback_output(original_prompt: str) -> PromptEngineerOutput:
    """Create a fallback output when LLM fails."""
    return PromptEngineerOutput(
        analysis=PromptAnalysis(
            clarity_score=0.5,
            specificity_score=0.5,
            context_score=0.5,
            actionability_score=0.5,
            issues_identified=["Unable to perform full analysis due to processing error"],
            strengths=["Original prompt preserved"]
        ),
        refined_prompt=f"Please help me with the following:\n\n{original_prompt}\n\nPlease provide a detailed and well-structured response.",
        changes_made=["Added basic structure", "Added politeness markers"],
        reasoning="Fallback refinement due to processing errors. Basic improvements applied.",
        alternative_prompts=[]
    )


def quick_refine(user_prompt: str) -> str:
    """
    Quick convenience function to get just the refined prompt.

    Args:
        user_prompt: The original prompt

    Returns:
        The refined prompt string
    """
    output, _ = refine_prompt(user_prompt)
    return output.refined_prompt


if __name__ == "__main__":
    # Test the prompt engineer
    test_prompts = [
        "write code for sorting",
        "Explain machine learning",
        "I need help with my project about AI stuff, it's not working right and I've tried a bunch of things but nothing works, can you help?",
    ]

    print("Testing Prompt Engineering Specialist Agent...\n")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n### Test {i}: Original Prompt ###")
        print(f'"{prompt}"')
        print()
        sub_query_prompt = {
            "lazy_prompt": prompt,
            "messages": [HumanMessage(content=prompt)]
        }

        output, success = prompt_engineer_chain.invoke(sub_query_prompt)

        print(f"Analysis Scores:")
        print(f"  - Clarity: {output.analysis.clarity_score:.2f}")
        print(f"  - Specificity: {output.analysis.specificity_score:.2f}")
        print(f"  - Context: {output.analysis.context_score:.2f}")
        print(f"  - Actionability: {output.analysis.actionability_score:.2f}")

        print(f"\nIssues Identified:")
        for issue in output.analysis.issues_identified:
            print(f"  - {issue}")

        print(f"\n### Refined Prompt ###")
        print(output.refined_prompt)

        print(f"\nChanges Made:")
        for change in output.changes_made:
            print(f"  - {change}")

        print(f"\nReasoning: {output.reasoning[:200]}...")
        print("=" * 60)
