"""
Chains for Reflection Agent - Using with_structured_output for direct Pydantic returns

This module uses LangChain's with_structured_output() method to get Pydantic objects
directly from the LLM, eliminating the need for manual parsing or text-based tool call extraction.

Key Benefits:
1. No manual JSON parsing from text
2. Automatic Pydantic validation
3. Direct object returns (AnswerQuestion, ReviseAnswer)
4. Built-in retry capability via .with_retry()
"""
import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from langgraph_examples.reflection_agent.schemas import AnswerQuestion, ReviseAnswer

load_dotenv(verbose=True)

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Base LLM configuration
LLM_CONFIG = {
    "model": "qwen3:30b-a3b",
    "temperature": 0,       # Deterministic for reliable structured output
    "num_ctx": 8192,        # Sufficient context for prompts + conversation
}

# Initialize base LLM
llm = ChatOllama(**LLM_CONFIG)


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

# Enhanced system prompt that explicitly instructs JSON output format
SYSTEM_PROMPT_TEMPLATE = """You are an expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend 1-3 search queries to research information and improve your answer.

IMPORTANT: You MUST respond with a valid JSON object matching the required schema.
"""

DRAFT_INSTRUCTION = "Provide a detailed and comprehensive answer (~250 words)."

REVISE_INSTRUCTION = """Revise your previous answer using the new information from the search results.

CRITICAL: Look at the FIRST message in the conversation - that is the user's original question.
You MUST answer ONLY that original question. Do NOT change topics or answer a different question.

Instructions:
- Use the search results to improve your answer about the ORIGINAL topic
- Include inline citations with actual source URLs next to corresponding text
- The 'references' field MUST contain actual URLs (starting with http:// or https://)
- If your answer is complete and accurate, set search_queries to an empty list to stop the process
"""

# Build prompt templates
actor_prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Answer the user's question above using the required JSON format."),
]).partial(time=lambda: datetime.datetime.now().isoformat())

# Specific prompts for each stage
draft_prompt = actor_prompt_template.partial(first_instruction=DRAFT_INSTRUCTION)
revise_prompt = actor_prompt_template.partial(first_instruction=REVISE_INSTRUCTION)


# ============================================================================
# STRUCTURED OUTPUT CHAINS
# ============================================================================

# These chains return Pydantic objects directly - no parsing needed!
# 
# For Ollama models, with_structured_output works best when the model
# supports tool calling. We don't specify 'method' to let LangChain
# auto-detect the best approach.

# Draft chain: Returns AnswerQuestion object
first_responder_structured = (
    draft_prompt 
    | llm.with_structured_output(
        AnswerQuestion,
        include_raw=False  # Only return the parsed object
    )
)

# Reviser chain: Returns ReviseAnswer object  
reviser_structured = (
    revise_prompt 
    | llm.with_structured_output(
        ReviseAnswer,
        include_raw=False
    )
)


# ============================================================================
# CHAINS WITH RETRY - For production resilience
# ============================================================================

# Add automatic retry on validation errors
# This handles cases where the LLM output doesn't match the schema
first_responder = first_responder_structured.with_retry(
    stop_after_attempt=3,
    retry_if_exception_type=(ValidationError, ValueError, TypeError, KeyError, AttributeError),
)

reviser = reviser_structured.with_retry(
    stop_after_attempt=3,
    retry_if_exception_type=(ValidationError, ValueError, TypeError, KeyError, AttributeError),
)


# ============================================================================
# LEGACY EXPORTS (for backward compatibility during migration)
# ============================================================================
# These are no longer needed but kept for any code that imports them
from langchain_core.output_parsers import PydanticToolsParser
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion], first_tool_only=True)
parser_pydantic_reviser = PydanticToolsParser(tools=[ReviseAnswer], first_tool_only=True)


# ============================================================================
# TEST
# ============================================================================
if __name__ == '__main__':
    print("Testing structured output chains...")
    print(f"Using model: {LLM_CONFIG['model']}")
    
    human_message = HumanMessage(
        content="What are the best practices for managing API keys in Python projects?"
    )
    
    print("\n" + "="*60)
    print("Testing first_responder (AnswerQuestion):")
    print("="*60)
    
    try:
        result = first_responder.invoke({"messages": [human_message]})
        print(f"\n✅ Success!")
        print(f"Type: {type(result)}")
        print(f"Answer ({len(result.answer)} chars): {result.answer[:200]}...")
        print(f"Reflection: {result.reflection[:100]}...")
        print(f"Search queries: {result.search_queries}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Chain returns Pydantic object directly - no parsing needed!")
    print("="*60)
