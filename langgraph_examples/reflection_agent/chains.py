import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama

from langgraph_examples.reflection_agent.schemas import AnswerQuestion, ReviseAnswer

load_dotenv(verbose=True)
llm = ChatOllama(model='qwen3:30b-a3b',
                 validate_model_on_init=True,
                 temperature=0,  # Temperature 0 for deterministic, focused responses
                 reasoning=True,  # Enable thinking/reasoning mode
                 )

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])
parser_pydantic_reviser = PydanticToolsParser(tools=[ReviseAnswer])
actor_prompt_template = ChatPromptTemplate.from_messages(
    [(
        "system",
        """
        You're expert researcher.
        Current time: {time}
        1.{first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. Recommend search queries to research information and improve your answer.
        """

    ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

revise_instructions = """Revise your previous answer using the new information from the search results.

    CRITICAL: Look at the FIRST message in the conversation - that is the user's original question.
    You MUST answer ONLY that original question. Do NOT change topics or answer a different question.

    Instructions:
    - Use the search results to improve your answer about the ORIGINAL topic
    - Include inline citations with the actual source URL next to the corresponding text, like: "Salicylic acid helps unclog pores [https://example.com/skincare]"
    - The 'references' field MUST contain actual URLs (starting with http:// or https://), NOT citation names
    - If your answer is complete and accurate, set search_queries to an empty list to stop the process
"""

first_respond_prompt_template = actor_prompt_template.partial(first_instruction=" Provide a detailed and comprehensive answer")
first_responder = first_respond_prompt_template | llm.bind_tools(tools=[AnswerQuestion],
                                                                 tool_choice="AnswerQuestion")
reviser = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == '__main__':
    human_message = HumanMessage(
        content="Search and suggest for the skin care routine to prevent any breakouts on face and to clean out the acne on face")

    chain = (first_respond_prompt_template
             | llm.bind_tools(tools=[AnswerQuestion],
                              tool_choice="AnswerQuestion")


             )

    res = chain.invoke({"messages": [human_message]})
    print(res)
