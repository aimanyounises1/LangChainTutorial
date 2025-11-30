from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "A viral Twitter influencer grading tweet generates critique and recommendations to the user. Always provide detailed recommendations, including requests for length, virality, style, etc."
    ),
    MessagesPlaceholder(variable_name="messages")
])

generation_prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    "You are a Twitter teacher's influence tasked with writing excellent Twitter posts."
    "Generate the best Twitter posts possible for the user's request. If the user provides critique,"
    " respond with a revised version of your previous attempts."
    ),
    MessagesPlaceholder(variable_name="messages")

])


llm = ChatOllama(model='qwen3:30b-a3b',
                     validate_model_on_init=True,
                     temperature=0.8,
                     reasoning=True
                     )

generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm