from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from core.schemas import AgentResponse, REACT_PROMPT_TEMPLATE
from core.tools import search_tool
import dotenv
load_dotenv(verbose=True)

import os

def run_agent():
    llm = ChatOllama(
        model="qwen3:30b-a3b",
        reasoning=True,
        temperature=0.8,
        validate_model_on_init=True,
    )

    output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
    prompt = PromptTemplate(input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
                            template=REACT_PROMPT_TEMPLATE).partial(
        format_instructions=output_parser.get_format_instructions())
    agent = create_react_agent(llm=llm, tools=[search_tool], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[search_tool],
                                   handle_parsing_errors=True, verbose=True)
    extract_output = RunnableLambda(lambda x:x['output'])
    parse_output = RunnableLambda(lambda x:output_parser.parse(x))
    chain = agent_executor | extract_output | parse_output
    result = chain.invoke({
        "input": "What are the new for today?",
    })

    print(result)

if __name__ == '__main__':
    run_agent()
