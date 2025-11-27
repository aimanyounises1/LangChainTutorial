from typing import List

from pydantic import BaseModel, Field

REACT_PROMPT_TEMPLATE = """
Answer the Following  questions as best you can you can access the following tools:
{tools}


Use the following format:
Questions : the input question you must answer
Thought: Think always about what you should do.
Action : the action to take, should be one of the [{tool_names}]
Observation : the result of the action 
...(this Thought/Action/Action Input/Observation can repeat N times)
Though: I know now the final answer
Final Answer : the final answer to the original input question

Begin!
 
Questions : {input}

Thought: {agent_scratchpad}

"""

class Source(BaseModel):
    """Schema for the url sources used by the agent this must be a url to the source."""
    url: str = Field(description="the url of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources."""
    answer: str = Field(description="The agent answer to the query.")
    source: List[Source] = Field(default_factory=list ,description="The list of the url sources used to generate the answer to the query.")
