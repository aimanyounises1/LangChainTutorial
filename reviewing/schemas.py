from typing import List

from pydantic import Field, BaseModel


class Source(BaseModel):
    """"
    The source of the tool calling post execution
    """
    url: str = Field(description="URL of the source")


class AgentResponse(BaseModel):
    """
        The agent response consisting two elements :
         i) Answer
         ii) The sources of each as citations
    """
    answer: str = Field(description="Answer of the agent")
    sources: List[Source] = Field(default_factory=list, description="Sources of the agent")
