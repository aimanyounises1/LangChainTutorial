from typing import List

from pydantic import BaseModel
from pydantic import Field


class Reflection(BaseModel):
    """
    Reflection of findings
    """
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """ Answer the question"""
    answer: str = Field(description="~250 detailed answer of the question.")
    reflection: str = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements"
                                                  " to address the critique of your current answer. "
                                      )
class ReviseAnswer(BaseModel):
    """
    Revise the original answer to your question
    """
    references:List[str] = Field(description="Citations motivating your answer.")