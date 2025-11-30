from typing import List, Annotated
from pydantic import BaseModel, Field


# --- EXISTING SCHEMAS ---
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 detailed answer of the question.")
    reflection: str = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements.")


class ReviseAnswer(AnswerQuestion):
    references: List[str] = Field(description="Citations(the source url) motivating your updated answer .")


# --- TOOL INPUT SCHEMA ---
class SearchQueriesInput(BaseModel):
    """Input schema for the search tool - only contains search_queries."""
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements.")


# --- NEW OBJECT (The Fix) ---
class SearchResult(BaseModel):
    """Represents a single search result item."""
    url: str = Field(description="The source URL of the information")
    content: str = Field(description="The content snippet from the source")


# This is where we use Annotated for the list of objects, as you requested.
# It defines: "This variable holds a list of SearchResult objects."
SearchResultsList = Annotated[List[SearchResult], Field(description="A collection of validated search results")]