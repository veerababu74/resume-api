# resume_api/models/search.py
from pydantic import BaseModel, Field
from typing import List, Optional


class VectorSearchQuery(BaseModel):
    query: str = Field(..., description="Search query for semantic search")
    field: str = Field(
        ...,
        description="Field to search in (skills, experience_text, education_text, projects_text)",
    )
    num_results: int = Field(default=5, description="Number of results to return")
