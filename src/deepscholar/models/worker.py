from pydantic import BaseModel
from deepscholar.models.evidence import (
    Evidence,
)


class SourceFailure(BaseModel):
    url: str
    error: str


class ResearchWorkerResult(BaseModel):
    task_id: str
    query: str
    evidences: list[Evidence]
    searched_sources: int
    processed_sources: int
    failed_sources: list[SourceFailure]
    error: str | None = None
