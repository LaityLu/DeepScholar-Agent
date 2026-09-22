from pydantic import BaseModel, Field
from .research import SourceType


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    source_type: SourceType
    published_at: str | None = None


class SearchResponse(BaseModel):
    query: str
    provider: str
    results: list[SearchResult]
    response_time: float | None = None