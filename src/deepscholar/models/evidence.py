from pydantic import BaseModel, Field

from .research import SourceType


class Evidence(BaseModel):
    id: str
    task_id: str
    source_type: SourceType
    title: str
    url: str
    content: str
    quote: str | None = None
    published_at: str | None = None
    relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    source_quality_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    quality_label: str | None = None


class EvidenceExtractionResult(BaseModel):
    evidences: list[Evidence]
    source_sufficient: bool = False