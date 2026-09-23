from pydantic import BaseModel, Field


class CritiqueResult(BaseModel):

    sufficient: bool
    coverage_score: float = Field(
        ge=0,
        le=1,
    )
    covered_aspects: list[str]
    knowledge_gaps: list[str]
    assessment: str