from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    id: str
    text: str
    index: int = Field(
        ge=0,
    )
    start_char: int | None = None
    end_char: int | None = None


class ChunkCandidate(BaseModel):
    chunk: DocumentChunk
    bm25_score: float | None = None
    dense_score: float | None = None
    rrf_score: float = 0.0