from pydantic import BaseModel

from .research import SourceType


class SourceDocument(BaseModel):
    url: str
    title: str | None = None
    content: str
    source_type: SourceType
    published_at: str | None = None