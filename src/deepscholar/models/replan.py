from pydantic import BaseModel

from deepscholar.models.research import (
    ResearchTask,
)


class ReplanResult(BaseModel):
    new_tasks: list[ResearchTask]
    rationale: str