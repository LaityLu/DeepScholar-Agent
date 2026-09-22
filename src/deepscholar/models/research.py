from enum import Enum
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    WEB = "web"
    PAPER = "paper"
    GITHUB = "github"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchTask(BaseModel):
    id: str
    title: str
    intent: str
    query: str
    source_type: SourceType
    status: TaskStatus = TaskStatus.PENDING
    # Number of times the task has been retried
    retry_count: int = Field(
        default=0,
        ge=0,
    )


class ResearchPlan(BaseModel):
    goal: str
    tasks: list[ResearchTask]