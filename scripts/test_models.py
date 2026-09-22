from deepscholar.models.research import (
    ResearchTask,
    SourceType,
)
from deepscholar.models.evidence import Evidence


task = ResearchTask(
    id="task_001",
    title="GUI Agent 架构调研",
    intent="了解近期 GUI Agent 的核心架构",
    query="multimodal GUI agent architecture 2026",
    source_type=SourceType.PAPER,
)


evidence = Evidence(
    id="ev_001",
    task_id=task.id,
    source_type=SourceType.PAPER,
    title="Example Paper",
    url="https://example.com/paper",
    content="This paper proposes...",
    relevance_score=0.9,
)


print(task.model_dump())
print(evidence.model_dump())