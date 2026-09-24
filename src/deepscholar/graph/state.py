import operator

from typing import Annotated
from typing_extensions import TypedDict

from deepscholar.models.citation import CitationVerificationResult, Claim
from deepscholar.models.critique import (
    CritiqueResult,
)
from deepscholar.models.evidence import (
    Evidence,
)

from deepscholar.models.research import (
    ResearchPlan,
    ResearchTask
)

from deepscholar.models.worker import (
    ResearchWorkerResult,
)


class ResearchTaskState(TypedDict):
    task: ResearchTask


class ResearchState(TypedDict):

    user_query: str
    plan: ResearchPlan | None
    worker_results: Annotated[
        list[ResearchWorkerResult],
        operator.add,
    ]
    evidences: Annotated[
        list[Evidence],
        operator.add,
    ]
    critique: CritiqueResult | None
    previous_critique: CritiqueResult | None
    last_critic_task_index: int

    has_new_tasks: bool
    replan_count: int
    max_replans: int 

    processed_evidences: list[Evidence]
    claims: list[Claim]
    citation_verification: (
        CitationVerificationResult | None
    )
    verified_claims: list[Claim]

    report: str | None

