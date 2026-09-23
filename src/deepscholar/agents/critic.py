import json

from deepscholar.llm.client import (
    LLMClient,
)
from deepscholar.models.critique import (
    CritiqueResult,
)
from deepscholar.models.evidence import (
    Evidence,
)
from deepscholar.models.research import (
    ResearchPlan,
    ResearchTask,
)
from deepscholar.utils.structured_output import clean_json_text


class CriticAgent:

    def __init__(
        self,
        llm: LLMClient,
    ):
        self.llm = llm

    def evaluate(
        self,
        plan: ResearchPlan,
        evidences: list[Evidence],
    ) -> CritiqueResult:

        prompt = self._build_prompt(
            plan=plan,
            evidences=evidences,
        )

        content = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research coverage critic."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=800,
        )

        print(
            "\n===== Raw Critic Output ====="
        )
        print(repr(content))

        data = json.loads(
            clean_json_text(content)
        )

        return self._parse_result(
            data
        )


    def _build_prompt(
        self,
        plan: ResearchPlan,
        evidences: list[Evidence],
    ) -> str:

        task_lines = []
        for task in plan.tasks:
            task_lines.append(
                (
                    f"- {task.id}: "
                    f"{task.title}\n"
                    f"  Intent: {task.intent}"
                )
            )
        tasks_text = "\n".join(
            task_lines
        )
        evidence_lines = []
        for index, evidence in enumerate(
            evidences,
            start=1,
        ):
            evidence_lines.append(
                (
                    f"[Evidence {index}]\n"
                    f"Task: {evidence.task_id}\n"
                    f"Source: {evidence.title}\n"
                    f"URL: {evidence.url}\n"
                    f"Content: {evidence.content}\n"
                    f"Quote: "
                    f"{evidence.quote or ''}"
                )
            )
        evidence_text = "\n\n".join(
            evidence_lines
        )
        
        return f"""
    You are evaluating the coverage of a
    deep research process.

    Overall research goal:

    {plan.goal}


    Planned research tasks:

    {tasks_text}


    Collected evidence:

    {evidence_text}


    Evaluate whether the current evidence
    is sufficient to answer the overall
    research goal.

    Focus on research coverage rather than
    the number of evidence items.

    A research plan is sufficient only when:

    1. The major research dimensions are
    adequately covered.

    2. Important user-requested dimensions
    are not missing.

    3. The evidence is specific enough to
    support meaningful conclusions.

    4. Evidence is not merely repetitive
    statements of the same fact.

    5. Missing information would not
    materially change the final report.


    Return valid JSON only:

    {{
    "sufficient": false,
    "coverage_score": 0.0,
    "covered_aspects": [
        "..."
    ],
    "knowledge_gaps": [
        "..."
    ],
    "assessment": "..."
    }}


    Requirements:

    - coverage_score must be between 0 and 1.

    - covered_aspects should list research
    dimensions that are adequately
    supported.

    - knowledge_gaps should describe
    concrete missing information that
    should be researched next.

    - If sufficient is true,
    knowledge_gaps should normally be
    empty.

    - Do not propose a new research plan.

    - Do not write the final research
    report.

    - Do not invent information that is not
    present in the evidence.

    - assessment should be concise.
    """

    def _parse_result(
        self,
        data: dict,
    ) -> CritiqueResult:

        required_fields = [
            "sufficient",
            "coverage_score",
            "covered_aspects",
            "knowledge_gaps",
            "assessment",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in data
        ]

        if missing_fields:
            raise ValueError(
                "Critic returned invalid schema. "
                f"Missing fields: {missing_fields}. "
                f"Received keys: {list(data.keys())}"
            )

        return CritiqueResult(
            sufficient=data["sufficient"],
            coverage_score=data[
                "coverage_score"
            ],
            covered_aspects=data[
                "covered_aspects"
            ],
            knowledge_gaps=data[
                "knowledge_gaps"
            ],
            assessment=data[
                "assessment"
            ],
        )

    def evaluate_incremental(
        self,
        plan: ResearchPlan,
        previous_critique: CritiqueResult,
        new_tasks: list[ResearchTask],
        new_evidences: list[Evidence],
    ) -> CritiqueResult:

        prompt = (
            self._build_incremental_prompt(
                plan=plan,
                previous_critique=(
                    previous_critique
                ),
                new_tasks=new_tasks,
                new_evidences=new_evidences,
            )
        )

        content = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research "
                        "coverage critic."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=800,
        )

        print("\n===== Raw Incremental Critic Output =====")
        print(repr(content))

        data = json.loads(
            clean_json_text(
            content
        )
        )

        return CritiqueResult.model_validate(
            data
        )

    def _build_incremental_prompt(
        self,
        plan: ResearchPlan,
        previous_critique: CritiqueResult,
        new_tasks: list[ResearchTask],
        new_evidences: list[Evidence],
    ) -> str:

        gaps_text = "\n".join(
            f"- {gap}"
            for gap
            in previous_critique.knowledge_gaps
        )

        covered_text = "\n".join(
            f"- {item}"
            for item
            in previous_critique.covered_aspects
        )

        task_lines = []

        for task in new_tasks:

            task_lines.append(
                (
                    f"- {task.id}: "
                    f"{task.title}\n"
                    f"  Intent: {task.intent}\n"
                    f"  Query: {task.query}"
                )
            )

        tasks_text = "\n".join(
            task_lines
        )

        evidence_lines = []

        for index, evidence in enumerate(
            new_evidences,
            start=1,
        ):

            evidence_lines.append(
                (
                    f"[New Evidence {index}]\n"
                    f"Task: {evidence.task_id}\n"
                    f"Source: {evidence.title}\n"
                    f"Content: "
                    f"{evidence.content}\n"
                    f"Relevance: "
                    f"{evidence.relevance_score}"
                )
            )

        evidence_text = "\n\n".join(
            evidence_lines
        )

        return f"""
    You are performing an incremental
    evaluation of an ongoing deep research
    process.

    Overall research goal:

    {plan.goal}


    Previously covered aspects:

    {covered_text}


    Knowledge gaps identified in the
    previous evaluation:

    {gaps_text}


    New research tasks created to address
    those gaps:

    {tasks_text}


    Newly collected evidence:

    {evidence_text}


    Evaluate whether the NEW evidence
    resolves the PREVIOUS knowledge gaps.

    Important rules:

    1. Use the previous knowledge gaps as
    the primary evaluation checklist.

    2. Do not restart the research
    evaluation from scratch.

    3. Do not introduce unrelated new
    research dimensions.

    4. A previous knowledge gap may remain
    unresolved if the new evidence is
    weak, indirect, or from insufficiently
    relevant sources.

    5. Previously covered aspects should
    remain covered unless the new evidence
    directly contradicts them.

    6. The final sufficient decision should
    reflect whether the original research
    goal can now be answered with adequate
    evidence.

    7. Minor secondary details should not
    prevent sufficient=true.

    Return exactly one JSON object with:

    {{
    "sufficient": false,
    "coverage_score": 0.0,
    "covered_aspects": [
        "..."
    ],
    "knowledge_gaps": [
        "..."
    ],
    "assessment": "..."
    }}

    Interpretation:

    - covered_aspects:
    Return the cumulative aspects now
    considered adequately covered,
    including previously covered aspects.

    - knowledge_gaps:
    Return only the previous gaps that
    still remain materially unresolved.

    - coverage_score:
    Estimate cumulative research coverage,
    not merely the quality of this new
    batch.

    - assessment:
    Briefly state which previous gaps were
    resolved and which remain.

    Return JSON only.
    """