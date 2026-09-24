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

        # print(
        #     "\n===== Raw Critic Output ====="
        # )
        # print(repr(content))

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
Evaluate whether the collected evidence is sufficient
to answer the ORIGINAL research goal at a useful
research-report level.

The purpose of this evaluation is NOT to determine
whether every possible technical detail has been
researched.

The purpose is to determine whether the user can now
receive a useful, evidence-grounded answer to the
original research request.


==================================================
ORIGINAL RESEARCH GOAL
==================================================

{plan.goal}


==================================================
PLANNED RESEARCH TASKS
==================================================

{tasks_text}


==================================================
COLLECTED EVIDENCE
==================================================

{evidence_text}


==================================================
CORE EVALUATION PRINCIPLE
==================================================

Judge sufficiency against the ORIGINAL research goal,
not against an ideal exhaustive academic survey.

The research should be considered sufficient when the
major dimensions explicitly requested or materially
implied by the original goal have useful supporting
evidence.

Do NOT keep expanding the research simply because more
technical details could theoretically be investigated.


==================================================
SUFFICIENCY RULES
==================================================

1. Focus on the original research goal.

   Ask:

   - What major dimensions did the user actually request?
   - Does the evidence support a useful discussion of
     those dimensions?

2. Prefer sufficient=true when all major requested
   dimensions have reasonable supporting evidence.

3. The research does NOT need to be exhaustive.

4. Minor missing details must NOT cause
   sufficient=false.

5. Implementation-level details should normally NOT be
   treated as blocking knowledge gaps unless the
   original research goal explicitly requests them.

6. The following are normally NON-BLOCKING details:

   - exact reward-function formulas
   - exact training hyperparameters
   - exact teacher-model identities
   - distillation implementation details
   - transfer-fidelity metrics
   - exact optimizer settings
   - exhaustive benchmark scores
   - exhaustive ablation results
   - exhaustive pairwise model comparisons
   - complete architectural implementation details

7. Do NOT introduce new research requirements merely
   because the collected evidence mentions an interesting
   technique or subtopic.

8. A knowledge gap should be reported ONLY if its absence
   materially prevents answering an important part of the
   original research goal.

9. If a missing detail would only make the final report
   more comprehensive, but the report can already answer
   the user's main question, it is NOT a blocking gap.

10. Do not require direct comparisons between every pair
    of systems unless comparison is explicitly central to
    the original goal.

11. If representative examples are available for a major
    research dimension, exhaustive examples are not
    required.

12. If evidence supports a meaningful qualitative
    synthesis but lacks every possible quantitative
    detail, the research may still be sufficient.

13. coverage_score measures coverage of the ORIGINAL
    research goal.

    Suggested interpretation:

    0.00 - 0.39:
        Major requested dimensions are missing.

    0.40 - 0.69:
        Partial coverage. Important user-requested
        dimensions remain weak or missing.

    0.70 - 0.84:
        Most major requested dimensions are adequately
        covered and a useful report can be written.

    0.85 - 1.00:
        Strong coverage of the requested research goal.

14. Do NOT artificially keep coverage_score low because
    obscure technical details remain unavailable.

15. It is acceptable to return sufficient=true even when
    some secondary details remain unknown.


==================================================
KNOWLEDGE GAP RULES
==================================================

knowledge_gaps must contain ONLY major missing
information necessary to answer the original research
goal.

Good knowledge gap:

"The research goal asks for benchmark comparison, but
the evidence contains no benchmark results."

Bad knowledge gap:

"The exact reward coefficient used by one model is not
available."

Bad knowledge gap:

"The teacher model used in one distillation stage is
not identified."

Bad knowledge gap:

"No direct comparison exists between every model
mentioned in the evidence."


==================================================
FINAL DECISION
==================================================

Before returning sufficient=false, explicitly ask:

"Would the absence of this information prevent a useful
and responsible final report from answering the user's
original research question?"

If the answer is NO, do not use that missing detail as
a reason for sufficient=false.


Return exactly one JSON object:

{{
  "sufficient": true,
  "coverage_score": 0.8,
  "covered_aspects": [
    "..."
  ],
  "knowledge_gaps": [],
  "assessment": "..."
}}

Requirements:

- sufficient must be a boolean.
- coverage_score must be between 0 and 1.
- covered_aspects should summarize the major dimensions
  already supported by evidence.
- knowledge_gaps should contain only materially important
  unresolved gaps.
- assessment should briefly explain why the research is
  or is not sufficient.

Return JSON only.
""".strip()

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

        # print("\n===== Raw Incremental Critic Output =====")
        # print(repr(content))

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
Perform an INCREMENTAL evaluation of research coverage.

Do NOT restart the research evaluation from scratch.

Use the previous critique as a compact summary of the
earlier research state, and determine whether the NEW
research is sufficient to produce a useful final report
for the ORIGINAL research goal.


==================================================
ORIGINAL RESEARCH GOAL
==================================================

{plan.goal}


==================================================
PREVIOUSLY COVERED ASPECTS
==================================================

{covered_text}


==================================================
PREVIOUS KNOWLEDGE GAPS
==================================================

{gaps_text}


==================================================
NEW RESEARCH TASKS
==================================================

{tasks_text}


==================================================
NEW EVIDENCE
==================================================

{evidence_text}


==================================================
CORE INCREMENTAL EVALUATION PRINCIPLE
==================================================

Evaluate whether the ORIGINAL research goal can now be
answered adequately.

Previous knowledge gaps are NOT automatically mandatory
requirements.

You must reconsider whether each previous gap was truly
important to the original research goal.


==================================================
INCREMENTAL EVALUATION RULES
==================================================

1. Treat previously covered aspects as still covered
   unless the new evidence directly contradicts them.

2. Use previous knowledge gaps as a reference checklist,
   NOT as mandatory research requirements.

3. Remove a previous knowledge gap if it is:

   - implementation-level
   - overly specific
   - peripheral
   - merely interesting
   - unnecessary for answering the original goal

4. Do NOT preserve a previous gap simply because it was
   previously listed.

5. Do NOT introduce new research dimensions unless their
   absence materially prevents answering the original
   research goal.

6. Do NOT create new gaps merely because the new evidence
   mentions additional methods, models, benchmarks, or
   technical details.

7. The following should normally NOT block completion
   unless explicitly requested by the original goal:

   - exact reward formulas
   - training hyperparameters
   - teacher-model selection details
   - transfer-fidelity metrics
   - optimizer details
   - exhaustive ablation studies
   - exhaustive benchmark values
   - exhaustive model-by-model comparisons

8. A previous gap may be considered resolved when the new
   evidence provides enough information for a useful
   research-level discussion.

   Perfect or exhaustive evidence is NOT required.

9. Minor unresolved details should not force
   sufficient=false.

10. If the major research dimensions requested by the
    original goal are now adequately covered, return
    sufficient=true.

11. knowledge_gaps should contain ONLY unresolved gaps
    that still materially prevent the final report from
    answering the original goal.

12. covered_aspects should be cumulative:

    previously covered aspects
    +
    newly resolved important aspects.

13. coverage_score is cumulative coverage of the
    ORIGINAL research goal.

    It is NOT a score for the quality of only the new
    evidence.

14. coverage_score should normally increase when the new
    evidence materially resolves previous gaps.

15. Do not keep coverage_score unchanged simply because
    implementation-level details remain unavailable.

16. It is acceptable to return sufficient=true while
    mentioning secondary missing details in the
    assessment.


==================================================
IMPORTANT DECISION TEST
==================================================

Before keeping any item in knowledge_gaps, ask:

"If this information remains unavailable, can the final
report still provide a useful and evidence-grounded
answer to the user's original research request?"

If YES:

Do NOT keep it as a blocking knowledge gap.

If NO:

Keep it as a knowledge gap.


==================================================
OUTPUT
==================================================

Return exactly one JSON object:

{{
  "sufficient": true,
  "coverage_score": 0.85,
  "covered_aspects": [
    "..."
  ],
  "knowledge_gaps": [],
  "assessment": "..."
}}

Return JSON only.
""".strip()