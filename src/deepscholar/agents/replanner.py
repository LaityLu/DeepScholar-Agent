import json

from deepscholar.llm.client import (
    LLMClient,
)

from deepscholar.models.critique import (
    CritiqueResult,
)

from deepscholar.models.replan import (
    ReplanResult,
)

from deepscholar.models.research import (
    ResearchPlan,
    ResearchTask,
    SourceType,
)
from deepscholar.utils.structured_output import clean_json_text

class ReplannerAgent:

    def __init__(
        self,
        llm: LLMClient,
    ):
        self.llm = llm

    def replan(
        self,
        plan: ResearchPlan,
        critique: CritiqueResult,
    ) -> ReplanResult:

        if not critique.knowledge_gaps:
            return ReplanResult(
                new_tasks=[],
                rationale=(
                    "No explicit knowledge gaps were provided by the critic."
                ),
            )

        prompt = self._build_prompt(
            plan=plan,
            critique=critique,
        )
        content = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research replanning agent."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        # print(
        #     "\n===== Raw Replanner Output ====="
        # )
        # print(repr(content))
        
        data = json.loads(
            clean_json_text(content)
        )
        return self._parse_result(
            data=data,
            plan=plan,
        )

    def _build_prompt(
        self,
        plan: ResearchPlan,
        critique: CritiqueResult,
    ) -> str:

        task_lines = []
        for task in plan.tasks:
            task_lines.append(
                (
                    f"- {task.id}: {task.title}\n"
                    f"  Intent: {task.intent}\n"
                    f"  Query: {task.query}"
                )
            )

        tasks_text = "\n".join(
            task_lines
        )
        gaps_text = "\n".join(
            f"- {gap}"
            for gap in critique.knowledge_gaps
        )

        return f"""
    You are replanning an ongoing deep
    research process.

    Overall research goal:

    {plan.goal}


    Existing research tasks:

    {tasks_text}


    Critic assessment:

    {critique.assessment}


    Covered aspects:

    {chr(10).join(
        f"- {item}"
        for item in critique.covered_aspects
    )}


    Remaining knowledge gaps:

    {gaps_text}


    Create only the additional research
    tasks needed to address the remaining
    knowledge gaps.

    Requirements:

    1. Do not recreate tasks that are
    already adequately covered.

    2. New tasks must directly target the
    listed knowledge gaps.

    3. Avoid duplicate or highly
    overlapping tasks.

    4. Prefer 1 to 3 focused new tasks.

    5. Each task must be independently
    executable by the research worker.

    6. Search queries should contain
    concrete technical keywords.

    7. Use "web" as source_type for now.

    8. Do not write the final report.

    9. Do not modify completed tasks.


    Return valid JSON only:

    {{
    "new_tasks": [
        {{
        "title": "...",
        "intent": "...",
        "query": "...",
        "source_type": "web"
        }}
    ],
    "rationale": "concise explanation"
    }}
    """

    def _parse_result(
        self,
        data: dict,
        plan: ResearchPlan,
    ) -> ReplanResult:

        raw_tasks = data.get(
            "new_tasks",
            [],
        )

        if len(raw_tasks) > 5:
            raise ValueError(
                "Replanner returned too many new tasks."
            )

        start_index = (
            len(plan.tasks) + 1
        )

        tasks = []

        for offset, item in enumerate(
            raw_tasks
        ):

            task_id = (
                f"task_"
                f"{start_index + offset:03d}"
            )

            tasks.append(
                ResearchTask(
                    id=task_id,
                    title=item["title"],
                    intent=item["intent"],
                    query=item["query"],
                    source_type=SourceType(
                        item.get(
                            "source_type",
                            "web",
                        )
                    ),
                )
            )

        return ReplanResult(
            new_tasks=tasks,
            rationale=data.get(
                "rationale",
                "",
            ),
        )