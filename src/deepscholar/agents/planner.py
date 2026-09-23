import json
from deepscholar.llm.client import (
    LLMClient,
)
from deepscholar.models.research import (
    ResearchPlan,
    ResearchTask,
    SourceType,
)
from deepscholar.utils.structured_output import clean_json_text


class PlannerAgent:

    def __init__(
        self,
        llm: LLMClient,
    ):
        self.llm = llm

    def plan(
        self,
        query: str,
    ) -> ResearchPlan:

        query = query.strip()
        if not query:
            raise ValueError(
                "Research query cannot be empty."
            )
        prompt = self._build_prompt(
            query
        )
        content = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research planning agent."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        print(
            "\n===== Raw Planner Output ====="
        )
        print(repr(content))
        
        data = json.loads(
            clean_json_text(content)
        )
        return self._parse_plan(
            data
        )

    def _build_prompt(
        self,
        query: str,
    ) -> str:

        return f"""
    You are planning a deep research task.

    User research request:

    {query}


    Create a research plan containing
    3 to 5 independent and complementary
    research tasks.

    Each task should represent one
    meaningful research dimension that can
    be searched and investigated
    independently.

    For each task provide:

    - title:
    short task name

    - intent:
    what this task should determine

    - query:
    a focused search query suitable for
    retrieving relevant technical sources

    - source_type:
    use "web" for now


    Planning requirements:

    1. Cover the important dimensions of
    the user's research request.

    2. Avoid overlapping or duplicate tasks.

    3. Do not make tasks too broad.

    4. Do not make tasks unnecessarily
    fine-grained.

    5. Each task should be independently
    executable by a research worker.

    6. Search queries should contain useful
    technical keywords rather than simply
    repeating the user's full request.

    7. Do not answer the research question.
    Only create the research plan.


    Return valid JSON only:

    {{
    "goal": "normalized overall research goal",
    "tasks": [
        {{
        "title": "...",
        "intent": "...",
        "query": "...",
        "source_type": "web"
        }}
    ]
    }}
    """

    def _parse_plan(
        self,
        data: dict,
    ) -> ResearchPlan:

        raw_tasks = data.get(
            "tasks",
            []
        )

        if not raw_tasks:
            raise ValueError(
                "Planner returned no tasks."
            )

        tasks = []

        for index, item in enumerate(
            raw_tasks,
            start=1,
        ):

            task = ResearchTask(
                id=f"task_{index:03d}",
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

            tasks.append(task)

        return ResearchPlan(
            goal=data.get(
                "goal",
                "",
            ),
            tasks=tasks,
        )