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

        # print(
        #     "\n===== Raw Planner Output ====="
        # )
        # print(repr(content))
        
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
  the selected source type

- source_type:
  choose the most appropriate source type
  based on the research need


Source type selection rules:

Use PAPER for:
- original research papers
- model architectures
- training methods
- benchmark results
- quantitative evaluations
- ablation studies
- technical methodology

Use WEB for:
- news
- documentation
- project announcements
- product or organization information
- broad ecosystem information
- recent developments that may not yet
  appear in research papers

Use GITHUB for:
- open-source implementations
- official repositories
- released code
- checkpoints
- configuration
- installation instructions
- repository activity
- implementation details


Query generation rules:

For WEB tasks:
- Generate a concise natural-language
  web search query.
- Use useful technical keywords.
- Do not simply repeat the user's full request.
- Prefer specific terms such as model names,
  task names, benchmark names, methods,
  years, or application domains when relevant.

Example WEB query:

"GUI agent computer use latest developments 2026"


For PAPER tasks:
- The query MUST be directly executable
  by the arXiv search backend.
- Use arXiv field syntax with Boolean operators.
- Prefer the following format:

  all:keyword1 AND all:keyword2 AND all:keyword3

- Use 3 to 6 discriminative technical keywords.
- Do not generate a full natural-language question.
- Do not include unnecessary stop words such as:
  the, a, an, of, for, to, latest, research.
- Avoid quoted phrases unless truly necessary.
- Prefer English technical keywords.
- Use AND to combine the major concepts.

Good PAPER query examples:

all:GUI AND all:agent AND all:multimodal

all:GUI AND all:agent AND all:reinforcement AND all:training

all:computer AND all:use AND all:agent AND all:benchmark

Bad PAPER query examples:

"What are the latest GUI agent training methods?"

"Research GUI agents in 2025-2026"

"latest multimodal GUI agent papers"

For GITHUB tasks:
- Generate concise repository-oriented keywords.
- Prefer project names, model names, task names,
  implementation terms, or framework names.
- Do not generate full natural-language questions.

Good examples:

GUI agent multimodal
computer use agent
UI-TARS
Mobile-Agent
GUI agent reinforcement learning


Planning requirements:

1. Cover the important dimensions of
   the user's research request.

2. Avoid overlapping or duplicate tasks.

3. Do not make tasks too broad.

4. Do not make tasks unnecessarily
   fine-grained.

5. Each task should be independently
   executable by a research worker.

6. Match the query format to the selected
   source_type.

7. PAPER queries must already be valid
   arXiv-style search queries and must not
   require additional rewriting by the
   research worker.

8. WEB queries should remain concise,
   readable search queries.

9. Prefer PAPER when the task mainly needs
   primary technical evidence.

10. Prefer WEB when the task mainly needs
    ecosystem-level or announcement-level
    information.

11. Do not answer the research question.
    Only create the research plan.


Return valid JSON only:

{{
  "goal": "normalized overall research goal",
  "tasks": [
    {{
      "title": "...",
      "intent": "...",
      "query": "...",
      "source_type": "paper"
    }}
  ]
}}
""".strip()

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