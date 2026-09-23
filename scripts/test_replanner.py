from deepscholar.agents.replanner import (
    ReplannerAgent,
)

from deepscholar.llm.client import (
    LLMClient,
)

from deepscholar.models.critique import (
    CritiqueResult,
)

from deepscholar.models.research import (
    ResearchPlan,
    ResearchTask,
    SourceType,
)


def main():

    plan = ResearchPlan(
        goal=(
            "Investigate GUI agents "
            "including architecture, "
            "training, benchmarks and "
            "limitations."
        ),
        tasks=[
            ResearchTask(
                id="task_001",
                title="Architecture",
                intent=(
                    "Understand GUI-agent "
                    "architectures."
                ),
                query=(
                    "GUI agent architecture"
                ),
                source_type=SourceType.WEB,
            ),
            ResearchTask(
                id="task_002",
                title="Training",
                intent=(
                    "Understand GUI-agent "
                    "training methods."
                ),
                query="GUI agent training",
                source_type=SourceType.WEB,
            ),
        ],
    )

    critique = CritiqueResult(
        sufficient=False,
        coverage_score=0.55,
        covered_aspects=[
            "Architecture",
            "Training methods",
        ],
        knowledge_gaps=[
            (
                "Major GUI-agent benchmarks "
                "and their evaluation settings "
                "are not sufficiently covered."
            ),
            (
                "Current limitations and "
                "failure modes remain unclear."
            ),
        ],
        assessment=(
            "Architecture and training are "
            "covered, but benchmark and "
            "limitation evidence is missing."
        ),
    )

    llm = LLMClient()

    replanner = ReplannerAgent(
        llm=llm
    )

    result = replanner.replan(
        plan=plan,
        critique=critique,
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()