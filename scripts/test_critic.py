from deepscholar.agents.critic import (
    CriticAgent,
)

from deepscholar.llm.client import (
    LLMClient,
)

from deepscholar.models.evidence import (
    Evidence,
)

from deepscholar.models.research import (
    ResearchPlan,
    ResearchTask,
    SourceType,
)


def main():
    plan = ResearchPlan(
        goal=(
            "Investigate recent GUI agents "
            "covering architecture, training, "
            "benchmarks and limitations."
        ),
        tasks=[
            ResearchTask(
                id="task_001",
                title="Architecture",
                intent=(
                    "Understand major GUI agent "
                    "architectures."
                ),
                query="GUI agent architecture",
                source_type=SourceType.WEB,
            ),
            ResearchTask(
                id="task_002",
                title="Training",
                intent=(
                    "Understand training methods."
                ),
                query="GUI agent training",
                source_type=SourceType.WEB,
            ),
            ResearchTask(
                id="task_003",
                title="Benchmarks",
                intent=(
                    "Identify major benchmarks."
                ),
                query="GUI agent benchmark",
                source_type=SourceType.WEB,
            ),
            ResearchTask(
                id="task_004",
                title="Limitations",
                intent=(
                    "Identify current limitations."
                ),
                query="GUI agent limitations",
                source_type=SourceType.WEB,
            ),
        ],
    )

    evidences = [
        Evidence(
            id="ev_001",
            task_id="task_001",
            source_type=SourceType.WEB,
            title="Example Architecture",
            url="https://example.com/a",
            content=(
                "Recent GUI agents use "
                "multimodal language models "
                "for visual understanding and "
                "action generation."
            ),
            quote=(
                "GUI agents use multimodal "
                "language models."
            ),
            relevance_score=0.95,
        ),
        Evidence(
            id="ev_002",
            task_id="task_002",
            source_type=SourceType.WEB,
            title="Example Training",
            url="https://example.com/b",
            content=(
                "GUI agents can be trained "
                "using supervised fine-tuning "
                "and reinforcement learning."
            ),
            quote=(
                "supervised fine-tuning and "
                "reinforcement learning"
            ),
            relevance_score=0.9,
        ),
    ]

    llm = LLMClient()

    critic = CriticAgent(
        llm=llm
    )

    result = critic.evaluate(
        plan=plan,
        evidences=evidences,
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()