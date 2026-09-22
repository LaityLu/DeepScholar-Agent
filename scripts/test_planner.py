from deepscholar.agents.planner import (
    PlannerAgent,
)

from deepscholar.llm.client import (
    LLMClient,
)


def main():

    llm = LLMClient()

    planner = PlannerAgent(
        llm=llm
    )

    query = """
Research the recent progress of
multimodal GUI agents in 2025-2026.

Focus on model architectures,
training methods, representative
systems, benchmarks, and current
limitations.
"""

    plan = planner.plan(
        query
    )

    print(
        "\n===== Research Plan ====="
    )

    print(
        f"Goal:\n{plan.goal}"
    )

    print(
        f"\nTask Count: "
        f"{len(plan.tasks)}"
    )

    for task in plan.tasks:

        print(
            f"\n===== {task.id} ====="
        )

        print(
            f"Title: "
            f"{task.title}"
        )

        print(
            f"Intent: "
            f"{task.intent}"
        )

        print(
            f"Query: "
            f"{task.query}"
        )

        print(
            f"Source: "
            f"{task.source_type}"
        )

        print(
            f"Status: "
            f"{task.status}"
        )


if __name__ == "__main__":
    main()