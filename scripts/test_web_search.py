from deepscholar.tools.web_search import (
    TavilyWebSearchTool,
)


def main():
    tool = TavilyWebSearchTool()
    response = tool.search(
        query=(
            "multimodal GUI agent research 2026"
        ),
        max_results=3,
    )

    print(
        f"Query: {response.query}"
    )

    print(
        f"Provider: {response.provider}"
    )

    print(
        f"Response Time: "
        f"{response.response_time}"
    )

    print(
        f"Results: "
        f"{len(response.results)}"
    )

    for i, result in enumerate(
        response.results,
        start=1,
    ):
        print(
            f"\n===== Result {i} ====="
        )

        print(
            f"Title: {result.title}"
        )

        print(
            f"URL: {result.url}"
        )

        print(
            f"Score: {result.score}"
        )

        print(
            f"Source: "
            f"{result.source_type}"
        )

        print(
            f"Content: "
            f"{result.content[:300]}"
        )


if __name__ == "__main__":
    main()