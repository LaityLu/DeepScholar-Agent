from deepscholar.tools.github_fetcher import (
    GitHubFetcher,
)
from deepscholar.tools.github_search import (
    GitHubSearchTool,
)


def main():

    search_tool = GitHubSearchTool()

    fetcher = GitHubFetcher()

    response = search_tool.search(
        query="GUI agent multimodal",
        max_results=3,
    )

    print(
        "\n===== GitHub Search ====="
    )

    assert response.results

    for index, result in enumerate(
        response.results,
        start=1,
    ):
        print(
            f"\n[{index}]"
        )

        print(
            "Title:",
            result.title,
        )

        print(
            "URL:",
            result.url,
        )

        print(
            result.content[:500],
        )

    first = response.results[0]

    print(
        "\n===== Fetch Repository ====="
    )

    document = fetcher.fetch(
        first.url
    )

    print(
        "Title:",
        document.title,
    )

    print(
        "URL:",
        document.url,
    )

    print(
        "Content Length:",
        len(document.content),
    )

    print(
        "\n===== Content Preview =====\n"
    )

    print(
        document.content[:3000]
    )

    assert document.content

    print(
        "\n===== GitHub Tool Test Passed ====="
    )


if __name__ == "__main__":
    main()