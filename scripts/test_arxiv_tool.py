from deepscholar.tools.arxiv_fetcher import ArxivFetcher
from deepscholar.tools.arxiv_search import ArxivSearchTool


def main():
    search_tool = ArxivSearchTool()
    fetcher = ArxivFetcher()

    query = (
        "all:GUI AND "
        "all:agent AND "
        "all:multimodal"
    )

    # ==================================================
    # 1. Test arXiv Search
    # ==================================================

    response = search_tool.search(
        query=query,
        max_results=3,
    )

    print(
        "\n===== Search Response ====="
    )

    print(
        "Provider:",
        response.provider,
    )

    print(
        "Query:",
        response.query,
    )

    print(
        "Result Count:",
        len(response.results),
    )

    assert response.results, (
        "ArxivSearchTool returned "
        "no search results."
    )

    print(
        "\n===== Search Results ====="
    )

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
            "Published:",
            result.published_at,
        )

        print(
            "Source Type:",
            result.source_type,
        )

        print(
            "Abstract:",
            result.content[:500],
        )

        assert result.title
        assert result.url
        assert "arxiv.org" in result.url
        assert result.content

    # ==================================================
    # 2. Test arXiv Fetch
    # ==================================================

    first_result = response.results[0]

    print(
        "\n\n===== Fetching First Paper ====="
    )

    print(
        "URL:",
        first_result.url,
    )

    document = fetcher.fetch(
        first_result.url
    )

    print(
        "\n===== Source Document ====="
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
        "Source Type:",
        document.source_type,
    )

    print(
        "Published:",
        document.published_at,
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

    # ==================================================
    # 3. Assertions
    # ==================================================

    assert document.url
    assert "arxiv.org" in document.url

    assert document.content
    assert len(document.content) > 1000

    print(
        "\n===== arXiv Tool Test Passed ====="
    )


if __name__ == "__main__":
    main()