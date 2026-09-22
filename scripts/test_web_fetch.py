from deepscholar.tools.web_fetch import (
    TavilyWebFetcher,
)


def main():
    fetcher = TavilyWebFetcher()
    document = fetcher.fetch(
        "https://en.wikipedia.org/wiki/Large_language_model"
    )
    print(
        f"URL: {document.url}"
    )
    print(
        f"Source Type: "
        f"{document.source_type}"
    )
    print(
        f"Content Length: "
        f"{len(document.content)}"
    )
    print("\n===== Preview =====")
    print(
        document.content[:1000]
    )

if __name__ == "__main__":
    main()