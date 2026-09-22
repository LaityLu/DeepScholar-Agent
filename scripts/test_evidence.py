from deepscholar.llm.client import (
    LLMClient,
)
from deepscholar.models.research import (
    ResearchTask,
    SourceType,
)
from deepscholar.services.evidence_extractor import (
    EvidenceExtractor,
)
from deepscholar.tools.web_fetch import (
    TavilyWebFetcher,
)


def main():
    task = ResearchTask(
        id="task_001",
        title=(
            "Large Language Model architecture"
        ),
        intent=(
            "Understand the core architecture of LLMs."
        ),
        query=(
            "large language model transformer architecture"
        ),
        source_type=SourceType.WEB,
    )

    fetcher = TavilyWebFetcher()
    document = fetcher.fetch(
        "https://en.wikipedia.org/wiki/Large_language_model"
    )

    # 第一版防止页面太长
    document.content = (
        document.content[:12000]
    )
    llm = LLMClient()
    extractor = EvidenceExtractor(
        llm=llm
    )
    result = extractor.extract(
        task=task,
        document=document,
    )
    print(
        f"Evidence Count: "
        f"{len(result.evidences)}"
    )
    print(
        f"Source Sufficient: "
        f"{result.source_sufficient}"
    )
    for i, evidence in enumerate(
        result.evidences,
        start=1,
    ):
        print(
            f"\n===== Evidence {i} ====="
        )
        print(
            f"Content: "
            f"{evidence.content}"
        )
        print(
            f"Quote: "
            f"{evidence.quote}"
        )
        print(
            f"Score: "
            f"{evidence.relevance_score}"
        )
        print(
            f"URL: "
            f"{evidence.url}"
        )


if __name__ == "__main__":
    main()