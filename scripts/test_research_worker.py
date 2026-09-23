from deepscholar.config import (
    settings,
)
from deepscholar.services.context_builder import (
    ContextBuilder,
)
from deepscholar.llm.client import (
    LLMClient,
)

from deepscholar.models.research import (
    ResearchTask,
    SourceType,
)

from deepscholar.services.chunker import (
    DocumentChunker,
)

from deepscholar.services.chunk_selector import (
    HybridChunkSelector,
)

from deepscholar.services.evidence_extractor import (
    EvidenceExtractor,
)

from deepscholar.services.research_worker import (
    ResearchWorker,
)

from deepscholar.tools.web_fetch import (
    TavilyWebFetcher,
)

from deepscholar.tools.web_search import (
    TavilyWebSearchTool,
)


def main():
    context_builder = ContextBuilder(
        tokenizer_path=(
            settings.llm_tokenizer_path
        ),
        max_context_tokens=(
            settings.llm_max_context_tokens
        ),
        reserved_prompt_tokens=700,
        reserved_output_tokens=(
            settings.llm_max_output_tokens
        ),
    )

    search_tool = (
        TavilyWebSearchTool()
    )

    fetcher = TavilyWebFetcher()

    chunker = DocumentChunker(
        chunk_size=3000,
        chunk_overlap=300,
    )

    selector = HybridChunkSelector(
        embedding_model_name=(
            settings.embedding_model_path
        ),
        bm25_top_k=10,
        dense_top_k=10,
        final_top_k=5,
    )

    llm = LLMClient()

    extractor = EvidenceExtractor(
        llm=llm
    )

    worker = ResearchWorker(
        search_tool=search_tool,
        fetcher=fetcher,
        chunker=chunker,
        selector=selector,
        extractor=extractor,
        context_builder=context_builder,
        max_sources=3,
    )

    task = ResearchTask(
        id="task_001",
        title=(
            "Multimodal GUI Agent "
            "architecture"
        ),
        intent=(
            "Understand the major "
            "architectural approaches "
            "used by recent multimodal "
            "GUI agents."
        ),
        query=(
            "multimodal GUI agent "
            "architecture vision "
            "language model"
        ),
        source_type=SourceType.WEB,
    )

    result = worker.run(
        task
    )

    print(
        "\n===== Research Result ====="
    )

    print(
        f"Task ID: "
        f"{result.task_id}"
    )

    print(
        f"Query: "
        f"{result.query}"
    )

    print(
        f"Searched Sources: "
        f"{result.searched_sources}"
    )

    print(
        f"Processed Sources: "
        f"{result.processed_sources}"
    )

    print(
        f"Failed Sources: "
        f"{len(result.failed_sources)}"
    )

    print(
        f"Evidence Count: "
        f"{len(result.evidences)}"
    )

    for i, evidence in enumerate(
        result.evidences,
        start=1,
    ):

        print(
            f"\n===== Evidence {i} ====="
        )

        print(
            f"Title: "
            f"{evidence.title}"
        )

        print(
            f"URL: "
            f"{evidence.url}"
        )

        print(
            f"Score: "
            f"{evidence.relevance_score}"
        )

        print(
            f"Content: "
            f"{evidence.content}"
        )

        print(
            f"Quote: "
            f"{evidence.quote}"
        )

    if result.failed_sources:

        print(
            "\n===== Failed Sources ====="
        )

        for failure in (
            result.failed_sources
        ):

            print(
                f"{failure.url}"
            )

            print(
                f"Error: "
                f"{failure.error}"
            )


if __name__ == "__main__":
    main()