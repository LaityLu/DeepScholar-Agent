from deepscholar.config import settings

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

from deepscholar.tools.web_fetch import (
    TavilyWebFetcher,
)


def main():

    task = ResearchTask(
        id="task_001",
        title="LLM architecture",
        intent=(
            "Understand Transformer, "
            "self-attention and decoder "
            "architecture used by modern "
            "large language models."
        ),
        query=(
            "large language model "
            "transformer attention "
            "architecture"
        ),
        source_type=SourceType.WEB,
    )

    fetcher = TavilyWebFetcher()

    document = fetcher.fetch(
        "https://en.wikipedia.org/wiki/"
        "Large_language_model"
    )

    chunker = DocumentChunker(
        chunk_size=3000,
        chunk_overlap=300,
    )

    chunks = chunker.split(
        document
    )

    selector = HybridChunkSelector(
        embedding_model_name=(
            settings.embedding_model_path
        ),
        bm25_top_k=10,
        dense_top_k=10,
        final_top_k=5,
    )

    candidates = selector.select(
        task=task,
        chunks=chunks,
    )

    print(
        f"Total Chunks: {len(chunks)}"
    )

    print(
        f"Selected: {len(candidates)}"
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):

        chunk = candidate.chunk

        print(
            f"\n===== Rank {rank} ====="
        )

        print(
            f"Chunk Index: "
            f"{chunk.index}"
        )

        print(
            f"BM25: "
            f"{candidate.bm25_score}"
        )

        print(
            f"Dense: "
            f"{candidate.dense_score}"
        )

        print(
            f"RRF: "
            f"{candidate.rrf_score:.6f}"
        )

        print(
            "\nContent:"
        )

        print(
            chunk.text[:1000]
        )


if __name__ == "__main__":
    main()