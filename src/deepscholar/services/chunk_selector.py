import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
)
from deepscholar.models.chunk import (
    ChunkCandidate,
    DocumentChunk,
)
from deepscholar.models.research import (
    ResearchTask,
)


class HybridChunkSelector:
    def __init__(
        self,
        embedding_model_name: str,
        bm25_top_k: int = 10,
        dense_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ):
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.embedding_model = (
            SentenceTransformer(
                embedding_model_name
            )
        )

    def _build_query(
        self,
        task: ResearchTask,
    ) -> str:
        return "\n".join([
            task.title,
            task.intent,
            task.query,
        ])

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:

        return re.findall(
            r"[A-Za-z0-9_\-\.]+|[\u4e00-\u9fff]",
            text.lower(),
        )

    def _retrieve_bm25(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[tuple[DocumentChunk, float]]:

        tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        bm25 = BM25Okapi(
            tokenized_chunks
        )

        tokenized_query = (
            self._tokenize(query)
        )

        scores = bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            (chunk, float(score))
            for chunk, score in ranked[
                :self.bm25_top_k
            ]
        ]

    def _retrieve_dense(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[tuple[DocumentChunk, float]]:

        chunk_texts = [
            chunk.text
            for chunk in chunks
        ]
        chunk_embeddings = (
            self.embedding_model.encode(
                chunk_texts,
                normalize_embeddings=True,
            )
        )
        query_embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )[0]
        )
        scores = np.dot(
            chunk_embeddings,
            query_embedding,
        )
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            (chunk, float(score))
            for chunk, score in ranked[
                :self.dense_top_k
            ]
        ]

    def _rrf_fusion(
        self,
        bm25_results: list[
            tuple[DocumentChunk, float]
        ],
        dense_results: list[
            tuple[DocumentChunk, float]
        ],
    ) -> list[ChunkCandidate]:

        candidates = {}
        for rank, (
            chunk,
            score,
        ) in enumerate(
            bm25_results,
            start=1,
        ):
            candidate = candidates.setdefault(
                chunk.id,
                ChunkCandidate(
                    chunk=chunk
                )
            )
            candidate.bm25_score = score
            candidate.rrf_score += (
                1.0
                / (
                    self.rrf_k
                    + rank
                )
            )
        for rank, (
            chunk,
            score,
        ) in enumerate(
            dense_results,
            start=1,
        ):
            candidate = candidates.setdefault(
                chunk.id,
                ChunkCandidate(
                    chunk=chunk
                )
            )
            candidate.dense_score = score
            candidate.rrf_score += (
                1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        ranked = sorted(
            candidates.values(),
            key=lambda item: item.rrf_score,
            reverse=True,
        )
        return ranked[
            :self.final_top_k
        ]

    def select(
        self,
        task: ResearchTask,
        chunks: list[DocumentChunk],
    ) -> list[ChunkCandidate]:

        if not chunks:
            return []

        query = self._build_query(
            task
        )

        bm25_results = (
            self._retrieve_bm25(
                query=query,
                chunks=chunks,
            )
        )

        dense_results = (
            self._retrieve_dense(
                query=query,
                chunks=chunks,
            )
        )

        return self._rrf_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
        )