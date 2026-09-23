from transformers import (
    AutoTokenizer,
)

from deepscholar.models.chunk import (
    ChunkCandidate,
    DocumentChunk,
)


class ContextBuilder:

    def __init__(
        self,
        tokenizer_path: str,
        max_context_tokens: int = 4096,
        reserved_prompt_tokens: int = 700,
        reserved_output_tokens: int = 800,
    ):
        if max_context_tokens <= 0:
            raise ValueError(
                "max_context_tokens must be positive."
            )

        if reserved_prompt_tokens < 0:
            raise ValueError(
                "reserved_prompt_tokens cannot be negative."
            )

        if reserved_output_tokens < 0:
            raise ValueError(
                "reserved_output_tokens cannot be negative."
            )

        available_tokens = (
            max_context_tokens
            - reserved_prompt_tokens
            - reserved_output_tokens
        )

        if available_tokens <= 0:
            raise ValueError(
                "No token budget remains for source context."
            )

        self.max_context_tokens = (
            max_context_tokens
        )

        self.reserved_prompt_tokens = (
            reserved_prompt_tokens
        )

        self.reserved_output_tokens = (
            reserved_output_tokens
        )

        self.available_context_tokens = (
            available_tokens
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        )

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        if not text:
            return 0

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        return len(token_ids)


    def select_chunks(
        self,
        candidates: list[ChunkCandidate],
    ) -> list[DocumentChunk]:

        selected = []
        used_tokens = 0
        for candidate in candidates:
            chunk = candidate.chunk
            chunk_text = (
                f"[Chunk {chunk.index}]\n"
                f"{chunk.text}"
            )
            chunk_tokens = (
                self._count_tokens(
                    chunk_text
                )
            )
            if (
                used_tokens
                + chunk_tokens
                > self.available_context_tokens
            ):
                continue

            selected.append(
                chunk
            )

            used_tokens += (
                chunk_tokens
            )

        return selected

    def count_selected_tokens(
        self,
        chunks: list[DocumentChunk],
    ) -> int:

        total = 0

        for chunk in chunks:
            text = (
                f"[Chunk {chunk.index}]\n"
                f"{chunk.text}"
            )

            total += self._count_tokens(
                text
            )

        return total