from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from deepscholar.models.chunk import (
    DocumentChunk,
)
from deepscholar.models.source import (
    SourceDocument,
)


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 300,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be positive."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size."
            )

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )
        )

    def split(
        self,
        document: SourceDocument,
    ) -> list[DocumentChunk]:
        content = document.content.strip()
        if not content:
            return []
        texts = self.splitter.split_text(
            content
        )
        chunks = []
        for index, text in enumerate(texts):
            chunks.append(
                DocumentChunk(
                    id=f"chunk_{index}",
                    text=text,
                    index=index,
                )
            )
        return chunks