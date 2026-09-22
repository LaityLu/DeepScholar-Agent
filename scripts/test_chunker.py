from deepscholar.models.research import (
    SourceType,
)
from deepscholar.models.source import (
    SourceDocument,
)
from deepscholar.services.chunker import (
    DocumentChunker,
)


def main():
    document = SourceDocument(
        url="https://example.com/test",
        title="Test Document",
        source_type=SourceType.WEB,
        content="""
# Large Language Models

Large language models are neural networks
trained on large amounts of text data.

They are commonly based on the Transformer
architecture.

## Transformer

The Transformer architecture relies heavily
on self-attention mechanisms.

Self-attention allows the model to capture
relationships between tokens in a sequence.

Modern decoder-only language models stack
multiple Transformer blocks.

Each block usually contains self-attention
layers and feed-forward neural networks.

## Training

Large language models are commonly trained
using next-token prediction.

During training, the model learns to predict
the next token based on previous tokens.

## Inference

During inference, decoder-only models generate
tokens autoregressively.

The generated token is appended to the context
and used to predict the next token.
""",
    )

    chunker = DocumentChunker(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = chunker.split(
        document
    )

    print(
        f"Document Length: "
        f"{len(document.content)}"
    )

    print(
        f"Chunk Count: "
        f"{len(chunks)}"
    )

    for chunk in chunks:

        print(
            f"\n===== Chunk "
            f"{chunk.index} ====="
        )

        print(
            f"Length: "
            f"{len(chunk.text)}"
        )

        print(chunk.text)


if __name__ == "__main__":
    main()