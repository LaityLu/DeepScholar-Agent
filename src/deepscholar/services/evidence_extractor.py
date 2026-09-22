import json
import uuid

from deepscholar.llm.client import LLMClient
from deepscholar.models.evidence import (
    Evidence,
    EvidenceExtractionResult,
)
from deepscholar.models.research import (
    ResearchTask,
)
from deepscholar.models.source import (
    SourceDocument,
)
from deepscholar.models.chunk import (
    DocumentChunk,
)


class EvidenceExtractor:
    def __init__(
        self,
        llm: LLMClient,
    ):
        self.llm = llm

    def extract(
        self,
        task: ResearchTask,
        document: SourceDocument,
    ) -> EvidenceExtractionResult:
        prompt = self._build_prompt(
            task=task,
            document=document,
        )
        content = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence extraction system."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        data = json.loads(content)
        evidences = []
        for item in data.get(
            "evidences",
            [],
        ):
            evidences.append(
                Evidence(
                    id=str(uuid.uuid4()),
                    task_id=task.id,
                    source_type=(
                        document.source_type
                    ),
                    title=(
                        document.title
                        or document.url
                    ),
                    url=document.url,
                    content=item[
                        "content"
                    ],
                    quote=item.get(
                        "quote"
                    ),
                    relevance_score=(
                        item.get(
                            "relevance_score"
                        )
                    ),
                )
            )

        return EvidenceExtractionResult(
            evidences=evidences,
            source_sufficient=data.get(
                "source_sufficient",
                False,
            ),
        )


    def _build_prompt(
            self,
            task: ResearchTask,
            document: SourceDocument,
        ) -> str:

            return f"""
        Research task:

        Title:
        {task.title}

        Intent:
        {task.intent}

        Query:
        {task.query}


        Source document:

        URL:
        {document.url}

        Content:
        {document.content}


        Extract only evidence that is directly
        useful for completing the research task.

        Return valid JSON only:

        {{
        "evidences": [
            {{
            "content": "concise evidence summary",
            "quote": "short supporting text",
            "relevance_score": 0.0
            }}
        ],
        "source_sufficient": false
        }}

        Requirements:

        1. Do not invent facts.
        2. Evidence must come from the provided source.
        3. Ignore irrelevant information.
        4. relevance_score must be between 0 and 1.
        5. If no useful evidence exists, return an empty list.
        6. Return at most 5 high-quality evidence items.
        7. Return JSON only.
        """

    def extract_from_chunks(
        self,
        task: ResearchTask,
        document: SourceDocument,
        chunks: list[DocumentChunk],
    ) -> EvidenceExtractionResult:

        if not chunks:
            return EvidenceExtractionResult(
                evidences=[],
                source_sufficient=False,
            )

        combined_content = "\n\n".join(
            (
                f"[Chunk {chunk.index}]\n"
                f"{chunk.text}"
            )
            for chunk in chunks
        )

        selected_document = (
            document.model_copy(
                update={
                    "content": combined_content
                }
            )
        )

        return self.extract(
            task=task,
            document=selected_document,
        )