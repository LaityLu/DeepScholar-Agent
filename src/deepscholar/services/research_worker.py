from deepscholar.models.chunk import (
    DocumentChunk,
)
from deepscholar.services.context_builder import (
    ContextBuilder,
)
from deepscholar.models.research import (
    ResearchTask,
)
from deepscholar.models.worker import (
    ResearchWorkerResult,
    SourceFailure,
)
from deepscholar.models.search import (
    SearchResult,
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
from deepscholar.services.tool_router import ResearchToolRouter
from deepscholar.tools.base import (
    BaseSourceFetcher,
)



class ResearchWorker:
    def __init__(
        self,
        tool_router: ResearchToolRouter,
        chunker: DocumentChunker,
        selector: HybridChunkSelector,
        extractor: EvidenceExtractor,
        context_builder: ContextBuilder,
        max_sources: int = 5,
    ):
        self.tool_router = tool_router
        self.chunker = chunker
        self.selector = selector
        self.extractor = extractor
        self.context_builder = context_builder
        self.max_sources = max_sources

    def run(
        self,
        task: ResearchTask,
    ) -> ResearchWorkerResult:

        try:
            return self._run_task(task)
        except Exception as exc:
            error = str(exc) or type(exc).__name__

            return ResearchWorkerResult(
                task_id=task.id,
                query=task.query,
                evidences=[],
                searched_sources=0,
                processed_sources=0,
                failed_sources=[],
                error=error,
            )

    def _run_task(
        self,
        task: ResearchTask,
    ) -> ResearchWorkerResult:
        tools = self.tool_router.resolve(
            task.source_type
        )
        search_response = (
            tools.search_tool.search(
                query=task.query,
                max_results=self.max_sources,
            )
        )
        all_evidences = []
        failed_sources = []
        processed_sources = 0
        for search_result in (
            search_response.results
        ):
            try:
                evidences = (
                    self._process_source(
                        task=task,
                        search_result=search_result,
                        fetcher=tools.fetcher,
                    )
                )
                all_evidences.extend(
                    evidences
                )
                processed_sources += 1
            except Exception as e:
                failed_sources.append(
                    SourceFailure(
                        url=search_result.url,
                        error=str(e),
                    )
                )
        return ResearchWorkerResult(
            task_id=task.id,
            query=task.query,
            evidences=all_evidences,
            searched_sources=len(
                search_response.results
            ),
            processed_sources=(
                processed_sources
            ),
            failed_sources=failed_sources,
            error=None,
        )

    def _process_source(
        self,
        task: ResearchTask,
        search_result: SearchResult,
        fetcher: BaseSourceFetcher,
    ):

        document = fetcher.fetch(
            search_result.url
        )
        if not document.title:
            document = document.model_copy(
                update={
                    "title": search_result.title
                }
            )
        chunks = self.chunker.split(
            document
        )
        if not chunks:
            return []
        candidates = self.selector.select(
            task=task,
            chunks=chunks,
        )
        if not candidates:
            return []
        selected_chunks = (
            self.context_builder
            .select_chunks(
                candidates
            )
        )
        if not selected_chunks:
            return []
        extraction_result = (
            self.extractor
            .extract_from_chunks(
                task=task,
                document=document,
                chunks=selected_chunks,
            )
        )
        return extraction_result.evidences