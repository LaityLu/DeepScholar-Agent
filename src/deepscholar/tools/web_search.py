from tavily import TavilyClient

from deepscholar.config import settings
from deepscholar.models.research import (
    SourceType,
)
from deepscholar.models.search import (
    SearchResponse,
    SearchResult,
)
from deepscholar.tools.base import (
    BaseSearchTool,
)


class TavilyWebSearchTool(BaseSearchTool):
    def __init__(self):
        if settings.tavily_api_key is None:
            raise ValueError(
                "TAVILY_API_KEY is not configured."
            )
        self.client = TavilyClient(
            api_key=(
                settings
                .tavily_api_key
                .get_secret_value()
            )
        )

    def search(
        self,
        query: str,
        max_results: int = 3,
    ) -> SearchResponse:
        query = query.strip()
        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )
        if not 1 <= max_results <= 20:
            raise ValueError(
                "max_results must be "
                "between 1 and 20."
            )
        raw_response = self.client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        results = [
            self._normalize_result(item)
            for item in raw_response.get(
                "results",
                [],
            )
        ]
        response_time = (
            raw_response.get(
                "response_time"
            )
        )
        if response_time is not None:
            try:
                response_time = float(
                    response_time
                )
            except (
                TypeError,
                ValueError,
            ):
                response_time = None
        return SearchResponse(
            query=query,
            provider="tavily",
            results=results,
            response_time=response_time,
        )

    def _normalize_result(
        self,
        item: dict,
    ) -> SearchResult:

        return SearchResult(
            title=item.get(
                "title",
                "",
            ),
            url=item.get(
                "url",
                "",
            ),
            content=item.get(
                "content",
                "",
            ),
            score=item.get(
                "score",
            ),
            source_type=SourceType.WEB,
        )