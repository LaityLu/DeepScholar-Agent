import feedparser
import httpx

from deepscholar.models.research import SourceType
from deepscholar.models.search import (
    SearchResponse,
    SearchResult,
)
from deepscholar.tools.base import BaseSearchTool


class ArxivSearchTool(BaseSearchTool):

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ):
        super().__init__(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "DeepScholar-Agent/0.1 "
                    "(research client)"
                ),
                "Accept": "application/atom+xml",
            },
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> SearchResponse:

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = self._get_with_retry(
            params=params
        )

        # print(
        #     "arXiv request:",
        #     response.request.url,
        # )

        # print(
        #     "arXiv status:",
        #     response.status_code,
        # )

        response.raise_for_status()

        feed = feedparser.parse(
            response.text
        )

        if feed.bozo:
            raise RuntimeError(
                "Failed to parse arXiv feed: "
                f"{feed.bozo_exception}"
            )

        results: list[SearchResult] = []

        for entry in feed.entries:

            title = getattr(
                entry,
                "title",
                "",
            ).strip()

            summary = getattr(
                entry,
                "summary",
                "",
            ).strip()

            url = getattr(
                entry,
                "id",
                "",
            ).strip()

            url = self._normalize_url(
                url
            )

            published_at = getattr(
                entry,
                "published",
                None,
            )

            if not title:
                continue

            if not url:
                continue

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    content=summary,
                    score=None,
                    source_type=(
                        SourceType.PAPER
                    ),
                    published_at=(
                        published_at
                    ),
                )
            )

        return SearchResponse(
            query=query,
            provider="arxiv",
            results=results,
        )

    def _get_with_retry(
        self,
        params: dict,
    ) -> httpx.Response:
        return self._run_with_retry(
            lambda: self._request(params)
        )

    def _request(
        self,
        params: dict,
    ) -> httpx.Response:
        response = self.client.get(
            self.BASE_URL,
            params=params,
        )
        response.raise_for_status()
        return response

    @staticmethod
    def _normalize_url(
        url: str,
    ) -> str:

        if url.startswith(
            "http://arxiv.org/"
        ):
            return url.replace(
                "http://",
                "https://",
                1,
            )

        return url