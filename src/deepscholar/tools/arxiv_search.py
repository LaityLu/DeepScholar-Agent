import time

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
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ):
        if max_retries < 0:
            raise ValueError(
                "max_retries must be non-negative."
            )

        if backoff_factor < 0:
            raise ValueError(
                "backoff_factor must be non-negative."
            )

        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
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
        for attempt in range(
            self.max_retries + 1
        ):
            try:
                response = self.client.get(
                    self.BASE_URL,
                    params=params,
                )

                if (
                    response.status_code
                    not in self.RETRYABLE_STATUS_CODES
                ):
                    response.raise_for_status()
                    return response

                response.raise_for_status()
            except httpx.RequestError:
                if attempt >= self.max_retries:
                    raise
            except httpx.HTTPStatusError as exc:
                if (
                    exc.response.status_code
                    not in self.RETRYABLE_STATUS_CODES
                    or attempt >= self.max_retries
                ):
                    raise

            delay = (
                self.backoff_factor
                * (2 ** attempt)
            )
            time.sleep(delay)

        raise RuntimeError(
            "arXiv request retry loop exited "
            "unexpectedly."
        )

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