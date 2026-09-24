import httpx
from deepscholar.config import settings
from deepscholar.models.research import SourceType
from deepscholar.models.search import (
    SearchResponse,
    SearchResult,
)
from deepscholar.tools.base import BaseSearchTool


class GitHubSearchTool(BaseSearchTool):

    BASE_URL = "https://api.github.com/search/repositories"

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

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "DeepScholar-Agent/0.1",
        }

        token = settings.github_token.get_secret_value()
        headers["Authorization"] = (
            f"Bearer {token}"
        )

        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> SearchResponse:

        response = self._run_with_retry(
            lambda: self._request(
                query=query,
                max_results=max_results,
            )
        )

        data = response.json()

        results: list[SearchResult] = []

        for repo in data.get(
            "items",
            [],
        ):
            description = (
                repo.get("description")
                or ""
            )

            content = (
                f"Repository: "
                f"{repo['full_name']}\n"
                f"Description: "
                f"{description}\n"
                f"Stars: "
                f"{repo['stargazers_count']}\n"
                f"Language: "
                f"{repo.get('language')}\n"
                f"Updated: "
                f"{repo.get('updated_at')}"
            )

            results.append(
                SearchResult(
                    title=repo["full_name"],
                    url=repo["html_url"],
                    content=content,
                    score=None,
                    source_type=(
                        SourceType.GITHUB
                    ),
                    published_at=(
                        repo.get("created_at")
                    ),
                )
            )

        return SearchResponse(
            query=query,
            provider="github",
            results=results,
        )

    def _request(
        self,
        query: str,
        max_results: int,
    ) -> httpx.Response:
        response = self.client.get(
            self.BASE_URL,
            params={
                "q": query,
                "per_page": max_results,
                "sort": "stars",
                "order": "desc",
            },
        )
        response.raise_for_status()
        return response
