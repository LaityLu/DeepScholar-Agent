from tavily import TavilyClient

from deepscholar.config import settings
from deepscholar.models.research import SourceType
from deepscholar.models.source import SourceDocument
from deepscholar.tools.base import BaseSourceFetcher


class TavilyWebFetcher(BaseSourceFetcher):
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

    def fetch(
        self,
        url: str,
    ) -> SourceDocument:
        url = url.strip()
        if not url:
            raise ValueError(
                "URL cannot be empty."
            )
        raw_response = self.client.extract(
            url,
            extract_depth="basic",
            include_images=False,
            format="markdown",
        )
        results = raw_response.get(
            "results",
            [],
        )
        if not results:
            failed_results = raw_response.get(
                "failed_results",
                [],
            )
            raise RuntimeError(
                f"Failed to fetch URL: {url}. "
                f"Details: {failed_results}"
            )
        item = results[0]
        content = item.get(
            "raw_content",
            "",
        )
        if not content.strip():
            raise RuntimeError(
                f"Fetched empty content from: {url}"
            )
        return SourceDocument(
            url=item.get(
                "url",
                url,
            ),
            content=content,
            source_type=SourceType.WEB,
        )