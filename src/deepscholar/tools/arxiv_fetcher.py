import re

import httpx
from bs4 import BeautifulSoup

from deepscholar.models.research import (
    SourceType,
)
from deepscholar.models.source import (
    SourceDocument,
)
from deepscholar.tools.base import (
    BaseSourceFetcher,
)


class ArxivFetcher(
    BaseSourceFetcher
):

    def __init__(
        self,
        timeout: float = 30.0,
    ):
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "DeepScholar-Agent/0.1"
                )
            },
        )

    def fetch(
        self,
        url: str,
    ) -> SourceDocument:

        arxiv_id = (
            self._extract_arxiv_id(
                url
            )
        )

        html_url = (
            f"https://arxiv.org/html/"
            f"{arxiv_id}"
        )

        response = self.client.get(
            html_url
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        title = self._extract_title(
            soup
        )

        content = self._extract_content(
            soup
        )

        if not content.strip():
            raise ValueError(
                f"No content extracted "
                f"from arXiv paper: {url}"
            )

        return SourceDocument(
            url=url,
            title=title,
            content=content,
            source_type=(
                SourceType.PAPER
            ),
            published_at=None,
        )

    @staticmethod
    def _extract_arxiv_id(
        url: str,
    ) -> str:

        match = re.search(
            r"arxiv\.org/"
            r"(?:abs|html|pdf)/"
            r"([^?#]+)",
            url,
        )

        if not match:
            raise ValueError(
                f"Invalid arXiv URL: "
                f"{url}"
            )

        arxiv_id = match.group(1)

        arxiv_id = (
            arxiv_id
            .removesuffix(".pdf")
        )

        return arxiv_id

    @staticmethod
    def _extract_title(
        soup: BeautifulSoup,
    ) -> str | None:

        title = soup.find(
            "h1",
            class_="ltx_title"
        )

        if title is None:
            return None

        return title.get_text(
            " ",
            strip=True,
        )

    @staticmethod
    def _extract_content(
        soup: BeautifulSoup,
    ) -> str:

        article = soup.find(
            "article"
        )

        if article is None:
            article = soup.body

        if article is None:
            return ""

        # Remove elements that contribute little
        # useful research evidence.
        for tag in article.find_all(
            [
                "script",
                "style",
                "nav",
            ]
        ):
            tag.decompose()

        text = article.get_text(
            "\n",
            strip=True,
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)