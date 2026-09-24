import re

import httpx
from deepscholar.config import settings
from deepscholar.models.research import SourceType
from deepscholar.models.source import (
    SourceDocument,
)
from deepscholar.tools.base import (
    BaseSourceFetcher,
)


class GitHubFetcher(
    BaseSourceFetcher
):

    def __init__(
        self,
        timeout: float = 30.0,
    ):
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

    def fetch(
        self,
        url: str,
    ) -> SourceDocument:

        owner, repo = (
            self._parse_repo_url(
                url
            )
        )

        repo_api = (
            f"https://api.github.com/"
            f"repos/{owner}/{repo}"
        )

        repo_response = (
            self.client.get(
                repo_api
            )
        )

        repo_response.raise_for_status()

        repo_data = (
            repo_response.json()
        )

        readme = self._fetch_readme(
            owner=owner,
            repo=repo,
        )

        metadata = (
            f"# Repository Metadata\n\n"
            f"Name: {repo_data['full_name']}\n"
            f"Description: "
            f"{repo_data.get('description')}\n"
            f"Stars: "
            f"{repo_data.get('stargazers_count')}\n"
            f"Forks: "
            f"{repo_data.get('forks_count')}\n"
            f"Language: "
            f"{repo_data.get('language')}\n"
            f"Created: "
            f"{repo_data.get('created_at')}\n"
            f"Updated: "
            f"{repo_data.get('updated_at')}\n"
            f"Homepage: "
            f"{repo_data.get('homepage')}\n"
        )

        content = (
            metadata
            + "\n\n# README\n\n"
            + readme
        )

        return SourceDocument(
            url=url,
            title=repo_data[
                "full_name"
            ],
            content=content,
            source_type=(
                SourceType.GITHUB
            ),
            published_at=(
                repo_data.get(
                    "created_at"
                )
            ),
        )

    def _fetch_readme(
        self,
        owner: str,
        repo: str,
    ) -> str:

        url = (
            f"https://api.github.com/"
            f"repos/{owner}/{repo}/readme"
        )

        response = self.client.get(
            url,
            headers={
                "Accept": (
                    "application/vnd.github.raw+json"
                )
            },
        )

        if response.status_code == 404:
            return ""

        response.raise_for_status()

        return response.text

    @staticmethod
    def _parse_repo_url(
        url: str,
    ) -> tuple[str, str]:

        match = re.match(
            r"https?://github\.com/"
            r"([^/]+)/([^/?#]+)",
            url,
        )

        if not match:
            raise ValueError(
                f"Invalid GitHub repo URL: "
                f"{url}"
            )

        owner = match.group(1)

        repo = (
            match.group(2)
            .removesuffix(".git")
        )

        return owner, repo