from abc import ABC, abstractmethod
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
from deepscholar.models.search import SearchResponse
from deepscholar.models.source import SourceDocument


T = TypeVar("T")


class BaseSearchTool(ABC):
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
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

    def _run_with_retry(
        self,
        operation: Callable[[], T],
    ) -> T:
        for attempt in range(
            self.max_retries + 1
        ):
            try:
                return operation()
            except Exception as exc:
                if (
                    attempt >= self.max_retries
                    or not self._is_retryable(exc)
                ):
                    raise

            delay = (
                self.backoff_factor
                * (2 ** attempt)
            )
            time.sleep(delay)

        raise RuntimeError(
            "Search retry loop exited unexpectedly."
        )

    def _is_retryable(
        self,
        exc: Exception,
    ) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return (
                exc.response.status_code
                in self.RETRYABLE_STATUS_CODES
            )

        return True

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> SearchResponse:
        """
        Execute a search query and return normalized results.
        """
        raise NotImplementedError


class BaseSourceFetcher(ABC):
    @abstractmethod
    def fetch(
        self,
        url: str,
    ) -> SourceDocument:
        raise NotImplementedError