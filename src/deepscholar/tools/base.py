from abc import ABC, abstractmethod
from deepscholar.models.search import SearchResponse
from deepscholar.models.source import SourceDocument

class BaseSearchTool(ABC):
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