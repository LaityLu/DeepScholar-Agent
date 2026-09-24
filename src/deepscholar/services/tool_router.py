from dataclasses import dataclass
from deepscholar.models.research import SourceType
from deepscholar.tools.base import (
    BaseSearchTool,
    BaseSourceFetcher,
)


@dataclass
class ResearchToolPair:
    search_tool: BaseSearchTool
    fetcher: BaseSourceFetcher


class ResearchToolRouter:

    def __init__(
        self,
        tools: dict[
            SourceType,
            ResearchToolPair,
        ],
    ):
        self.tools = tools

    def resolve(
        self,
        source_type: SourceType,
    ) -> ResearchToolPair:

        tool_pair = self.tools.get(
            source_type
        )

        if tool_pair is None:
            raise ValueError(
                f"No research tools registered "
                f"for source type: {source_type}"
            )

        return tool_pair