from deepscholar.services.context_builder import ContextBuilder
from deepscholar.agents.critic import CriticAgent
from deepscholar.agents.planner import (
    PlannerAgent,
)

from deepscholar.agents.replanner import ReplannerAgent
from deepscholar.config import (
    settings,
)

from deepscholar.graph.research_graph import (
    ResearchGraph,
)

from deepscholar.llm.client import (
    LLMClient,
)

from deepscholar.services.chunker import (
    DocumentChunker,
)

from deepscholar.services.chunk_selector import (
    HybridChunkSelector,
)

from deepscholar.services.evidence_extractor import (
    EvidenceExtractor,
)

from deepscholar.services.research_worker import (
    ResearchWorker,
)

from deepscholar.tools.web_fetch import (
    TavilyWebFetcher,
)

from deepscholar.tools.web_search import (
    TavilyWebSearchTool,
)


def main():

    llm = LLMClient()
    planner = PlannerAgent(
        llm=llm
    )
    search_tool = TavilyWebSearchTool()
    fetcher = TavilyWebFetcher()
    chunker = DocumentChunker(
        chunk_size=3000,
        chunk_overlap=300,
    )
    selector = HybridChunkSelector(
        embedding_model_name=(
            settings.embedding_model_path
        ),
        bm25_top_k=5,
        dense_top_k=5,
        final_top_k=2,
    )
    context_builder = ContextBuilder(
        tokenizer_path=(
            settings.llm_tokenizer_path
        ),
        max_context_tokens=(
            settings.llm_max_context_tokens
        ),
        reserved_prompt_tokens=700,
        reserved_output_tokens=(
            settings.llm_max_output_tokens
        ),
    )
    extractor = EvidenceExtractor(
        llm=llm
    )
    replanner = ReplannerAgent(
        llm=llm
    )
    critic = CriticAgent(
        llm=llm
    )

    worker = ResearchWorker(
        search_tool=search_tool,
        fetcher=fetcher,
        chunker=chunker,
        selector=selector,
        extractor=extractor,
        context_builder=context_builder,
        max_sources=2,
    )

    research_graph = ResearchGraph(
        planner=planner,
        worker=worker,
        critic=critic,
        replanner=replanner,
    )

    initial_state = {
        "user_query": (
            "Research recent multimodal GUI "
            "agent developments, focusing on "
            "architecture, training methods, "
            "benchmarks and limitations."
        ),
        "plan": None,
        "current_task_index": 0,
        "worker_results": [],
        "evidences": [],
        "critique": None,
        "has_new_tasks": False,

        "replan_count": 0,
        "max_replans": 1,

        "previous_critique": None,
        "last_critic_task_index": 0,

    }

    for event in (
        research_graph
        .graph
        .stream(
            initial_state,
            stream_mode="updates",
        )
    ):
        print(
            "\n===== Graph Update ====="
        )

        print(event)


if __name__ == "__main__":
    main()