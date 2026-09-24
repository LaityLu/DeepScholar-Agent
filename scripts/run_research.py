import argparse
from uuid import uuid4

from deepscholar.agents.citation_verifier import (
    CitationVerifierAgent,
)
from deepscholar.agents.claim_generator import (
    ClaimGeneratorAgent,
)
from deepscholar.agents.critic import CriticAgent
from deepscholar.agents.planner import PlannerAgent
from deepscholar.agents.replanner import ReplannerAgent
from deepscholar.agents.report_writer import (
    ReportWriterAgent,
)
from deepscholar.config import settings
from deepscholar.graph.research_graph import (
    ResearchGraph,
)
from deepscholar.graph.state import ResearchState
from deepscholar.llm.client import LLMClient
from deepscholar.models.research import SourceType
from deepscholar.services.checkpoint import (
    create_sqlite_checkpointer,
)
from deepscholar.services.chunk_selector import (
    HybridChunkSelector,
)
from deepscholar.services.chunker import (
    DocumentChunker,
)
from deepscholar.services.context_builder import (
    ContextBuilder,
)
from deepscholar.services.evidence_extractor import (
    EvidenceExtractor,
)
from deepscholar.services.evidence_processor import (
    EvidenceProcessor,
)
from deepscholar.services.research_worker import (
    ResearchWorker,
)
from deepscholar.services.tool_router import ResearchToolPair, ResearchToolRouter
from deepscholar.tools.web_fetch import (
    TavilyWebFetcher,
)
from deepscholar.tools.web_search import (
    TavilyWebSearchTool,
)
from deepscholar.tools.arxiv_search import (
    ArxivSearchTool,
)
from deepscholar.tools.arxiv_fetcher import (
    ArxivFetcher,
)


CHECKPOINT_DB = "runtime/checkpoints.sqlite"
MAX_CONCURRENCY = 3


def build_graph():
    # ==================================================
    # LLM
    # ==================================================

    llm = LLMClient()

    # ==================================================
    # Agents
    # ==================================================

    planner = PlannerAgent(
        llm=llm
    )

    critic = CriticAgent(
        llm=llm
    )

    replanner = ReplannerAgent(
        llm=llm
    )

    claim_generator = ClaimGeneratorAgent(
        llm=llm,
        max_output_tokens=1200,
        max_claims_per_task=5,
    )

    citation_verifier = (
        CitationVerifierAgent(
            llm=llm,
            max_output_tokens=400,
        )
    )

    report_writer = ReportWriterAgent(
        llm=llm,
        max_output_tokens=2000,
    )

    # ==================================================
    # Search / Fetch
    # ==================================================

    tavily_search = TavilyWebSearchTool()
    tavily_fetcher = TavilyWebFetcher()
    arxiv_search = ArxivSearchTool()
    arxiv_fetcher = ArxivFetcher()
    tool_router = ResearchToolRouter(
        tools={
            SourceType.WEB: (
                ResearchToolPair(
                    search_tool=tavily_search,
                    fetcher=tavily_fetcher,
                )
            ),
            SourceType.PAPER: (
                ResearchToolPair(
                    search_tool=arxiv_search,
                    fetcher=arxiv_fetcher,
                )
            )
        }
    )

    # ==================================================
    # Research services
    # ==================================================

    chunker = DocumentChunker(
        chunk_size=3000,
        chunk_overlap=300,
    )

    selector = HybridChunkSelector(
        embedding_model_name=(
            settings.embedding_model_path
        ),
        bm25_top_k=10,
        dense_top_k=10,
        final_top_k=5,
        rrf_k=60,
    )

    context_builder = ContextBuilder(
        tokenizer_path=(
            settings.llm_tokenizer_path
        ),
        max_context_tokens=(
            settings.llm_max_context_tokens
        ),
        reserved_prompt_tokens=700,
        reserved_output_tokens=800,
    )

    extractor = EvidenceExtractor(
        llm=llm
    )

    worker = ResearchWorker(
        tool_router=tool_router,
        chunker=chunker,
        selector=selector,
        context_builder=context_builder,
        extractor=extractor,
        max_sources=3,
    )

    evidence_processor = (
        EvidenceProcessor()
    )

    # ==================================================
    # Persistent checkpointer
    # ==================================================

    checkpointer = (
        create_sqlite_checkpointer(
            CHECKPOINT_DB
        )
    )

    # ==================================================
    # Graph
    # ==================================================

    research_graph = ResearchGraph(
        planner=planner,
        worker=worker,
        critic=critic,
        replanner=replanner,
        evidence_processor=(
            evidence_processor
        ),
        claim_generator=(
            claim_generator
        ),
        citation_verifier=(
            citation_verifier
        ),
        report_writer=(
            report_writer
        ),
    )

    graph = research_graph.build(
        checkpointer=checkpointer
    )

    return graph


def build_initial_state(
    query: str,
) -> ResearchState:
    return {
        "user_query": query,

        "plan": None,

        "worker_results": [],

        "evidences": [],

        "critique": None,

        "previous_critique": None,

        "last_critic_task_index": 0,

        "has_new_tasks": False,

        "replan_count": 0,

        "max_replans": 1,

        "processed_evidences": [],

        "claims": [],

        "citation_verification": None,

        "verified_claims": [],

        "report": None,
    }


def make_config(
    thread_id: str,
) -> dict:
    return {
        "max_concurrency": MAX_CONCURRENCY,
        "configurable": {
            "thread_id": thread_id
        }
    }


def run_new_research(
    graph,
    query: str,
):
    thread_id = (
        f"research_{uuid4().hex[:12]}"
    )

    config = make_config(
        thread_id
    )

    initial_state = (
        build_initial_state(
            query=query
        )
    )

    print(
        "\n===== New Research Session ====="
    )

    print(
        f"Thread ID: {thread_id}"
    )

    print(
        f"Query: {query}"
    )

    print(
        "\nSave this Thread ID if "
        "you need to resume later.\n"
    )

    final_state = graph.invoke(
        initial_state,
        config=config,
    )

    print_final_result(
        final_state=final_state,
        thread_id=thread_id,
    )


def resume_research(
    graph,
    thread_id: str,
):
    config = make_config(
        thread_id
    )

    print(
        "\n===== Resume Research ====="
    )

    print(
        f"Thread ID: {thread_id}"
    )

    # First inspect the current checkpoint.
    snapshot = graph.get_state(
        config
    )

    if not snapshot.values:
        raise RuntimeError(
            f"No checkpoint found for "
            f"thread '{thread_id}'."
        )

    print_state_summary(
        snapshot.values
    )

    # IMPORTANT:
    #
    # None means:
    # "continue from the latest checkpoint"
    #
    # Do NOT pass build_initial_state() again here.
    final_state = graph.invoke(
        None,
        config=config,
    )

    print_final_result(
        final_state=final_state,
        thread_id=thread_id,
    )


def print_state_summary(
    state: dict,
):
    print(
        "\n===== Saved State ====="
    )

    plan = state.get(
        "plan"
    )

    if plan is not None:
        print(
            f"Tasks: {len(plan.tasks)}"
        )

    worker_results = state.get(
        "worker_results",
        [],
    )

    completed_task_ids = {
        result.task_id
        for result in worker_results
    }

    print(
        "Completed Research Tasks:",
        len(completed_task_ids),
    )

    print(
        "Raw Evidences:",
        len(
            state.get(
                "evidences",
                [],
            )
        ),
    )

    print(
        "Replan Count:",
        state.get(
            "replan_count"
        ),
    )

    print(
        "Report Generated:",
        bool(
            state.get(
                "report"
            )
        ),
    )


def print_final_result(
    final_state: dict,
    thread_id: str,
):
    print(
        "\n===== Research Finished ====="
    )

    print(
        f"Thread ID: {thread_id}"
    )

    report = final_state.get(
        "report"
    )

    if report:
        print(
            "\n===== Final Report =====\n"
        )

        print(
            report
        )
    else:
        print(
            "\nNo final report was generated."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume a "
            "DeepScholar research session."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        help=(
            "Research query for a new session."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        help=(
            "Resume an existing research "
            "session using its thread ID."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume and args.query:
        raise SystemExit(
            "Use either a new query or --resume, not both."
        )

    if not args.resume and not args.query:
        raise SystemExit(
            "Provide a research query or --resume THREAD_ID."
        )

    graph = build_graph()

    if args.resume:
        resume_research(
            graph=graph,
            thread_id=args.resume,
        )

    else:
        run_new_research(
            graph=graph,
            query=args.query,
        )


if __name__ == "__main__":
    main()