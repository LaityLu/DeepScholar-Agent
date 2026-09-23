from deepscholar.agents.citation_verifier import (
    CitationVerifierAgent,
)
from deepscholar.agents.claim_generator import (
    ClaimGeneratorAgent,
)
from deepscholar.agents.critic import CriticAgent
from deepscholar.agents.planner import PlannerAgent
from deepscholar.agents.replanner import (
    ReplannerAgent,
)
from deepscholar.agents.report_writer import (
    ReportWriterAgent,
)
from deepscholar.config import settings
from deepscholar.graph.research_graph import (
    ResearchGraph,
)
from deepscholar.graph.state import ResearchState
from deepscholar.llm.client import LLMClient
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
from deepscholar.tools.web_fetch import TavilyWebFetcher
from deepscholar.tools.web_search import TavilyWebSearchTool


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

    claim_generator = (
        ClaimGeneratorAgent(
            llm=llm,
            max_output_tokens=1200,
            max_claims_per_task=5,
        )
    )

    citation_verifier = (
        CitationVerifierAgent(
            llm=llm,
            max_output_tokens=400,
        )
    )

    report_writer = (
        ReportWriterAgent(
            llm=llm,
            max_output_tokens=2000,
        )
    )

    # ==================================================
    # Tools
    # ==================================================

    search_tool = TavilyWebSearchTool()

    fetcher = TavilyWebFetcher()

    # ==================================================
    # Research Services
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
            settings
            .llm_max_context_tokens
        ),
        reserved_prompt_tokens=700,
        reserved_output_tokens=800,
    )

    extractor = EvidenceExtractor(
        llm=llm
    )

    worker = ResearchWorker(
        search_tool=search_tool,
        fetcher=fetcher,
        chunker=chunker,
        selector=selector,
        context_builder=(
            context_builder
        ),
        extractor=extractor,

        # 第一次端到端测试建议少一点，
        # 减少时间与 API 调用。
        max_sources=3,
    )

    evidence_processor = (
        EvidenceProcessor()
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

    return research_graph.build()


def main():
    graph = build_graph()

    query = (
        "Research the latest progress in "
        "multimodal GUI agents during 2025-2026, "
        "focusing on model architectures, "
        "training methods, benchmarks, "
        "and current limitations."
    )

    initial_state: ResearchState = {
        "user_query": query,

        "plan": None,

        "current_task_index": 0,

        "worker_results": [],

        "evidences": [],

        "critique": None,

        "previous_critique": None,

        "last_critic_task_index": 0,

        "has_new_tasks": False,

        "replan_count": 0,

        # 第一次全链路测试只允许
        # 一轮 Replanner。
        "max_replans": 1,

        "processed_evidences": [],

        "claims": [],

        "citation_verification": None,

        "verified_claims": [],

        "report": None,
    }

    print(
        "\n===== DeepScholar Research ====="
    )

    print(
        f"Query:\n{query}\n"
    )

    final_state = graph.invoke(
        initial_state
    )

    # ==================================================
    # Plan
    # ==================================================

    print(
        "\n===== Final Research Plan ====="
    )

    plan = final_state[
        "plan"
    ]

    if plan is not None:
        print(
            f"Goal: {plan.goal}"
        )

        for task in plan.tasks:
            print(
                f"\n{task.id}"
            )
            print(
                f"Title: {task.title}"
            )
            print(
                f"Intent: {task.intent}"
            )
            print(
                f"Query: {task.query}"
            )

    # ==================================================
    # Evidence
    # ==================================================

    print(
        "\n===== Evidence Statistics ====="
    )

    print(
        "Raw Evidence:",
        len(
            final_state[
                "evidences"
            ]
        ),
    )

    print(
        "Processed Evidence:",
        len(
            final_state[
                "processed_evidences"
            ]
        ),
    )

    # ==================================================
    # Critique
    # ==================================================

    print(
        "\n===== Final Critique ====="
    )

    critique = final_state[
        "critique"
    ]

    if critique is not None:
        print(
            f"Sufficient: "
            f"{critique.sufficient}"
        )

        print(
            f"Coverage: "
            f"{critique.coverage_score}"
        )

        print(
            "\nKnowledge Gaps:"
        )

        for gap in (
            critique.knowledge_gaps
        ):
            print(
                f"- {gap}"
            )

        print(
            "\nAssessment:"
        )

        print(
            critique.assessment
        )

    # ==================================================
    # Claims
    # ==================================================

    print(
        "\n===== Claims ====="
    )

    claims = final_state[
        "claims"
    ]

    print(
        f"Generated Claims: "
        f"{len(claims)}"
    )

    for claim in claims:
        print(
            f"\n{claim.id}"
        )
        print(
            f"[{claim.section}] "
            f"{claim.text}"
        )
        print(
            "Candidate Evidence:",
            claim.evidence_ids,
        )

    # ==================================================
    # Citation Verification
    # ==================================================

    print(
        "\n===== Citation Verification ====="
    )

    verification_result = (
        final_state[
            "citation_verification"
        ]
    )

    if verification_result:
        supported_count = sum(
            1
            for item
            in verification_result.verifications
            if item.supported
        )

        print(
            f"Supported: "
            f"{supported_count} / "
            f"{len(
                verification_result
                .verifications
            )}"
        )

        for item in (
            verification_result
            .verifications
        ):
            print(
                f"\n{item.claim_id}: "
                f"{item.supported}"
            )

            if (
                item
                .supporting_evidence_ids
            ):
                print(
                    "Evidence:",
                    item
                    .supporting_evidence_ids,
                )

            if (
                item
                .unsupported_reason
            ):
                print(
                    "Reason:",
                    item
                    .unsupported_reason,
                )

    # ==================================================
    # Verified Claims
    # ==================================================

    print(
        "\n===== Verified Claims ====="
    )

    verified_claims = (
        final_state[
            "verified_claims"
        ]
    )

    print(
        f"Verified Claims: "
        f"{len(verified_claims)}"
    )

    for claim in verified_claims:
        print(
            f"\n{claim.id}"
        )
        print(
            f"[{claim.section}] "
            f"{claim.text}"
        )
        print(
            "Evidence:",
            claim
            .supporting_evidence_ids,
        )

    # ==================================================
    # Final Report
    # ==================================================

    print(
        "\n===== Final Report =====\n"
    )

    report = final_state[
        "report"
    ]

    print(
        report
    )

    # ==================================================
    # Assertions
    # ==================================================

    print(
        "\n===== Assertions ====="
    )

    assert plan is not None

    assert len(
        final_state[
            "evidences"
        ]
    ) > 0

    assert len(
        final_state[
            "processed_evidences"
        ]
    ) > 0

    assert len(
        claims
    ) > 0

    assert (
        verification_result
        is not None
    )

    assert len(
        verified_claims
    ) > 0

    assert (
        report is not None
        and report.strip()
    )

    assert (
        "# References"
        in report
    )

    print(
        "Full DeepScholar graph "
        "test passed."
    )


if __name__ == "__main__":
    main()