from deepscholar.agents.report_writer import (
    ReportWriterAgent,
)
from deepscholar.llm.client import LLMClient
from deepscholar.models.citation import (
    VerifiedClaim,
)
from deepscholar.models.critique import (
    CritiqueResult,
)
from deepscholar.models.evidence import Evidence
from deepscholar.models.research import SourceType


def build_test_evidences() -> list[Evidence]:
    return [
        Evidence(
            id="ev_001",
            task_id="task_001",
            source_type=SourceType.PAPER,
            title="D-GARA",
            url="https://example.com/dgara",
            content=(
                "D-GARA evaluates GUI agent "
                "robustness under dynamic "
                "real-world anomalies."
            ),
            quote=(
                "We propose a dynamic benchmark "
                "for GUI agent robustness."
            ),
            published_at="2026",
            relevance_score=0.95,
        ),
        Evidence(
            id="ev_002",
            task_id="task_002",
            source_type=SourceType.PAPER,
            title="AgentTrek",
            url="https://example.com/agenttrek",
            content=(
                "AgentTrek collected 10,398 "
                "successful trajectories across "
                "127 websites."
            ),
            quote=(
                "We obtained 10,398 successful "
                "trajectories across 127 websites."
            ),
            published_at="2025",
            relevance_score=0.94,
        ),
        Evidence(
            id="ev_003",
            task_id="task_002",
            source_type=SourceType.PAPER,
            title="AgentTrek",
            url="https://example.com/agenttrek",
            content=(
                "AgentTrek uses tutorial filtering "
                "and VLM-guided replay."
            ),
            quote=(
                "The pipeline uses tutorial filtering "
                "and guided replay."
            ),
            published_at="2025",
            relevance_score=0.90,
        ),
    ]


def build_verified_claims() -> list[
    VerifiedClaim
]:
    return [
        VerifiedClaim(
            id="claim_001",
            text=(
                "D-GARA evaluates GUI agent "
                "robustness under dynamic "
                "real-world anomalies."
            ),
            section="Benchmarks",
            supporting_evidence_ids=[
                "ev_001"
            ],
        ),
        VerifiedClaim(
            id="claim_002",
            text=(
                "AgentTrek collected 10,398 "
                "successful trajectories across "
                "127 websites."
            ),
            section="Training",
            supporting_evidence_ids=[
                "ev_002"
            ],
        ),
        VerifiedClaim(
            id="claim_003",
            text=(
                "AgentTrek uses tutorial filtering "
                "and VLM-guided replay in its "
                "data pipeline."
            ),
            section="Training",
            supporting_evidence_ids=[
                "ev_003"
            ],
        ),
    ]


def main():
    llm = LLMClient()

    writer = ReportWriterAgent(
        llm=llm,
        max_output_tokens=1800,
    )

    evidences = build_test_evidences()

    claims = build_verified_claims()

    critique = CritiqueResult(
        sufficient=False,
        coverage_score=0.8,
        covered_aspects=[
            "training data pipeline",
            "dynamic benchmark",
        ],
        knowledge_gaps=[
            (
                "Detailed quantitative comparison "
                "between different GUI agent "
                "architectures remains limited."
            )
        ],
        assessment=(
            "Most major aspects are covered, "
            "but architecture comparison remains "
            "incomplete."
        ),
    )

    report = writer.write(
        goal=(
            "Research recent multimodal GUI agents "
            "with emphasis on training methods "
            "and benchmarks."
        ),
        claims=claims,
        evidences=evidences,
        critique=critique,
    )

    print(
        "\n===== Final Report =====\n"
    )

    print(
        report
    )

    print(
        "\n===== Assertions ====="
    )

    assert report.strip(), (
        "ReportWriter returned empty output."
    )

    assert "# References" in report, (
        "Report does not contain References."
    )

    assert "[1]" in report, (
        "Reference [1] is missing."
    )

    assert (
        "https://example.com/dgara"
        in report
    )

    assert (
        "https://example.com/agenttrek"
        in report
    )

    # 当前只有三条 Evidence，
    # Writer 不应该凭空生成 [4]
    assert "[4]" not in report, (
        "Report contains an invalid "
        "citation number [4]."
    )

    print(
        "All ReportWriter tests passed."
    )


if __name__ == "__main__":
    main()