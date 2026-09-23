from deepscholar.agents.citation_verifier import (
    CitationVerifierAgent,
)
from deepscholar.llm.client import LLMClient
from deepscholar.models.citation import Claim
from deepscholar.models.evidence import Evidence
from deepscholar.models.research import SourceType


def main():
    llm = LLMClient()

    verifier = CitationVerifierAgent(
        llm=llm,
        max_output_tokens=400,
    )

    evidences = [
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
                "for GUI agent robustness under "
                "real-world anomalies."
            ),
            published_at="2026",
            relevance_score=0.95,
        ),
        Evidence(
            id="ev_002",
            task_id="task_002",
            source_type=SourceType.PAPER,
            title="RAA",
            url="https://example.com/raa",
            content=(
                "RAA improves attack success rate "
                "by 30% under localized "
                "perturbations."
            ),
            quote=(
                "Attack success rate improves "
                "by approximately 30%."
            ),
            published_at="2026",
            relevance_score=0.92,
        ),
    ]

    claims = [
        Claim(
            id="claim_001",
            text=(
                "D-GARA evaluates GUI agent "
                "robustness under dynamic "
                "real-world anomalies."
            ),
            section="Benchmarks",
            evidence_ids=[
                "ev_001"
            ],
        ),

        # 应该被拒绝：
        # attack success rate
        # != GUI task performance
        Claim(
            id="claim_002",
            text=(
                "RAA improves GUI agent task "
                "performance by 30%."
            ),
            section="Limitations",
            evidence_ids=[
                "ev_002"
            ],
        ),

        Claim(
            id="claim_003",
            text=(
                "GUI agents achieve perfect "
                "generalization."
            ),
            section="Limitations",
            evidence_ids=[],
        ),
    ]

    result = verifier.verify(
        claims=claims,
        evidences=evidences,
    )

    print(
        "\n===== Citation Verification ====="
    )

    for verification in (
        result.verifications
    ):
        print(
            f"\n{verification.claim_id}"
        )
        print(
            f"supported = "
            f"{verification.supported}"
        )
        print(
            "supporting evidence =",
            verification
            .supporting_evidence_ids,
        )
        print(
            "reason =",
            verification
            .unsupported_reason,
        )

    verification_map = {
        item.claim_id: item
        for item in result.verifications
    }

    assert (
        verification_map[
            "claim_001"
        ].supported
        is True
    )

    assert (
        verification_map[
            "claim_002"
        ].supported
        is False
    )

    assert (
        verification_map[
            "claim_003"
        ].supported
        is False
    )

    assert (
        result.all_supported
        is False
    )

    print(
        "\nAll CitationVerifier "
        "tests passed."
    )


if __name__ == "__main__":
    main()