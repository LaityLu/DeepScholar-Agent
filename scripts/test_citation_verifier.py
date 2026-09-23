from deepscholar.agents.citation_verifier import (
    CitationVerifierAgent,
)
from deepscholar.llm.client import LLMClient
from deepscholar.models.citation import Claim
from deepscholar.models.evidence import Evidence
from deepscholar.models.research import SourceType


def build_test_evidences() -> list[Evidence]:
    return [
        Evidence(
            id="ev_dgara",
            task_id="task_003",
            source_type=SourceType.PAPER,
            title=(
                "D-GARA: A Dynamic Benchmarking "
                "Framework for GUI Agents"
            ),
            url=(
                "https://ojs.aaai.org/"
                "index.php/AAAI/article/view/38795"
            ),
            content=(
                "D-GARA is a dynamic benchmarking "
                "framework designed to evaluate "
                "GUI agent robustness under "
                "real-world anomalies."
            ),
            quote=(
                "We propose a Dynamic benchmarking "
                "framework for GUI Agent Robustness "
                "in real-world Anomalies."
            ),
            published_at="2026",
            relevance_score=0.95,
        ),

        Evidence(
            id="ev_attack",
            task_id="task_004",
            source_type=SourceType.PAPER,
            title=(
                "Breaking GUI Agent Actions: "
                "Realistic Adversarial Attacks"
            ),
            url=(
                "https://example.org/"
                "gui-adversarial-attacks"
            ),
            content=(
                "The attack framework improves "
                "attack success rates by 30% under "
                "localized perturbations."
            ),
            quote=(
                "RAA improves attack success rates "
                "by an average of 30% under "
                "localized perturbations."
            ),
            published_at="2026",
            relevance_score=0.93,
        ),
    ]


def build_test_claims() -> list[Claim]:
    return [
        # Case 1:
        # Evidence 直接支持
        Claim(
            id="claim_001",
            text=(
                "D-GARA evaluates GUI agent "
                "robustness under dynamic "
                "real-world anomalies."
            ),
            section="Benchmarks",
            evidence_ids=[
                "ev_dgara",
            ],
        ),

        # Case 2:
        # 关键词相似，但语义不支持
        #
        # Evidence 说的是：
        # attack success rate +30%
        #
        # Claim 却说：
        # GUI agent performance +30%
        Claim(
            id="claim_002",
            text=(
                "RAA improves GUI agent "
                "task performance by 30%."
            ),
            section="Limitations",
            evidence_ids=[
                "ev_attack",
            ],
        ),

        # Case 3:
        # 没有 evidence
        Claim(
            id="claim_003",
            text=(
                "GUI agents achieve perfect "
                "generalization across unseen "
                "interfaces."
            ),
            section="Limitations",
            evidence_ids=[],
        ),

        # Case 4:
        # 引用了不存在的 evidence id
        Claim(
            id="claim_004",
            text=(
                "GUI agent benchmarks use "
                "dynamic anomaly injection."
            ),
            section="Benchmarks",
            evidence_ids=[
                "ev_missing",
            ],
        ),
    ]


def main():
    llm = LLMClient()

    verifier = CitationVerifierAgent(
        llm=llm,
        max_output_tokens=400,
    )

    evidences = build_test_evidences()
    claims = build_test_claims()

    print(
        "\n===== Citation Verifier Input ====="
    )

    print(
        f"Claims: {len(claims)}"
    )

    print(
        f"Evidences: {len(evidences)}"
    )

    result = verifier.verify(
        claims=claims,
        evidences=evidences,
    )

    print(
        "\n===== Citation Verification ====="
    )

    for item in result.verifications:
        print(
            f"\nClaim: {item.claim_id}"
        )

        print(
            f"Supported: {item.supported}"
        )

        print(
            "Supporting Evidence:",
            item.supporting_evidence_ids,
        )

        print(
            "Reason:",
            item.unsupported_reason,
        )

    print(
        "\nAll Supported:",
        result.all_supported,
    )

    print(
        "\n===== Assertions ====="
    )

    verification_map = {
        item.claim_id: item
        for item in result.verifications
    }

    # Case 1
    claim_001 = verification_map[
        "claim_001"
    ]

    assert claim_001.supported is True, (
        "claim_001 should be supported."
    )

    assert (
        "ev_dgara"
        in claim_001.supporting_evidence_ids
    ), (
        "claim_001 should cite ev_dgara."
    )

    # Case 2
    claim_002 = verification_map[
        "claim_002"
    ]

    assert claim_002.supported is False, (
        "claim_002 should be unsupported "
        "because attack success rate is not "
        "GUI agent task performance."
    )

    assert (
        len(
            claim_002.supporting_evidence_ids
        )
        == 0
    )

    # Case 3
    claim_003 = verification_map[
        "claim_003"
    ]

    assert claim_003.supported is False, (
        "claim_003 should be unsupported "
        "because it has no evidence."
    )

    assert (
        len(
            claim_003.supporting_evidence_ids
        )
        == 0
    )

    # Case 4
    claim_004 = verification_map[
        "claim_004"
    ]

    assert claim_004.supported is False, (
        "claim_004 should be unsupported "
        "because its evidence ID does not exist."
    )

    assert (
        len(
            claim_004.supporting_evidence_ids
        )
        == 0
    )

    assert result.all_supported is False

    print(
        "All CitationVerifier tests passed."
    )


if __name__ == "__main__":
    main()