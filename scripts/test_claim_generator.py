from deepscholar.agents.claim_generator import (
    ClaimGeneratorAgent,
)
from deepscholar.llm.client import LLMClient
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
                "D-GARA is a dynamic benchmark "
                "designed to evaluate GUI agent "
                "robustness under real-world anomalies."
            ),
            quote=(
                "We propose a dynamic benchmarking "
                "framework for GUI agent robustness "
                "under real-world anomalies."
            ),
            published_at="2026",
            relevance_score=0.96,
        ),
        Evidence(
            id="ev_002",
            task_id="task_002",
            source_type=SourceType.PAPER,
            title="AgentTrek",
            url="https://example.com/agenttrek",
            content=(
                "AgentTrek filtered 23,430 tutorials "
                "and collected 10,398 successful "
                "trajectories across 127 websites."
            ),
            quote=(
                "We obtained 10,398 successful "
                "trajectories across 127 websites."
            ),
            published_at="2025",
            relevance_score=0.95,
        ),
        Evidence(
            id="ev_003",
            task_id="task_002",
            source_type=SourceType.PAPER,
            title="AgentTrek",
            url="https://example.com/agenttrek",
            content=(
                "The AgentTrek data pipeline uses "
                "tutorial filtering, structured tutorial "
                "generation, and VLM-guided replay."
            ),
            quote=(
                "The pipeline consists of tutorial "
                "filtering, structured tutorials, "
                "and guided replay."
            ),
            published_at="2025",
            relevance_score=0.92,
        ),
    ]


def main():
    llm = LLMClient()

    generator = ClaimGeneratorAgent(
        llm=llm,
        max_output_tokens=1200,
        max_claims_per_task=5,
    )

    evidences = build_test_evidences()

    print(
        "\n===== Claim Generator Input ====="
    )

    print(
        f"Evidence Count: {len(evidences)}"
    )

    claims = generator.generate(
        goal=(
            "Research recent multimodal GUI agents, "
            "including training methods, benchmarks, "
            "and limitations."
        ),
        evidences=evidences,
    )

    print(
        "\n===== Generated Claims ====="
    )

    for claim in claims:
        print(
            f"\n{claim.id}"
        )
        print(
            f"Section: {claim.section}"
        )
        print(
            f"Text: {claim.text}"
        )
        print(
            f"Evidence IDs: "
            f"{claim.evidence_ids}"
        )

    print(
        "\n===== Assertions ====="
    )

    assert len(claims) > 0, (
        "ClaimGenerator produced no claims."
    )

    valid_evidence_ids = {
        evidence.id
        for evidence in evidences
    }

    valid_sections = {
        "Overview",
        "Architecture",
        "Training",
        "Benchmarks",
        "Limitations",
        "Open Problems",
    }

    claim_ids = set()

    for claim in claims:
        assert claim.id not in claim_ids, (
            f"Duplicate claim ID: {claim.id}"
        )

        claim_ids.add(
            claim.id
        )

        assert claim.text.strip(), (
            f"{claim.id} has empty text."
        )

        assert (
            claim.section
            in valid_sections
        ), (
            f"{claim.id} has invalid section: "
            f"{claim.section}"
        )

        assert claim.evidence_ids, (
            f"{claim.id} has no evidence IDs."
        )

        for evidence_id in (
            claim.evidence_ids
        ):
            assert (
                evidence_id
                in valid_evidence_ids
            ), (
                f"{claim.id} references "
                f"invalid evidence ID: "
                f"{evidence_id}"
            )

    print(
        "All ClaimGenerator tests passed."
    )


if __name__ == "__main__":
    main()