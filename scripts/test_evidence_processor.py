from deepscholar.models.evidence import Evidence
from deepscholar.models.research import SourceType
from deepscholar.services.evidence_processor import (
    EvidenceProcessor,
)


def build_test_evidences() -> list[Evidence]:
    return [
        Evidence(
            id="ev_001",
            task_id="task_001",
            source_type=SourceType.PAPER,
            title="GUI Agents: A Survey",
            url=(
                "https://aclanthology.org/"
                "2025.findings-acl.1158/"
            ),
            content=(
                "Task completion rate is one of the "
                "primary metrics used for evaluating "
                "GUI agents."
            ),
            quote=(
                "The majority of benchmarks use task "
                "completion rate as the primary metric."
            ),
            published_at="2025",
            relevance_score=0.88,
        ),

        # 与 ev_001 内容完全相同，
        # 但 URL 带 query 和末尾 /
        # 应被 exact dedup 去掉
        Evidence(
            id="ev_002",
            task_id="task_001",
            source_type=SourceType.PAPER,
            title="GUI Agents: A Survey",
            url=(
                "https://aclanthology.org/"
                "2025.findings-acl.1158/"
                "?utm_source=test"
            ),
            content=(
                "Task completion rate is one of the "
                "primary metrics used for evaluating "
                "GUI agents."
            ),
            quote=(
                "The majority of benchmarks use task "
                "completion rate as the primary metric."
            ),
            published_at="2025",
            relevance_score=0.90,
        ),

        # 同一 source，不同 evidence
        Evidence(
            id="ev_003",
            task_id="task_001",
            source_type=SourceType.PAPER,
            title="GUI Agents: A Survey",
            url=(
                "https://aclanthology.org/"
                "2025.findings-acl.1158/"
            ),
            content=(
                "GUI evaluation also considers "
                "generalization, robustness, "
                "and efficiency."
            ),
            quote=(
                "Existing benchmarks include metrics "
                "for efficiency, generalization, "
                "and robustness."
            ),
            published_at="2025",
            relevance_score=0.84,
        ),

        Evidence(
            id="ev_004",
            task_id="task_002",
            source_type=SourceType.WEB,
            title="Neuro-Symbolic AI Makes Agents Accurate",
            url=(
                "https://medium.com/"
                "some-neuro-symbolic-article"
            ),
            content=(
                "Neuro-symbolic methods are reported "
                "to improve multi-step reasoning "
                "accuracy."
            ),
            quote=(
                "Adding a neuro-symbolic layer "
                "can improve accuracy."
            ),
            published_at="2024",
            relevance_score=0.95,
        ),

        Evidence(
            id="ev_005",
            task_id="task_002",
            source_type=SourceType.GITHUB,
            title="Official GUI Agent Repository",
            url=(
                "https://github.com/"
                "example/gui-agent"
            ),
            content=(
                "The repository provides the official "
                "implementation and training scripts "
                "for the GUI agent."
            ),
            quote=None,
            published_at=None,
            relevance_score=0.90,
        ),

        Evidence(
            id="ev_006",
            task_id="task_003",
            source_type=SourceType.WEB,
            title="AgentTrek",
            url="https://agenttrek.github.io/",
            content=(
                "AgentTrek collects multimodal agent "
                "trajectories through guided replay "
                "in real digital environments."
            ),
            quote=(
                "A VLM agent interacts with the real "
                "digital environment guided by tutorials."
            ),
            published_at=None,
            relevance_score=0.93,
        ),
    ]


def main():
    processor = EvidenceProcessor()

    evidences = build_test_evidences()

    print("\n===== Raw Evidences =====")
    print(f"Count: {len(evidences)}")

    for evidence in evidences:
        print(
            f"- {evidence.id}: "
            f"{evidence.title} "
            f"(relevance={evidence.relevance_score})"
        )

    processed = processor.process(
        evidences
    )

    print("\n===== Processed Evidences =====")
    print(f"Count: {len(processed)}")

    for evidence in processed:
        print(
            f"- {evidence.id}\n"
            f"  title: {evidence.title}\n"
            f"  url: {evidence.url}\n"
            f"  relevance: "
            f"{evidence.relevance_score}\n"
            f"  source_quality: "
            f"{getattr(
                evidence,
                'source_quality_score',
                None,
            )}\n"
            f"  quality_label: "
            f"{getattr(
                evidence,
                'quality_label',
                None,
            )}"
        )

    print("\n===== Assertions =====")

    # ev_001 和 ev_002 应视为重复
    processed_ids = {
        evidence.id
        for evidence in processed
    }

    assert not (
        "ev_001" in processed_ids
        and "ev_002" in processed_ids
    ), (
        "Exact duplicate evidence was not removed."
    )

    # 原始 6 条，至少应该少 1 条
    assert len(processed) < len(evidences), (
        "EvidenceProcessor did not remove "
        "any duplicate evidence."
    )

    # 所有处理后的 Evidence 都应具有
    # source quality 信息
    for evidence in processed:
        assert (
            getattr(
                evidence,
                "source_quality_score",
                None,
            )
            is not None
        ), (
            f"{evidence.id} has no "
            "source_quality_score."
        )

        assert (
            getattr(
                evidence,
                "quality_label",
                None,
            )
            is not None
        ), (
            f"{evidence.id} has no "
            "quality_label."
        )

    # ACL 论文质量应该高于 Medium
    academic = next(
        (
            evidence
            for evidence in processed
            if "aclanthology.org"
            in evidence.url
        ),
        None,
    )

    medium = next(
        (
            evidence
            for evidence in processed
            if "medium.com"
            in evidence.url
        ),
        None,
    )

    assert academic is not None
    assert medium is not None

    assert (
        academic.source_quality_score
        > medium.source_quality_score
    ), (
        "Academic source should have "
        "higher quality than Medium."
    )

    print(
        "All EvidenceProcessor tests passed."
    )


if __name__ == "__main__":
    main()