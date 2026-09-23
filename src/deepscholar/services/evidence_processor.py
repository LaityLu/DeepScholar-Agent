from collections import defaultdict
from urllib.parse import (
    urlsplit,
    urlunsplit,
)

from deepscholar.models.evidence import (
    Evidence,
)


class EvidenceProcessor:

    def process(
        self,
        evidences: list[Evidence],
    ) -> list[Evidence]:
        evidences = self._deduplicate(
            evidences
        )
        evidences = [
            self._attach_source_quality(
                evidence
            )
            for evidence in evidences
        ]

        return evidences

    def _normalize_url(
        self,
        url: str,
    ) -> str:

        parts = urlsplit(
            url.strip()
        )
        path = parts.path.rstrip("/")
        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                path,
                "",
                "",
            )
    )

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        return " ".join(
            text.lower().split()
        )

    def _deduplicate(
        self,
        evidences: list[Evidence],
    ) -> list[Evidence]:

        seen = set()
        result = []
        for evidence in evidences:
            key = (
                self._normalize_url(
                    evidence.url
                ),
                self._normalize_text(
                    evidence.content
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(
                evidence
            )
        return result


    def _limit_per_source(
        self,
        evidences: list[Evidence],
        max_per_source: int = 5,
    ) -> list[Evidence]:

        groups = defaultdict(list)
        for evidence in evidences:
            key = (
                evidence.task_id,
                self._normalize_url(
                    evidence.url
                ),
            )

            groups[key].append(
                evidence
            )
        result = []
        for group in groups.values():
            ranked = sorted(
                group,
                key=lambda item: (
                    item.relevance_score
                    or 0.0
                ),
                reverse=True,
            )
            result.extend(
                ranked[:max_per_source]
            )
        return result

    def _score_source_quality(
        self,
        evidence: Evidence,
    ) -> tuple[float, str]:

        domain = urlsplit(
            evidence.url
        ).netloc.lower()

        if (
            "arxiv.org" in domain
            or "aclanthology.org" in domain
            or "neurips.cc" in domain
            or "aaai.org" in domain
        ):
            return 0.95, "primary_academic"

        if "github.com" in domain:
            return 0.85, "repository"

        if "medium.com" in domain:
            return 0.55, "secondary_blog"

        if (
            "youtube.com" in domain
            or "youtu.be" in domain
        ):
            return 0.50, "secondary_media"

        return 0.70, "web_source"

    def _attach_source_quality(
        self,
        evidence: Evidence,
    ) -> Evidence:

        score, label = (
            self._score_source_quality(
                evidence
            )
        )

        return evidence.model_copy(
            update={
                "source_quality_score": score,
                "quality_label": label,
            }
        )

    def _ranking_score(
        self,
        evidence: Evidence,
    ) -> float:

        relevance = (
            evidence.relevance_score
            or 0.0
        )

        quality = (
            evidence.source_quality_score
            or 0.0
        )

        return (
            0.6 * relevance
            + 0.4 * quality
        )