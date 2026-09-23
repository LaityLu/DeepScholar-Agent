import json
from collections import defaultdict

from deepscholar.llm.client import LLMClient
from deepscholar.models.citation import Claim
from deepscholar.models.evidence import Evidence
from deepscholar.utils.structured_output import clean_json_text


class ClaimGeneratorAgent:
    """
    Generate atomic factual claims from curated evidences.

    Responsibilities:
    - Convert curated evidences into report-ready claims.
    - Bind each claim to candidate evidence IDs.
    - Assign claims to predefined report sections.
    - Keep claims atomic and directly grounded in evidence.

    This agent does NOT:
    - Search for new information.
    - Verify whether citations are actually sufficient.
    - Generate the final report.
    """

    ALLOWED_SECTIONS = {
        "Overview",
        "Architecture",
        "Training",
        "Benchmarks",
        "Limitations",
        "Open Problems",
    }

    def __init__(
        self,
        llm: LLMClient,
        max_output_tokens: int = 1200,
        max_claims_per_task: int = 6,
    ):
        self.llm = llm
        self.max_output_tokens = max_output_tokens
        self.max_claims_per_task = max_claims_per_task

    def generate(
        self,
        goal: str,
        evidences: list[Evidence],
    ) -> list[Claim]:
        """
        Generate claims from curated evidences.

        Evidences are grouped by task_id so that each LLM
        call only processes a relatively small evidence
        subset.

        Args:
            goal:
                Overall research goal.

            evidences:
                Curated evidences produced by
                EvidenceProcessor.

        Returns:
            A list of Claim objects with program-generated IDs.
        """

        goal = goal.strip()

        if not goal:
            raise ValueError(
                "Research goal cannot be empty."
            )

        if not evidences:
            return []

        evidence_map = {
            evidence.id: evidence
            for evidence in evidences
        }

        grouped_evidences = self._group_by_task(
            evidences
        )

        generated_claims: list[dict] = []

        for task_id, task_evidences in (
            grouped_evidences.items()
        ):
            task_claims = self._generate_for_task(
                goal=goal,
                task_id=task_id,
                evidences=task_evidences,
            )

            generated_claims.extend(
                task_claims
            )

        return self._build_claim_models(
            raw_claims=generated_claims,
            evidence_map=evidence_map,
        )

    @staticmethod
    def _group_by_task(
        evidences: list[Evidence],
    ) -> dict[str, list[Evidence]]:
        """
        Group evidences by ResearchTask ID.

        Grouping keeps ClaimGenerator prompts smaller and
        naturally aligns claims with research dimensions.
        """

        groups: dict[
            str,
            list[Evidence],
        ] = defaultdict(list)

        for evidence in evidences:
            groups[
                evidence.task_id
            ].append(
                evidence
            )

        return dict(groups)

    def _generate_for_task(
        self,
        goal: str,
        task_id: str,
        evidences: list[Evidence],
    ) -> list[dict]:
        """
        Generate candidate claims for one research task.
        """

        if not evidences:
            return []

        prompt = self._build_prompt(
            goal=goal,
            task_id=task_id,
            evidences=evidences,
        )

        content = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence-grounded "
                        "research claim generation agent. "
                        "Generate only factual claims that "
                        "are directly grounded in the "
                        "provided evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=self.max_output_tokens,
        )

        cleaned_content = clean_json_text(
            content
        )

        data = json.loads(
            cleaned_content
        )

        raw_claims = data.get(
            "claims",
            []
        )

        if not isinstance(
            raw_claims,
            list,
        ):
            raise ValueError(
                "ClaimGenerator output field "
                "'claims' must be a list."
            )

        return raw_claims[
            :self.max_claims_per_task
        ]

    def _build_claim_models(
        self,
        raw_claims: list[dict],
        evidence_map: dict[str, Evidence],
    ) -> list[Claim]:
        """
        Convert LLM output into validated Claim models.

        Python owns:
        - claim IDs
        - valid section names
        - valid evidence IDs
        - duplicate suppression
        """

        claims: list[Claim] = []

        seen_claim_keys: set[
            tuple[str, tuple[str, ...]]
        ] = set()

        for raw_claim in raw_claims:
            if not isinstance(
                raw_claim,
                dict,
            ):
                continue

            text = str(
                raw_claim.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            section = str(
                raw_claim.get(
                    "section",
                    "Overview",
                )
            ).strip()

            if (
                section
                not in self.ALLOWED_SECTIONS
            ):
                section = "Overview"

            raw_evidence_ids = (
                raw_claim.get(
                    "evidence_ids",
                    [],
                )
            )

            if not isinstance(
                raw_evidence_ids,
                list,
            ):
                continue

            valid_evidence_ids: list[str] = []
            seen_evidence_ids: set[str] = set()

            for evidence_id in raw_evidence_ids:
                evidence_id = str(
                    evidence_id
                ).strip()

                if not evidence_id:
                    continue

                if (
                    evidence_id
                    not in evidence_map
                ):
                    continue

                if (
                    evidence_id
                    in seen_evidence_ids
                ):
                    continue

                seen_evidence_ids.add(
                    evidence_id
                )

                valid_evidence_ids.append(
                    evidence_id
                )

            # A claim without candidate evidence has no
            # value for the downstream CitationVerifier.
            if not valid_evidence_ids:
                continue

            normalized_text = self._normalize_text(
                text
            )

            dedup_key = (
                normalized_text,
                tuple(
                    sorted(
                        valid_evidence_ids
                    )
                ),
            )

            if dedup_key in seen_claim_keys:
                continue

            seen_claim_keys.add(
                dedup_key
            )

            claim_id = (
                f"claim_{len(claims) + 1:03d}"
            )

            claims.append(
                Claim(
                    id=claim_id,
                    text=text,
                    section=section,
                    evidence_ids=(
                        valid_evidence_ids
                    ),
                )
            )

        return claims

    def _build_prompt(
        self,
        goal: str,
        task_id: str,
        evidences: list[Evidence],
    ) -> str:
        """
        Build the prompt for one task-level evidence group.
        """

        evidence_blocks = []

        for index, evidence in enumerate(
            evidences,
            start=1,
        ):
            evidence_blocks.append(
                self._format_evidence(
                    index=index,
                    evidence=evidence,
                )
            )

        evidence_text = "\n\n".join(
            evidence_blocks
        )

        allowed_sections = "\n".join(
            f"- {section}"
            for section in sorted(
                self.ALLOWED_SECTIONS
            )
        )

        return f"""
Generate atomic factual research claims from the supplied
evidence.

Overall Research Goal:

{goal}


Current Research Task:

{task_id}


Available Evidence:

{evidence_text}


Claim generation rules:

1. Every claim must be directly grounded in one or more
   supplied evidences.

2. Do not use outside knowledge.

3. Do not infer a stronger conclusion than the evidence
   explicitly supports.

4. Each claim must express ONE primary factual proposition.

5. Do not combine multiple independently verifiable facts
   into a single claim.

6. Numerical claims must preserve the exact:
   - metric
   - value
   - comparison target
   - direction
   - scope
   - conditions
   supported by the evidence.

7. evidence_ids may contain only Evidence IDs appearing
   in the supplied evidence.

8. Prefer concise claims that can later be independently
   checked by a citation verifier.

9. Do not generate claims that are merely opinions,
   speculation, or generic background statements.

10. Avoid generating multiple claims that state essentially
    the same fact.

11. Generate at most {self.max_claims_per_task} claims.

12. section must be exactly one of:

{allowed_sections}


Section guidance:

- Overview:
  broad factual developments or representative trends

- Architecture:
  model architecture, system design, perception,
  reasoning, action, or agent structure

- Training:
  datasets, data pipelines, supervised learning,
  reinforcement learning, alignment, or training methods

- Benchmarks:
  benchmark design, evaluation settings, metrics,
  datasets, or quantitative evaluation findings

- Limitations:
  experimentally observed weaknesses, robustness issues,
  failure modes, or deployment limitations

- Open Problems:
  evidence-backed unresolved challenges or clearly stated
  future research directions from the supplied sources


Return exactly one JSON object:

{{
  "claims": [
    {{
      "text": "A single atomic factual claim.",
      "section": "Architecture",
      "evidence_ids": [
        "evidence_id"
      ]
    }}
  ]
}}

If the supplied evidence does not support any useful
factual claim, return:

{{
  "claims": []
}}

Return JSON only.
""".strip()

    @staticmethod
    def _format_evidence(
        index: int,
        evidence: Evidence,
    ) -> str:
        """
        Format one Evidence for claim generation.

        ClaimGenerator mainly relies on the extracted
        evidence summary, but the quote is also retained
        to reduce semantic drift.
        """

        parts = [
            f"[Evidence {index}]",
            f"Evidence ID: {evidence.id}",
            f"Task ID: {evidence.task_id}",
            f"Source: {evidence.title}",
            f"URL: {evidence.url}",
            f"Content: {evidence.content}",
        ]

        if evidence.quote:
            parts.append(
                f"Quote: {evidence.quote}"
            )

        if evidence.published_at:
            parts.append(
                f"Published At: "
                f"{evidence.published_at}"
            )

        if (
            evidence.relevance_score
            is not None
        ):
            parts.append(
                f"Relevance Score: "
                f"{evidence.relevance_score:.2f}"
            )

        if (
            getattr(
                evidence,
                "source_quality_score",
                None,
            )
            is not None
        ):
            parts.append(
                f"Source Quality Score: "
                f"{evidence.source_quality_score:.2f}"
            )

        return "\n".join(
            parts
        )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Lightweight normalization for exact-ish claim
        deduplication.
        """

        return " ".join(
            text.lower().split()
        )