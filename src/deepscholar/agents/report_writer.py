from deepscholar.llm.client import LLMClient
from deepscholar.models.citation import VerifiedClaim
from deepscholar.models.critique import CritiqueResult
from deepscholar.models.evidence import Evidence


class ReportWriterAgent:
    """
    Generate the final research report from verified claims.

    Responsibilities:
    - Organize verified claims into a coherent report.
    - Preserve program-generated citation markers.
    - Mention unresolved research gaps when present.

    This agent does NOT:
    - Search for information.
    - Generate new evidence.
    - Verify citations.
    - Introduce unsupported factual claims.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_output_tokens: int = 2000,
    ):
        self.llm = llm
        self.max_output_tokens = max_output_tokens

    def write(
        self,
        goal: str,
        claims: list[VerifiedClaim],
        evidences: list[Evidence],
        critique: CritiqueResult | None = None,
    ) -> str:
        """
        Generate the final Markdown research report.
        """

        if not claims:
            raise ValueError(
                "No verified claims are available "
                "for report generation."
            )

        evidence_map = {
            evidence.id: evidence
            for evidence in evidences
        }

        used_evidence_ids = (
            self._collect_used_evidence_ids(
                claims=claims,
                evidence_map=evidence_map,
            )
        )

        reference_numbers = {
            evidence_id: index
            for index, evidence_id in enumerate(
                used_evidence_ids,
                start=1,
            )
        }

        used_evidences = [
            evidence_map[evidence_id]
            for evidence_id in used_evidence_ids
        ]

        claims_text = self._format_claims(
            claims=claims,
            reference_numbers=reference_numbers,
        )

        references_text = self._format_references(
            evidences=used_evidences,
            reference_numbers=reference_numbers,
        )

        unresolved_gaps = (
            critique.knowledge_gaps
            if critique is not None
            else []
        )

        prompt = self._build_prompt(
            goal=goal,
            claims_text=claims_text,
            unresolved_gaps=unresolved_gaps,
        )

        report_body = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence-grounded "
                        "research report writer. "
                        "Use only the verified claims "
                        "provided by the user."
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

        report_body = report_body.strip()

        return (
            f"{report_body}\n\n"
            f"# References\n\n"
            f"{references_text}"
        )

    @staticmethod
    def _collect_used_evidence_ids(
        claims: list[VerifiedClaim],
        evidence_map: dict[str, Evidence],
    ) -> list[str]:
        """
        Collect unique evidence IDs in deterministic order.
        """

        used_ids: list[str] = []
        seen_ids: set[str] = set()

        for claim in claims:
            for evidence_id in (
                claim.supporting_evidence_ids
            ):
                if evidence_id not in evidence_map:
                    continue

                if evidence_id in seen_ids:
                    continue

                seen_ids.add(evidence_id)
                used_ids.append(evidence_id)

        if not used_ids:
            raise ValueError(
                "Verified claims contain no valid "
                "supporting evidence IDs."
            )

        return used_ids

    @staticmethod
    def _format_claims(
        claims: list[VerifiedClaim],
        reference_numbers: dict[str, int],
    ) -> str:
        """
        Format verified claims together with citation markers.
        """

        blocks: list[str] = []

        for claim in claims:
            citation_numbers = []

            for evidence_id in (
                claim.supporting_evidence_ids
            ):
                number = reference_numbers.get(
                    evidence_id
                )

                if number is not None:
                    citation_numbers.append(
                        number
                    )

            citation_numbers = sorted(
                set(citation_numbers)
            )

            citations = "".join(
                f"[{number}]"
                for number in citation_numbers
            )

            block = (
                f"Claim ID: {claim.id}\n"
                f"Section: {claim.section}\n"
                f"Claim: {claim.text}"
            )

            if citations:
                block += f" {citations}"

            blocks.append(block)

        return "\n\n".join(blocks)

    @staticmethod
    def _format_references(
        evidences: list[Evidence],
        reference_numbers: dict[str, int],
    ) -> str:
        """
        Build the final reference list programmatically.
        """

        references: list[str] = []

        for evidence in evidences:
            number = reference_numbers[
                evidence.id
            ]

            reference = (
                f"[{number}] "
                f"{evidence.title}\n"
                f"{evidence.url}"
            )

            if evidence.published_at:
                reference += (
                    f"\nPublished: "
                    f"{evidence.published_at}"
                )

            references.append(
                reference
            )

        return "\n\n".join(references)

    @staticmethod
    def _build_prompt(
        goal: str,
        claims_text: str,
        unresolved_gaps: list[str],
    ) -> str:
        """
        Build the final report writing prompt.
        """

        if unresolved_gaps:
            gaps_text = "\n".join(
                f"- {gap}"
                for gap in unresolved_gaps
            )
        else:
            gaps_text = "None"

        return f"""
Write a structured research report based ONLY on the
verified claims supplied below.

Research Goal:

{goal}


Verified Claims:

{claims_text}


Remaining Unresolved Research Gaps:

{gaps_text}


Writing requirements:

1. Use only the verified claims supplied above.

2. Do not introduce factual claims that are not contained
   in the verified claim set.

3. You may summarize, connect, and reorganize claims, but
   do not strengthen, generalize, or extrapolate beyond
   their supported scope.

4. Preserve citation markers such as [1], [2], and [3]
   exactly where they support the corresponding factual
   statements.

5. Do not invent citation numbers.

6. Do not create a References section. The reference list
   will be appended programmatically.

7. Avoid repeating the same claim in multiple sections.

8. If unresolved research gaps remain, explicitly mention
   them in the limitations or open-problems discussion.

9. Distinguish established evidence from limitations and
   unresolved issues.

10. Produce a professional research report in Markdown.

Use an appropriate structure such as:

# Executive Summary

# Architecture

# Training Methodologies

# Evaluation Benchmarks

# Current Limitations

# Open Problems

Sections without meaningful supporting claims may be
omitted.

Return Markdown only.
""".strip()