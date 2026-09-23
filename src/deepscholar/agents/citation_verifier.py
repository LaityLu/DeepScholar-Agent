import json

from deepscholar.llm.client import LLMClient
from deepscholar.models.citation import (
    Claim,
    CitationVerification,
    CitationVerificationResult,
)
from deepscholar.models.evidence import Evidence
from deepscholar.utils.structured_output import clean_json_text


class CitationVerifierAgent:
    """
    Verify whether generated research claims are
    directly supported by their cited evidences.

    Responsibilities:
    - Validate Claim -> Evidence support.
    - Reject topic-related but unsupported citations.
    - Validate supporting evidence IDs.
    - Return structured verification results.

    This agent does NOT:
    - Search for new evidence.
    - Modify claims.
    - Generate reports.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_output_tokens: int = 400,
    ):
        self.llm = llm
        self.max_output_tokens = max_output_tokens

    def verify(
        self,
        claims: list[Claim],
        evidences: list[Evidence],
    ) -> CitationVerificationResult:
        """
        Verify all claims against their candidate evidences.

        Args:
            claims:
                Claims produced by ClaimGeneratorAgent.

            evidences:
                Curated evidence pool produced by
                EvidenceProcessor.

        Returns:
            CitationVerificationResult containing
            verification results for all claims.
        """

        evidence_map = {
            evidence.id: evidence
            for evidence in evidences
        }

        verifications: list[
            CitationVerification
        ] = []

        for claim in claims:
            candidate_evidences = self._get_candidate_evidences(
                claim=claim,
                evidence_map=evidence_map,
            )

            verification = self._verify_claim(
                claim=claim,
                evidences=candidate_evidences,
            )

            verifications.append(
                verification
            )

        all_supported = (
            len(verifications) > 0
            and all(
                item.supported
                for item in verifications
            )
        )

        return CitationVerificationResult(
            verifications=verifications,
            all_supported=all_supported,
        )

    def _get_candidate_evidences(
        self,
        claim: Claim,
        evidence_map: dict[str, Evidence],
    ) -> list[Evidence]:
        """
        Resolve evidence IDs attached to a claim.

        Invalid or missing evidence IDs are ignored here.
        If no valid evidence remains, the claim will be
        marked unsupported without calling the LLM.
        """

        result: list[Evidence] = []

        seen_ids: set[str] = set()

        for evidence_id in claim.evidence_ids:
            if evidence_id in seen_ids:
                continue

            evidence = evidence_map.get(
                evidence_id
            )

            if evidence is None:
                continue

            seen_ids.add(
                evidence_id
            )

            result.append(
                evidence
            )

        return result

    def _verify_claim(
        self,
        claim: Claim,
        evidences: list[Evidence],
    ) -> CitationVerification:
        """
        Verify one claim against its candidate evidences.
        """

        if not evidences:
            return CitationVerification(
                claim_id=claim.id,
                supported=False,
                supporting_evidence_ids=[],
                unsupported_reason=(
                    "No valid candidate evidence "
                    "is available for this claim."
                ),
            )

        prompt = self._build_prompt(
            claim=claim,
            evidences=evidences,
        )

        content = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict citation "
                        "verification agent. "
                        "Judge whether evidence directly "
                        "supports a factual claim. "
                        "Do not use outside knowledge."
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

        result = CitationVerification.model_validate(
            data
        )

        return self._sanitize_result(
            claim=claim,
            evidences=evidences,
            result=result,
        )

    def _sanitize_result(
        self,
        claim: Claim,
        evidences: list[Evidence],
        result: CitationVerification,
    ) -> CitationVerification:
        """
        Apply program-side validation to LLM output.

        The LLM is responsible for semantic judgment,
        while Python owns IDs and state integrity.
        """

        valid_evidence_ids = {
            evidence.id
            for evidence in evidences
        }

        supporting_evidence_ids = []

        seen_ids: set[str] = set()

        for evidence_id in (
            result.supporting_evidence_ids
        ):
            if evidence_id not in valid_evidence_ids:
                continue

            if evidence_id in seen_ids:
                continue

            seen_ids.add(
                evidence_id
            )

            supporting_evidence_ids.append(
                evidence_id
            )

        # A supported claim must contain at least
        # one valid supporting evidence.
        if (
            result.supported
            and not supporting_evidence_ids
        ):
            return CitationVerification(
                claim_id=claim.id,
                supported=False,
                supporting_evidence_ids=[],
                unsupported_reason=(
                    "The verifier marked the claim "
                    "as supported but returned no valid "
                    "supporting evidence IDs."
                ),
            )

        # Unsupported claims should not retain citations.
        if not result.supported:
            return CitationVerification(
                claim_id=claim.id,
                supported=False,
                supporting_evidence_ids=[],
                unsupported_reason=(
                    result.unsupported_reason
                    or (
                        "The supplied evidence does "
                        "not directly support the claim."
                    )
                ),
            )

        return CitationVerification(
            claim_id=claim.id,
            supported=True,
            supporting_evidence_ids=(
                supporting_evidence_ids
            ),
            unsupported_reason=None,
        )

    def _build_prompt(
        self,
        claim: Claim,
        evidences: list[Evidence],
    ) -> str:
        """
        Build the verification prompt for one claim.
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

        return f"""
Verify whether the candidate evidence directly supports
the following research claim.

Claim ID:
{claim.id}

Claim:
{claim.text}


Candidate Evidence:

{evidence_text}


Verification rules:

1. A claim is supported only when one or more supplied
   evidences directly provide factual support for the
   same proposition.

2. Topic similarity is not sufficient. Evidence that
   merely discusses the same subject must not be treated
   as support.

3. Carefully compare:
   - subject
   - object
   - action or relationship
   - metric
   - numerical value
   - direction of change
   - comparison target
   - population or system
   - scope
   - conditions

4. Do not strengthen, generalize, or extrapolate beyond
   what the evidence explicitly supports.

5. If an important part of the claim is unsupported,
   return supported=false.

6. If the claim contains a numerical statement, the
   number, metric, direction, and context must all be
   supported by the evidence.

7. supporting_evidence_ids may contain only Evidence IDs
   explicitly provided above.

8. Include only evidence that directly supports the claim.
   Do not include evidence merely because it is relevant
   to the topic.

9. Do not use outside knowledge.

10. If multiple pieces of evidence together are required
    to support the claim, all necessary evidence IDs may
    be returned.

Return exactly one JSON object using this schema:

{{
  "claim_id": "{claim.id}",
  "supported": true,
  "supporting_evidence_ids": [
    "evidence_id"
  ],
  "unsupported_reason": null
}}

If the claim is not sufficiently supported, return:

{{
  "claim_id": "{claim.id}",
  "supported": false,
  "supporting_evidence_ids": [],
  "unsupported_reason": "Brief explanation of what is unsupported."
}}

Return JSON only.
""".strip()

    @staticmethod
    def _format_evidence(
        index: int,
        evidence: Evidence,
    ) -> str:
        """
        Format one evidence item for citation verification.

        Both the extracted summary and original quote are
        provided because citation verification requires
        stronger grounding than coverage evaluation.
        """

        parts = [
            f"[Evidence {index}]",
            f"Evidence ID: {evidence.id}",
            f"Source: {evidence.title}",
            f"URL: {evidence.url}",
            f"Summary: {evidence.content}",
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