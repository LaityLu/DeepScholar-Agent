from pydantic import BaseModel, Field


class Claim(BaseModel):
    id: str
    text: str
    section: str
    evidence_ids: list[str] = Field(
        default_factory=list
    )


class VerifiedClaim(BaseModel):
    id: str
    text: str
    section: str
    supporting_evidence_ids: list[str] = Field(
        default_factory=list
    )


class CitationVerification(BaseModel):
    claim_id: str
    supported: bool
    supporting_evidence_ids: list[str] = Field(
        default_factory=list
    )
    unsupported_reason: str | None = None


class CitationVerificationResult(BaseModel):
    verifications: list[CitationVerification]
    all_supported: bool

