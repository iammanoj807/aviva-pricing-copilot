from typing import Literal
from pydantic import BaseModel, Field


class Passage(BaseModel):
    """A retrieved chunk of unstructured evidence (market doc or verbatim)."""

    source: str          # filename for market docs, feedback_id for verbatims — the citation
    segment: str
    kind: Literal["market_doc", "verbatim"]
    text: str


class Evidence(BaseModel):
    """One claim tied to one source. No source, no evidence."""

    claim: str
    source: str


class Finding(BaseModel):
    """What a specialist agent reports back.

    `metrics` = pandas-computed numbers. `summary` = LLM narration of those.
    `flags` = short codes the supervisor uses for routing decisions.
    """

    agent: str
    segment: str
    summary: str
    metrics: dict = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """Final output for a human analyst to review.

    requires_human_approval is always True. The system proposes, a person decides.
    """

    segment: str
    recommended_action: str
    rationale: str
    evidence: list[Evidence]
    confidence: Literal["low", "medium", "high"]
    caveats: list[str] = Field(default_factory=list)
    requires_human_approval: bool = True
