from schemas import Finding, Recommendation

# Known CSV sources. Unstructured sources are checked against what the agents actually retrieved.
KNOWN_STRUCTURED_SOURCES = {
    "claims_performance.csv",
    "conversion_performance.csv",
    "previous_pricing_actions.csv",
    "competitor_pricing.csv",
}


def _known_sources(findings: dict[str, Finding]) -> set[str]:
    """All source ids the specialists actually used this run."""
    sources = set(KNOWN_STRUCTURED_SOURCES)
    for finding in findings.values():
        sources.update(finding.sources)
    return sources


def apply_guardrails(rec: Recommendation, findings: dict[str, Finding]) -> tuple[Recommendation, list[str]]:
    """Clean the recommendation and return a log of what changed.

    Drops evidence citing unknown sources, forces human approval on.
    """
    known = _known_sources(findings)
    notes: list[str] = []

    supported, dropped = [], []
    for item in rec.evidence:
        (supported if item.source in known else dropped).append(item)

    for item in dropped:
        notes.append(f"Dropped unsupported evidence (source '{item.source}' not in retrieved data): "
                     f"\"{item.claim}\"")

    # Always force human approval regardless of what upstream did.
    if not rec.requires_human_approval:
        notes.append("Forced requires_human_approval=True (propose-only enforced).")

    cleaned = rec.model_copy(update={
        "evidence": supported,
        "requires_human_approval": True,
    })

    if not notes:
        notes.append("All evidence cited a valid source; propose-only intact. No changes.")

    return cleaned, notes
