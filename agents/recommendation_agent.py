import json

from schemas import Evidence, Finding, Recommendation


def _detect_adverse_selection(findings: dict[str, Finding]) -> bool:
    """Check for worsening risk + rising sales — the combination no single agent sees.

    Uses code-set flags, not narration, so it's reliable.
    """
    claims = findings.get("Claims")
    conversion = findings.get("Conversion")
    return bool(
        claims and "loss_ratio_deteriorating" in claims.flags
        and conversion and "conversion_up_without_price_cut" in conversion.flags
    )


def _build_evidence(findings: dict[str, Finding]) -> list[Evidence]:
    """Build evidence list from computed metrics and real sources.

    Built in code (not by the LLM) so citations can't be invented.
    """
    evidence: list[Evidence] = []

    claims = findings.get("Claims")
    if claims and claims.metrics:
        m = claims.metrics
        evidence.append(Evidence(
            claim=f"Loss ratio moved {m['loss_ratio_start_6mo']:.0%} → {m['loss_ratio_latest']:.0%} "
                  f"({m['loss_ratio_change_pts'] * 100:+.0f}pts) over 6 months",
            source="claims_performance.csv",
        ))
        if "frequency_rising" in claims.flags:
            evidence.append(Evidence(
                claim=f"Claim frequency rose {m['frequency_prior']:.3f} → {m['frequency_recent']:.3f} claims/policy",
                source="claims_performance.csv",
            ))

    conversion = findings.get("Conversion")
    if conversion and conversion.metrics:
        m = conversion.metrics
        evidence.append(Evidence(
            claim=f"Conversion moved {m['conversion_start_6mo']:.1%} → {m['conversion_latest']:.1%} "
                  f"({m['conversion_change_pts'] * 100:+.1f}pts) while average premium changed "
                  f"{m['avg_premium_change_pct']:+.1f}%",
            source="conversion_performance.csv",
        ))

    market = findings.get("Market")
    if market:
        # Split the retrieved sources into documents and verbatims by their id shape.
        docs = [s for s in market.sources if s.endswith(".md")]
        verbatims = [s for s in market.sources if s.startswith("FB")]
        if "competitor_rate_move" in market.flags and docs:
            evidence.append(Evidence(
                claim="Competitor activity / rate movement reported in this segment",
                source=docs[0],
            ))
        if verbatims:
            evidence.append(Evidence(
                claim=f"Customer verbatims tagged to this segment corroborate the picture "
                      f"({len(verbatims)} retrieved)",
                source=verbatims[0],
            ))

    return evidence


def recommend(segment: str, findings: dict[str, Finding], llm) -> Recommendation:
    """Synthesise specialist findings into a validated recommendation."""
    adverse = _detect_adverse_selection(findings)
    evidence = _build_evidence(findings)

    cross_signal = (
        "ADVERSE SELECTION DETECTED: loss ratio is worsening while conversion rises without "
        "a price cut. This means we are increasingly winning business we are underpricing "
        "for the risk — the recommendation should lean towards a targeted, risk-justified "
        "rate correction and/or underwriting review, NOT chasing further volume."
        if adverse else
        "No adverse-selection pattern detected across the signals."
    )

    claims = findings.get("Claims")
    conversion = findings.get("Conversion")
    market = findings.get("Market")

    prompt = f"""You are the lead pricing recommendation agent. Synthesise the specialist
findings below into ONE recommendation for the {segment} segment.

Return ONLY valid JSON with these keys:
  "recommended_action": string (a concrete, specific pricing action),
  "rationale": string (2-4 sentences; explicitly connect the claims, conversion and market signals),
  "confidence": one of "low", "medium", "high",
  "caveats": array of short strings (risks, unknowns, or things to check before acting).

Claims finding: {claims.summary if claims else 'n/a'}
Conversion finding: {conversion.summary if conversion else 'n/a'}
Market finding: {market.summary if market else 'n/a'}

Cross-signal analysis: {cross_signal}

JSON:"""
    raw = llm.invoke(prompt).content.strip()
    draft = _parse_json(raw)

    return Recommendation(
        segment=segment,
        recommended_action=draft.get("recommended_action", "Investigate further — see caveats"),
        rationale=draft.get("rationale", cross_signal),
        evidence=evidence,
        confidence=_coerce_confidence(draft.get("confidence"), adverse),
        caveats=draft.get("caveats", []) or [],
        # Human approval is always on. The model doesn't get to change this.
        requires_human_approval=True,
    )


def _parse_json(raw: str) -> dict:
    """Extract JSON from the model's reply, handling stray prose around it.

    Slices from first { to last } since local models often wrap JSON in
    code fences or preamble. Falls back to empty dict on failure.
    """
    print("raw", raw)
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}


def _coerce_confidence(value, adverse: bool) -> str:
    """Keep confidence in the allowed set. Default based on whether adverse selection was found."""
    if value in ("low", "medium", "high"):
        return value
    # A clean multi-signal agreement is a high-confidence read; otherwise stay modest.
    return "high" if adverse else "medium"
