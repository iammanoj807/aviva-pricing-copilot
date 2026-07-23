from schemas import Finding
from retriever import Retriever


def analyse_market(segment: str, llm, retriever: Retriever, focus: str = "") -> Finding:
    """Retrieve segment-tagged passages and summarise them with per-claim citations."""
    passages = retriever.retrieve(segment, query=focus or segment)
    if not passages:
        return Finding(agent="Market", segment=segment,
                       summary=f"No market intelligence tagged for segment '{segment}'.")

    # Use real source ids (filename / feedback id) so citations are traceable.
    context = "\n\n".join(f"[{p.source}] ({p.kind})\n{p.text}" for p in passages)

    focus_line = (
        f"\nThe supervisor has asked you to focus specifically on: {focus}\n"
        if focus else ""
    )
    prompt = f"""You are a market intelligence analyst for insurance pricing.
Summarise what the sources below say about the {segment} segment.
Rules:
- Use ONLY the sources provided. If something isn't in them, don't say it.
- After EVERY claim, cite the source in square brackets, e.g. [competitor_motor_young_driver.md] or [FB001].
- Pull out competitor pricing moves and any cluster of customer complaints.
- 3-5 sentences, plain English.{focus_line}
Sources:
{context}

Summary:"""
    summary = llm.invoke(prompt).content.strip()

    flags = []
    # Quick check: any competitor rate moves mentioned in the docs for this segment?
    if any("cut" in p.text.lower() or "aggressive" in p.text.lower()
           for p in passages if p.kind == "market_doc"):
        flags.append("competitor_rate_move")

    return Finding(
        agent="Market",
        segment=segment,
        summary=summary,
        metrics={"passages_retrieved": len(passages)},
        sources=[p.source for p in passages],
        flags=flags,
    )
