from operator import add
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.claims_agent import analyse_claims
from agents.conversion_agent import analyse_conversion
from agents.market_agent import analyse_market
from agents.recommendation_agent import recommend
from guardrails import apply_guardrails
from schemas import Finding, Recommendation

SEGMENTS = [
    "Motor 25-34", "Motor 35-49", "Motor 50+",
    "Home Standard", "Home Premium", "Van Commercial",
]


class PricingState(TypedDict):
    """State dict passed between graph nodes."""

    question: str
    segment: str
    claims: Optional[Finding]
    conversion: Optional[Finding]
    market: Optional[Finding]
    recommendation: Optional[Recommendation]
    guardrail_notes: list[str]
    second_round: bool
    trace: Annotated[list[dict], add]  # append-only across parallel branches


def _resolve_segment(question: str, llm) -> str:
    """Use the LLM to map free-text to one of our known segments.

    Validates against SEGMENTS so the model can't invent one we don't price.
    Falls back to Motor 25-34 if the reply doesn't match.
    """
    options = ", ".join(SEGMENTS)
    prompt = (
        "You route a pricing question to exactly one insurance segment.\n"
        f"Reply with EXACTLY one of these, verbatim, and nothing else: {options}\n\n"
        f"Question: {question}\n\nSegment:"
    )
    guess = llm.invoke(prompt).content.strip()
    return guess if guess in SEGMENTS else "Motor 25-34"


def _trace(agent: str, round_no: int, event: str, detail: str = "") -> dict:
    """Structured trace entry for the UI."""
    return {"agent": agent, "round": round_no, "event": event, "detail": detail}


def build_graph(llm, retriever):
    """Compile the supervisor graph. LLM and retriever are injected for testability."""

    def intake(state: PricingState) -> dict:
        segment = _resolve_segment(state["question"], llm)
        return {
            "segment": segment,
            "second_round": False,
            "trace": [_trace("Supervisor", 1, "intake",
                             f"Resolved target segment '{segment}'. Dispatching parallel sweep.")],
        }

    # --- Round 1: three specialists in parallel ---------------------------------------

    def claims_node(state: PricingState) -> dict:
        f = analyse_claims(state["segment"], llm)
        return {"claims": f,
                "trace": [_trace("Claims", 1, "ran", f"flags={f.flags or 'none'}")]}

    def conversion_node(state: PricingState) -> dict:
        f = analyse_conversion(state["segment"], llm)
        return {"conversion": f,
                "trace": [_trace("Conversion", 1, "ran", f"flags={f.flags or 'none'}")]}

    def market_node(state: PricingState) -> dict:
        f = analyse_market(state["segment"], llm, retriever)
        return {"market": f,
                "trace": [_trace("Market", 1, "ran",
                                 f"retrieved {f.metrics.get('passages_retrieved', 0)} passages")]}

    # --- Supervisor checks round 1, decides if we need a deeper look ----------------

    def inspect(state: PricingState) -> dict:
        claims = state.get("claims")
        material = bool(claims and "loss_ratio_deteriorating" in claims.flags)
        if material:
            return {"second_round": True,
                    "trace": [_trace("Supervisor", 1, "inspect",
                                     "Claims flagged loss-ratio deterioration — triggering targeted "
                                     "second-round market re-query.")]}
        return {"second_round": False,
                "trace": [_trace("Supervisor", 1, "inspect",
                                 "No material deterioration flagged — proceeding straight to synthesis.")]}

    def route_after_inspect(state: PricingState) -> str:
        """Conditional edge: did round 1 warrant a second pass?"""
        return "second_round" if state["second_round"] else "recommend"

    # --- Round 2 (conditional): re-query market with a specific focus ------------------

    def market_round2(state: PricingState) -> dict:
        claims = state.get("claims")
        drivers = ", ".join(claims.flags) if claims else "loss-ratio deterioration"
        focus = (f"Claims flagged {drivers} in {state['segment']}. Look specifically for competitor "
                 f"rate cuts and renewal-price complaints that would explain worsening risk.")
        f = analyse_market(state["segment"], llm, retriever, focus=focus)
        return {"market": f,
                "trace": [_trace("Market", 2, "re-queried",
                                 "Focused on competitor rate moves + renewal complaints per Claims flag.")]}

    # --- Synthesis + guardrails -------------------------------------------------------

    def recommend_node(state: PricingState) -> dict:
        findings = _findings_dict(state)
        rec = recommend(state["segment"], findings, llm)
        return {"recommendation": rec,
                "trace": [_trace("Recommendation", 2 if state["second_round"] else 1, "synthesised",
                                 f"action set, confidence={rec.confidence}")]}

    def guardrail_node(state: PricingState) -> dict:
        findings = _findings_dict(state)
        cleaned, notes = apply_guardrails(state["recommendation"], findings)
        return {"recommendation": cleaned, "guardrail_notes": notes,
                "trace": [_trace("Guardrail", 2 if state["second_round"] else 1, "checked",
                                 notes[0])]}

    # --- Wiring -----------------------------------------------------------------------

    g = StateGraph(PricingState)
    g.add_node("intake", intake)
    g.add_node("claims", claims_node)
    g.add_node("conversion", conversion_node)
    g.add_node("market", market_node)
    g.add_node("inspect", inspect)
    g.add_node("market_round2", market_round2)
    g.add_node("recommend", recommend_node)
    g.add_node("guardrail", guardrail_node)

    g.add_edge(START, "intake")
    # Fan-out: the three specialists run in parallel off intake.
    g.add_edge("intake", "claims")
    g.add_edge("intake", "conversion")
    g.add_edge("intake", "market")
    # Fan-in: inspect waits for all three before it runs.
    g.add_edge("claims", "inspect")
    g.add_edge("conversion", "inspect")
    g.add_edge("market", "inspect")
    # The conditional bit.
    g.add_conditional_edges("inspect", route_after_inspect, {
        "second_round": "market_round2",
        "recommend": "recommend",
    })
    g.add_edge("market_round2", "recommend")
    g.add_edge("recommend", "guardrail")
    g.add_edge("guardrail", END)

    return g.compile()


def _findings_dict(state: PricingState) -> dict[str, Finding]:
    """Collect non-null specialist findings into a dict."""
    return {
        name: state[key]
        for name, key in [("Claims", "claims"), ("Conversion", "conversion"), ("Market", "market")]
        if state.get(key) is not None
    }
