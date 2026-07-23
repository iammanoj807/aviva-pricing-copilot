import streamlit as st

from graph import build_graph
from llm import get_llm
from retriever import TagFilterRetriever

st.set_page_config(page_title="Aviva Pricing Analyst Copilot", layout="wide")


@st.cache_resource
def get_app():
    """Build the graph once and reuse across reruns."""
    return build_graph(get_llm(), TagFilterRetriever())


EXAMPLE_QUESTIONS = [
    "Should we take any pricing action on Motor 25-34?",
    "Is there anything to worry about in Motor 50+?",
    "Review Van Commercial pricing.",
    "How is Home Premium performing?",
]

st.title("🧭 Aviva Pricing Analyst Copilot")
st.caption("A multi-agent copilot that gathers, synthesises and **recommends** — a human analyst always makes the final call.")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Four specialist agents coordinated by a LangGraph supervisor:\n"
        "- **Claims** — loss-ratio / frequency / severity (pandas)\n"
        "- **Conversion** — conversion trend + elasticity (pandas)\n"
        "- **Market** — tag-filtered retrieval, cited\n"
        "- **Recommendation** — synthesis, Pydantic-validated\n\n"
        "Numbers are computed in code; the LLM only narrates. Retrieval is tag-filtered — "
        "no vector DB at this data size."
    )
    st.info("Every recommendation is **propose-only** and requires human approval.")

question = st.text_input(
    "Ask the copilot a pricing question",
    value=EXAMPLE_QUESTIONS[0],
    help="Mention a segment (e.g. 'Motor 25-34') or the copilot defaults to it.",
)
st.caption("Try: " + "  ·  ".join(f"_{q}_" for q in EXAMPLE_QUESTIONS[1:]))

if st.button("Run analysis", type="primary"):
    with st.spinner("Agents working — parallel sweep, then conditional follow-up…"):
        result = get_app().invoke({"question": question})
    st.session_state["result"] = result

result = st.session_state.get("result")
if result:
    rec = result["recommendation"]

    # --- Execution trace: the headline element ---------------------------------------
    st.subheader("🔍 Execution trace")
    second = result.get("second_round")
    st.markdown(
        f"**Segment:** `{rec.segment}`  ·  "
        f"**Second round triggered:** {'✅ yes — a specialist finding changed the plan' if second else '— no'}"
    )
    st.table([
        {"Round": t["round"], "Agent": t["agent"], "Event": t["event"], "Detail": t["detail"]}
        for t in result["trace"]
    ])

    # --- Per-agent findings -----------------------------------------------------------
    st.subheader("🧩 Specialist findings")
    for key, label in [("claims", "Claims Analysis"), ("conversion", "Conversion Analysis"), ("market", "Market Intelligence")]:
        finding = result.get(key)
        if not finding:
            continue
        with st.expander(f"{label}  ·  flags: {', '.join(finding.flags) or 'none'}"):
            st.write(finding.summary)
            if finding.metrics:
                st.markdown("**Computed figures (pandas, not the LLM):**")
                st.json(finding.metrics)
            st.caption("Sources: " + ", ".join(finding.sources))

    # --- Recommendation ---------------------------------------------------------------
    st.subheader("✅ Recommendation")
    st.warning("⚠️ Proposal only — requires human analyst approval before any action.")
    st.markdown(f"### {rec.recommended_action}")
    st.write(rec.rationale)
    st.markdown(f"**Confidence:** `{rec.confidence}`")

    with st.expander("📎 Evidence trail (every claim cites a source)", expanded=True):
        for e in rec.evidence:
            st.markdown(f"- {e.claim}  — `{e.source}`")

    if rec.caveats:
        with st.expander("⚠️ Caveats"):
            for c in rec.caveats:
                st.markdown(f"- {c}")

    with st.expander("🛡️ Guardrail log"):
        for note in result["guardrail_notes"]:
            st.markdown(f"- {note}")
