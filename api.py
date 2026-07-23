from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph import build_graph
from llm import get_llm
from retriever import TagFilterRetriever
from schemas import Recommendation

app = FastAPI(
    title="Aviva Pricing Analyst Copilot API",
    description=(
        "Multi-agent pricing copilot. Claims, Conversion, Market and Recommendation "
        "agents coordinated by a LangGraph supervisor. Recommendations are propose-only "
        "and require human approval."
    ),
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_app_graph():
    """Compile the graph once and reuse across requests. Lazy-loads on first call."""
    return build_graph(get_llm(), TagFilterRetriever())


# --- Request / response contracts ----------------------------------------------------

class AskRequest(BaseModel):
    """A pricing question. The graph resolves it to a known segment."""

    question: str = Field(
        ...,
        min_length=1,
        examples=["Should we take any pricing action on Motor 25-34?"],
        description="Mention a segment (e.g. 'Motor 25-34') or it defaults to Motor 25-34.",
    )


class TraceStep(BaseModel):
    """One entry in the execution trace."""

    round: int
    agent: str
    event: str
    detail: str = ""


class AgentFinding(BaseModel):
    """A specialist's report: LLM summary over pandas-computed metrics."""

    agent: str
    summary: str
    metrics: dict = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Full result: recommendation + findings + trace + guardrail log."""

    segment: str
    requires_human_approval: bool
    second_round_triggered: bool
    recommendation: Recommendation
    findings: dict[str, AgentFinding]
    trace: list[TraceStep]
    guardrail_notes: list[str]


# --- Routes --------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe. Doesn't touch the model."""
    return {"status": "ok"}


@app.get("/segments", tags=["meta"])
def segments() -> dict:
    """List the segments the copilot knows about."""
    from graph import SEGMENTS

    return {"segments": SEGMENTS}


@app.post("/ask", response_model=AskResponse, tags=["copilot"])
def ask(req: AskRequest) -> AskResponse:
    """Run the full multi-agent analysis for one pricing question.

    Synchronous call that runs several LLM inferences, so expect some latency.
    """
    try:
        result = get_app_graph().invoke({"question": req.question})
    except Exception as exc:  # surface a clean 500 rather than a stack trace to the caller
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    rec: Recommendation = result["recommendation"]

    findings: dict[str, AgentFinding] = {}
    for key in ("claims", "conversion", "market"):
        f = result.get(key)
        if f is not None:
            findings[key] = AgentFinding(
                agent=f.agent, summary=f.summary, metrics=f.metrics,
                sources=f.sources, flags=f.flags,
            )

    return AskResponse(
        segment=rec.segment,
        requires_human_approval=rec.requires_human_approval,
        second_round_triggered=bool(result.get("second_round")),
        recommendation=rec,
        findings=findings,
        trace=[TraceStep(**t) for t in result["trace"]],
        guardrail_notes=result["guardrail_notes"],
    )
