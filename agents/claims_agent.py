import os
import pandas as pd

from schemas import Finding

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# 5-point loss-ratio jump over 6 months triggers a second-round investigation.
DETERIORATION_PTS = 0.05


def _compute_metrics(seg_df: pd.DataFrame) -> dict:
    """Deterministic maths for one segment. Splits into recent 6mo vs prior 6mo."""
    seg_df = seg_df.sort_values("month")
    recent = seg_df.tail(6)
    prior = seg_df.head(len(seg_df) - 6) if len(seg_df) > 6 else seg_df

    # Recompute loss ratio from source columns, don't trust the CSV's stored value.
    recent_lr = (recent["claims_cost"].sum() / recent["earned_premium"].sum())
    prior_lr = (prior["claims_cost"].sum() / prior["earned_premium"].sum())
    start_lr = seg_df.iloc[-6]["loss_ratio"] if len(seg_df) >= 6 else seg_df.iloc[0]["loss_ratio"]
    latest_lr = seg_df.iloc[-1]["loss_ratio"]

    # Frequency = claims/policy, severity = cost/claim. Helps explain *why* LR moved.
    recent_freq = recent["claims_count"].sum() / recent["policies"].sum()
    prior_freq = prior["claims_count"].sum() / prior["policies"].sum()
    recent_sev = recent["claims_cost"].sum() / recent["claims_count"].sum()
    prior_sev = prior["claims_cost"].sum() / prior["claims_count"].sum()

    return {
        "latest_month": seg_df.iloc[-1]["month"],
        "loss_ratio_start_6mo": round(float(start_lr), 4),
        "loss_ratio_latest": round(float(latest_lr), 4),
        "loss_ratio_change_pts": round(float(latest_lr - start_lr), 4),
        "loss_ratio_recent_avg": round(float(recent_lr), 4),
        "loss_ratio_prior_avg": round(float(prior_lr), 4),
        "frequency_recent": round(float(recent_freq), 4),
        "frequency_prior": round(float(prior_freq), 4),
        "severity_recent": round(float(recent_sev), 2),
        "severity_prior": round(float(prior_sev), 2),
    }


def analyse_claims(segment: str, llm) -> Finding:
    """Compute claims trends for a segment, then have the LLM narrate the result."""
    df = pd.read_csv(os.path.join(DATA_DIR, "claims_performance.csv"))
    seg_df = df[df["segment"] == segment]
    if seg_df.empty:
        return Finding(agent="Claims", segment=segment,
                       summary=f"No claims data for segment '{segment}'.")

    m = _compute_metrics(seg_df)

    # Flags are set in code, not by the model, since the supervisor routes on them.
    # Deterioration = rising LR. Frequency/severity flags explain the driver.
    flags = []
    if m["loss_ratio_change_pts"] >= DETERIORATION_PTS:
        flags.append("loss_ratio_deteriorating")
    if m["frequency_recent"] > m["frequency_prior"]:
        flags.append("frequency_rising")
    if m["severity_recent"] > m["severity_prior"]:
        flags.append("severity_rising")

    summary = _narrate(segment, m, llm)
    return Finding(
        agent="Claims",
        segment=segment,
        summary=summary,
        metrics=m,
        sources=["claims_performance.csv"],
        flags=flags,
    )


def _narrate(segment: str, m: dict, llm) -> str:
    """Turn computed figures into 2-3 sentences. The LLM only gets to pick the wording."""
    prompt = f"""You are a claims analyst summarising pre-computed figures for a pricing colleague.
Use ONLY the numbers given below. Do NOT calculate, estimate, or invent any new number.
Write 2-3 plain-English sentences. State the loss-ratio move, then whether frequency or
severity (or both) is driving it.

Segment: {segment}
Loss ratio 6 months ago: {m['loss_ratio_start_6mo']:.0%}
Loss ratio latest ({m['latest_month']}): {m['loss_ratio_latest']:.0%}
Loss ratio change: {m['loss_ratio_change_pts'] * 100:+.1f} points
Claim frequency (recent vs prior 6mo): {m['frequency_recent']:.3f} vs {m['frequency_prior']:.3f} claims/policy
Claim severity (recent vs prior 6mo): £{m['severity_recent']:,.0f} vs £{m['severity_prior']:,.0f} per claim

Summary:"""
    return llm.invoke(prompt).content.strip()
