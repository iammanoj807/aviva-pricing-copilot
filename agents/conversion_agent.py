import os

import pandas as pd

from schemas import Finding

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# +2pts over 6 months counts as a real conversion shift.
CONVERSION_MOVE_PTS = 0.02


def _compute_metrics(seg_df: pd.DataFrame, actions: pd.DataFrame) -> dict:
    """Conversion + premium trends, plus the last pricing action for elasticity context."""
    seg_df = seg_df.sort_values("month")
    recent = seg_df.tail(6)
    prior = seg_df.head(len(seg_df) - 6) if len(seg_df) > 6 else seg_df

    start_conv = float(seg_df.iloc[-6]["conversion_rate"]) if len(seg_df) >= 6 else float(seg_df.iloc[0]["conversion_rate"])
    latest_conv = float(seg_df.iloc[-1]["conversion_rate"])

    start_prem = float(seg_df.iloc[-6]["avg_premium"]) if len(seg_df) >= 6 else float(seg_df.iloc[0]["avg_premium"])
    latest_prem = float(seg_df.iloc[-1]["avg_premium"])
    prem_change_pct = (latest_prem - start_prem) / start_prem * 100

    m = {
        "latest_month": seg_df.iloc[-1]["month"],
        "conversion_start_6mo": round(start_conv, 4),
        "conversion_latest": round(latest_conv, 4),
        "conversion_change_pts": round(latest_conv - start_conv, 4),
        "conversion_recent_avg": round(float(recent["conversion_rate"].mean()), 4),
        "conversion_prior_avg": round(float(prior["conversion_rate"].mean()), 4),
        "avg_premium_change_pct": round(prem_change_pct, 2),
    }

    # Join against last pricing action for this segment to read elasticity.
    # Conversion rising without a price cut is a flag worth noting.
    seg_actions = actions[actions["segment"] == seg_df.iloc[0]["segment"]]
    if not seg_actions.empty:
        last = seg_actions.sort_values("date").iloc[-1]
        m["last_action"] = last["action"]
        m["last_action_rate_change_pct"] = float(last["rate_change_pct"])
        m["last_action_date"] = last["date"]
        m["last_action_outcome"] = last["observed_outcome"]
    else:
        m["last_action"] = "none"
        m["last_action_rate_change_pct"] = 0.0

    return m


def analyse_conversion(segment: str, llm) -> Finding:
    """Compute conversion + elasticity signal for a segment, then narrate it."""
    df = pd.read_csv(os.path.join(DATA_DIR, "conversion_performance.csv"))
    actions = pd.read_csv(os.path.join(DATA_DIR, "previous_pricing_actions.csv"))
    seg_df = df[df["segment"] == segment]
    if seg_df.empty:
        return Finding(agent="Conversion", segment=segment,
                       summary=f"No conversion data for segment '{segment}'.")

    m = _compute_metrics(seg_df, actions)

    flags = []
    if m["conversion_change_pts"] >= CONVERSION_MOVE_PTS:
        flags.append("conversion_rising")
    # Conversion rising without us having cut price = possible adverse selection signal.
    if m["conversion_change_pts"] >= CONVERSION_MOVE_PTS and m["avg_premium_change_pct"] > -1.0:
        flags.append("conversion_up_without_price_cut")

    summary = _narrate(segment, m, llm)
    return Finding(
        agent="Conversion",
        segment=segment,
        summary=summary,
        metrics=m,
        sources=["conversion_performance.csv", "previous_pricing_actions.csv"],
        flags=flags,
    )


def _narrate(segment: str, m: dict, llm) -> str:
    """Narrate conversion trend and what the last pricing action implies."""
    prompt = f"""You are a pricing analyst summarising pre-computed conversion figures.
Use ONLY the numbers given. Do NOT calculate or invent any new number. 2-3 sentences.
State the conversion move, note the average-premium change, and interpret the last
pricing action's elasticity signal.

Segment: {segment}
Conversion 6 months ago: {m['conversion_start_6mo']:.1%}
Conversion latest ({m['latest_month']}): {m['conversion_latest']:.1%}
Conversion change: {m['conversion_change_pts'] * 100:+.1f} points
Average premium change over window: {m['avg_premium_change_pct']:+.1f}%
Last pricing action: {m.get('last_action')} ({m.get('last_action_rate_change_pct'):+.1f}%) on {m.get('last_action_date', 'n/a')}
Observed outcome of that action: {m.get('last_action_outcome', 'n/a')}

Summary:"""
    return llm.invoke(prompt).content.strip()
