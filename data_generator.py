import os
import random

import pandas as pd

# Fixed seed so volume noise is reproducible across runs.
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 12 months ending June 2026. "Last 6 months" = 2026-01..2026-06.
MONTHS = [
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
]

SEGMENTS = [
    "Motor 25-34",   # the story segment
    "Motor 35-49",
    "Motor 50+",
    "Home Standard",
    "Home Premium",
    "Van Commercial",
]

# Loss-ratio trajectory per segment. Motor 25-34 deliberately climbs from ~60% to 74%.
# The rest stay flat.
LOSS_RATIO = {
    "Motor 25-34":   [0.60, 0.59, 0.61, 0.60, 0.62, 0.61, 0.62, 0.65, 0.67, 0.70, 0.72, 0.74],
    "Motor 35-49":   [0.57, 0.58, 0.56, 0.59, 0.57, 0.58, 0.57, 0.58, 0.59, 0.57, 0.58, 0.58],
    "Motor 50+":     [0.52, 0.51, 0.53, 0.52, 0.51, 0.52, 0.53, 0.52, 0.51, 0.52, 0.53, 0.52],
    "Home Standard": [0.55, 0.54, 0.56, 0.55, 0.56, 0.54, 0.55, 0.56, 0.55, 0.54, 0.56, 0.55],
    "Home Premium":  [0.61, 0.60, 0.62, 0.61, 0.60, 0.61, 0.62, 0.61, 0.60, 0.61, 0.62, 0.61],
    "Van Commercial":[0.64, 0.65, 0.63, 0.66, 0.64, 0.65, 0.64, 0.66, 0.65, 0.64, 0.66, 0.65],
}

# Claims frequency per month. Motor 25-34 drifts up in the back half
# so the LR decomposition has something real to say.
FREQUENCY = {
    "Motor 25-34":   [0.110, 0.109, 0.112, 0.111, 0.113, 0.112, 0.114, 0.118, 0.121, 0.125, 0.128, 0.132],
    "Motor 35-49":   [0.085] * 12,
    "Motor 50+":     [0.060] * 12,
    "Home Standard": [0.040] * 12,
    "Home Premium":  [0.045] * 12,
    "Van Commercial":[0.095] * 12,
}

# Conversion rate. Motor 25-34 rises 19% -> 24% in the back half.
# Everyone else stays flat.
CONVERSION = {
    "Motor 25-34":   [0.17, 0.18, 0.17, 0.18, 0.18, 0.19, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24],
    "Motor 35-49":   [0.22, 0.21, 0.22, 0.21, 0.20, 0.21, 0.20, 0.21, 0.20, 0.21, 0.20, 0.20],
    "Motor 50+":     [0.26, 0.27, 0.26, 0.27, 0.26, 0.27, 0.26, 0.27, 0.26, 0.27, 0.26, 0.27],
    "Home Standard": [0.19, 0.20, 0.19, 0.20, 0.19, 0.20, 0.19, 0.20, 0.19, 0.20, 0.19, 0.20],
    "Home Premium":  [0.15, 0.15, 0.16, 0.15, 0.14, 0.15, 0.16, 0.15, 0.15, 0.16, 0.15, 0.15],
    "Van Commercial":[0.23, 0.22, 0.23, 0.22, 0.23, 0.22, 0.23, 0.22, 0.23, 0.22, 0.23, 0.22],
}

# Average written premium (£). Motor 25-34 stays flat while risk climbs
# = we didn't reprice, so we kept winning underpriced business.
AVG_PREMIUM = {
    "Motor 25-34":   [785, 784, 786, 783, 782, 781, 780, 779, 778, 779, 777, 776],
    "Motor 35-49":   [690] * 12,
    "Motor 50+":     [610] * 12,
    "Home Standard": [430] * 12,
    "Home Premium":  [720] * 12,
    "Van Commercial":[1150] * 12,
}

# Starting policy/quote volumes per segment.
BASE_POLICIES = {
    "Motor 25-34": 12000, "Motor 35-49": 18000, "Motor 50+": 15000,
    "Home Standard": 22000, "Home Premium": 9000, "Van Commercial": 6000,
}
BASE_QUOTES = {
    "Motor 25-34": 68000, "Motor 35-49": 74000, "Motor 50+": 52000,
    "Home Standard": 90000, "Home Premium": 40000, "Van Commercial": 24000,
}


def build_claims() -> pd.DataFrame:
    """Claims performance CSV. The agent recomputes loss_ratio from cost/premium."""
    rows = []
    for seg in SEGMENTS:
        for i, month in enumerate(MONTHS):
            # ~0.4% monthly book growth plus a little noise, so volumes look alive.
            growth = 1 + 0.004 * i + random.uniform(-0.01, 0.01)
            policies = int(BASE_POLICIES[seg] * growth)
            earned_premium = round(policies * AVG_PREMIUM[seg][i] / 12, 2)  # monthly earned

            lr = LOSS_RATIO[seg][i]
            claims_cost = round(lr * earned_premium, 2)
            claims_count = max(1, int(policies * FREQUENCY[seg][i]))

            rows.append({
                "segment": seg,
                "month": month,
                "policies": policies,
                "claims_count": claims_count,
                "claims_cost": claims_cost,
                "earned_premium": earned_premium,
                "loss_ratio": round(claims_cost / earned_premium, 4),
            })
    return pd.DataFrame(rows)


def build_conversion() -> pd.DataFrame:
    """Conversion performance — quotes, sales, conversion rate and average premium."""
    rows = []
    for seg in SEGMENTS:
        for i, month in enumerate(MONTHS):
            growth = 1 + 0.003 * i + random.uniform(-0.015, 0.015)
            quotes = int(BASE_QUOTES[seg] * growth)
            rate = CONVERSION[seg][i]
            sales = int(quotes * rate)
            rows.append({
                "segment": seg,
                "month": month,
                "quotes": quotes,
                "sales": sales,
                "conversion_rate": round(sales / quotes, 4),
                "avg_premium": AVG_PREMIUM[seg][i],
            })
    return pd.DataFrame(rows)


def build_competitor() -> pd.DataFrame:
    """Competitor pricing. Directwise cuts Motor 25-34 rates from ~Feb 2026."""
    competitors = ["Directwise", "Hastwell", "Churchbridge"]
    rows = []
    for seg in SEGMENTS:
        for i, month in enumerate(MONTHS):
            our_price = AVG_PREMIUM[seg][i]
            for c_idx, comp in enumerate(competitors):
                # Competitors normally sit within +/- a few percent of our price.
                offset = [1.02, 0.98, 1.05][c_idx]
                premium = our_price * offset

                # The Motor 25-34 rate cut: Directwise drops hard from Feb 2026.
                if seg == "Motor 25-34" and comp == "Directwise" and i >= 7:
                    premium = our_price * 0.90

                premium = round(premium + random.uniform(-8, 8), 2)
                rows.append({
                    "segment": seg,
                    "month": month,
                    "competitor": comp,
                    "indicative_premium": premium,
                })

    df = pd.DataFrame(rows)
    # Rank position is cheapest-first within each segment/month across the panel.
    df["rank_position"] = (
        df.groupby(["segment", "month"])["indicative_premium"].rank(method="min").astype(int)
    )
    return df


def build_pricing_actions() -> pd.DataFrame:
    """Previous pricing actions with observed outcomes. Used for elasticity reads."""
    rows = [
        {"date": "2025-09-15", "segment": "Motor 25-34", "action": "hold",
         "rate_change_pct": 0.0,
         "rationale": "Protect new-business volume against competitor activity",
         "observed_outcome": "Conversion continued rising; loss ratio began drifting up"},
        {"date": "2025-11-01", "segment": "Motor 35-49", "action": "increase",
         "rate_change_pct": 4.0,
         "rationale": "Margin protection on a stable-risk book",
         "observed_outcome": "Conversion softened ~1-2pts, margin held"},
        {"date": "2026-01-10", "segment": "Home Premium", "action": "decrease",
         "rate_change_pct": -3.0,
         "rationale": "Respond to competitor pressure on high-value home",
         "observed_outcome": "Conversion improved ~1pt, volume up modestly"},
        {"date": "2025-08-20", "segment": "Motor 50+", "action": "increase",
         "rate_change_pct": 2.0,
         "rationale": "Small inflationary adjustment on a low-risk book",
         "observed_outcome": "Negligible conversion impact"},
        {"date": "2025-10-05", "segment": "Van Commercial", "action": "increase",
         "rate_change_pct": 5.0,
         "rationale": "Rising repair-cost inflation on commercial vehicles",
         "observed_outcome": "Conversion dipped ~2pts, loss ratio stabilised"},
    ]
    return pd.DataFrame(rows)


# Customer verbatims, tagged by segment. Motor 25-34 has a visible
# cluster of renewal-price complaints.
FEEDBACK = [
    ("Motor 25-34", "renewal", "My renewal jumped again and I've had no claims — I'm shopping around this year."),
    ("Motor 25-34", "renewal", "Price went up at renewal for no reason I can see. Found cheaper elsewhere in minutes."),
    ("Motor 25-34", "renewal", "Loyal for three years and the renewal price is a joke. Feels like being punished for staying."),
    ("Motor 25-34", "quote", "Actually your quote came out cheapest for my age group, pleasantly surprised."),
    ("Motor 25-34", "renewal", "Another year, another renewal hike. Why does it keep climbing?"),
    ("Motor 25-34", "quote", "Cheapest I found as a young driver, signed up straight away."),
    ("Motor 25-34", "claim", "Claim was handled fine but the renewal afterwards was brutal."),
    ("Motor 25-34", "renewal", "Renewal price crept up 15%. No accidents, no tickets. Makes no sense."),
    ("Motor 25-34", "service", "App is easy to use, no complaints there."),
    ("Motor 35-49", "renewal", "Renewal was reasonable this year, stayed put."),
    ("Motor 35-49", "quote", "Middle of the pack on price but the cover was better."),
    ("Motor 35-49", "claim", "Windscreen claim sorted quickly, happy enough."),
    ("Motor 35-49", "service", "Call centre wait was long but the agent was helpful."),
    ("Motor 35-49", "renewal", "Slight increase at renewal, still competitive so I stayed."),
    ("Motor 50+", "renewal", "Been with you years, price is fair, no reason to move."),
    ("Motor 50+", "quote", "Good value for a safe driver, easy sign-up."),
    ("Motor 50+", "service", "Friendly service, sorted my mid-term change with no fuss."),
    ("Motor 50+", "claim", "Bumper claim handled smoothly, would recommend."),
    ("Motor 50+", "renewal", "Renewal held steady, appreciate the consistency."),
    ("Home Standard", "renewal", "Home renewal was flat this year, all good."),
    ("Home Standard", "quote", "Decent price for buildings and contents together."),
    ("Home Standard", "claim", "Leak claim took a while to assess but paid out fairly."),
    ("Home Standard", "service", "Website was down when I tried to update my address."),
    ("Home Standard", "renewal", "Small increase, still cheaper than the comparison sites."),
    ("Home Standard", "quote", "Straightforward quote, no hidden extras."),
    ("Home Premium", "renewal", "High-value home cover renewed with a small drop, nice surprise."),
    ("Home Premium", "quote", "Competitive on premium contents, good limits."),
    ("Home Premium", "claim", "Jewellery claim handled discreetly and quickly."),
    ("Home Premium", "service", "Dedicated line for premium customers is a nice touch."),
    ("Home Premium", "renewal", "Renewal came down slightly this year, no complaints."),
    ("Van Commercial", "renewal", "Van renewal went up with repair costs, understandable but stings."),
    ("Van Commercial", "quote", "Priced a bit high for my trade but cover was solid."),
    ("Van Commercial", "claim", "Tools theft claim paid out, saved my business honestly."),
    ("Van Commercial", "service", "Took two calls to get my fleet details changed."),
    ("Van Commercial", "renewal", "Commercial renewal keeps climbing, watching the market now."),
    ("Motor 25-34", "renewal", "Third year running the renewal has gone up. Time to switch."),
    ("Motor 25-34", "quote", "Beat every other quote for a 26-year-old, thanks."),
    ("Home Standard", "service", "Easy to reach someone, sorted in one call."),
    ("Motor 50+", "quote", "Best price for my no-claims history, very happy."),
    ("Van Commercial", "quote", "Reasonable for a new van, decent breakdown cover included."),
]


def build_feedback() -> pd.DataFrame:
    """Customer feedback verbatims, tagged by segment, dated across the window."""
    rows = []
    for i, (seg, channel, text) in enumerate(FEEDBACK):
        # Spread the dates across the 12 months deterministically.
        month = MONTHS[i % len(MONTHS)]
        rows.append({
            "feedback_id": f"FB{i + 1:03d}",
            "segment": seg,
            "date": f"{month}-15",
            "channel": channel,
            "verbatim": text,
        })
    return pd.DataFrame(rows)


# Market intelligence markdown docs with segment tags for the retriever.
MARKET_DOCS = {
    "competitor_motor_young_driver.md": """---
segment: Motor 25-34
source: Competitor Watch — Weekly Digest
date: 2026-03-02
---

# Directwise cuts young-driver headline rates

Directwise has launched an aggressive new-business campaign on the 25-34 motor
segment, cutting headline premiums by roughly 10% since February. Trade press
suggests the cut is paired with tighter underwriting criteria on the highest-risk
young drivers — i.e. cheap headline pricing to win the cleaner risks while
declining the worst.

Market read: rivals are actively repricing and re-selecting risk in this segment.
Insurers that hold rates flat may find themselves absorbing the risks competitors
are shedding, without the premium to match.
""",
    "reinsurance_motor.md": """---
segment: Motor 25-34
source: Reinsurance Market Note
date: 2026-02-18
---

# Motor bodily-injury severity trending up

Reinsurers are flagging continued upward pressure on motor bodily-injury claim
severity, driven by care-cost inflation and longer settlement times. Young-driver
books are disproportionately exposed given higher accident frequency.

Implication: expect claims severity on younger motor cohorts to keep drifting up
into 2026; pricing that was adequate 12 months ago may no longer be.
""",
    "home_market_stable.md": """---
segment: Home Premium
source: Household Market Commentary
date: 2026-01-25
---

# High-value home market remains competitive but stable

The premium household segment is competitive but broadly rational. Weather-related
claims have been benign over the period and no major competitor repricing has been
observed. Modest rate reductions are being used tactically to defend retention on
high-value books.
""",
    "regulatory_consumer_duty.md": """---
segment: all
source: Regulatory Briefing
date: 2026-02-10
---

# Consumer Duty and fair value in pricing

Under FCA Consumer Duty and the general insurance pricing rules, renewal pricing
must deliver fair value and cannot systematically penalise loyal customers relative
to new business ('price walking' remains prohibited). Any rate action — especially
increases at renewal — must be justifiable on a cost/risk basis and evidenced.

Implication: a risk-justified rate correction is defensible; an increase that looks
like it targets inertia is not. Human sign-off on the rationale is required.
""",
}


def write_all():
    """Generate every file into ./data. Safe to re-run — it overwrites cleanly."""
    os.makedirs(os.path.join(DATA_DIR, "market_intel"), exist_ok=True)

    build_claims().to_csv(os.path.join(DATA_DIR, "claims_performance.csv"), index=False)
    build_conversion().to_csv(os.path.join(DATA_DIR, "conversion_performance.csv"), index=False)
    build_competitor().to_csv(os.path.join(DATA_DIR, "competitor_pricing.csv"), index=False)
    build_pricing_actions().to_csv(os.path.join(DATA_DIR, "previous_pricing_actions.csv"), index=False)
    build_feedback().to_csv(os.path.join(DATA_DIR, "customer_feedback.csv"), index=False)

    for filename, content in MARKET_DOCS.items():
        with open(os.path.join(DATA_DIR, "market_intel", filename), "w") as f:
            f.write(content)

    print("✅  Generated synthetic dataset in ./data")
    print("   • claims_performance.csv       (72 rows — 6 segments × 12 months)")
    print("   • conversion_performance.csv   (72 rows)")
    print("   • competitor_pricing.csv       (216 rows — 3 competitors)")
    print("   • previous_pricing_actions.csv (5 rows)")
    print("   • customer_feedback.csv        (40 verbatims)")
    print(f"   • market_intel/               ({len(MARKET_DOCS)} tagged .md documents)")


if __name__ == "__main__":
    write_all()
