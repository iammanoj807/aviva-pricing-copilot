# Aviva Pricing Analyst Copilot

Multi-agent copilot for pricing analysts. Ask it about a segment
(e.g. "Should we take any pricing action on Motor 25-34?") and it pulls together
claims trends, conversion data, and market intel, then proposes an action.
It never decides on its own. Every recommendation needs an analyst to sign it off.

Built for the Aviva GenAI case study (Challenge B).

## The problem it targets

Pricing analysts spend most of the day pulling numbers from different systems before
they can even start thinking. This copilot handles the gathering and first-pass
synthesis. The analyst keeps the decision.

## What it does on one question

```
                              question
                                 │
                          ┌──────────────┐
                          │   intake     │  resolve which segment
                          └──────┬───────┘
             ┌───────────────────┼───────────────────┐    round 1: parallel
        ┌─────────┐        ┌────────────┐      ┌────────────┐
        │ Claims  │        │ Conversion │      │  Market    │
        │ (pandas)│        │  (pandas)  │      │ (retrieval)│
        └────┬────┘        └─────┬──────┘      └─────┬──────┘
             └───────────────────┼───────────────────┘
                          ┌──────────────┐
                          │  inspect     │  supervisor reads the findings
                          └──────┬───────┘
                     deterioration?  ──── no ───────────┐
                                 │ yes                   │
                          ┌──────────────┐               │
                          │ market_round2│  re-query     │   round 2 only
                          │ (targeted)   │  with a focus  │   when warranted
                          └──────┬───────┘               │
                                 └───────────┬───────────┘
                          ┌──────────────┐
                          │ recommend    │  synthesise → Pydantic recommendation
                          └──────┬───────┘
                          ┌──────────────┐
                          │ guardrail    │  drop unsupported claims, force human sign-off
                          └──────┬───────┘
                              recommendation (propose-only)
```

The key bit is the `inspect` → `market_round2` branch. Round 1 is a plain
parallel sweep. If Claims flags a loss-ratio deterioration, the supervisor sends
Market back with a targeted question ("look for competitor rate cuts that explain this")
instead of leaving it with its generic first pass. So one agent's output actually
changes what another goes looking for.

## Two things to know before reading the code

**No vector database.** Four market notes and forty verbatims. It all fits in a prompt,
so there's nothing for semantic search to solve here. Retrieval is just tag-filtering on
a `segment` field. There's a `Retriever` interface so a vector backend can slot in later
if the corpus grows. See `retriever.py`.

**pandas computes, the LLM narrates.** Loss ratios, trends, conversion rates are all
calculated in pandas. The LLM gets the finished figures and puts them into English.
It never does arithmetic. LLMs are bad at maths and confident about it, which is a
problem when you're making pricing decisions.

## Running it

Everything runs against a local [Ollama](https://ollama.com) model — no API key, nothing
network-dependent to fail.

```bash
# 1. model
ollama pull qwen2.5:14b

# 2. deps
pip install -r requirements.txt

# 3. generate the synthetic data (writes into ./data)
python data_generator.py

# 4a. the UI
streamlit run app.py

# 4b. or the JSON API
uvicorn api:app --reload      # docs at http://localhost:8000/docs

# 5. the eval harness (writes EVAL_RESULTS.md)
python evaluate.py
```

Model and host are overridable in `.env` (see `.env.example`); the defaults work as-is.

## The data (all synthetic, one deliberate story)

The generator plants one story in **Motor 25-34**: loss ratio climbing 62% → 74%
over six months while conversion rises 17% → 24%, a competitor cutting young-driver
rates, and renewal complaints clustering in that segment. That's adverse selection
(we've become the cheap option for a worsening book). No single agent can spot it alone.
The other five segments are stable controls that shouldn't trigger a second round.

| File | Feeds | What it is |
|------|-------|------------|
| `data/claims_performance.csv` | Claims | loss ratio / frequency / severity, 12 months × 6 segments |
| `data/conversion_performance.csv` | Conversion | conversion rate + avg premium |
| `data/previous_pricing_actions.csv` | Conversion | past rate changes + outcomes (elasticity read) |
| `data/competitor_pricing.csv` | Market | competitor premiums + rank |
| `data/customer_feedback.csv` | Market | ~40 verbatims, tagged by segment |
| `data/market_intel/*.md` | Market | 4 market/regulatory notes, tagged by segment |

## Layout

```
graph.py         supervisor graph — nodes, edges, the conditional second round
agents/          the four specialists (claims, conversion, market, recommendation)
retriever.py     Retriever interface + tag-filter implementation
guardrails.py    last check before display: drop unsupported claims, enforce propose-only
schemas.py       Pydantic contracts passed between agents
llm.py           single place the model is constructed
app.py           Streamlit UI, built around the execution trace
api.py           same graph behind a FastAPI endpoint
evaluate.py      golden-set scoring + data-drift check
data_generator.py  regenerates everything in ./data
golden_set.json  14 questions with known-correct answers
```
