<div align="center">

# 🔎 Investigation Certificates

### Your RAG pipeline can't tell you what it *didn't* retrieve. This one proves it saw everything.

[![License: MIT](https://img.shields.io/badge/License-MIT-A371F7.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![TigerGraph](https://img.shields.io/badge/TigerGraph-Savanna%20%2B%20GSQL-f7941e?style=flat-square)](agrag/graph/queries.gsql)
[![TypeSafe Jev](https://img.shields.io/badge/Decision%20Model-TypeSafe%20Jev%20System%20One-blue?style=flat-square)](agrag/decision.py)
[![Accuracy](https://img.shields.io/badge/accuracy-99%25%20vs%2043%25-3fb950?style=flat-square)](results/summary.json)
[![Tests](https://img.shields.io/badge/tests-65%20offline-3fb950?style=flat-square)](tests/)
[![Hackathon](https://img.shields.io/badge/TigerGraph-Agentic%20GraphRAG%20Hackathon-0d1117?style=flat-square)](docs/hackathon-brief.md)

**[How it works](#how-it-works)** · **[Architecture](#architecture)** · **[The benchmark](#the-benchmark)** · **[JEV addition](#-jev-addition)** · **[The certificate](#the-certificate)** · **[Try it](#try-it)**

</div>

<div align="center">
  <img src="docs/assets/demo.gif" alt="One benchmark question answered by three pipelines. RAG sees 4 of 43 matching events and answers 1; GraphRAG sees 12 and answers 2; the agentic pipeline counts all 43 and answers 20, the gold answer." width="880">
</div>

---

## The question nobody's pipeline can answer honestly

> *How many athletics events at the 2004 Summer Olympics had more than 41 competitors?*

Plain RAG retrieves the top 4 chunks, reads them, and confidently answers **1**. GraphRAG expands one hop, sees 12 chunks, and confidently answers **2**.

The real answer is **20**. There were **43** matching events in the graph.

Both pipelines were wrong for the same reason, and it isn't the model: **similarity search retrieves what looks relevant, and counting questions need everything that *is* relevant.** No value of `k` fixes that. Worse, neither pipeline had any way to know it was working from 9% of the evidence — it reports the same confident tone either way.

So stop grading the answer and start proving the evidence.

## How it works

The graph already knows how many events match `sport = athletics AND games = 2004 Summer`. It's a `COUNT`. That number is a **structural bound** — the size of the complete answer set, straight from TigerGraph.

Every answer this pipeline emits ships an **Investigation Certificate**: a JSON record putting the structural bound next to the evidence the agent actually inspected. If they match, the answer is provably complete. If they don't, the certificate says so instead of hiding it.

A router sorts each question into a completeness class and picks the cheapest tool that can satisfy it — the LLM is called only to break a tie or recover a missing field, never to count:

| Completeness class | Question types | How it's answered | Certificate proves |
|---|---|---|---|
| **existential** | lookup, multi-hop | exact match / venue+date traversal | the entity resolved uniquely |
| **chained** | temporal | `PREV`/`NEXT` hop chain | the hop sequence was actually walked |
| **exhaustive** | aggregation, superlative | GSQL structural scan: `COUNT` + every match | evidence set **==** the graph's own count |

The router combines five benchmark-scoped templates with **Jev System One** fallback classification so non-templated natural queries preserve their structural guarantees. Anything completely unclassifiable falls through to a GraphRAG baseline rather than guessing.

## Architecture

```mermaid
flowchart LR
    Q[Question] --> R[Router: 5 Regex Templates]
    R -->|matched| C{Completeness Class}
    R -->|non-template| J1[Jev System One: Intent Router]
    J1 -->|classified| C
    J1 -->|unclassified| F[GraphRAG fallback]

    C -->|existential| T1[exact match / venue+date traversal]
    C -->|chained| T2[PREV hop chain]
    C -->|exhaustive| T3[GSQL structural scan: COUNT + all events]

    T1 & T2 & T3 --> E[Evidence evaluator]
    E -->|complete| A[Answer + Certificate]
    E -->|ambiguous tie| J2[Jev System One: Typed Disambiguation]
    J2 -->|resolved| A
    J2 -->|gap/fallback| L[LLM: disambiguate / extract] --> A

    subgraph TigerGraph Savanna
        G[(Event / Games / Venue / Athlete graph)]
        V[(Chunk vectors)]
    end
    T1 & T2 & T3 --> G
    F --> V
```

Exported image (for the demo video / slide deck / hackathon submission form): [`docs/diagrams/architecture.png`](docs/diagrams/architecture.png) (also available as [`.svg`](docs/diagrams/architecture.svg)). Rendered from the mermaid source above via `docs/diagrams/architecture.mmd`. Full architectural rationale: [`docs/architecture-jev.md`](docs/architecture-jev.md).

Three pipelines answer the same 150 benchmark questions (100 public with gold answers, 50 hidden):

- **RAG** — vector top-k over text chunks, one LLM call.
- **GraphRAG** — top-k chunks plus a fixed one-hop graph expansion (PREV/NEXT/same-venue), one LLM call.
- **Agentic** — routes by completeness class (with Jev System One fallback), uses cheap deterministic graph queries where they suffice, calls Jev or the LLM only to disambiguate ties or recover a missing field, and emits a certificate.

## 🆒 JEV addition

Integrated via PR #1, TypeSafe AI's **Jev System One** decision model ([`agrag/decision.py`](agrag/decision.py)) brings fast, non-autoregressive typed decisions into the TigerGraph Agentic GraphRAG pipeline. Instead of relying solely on heavy generative text models, Jev provides discrete decision intelligence consumed directly by software control flow at zero output token cost. Complete technical specifications are detailed in [`JEV.md`](JEV.md) and [`docs/architecture-jev.md`](docs/architecture-jev.md).

### Capabilities

1. **Non-Template Intent Classification**
   - The primary router uses rigid regex heuristics over benchmark phrasing. Arbitrary natural language questions outside template syntax previously defaulted to unverified RAG.
   - Jev System One acts as an intent classifier across the five evidential classes (`lookup`, `multi_hop`, `temporal`, `aggregation`, `superlative`), achieving **100% accuracy (5/5)** on un-templated test queries with zero output tokens generated.

| Test Query | Regex Router | Jev System One Route | Result |
|---|---|---|---|
| *How many countries participated in the 1996 Summer Olympics?* | `unknown` (Unverified RAG) | `lookup` (existential) | **Pass** (conf=1.00) |
| *Tell me who took the gold medal at ExCeL on 30 July 2012* | `unknown` (Unverified RAG) | `multi_hop` (existential) | **Pass** (conf=1.00) |
| *Which athlete won the 100m sprint right before the 2012 Games?* | `unknown` (Unverified RAG) | `temporal` (chained) | **Pass** (conf=1.00) |
| *Find the number of wrestling events in 2004 with more than 20 competitors* | `unknown` (Unverified RAG) | `aggregation` (exhaustive) | **Pass** (conf=1.00) |
| *What was the largest event by participant count in 2008?* | `unknown` (Unverified RAG) | `superlative` (exhaustive) | **Pass** (conf=1.00) |

2. **Zero-Token Typed Candidate Disambiguation**
   - Resolves multi-candidate graph collisions (such as `pub-060` with three fencing/judo events on 30 July 2012 at ExCeL) by evaluating structured candidate criteria.
   - Outputs discrete document selections with calibrated confidence and probabilities, avoiding the hallucination and token overhead of generative models.

3. **Verifiable Certificate Step Logging**
   - Every Jev decision emits structured telemetry into the **Investigation Certificate**:

```json
{
  "tool": "jev_disambiguate",
  "note": "Jev (jev-latest) 3 candidates -> Q1064016 (conf=0.98)",
  "tokens": {"input": 182, "output": 0},
  "latency_ms": 312
}
```

### Jev Quickstart & Benchmarking

Configure in `.env`:
```env
JEV_API_KEY=your_typesafe_jev_key
JEV_MODEL=jev-latest
JEV_TIMEOUT_MS=5000
```

Run the benchmark and decision model tests:
```bash
python scripts/benchmark_jev.py          # benchmark intent routing and pub-060 disambiguation
python -m pytest tests/test_decision_jev.py # verify unit tests for mock and live Jev models
```

## The benchmark

<div align="center">
  <img src="docs/assets/results.png" alt="Accuracy by question type for RAG, GraphRAG and the agentic pipeline. Both baselines score 0% on aggregation and superlative; the agentic pipeline scores 100%. Overall 99% accuracy at 18 tokens and 0.41s per answer." width="880">
</div>

Three pipelines, the same 100 public questions, the same live TigerGraph graph and the same live Groq model. The whole thesis is in two rows of that chart:

- **`lookup` is where plain RAG works** — 84%, because one relevant chunk is usually retrievable by similarity. This is the shape of question RAG was designed for.
- **`aggregation` and `superlative` are where it collapses** — **0%** for RAG *and* GraphRAG. Not "low": zero out of 31. These need all 8–43 matching events, and top-k cannot produce an exhaustive set at any `k`.
- **The agentic pipeline answers those deterministically** — a `COUNT` and a scan over the exact filter predicate instead of a guess from a handful of chunks. 100% on both.

And it is *cheaper*, not more expensive, because the expensive part was never the graph — it was stuffing chunks into a context window:

| | RAG | GraphRAG | **Agentic** |
|---|---|---|---|
| Accuracy | 25% | 43% | **99%** |
| Tokens per answer | 1,389 | 2,203 | **18** |
| Latency per answer | 9.3s | 15.5s | **0.41s** |

That's **~120× fewer tokens** and **~38× faster** than the GraphRAG baseline, at more than double the accuracy. Full per-type numbers in [`results/summary.json`](results/summary.json); raw per-question traces, including every certificate, in [`results/`](results/).

The single agentic miss is honest and instructive: `pub-060`, a date+venue tie among 37 candidate events at ExCeL. Jev System One made the pick (conf=0.90) and chose the wrong event, so Jev did not fix this case; the LLM guessed wrong here in the earlier run too. Its certificate reports `pass_with_llm_recovery`, not a clean deterministic `pass` — because the certificate measures *"was the evidence complete,"* not *"was the answer right."* Those are deliberately different axes, and this is exactly where they diverge.

## The certificate

```json
{
  "qid": "pub-045",
  "completeness_class": "exhaustive",
  "retrieval_mode": "structural_scan",
  "predicate": { "sport": "Athletics", "games": "2004 Summer", "threshold": 41 },
  "structural_bound": 43,
  "evidence_set_size": 43,
  "completeness_check": "pass",
  "stop_reason": "structural_bound_met",
  "tokens": { "total": 0 },
  "latency_ms": 227
}
```

`structural_bound` came from the graph. `evidence_set_size` is what the agent actually looked at. They match, so the answer is complete — and you can check that yourself without trusting the model.

## Try it

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest          # 65 tests, in-memory backend, no TigerGraph needed
```

Point it at a live graph by copying `.env.example` to `.env` and filling in `TG_HOST` / `TG_GRAPH` / `TG_SECRET` (a TigerGraph Savanna workspace), `GROQ_API_KEY` (free, no card), and optionally `JEV_API_KEY`. Then:

```bash
python scripts/tg_setup.py --schema --queries   # install schema + GSQL queries
python scripts/tg_load.py                       # ~2,951 docs, ~20k chunks
python scripts/reconcile.py --backend tigergraph

python -m agrag.eval.run --pipeline rag       --backend tigergraph
python -m agrag.eval.run --pipeline graphrag  --backend tigergraph
python -m agrag.eval.run --pipeline agentic   --backend tigergraph
python -m agrag.eval.report

streamlit run dashboard/app.py                  # side-by-side comparison
```

Completeness was validated, not assumed: reconciliation ran twice — once locally against the parsed corpus before any agent code existed, and again against the loaded graph — and both records are committed in [`data/`](data/).

## Built with

Demo: https://www.loom.com/share/6c4edb59ed714f1694d1bc65cca2d934  
**TigerGraph Savanna** (graph + vector attributes, GSQL structural scans) · **TypeSafe AI Jev** (System One non-autoregressive decision model) · **Groq** for the few LLM calls that survive routing · **Streamlit** for the comparison dashboard. Built for the TigerGraph Agentic GraphRAG Hackathon.

Deeper reading: [architecture diagram](docs/diagrams/architecture.png) · [Jev architecture explanation](docs/architecture-jev.md) · [Jev technical notes](JEV.md) · [idea spec & council review](docs/idea-spec.md) · [literature scan](docs/research-scan-agentic-graphrag.md) · [hackathon brief](docs/hackathon-brief.md) · [build status](docs/status.md)

<div align="center">
<br>
<b>Investigation Certificates</b> — prove the evidence, don't just trust the answer.
</div>
