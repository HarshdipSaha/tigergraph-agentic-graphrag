# Investigation Certificates for Agentic GraphRAG

Built for the TigerGraph Agentic GraphRAG Hackathon (Unstop). Full rules and dataset facts in `docs/hackathon-brief.md`; the idea, its literature grounding, and the LLM-council review that shaped it are in `docs/idea-spec.md` and `docs/research-scan-agentic-graphrag.md`.

## The pitch

Every answer the agentic pipeline gives ships an **Investigation Certificate**: a JSON record proving, using the graph's own structure, that the evidence behind the answer is as complete as the question requires — not just that the final answer looks right. For counting and superlative questions (which need every matching event, not just a plausible few), the certificate reports TigerGraph's own structural bound (a `COUNT` over the exact filter predicate) next to the evidence set the agent actually inspected. Plain RAG and vanilla GraphRAG have no way to know they only saw 8 of 41 matching events; this pipeline proves it did or didn't.

**Scoping note, stated plainly rather than discovered under questioning:** the question router is a benchmark-scoped heuristic — a template matcher over this hackathon's five known question types, not a general-purpose learned router. It works because this corpus's completeness is structurally expressible (an exact `sport + games` filter bounds the answer set). That property doesn't hold for every corpus, and the write-up doesn't claim otherwise.

## Architecture

```mermaid
flowchart LR
    Q[Question] --> R[Router: 5 templates -> completeness class]
    R -->|existential| T1[exact match / venue+date traversal]
    R -->|chained| T2[PREV hop chain]
    R -->|exhaustive| T3[GSQL structural scan: COUNT + all events]
    R -->|unknown| F[GraphRAG fallback]
    T1 & T2 & T3 --> E[Evidence evaluator]
    E -->|complete| A[Answer + Certificate]
    E -->|gap| L[LLM: disambiguate / extract] --> A
    subgraph TigerGraph Savanna
        G[(Event / Games / Venue / Athlete graph)]
        V[(Chunk vectors)]
    end
    T1 & T2 & T3 --> G
    F --> V
```

Three pipelines answer the same 150 benchmark questions (100 public with gold answers, 50 hidden):

- **RAG** — vector top-k over text chunks, one LLM call.
- **GraphRAG** — top-k chunks plus a fixed one-hop graph expansion (PREV/NEXT/same-venue), one LLM call.
- **Agentic** — routes by completeness class, uses cheap deterministic graph queries where they suffice, calls the LLM only to disambiguate ties or recover a missing field, and emits a certificate.

## Reproduction

```bash
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py          # or use the data/ files already in this repo
```

Copy `.env.example` to `.env` and fill in:
- `TG_HOST`, `TG_GRAPH`, `TG_SECRET` — a TigerGraph Savanna workspace with the graph and vector attribute created (`agrag/graph/schema.gsql`) and the queries installed (`agrag/graph/queries.gsql`) — paste both into the workspace's Query Editor, or use `python scripts/tg_setup.py --schema` / `--queries`.
- `GROQ_API_KEY` — free, no card required, at https://console.groq.com/keys (default provider; `.env` also supports switching to Anthropic).

Then:

```bash
python scripts/tg_smoke.py                              # confirm the connection
# If tg_smoke.py fails with a 500/502, the Savanna free-tier workspace has likely auto-suspended
# (it does this after ~60 min idle). Resume it in the Savanna dashboard, then:
python scripts/tg_wait.py                               # polls until it's back, or times out
python scripts/tg_load.py                                # load corpus.jsonl into the graph (~2,951 docs, ~20k chunks)
python scripts/reconcile.py --backend tigergraph          # council-mandated completeness check (see docs/idea-spec.md §6)
python -m agrag.eval.run --pipeline rag --backend tigergraph
python -m agrag.eval.run --pipeline graphrag --backend tigergraph
python -m agrag.eval.run --pipeline agentic --backend tigergraph
python -m agrag.eval.run --pipeline rag --backend tigergraph --questions data/eval_hidden.jsonl
python -m agrag.eval.run --pipeline graphrag --backend tigergraph --questions data/eval_hidden.jsonl
python -m agrag.eval.run --pipeline agentic --backend tigergraph --questions data/eval_hidden.jsonl
python -m agrag.eval.report
streamlit run dashboard/app.py
```

Run the test suite (no TigerGraph needed — everything except `tests/integration/` runs against an in-memory backend):

```bash
python -m pytest
```

## Certificate schema (v0)

```json
{
  "qid": "pub-045",
  "qtype": "aggregation",
  "completeness_class": "exhaustive",
  "retrieval_mode": "structural_scan",
  "predicate": {"sport": "Athletics", "games": "2004 Summer", "threshold": 41},
  "structural_bound": 41,
  "evidence_set_size": 41,
  "completeness_check": "pass",
  "docs_inspected": ["Q1", "Q2", "..."],
  "steps": [{"tool": "aggregation_scan", "note": "bound=41; regex-recovered=[]"}],
  "tokens": {"input": 0, "output": 0, "context": 0, "total": 0},
  "latency_ms": 340,
  "stop_reason": "structural_bound_met"
}
```

Every question class gets a certificate, not just counting questions — `existential` (lookup/multi-hop) certifies exact-match resolution, `chained` (temporal) certifies the PREV/NEXT hop sequence was actually walked, `exhaustive` (aggregation/superlative) certifies the evidence set equals the graph's own structural count.

## Reconciliation

Reconciliation ran twice, per `docs/idea-spec.md` §6: locally against the parsed corpus before any agent/orchestrator code existed (`data/reconciliation-local.json`), and again against the loaded TigerGraph graph after loading (`data/reconciliation-tigergraph.json`). Both are committed as evidence the completeness claim was validated, not assumed.

## Results (live TigerGraph + live Groq, public set, n=100 per pipeline)

| Pipeline | Accuracy | Avg tokens/answer | lookup | multi_hop | temporal | aggregation | superlative |
|---|---|---|---|---|---|---|---|
| RAG | 25% | 1,389 | 84% (16/19) | 11% (3/28) | 27% (6/22) | **0%** (0/21) | **0%** (0/10) |
| GraphRAG | 43% | 2,203 | 89% (17/19) | 32% (9/28) | 77% (17/22) | **0%** (0/21) | **0%** (0/10) |
| **Agentic** | **99%** | **42** | 100% | 96% (27/28) | 100% | 100% | 100% |

This is the whole thesis in one table. `lookup` (a single fact) is the one question type plain RAG handles reasonably — 84% correct, because a single relevant chunk is usually retrievable by similarity. GraphRAG's one-hop expansion (PREV/NEXT/same-venue facts) roughly doubles `multi_hop` and nearly triples `temporal` accuracy over RAG, since those facts are often already sitting in the immediate neighbourhood of a similarity hit. But **neither baseline gets a single `aggregation` or `superlative` question right** — these need every matching event for a `sport + games` filter (8–43 documents in this dataset), and no top-k similarity search, however the k is tuned, retrieves an exhaustive set. The agentic pipeline answers those with a deterministic `COUNT`/scan against the exact filter predicate instead of guessing from a handful of chunks, at **zero LLM tokens** for every question type except `multi_hop` (where venue+date ties occasionally need a disambiguating LLM call).

The one agentic miss (`multi_hop`, 27/28) is `pub-060` — a three-way date tie among fencing events at ExCeL, where disambiguation fell to the LLM and it guessed wrong. Its certificate honestly reports `pass_with_llm_recovery` rather than a deterministic `pass`, which is the point: the certificate's 100% pass rate measures "was the evidence-completeness check satisfied," not "was the answer correct" — those are deliberately different axes, and this case is exactly where they can diverge.

Full per-question-type numbers: `results/summary.json`. Raw per-question traces (answer, tokens, and for the agentic pipeline, the full certificate) for both the 100 public and 50 hidden questions: `results/*_public.jsonl` and `results/*_hidden.jsonl`.

## Repo layout

- `agrag/` — the package: infobox parsing, router, deterministic tools, certificate model, three pipelines, TigerGraph backend.
- `agrag/graph/` — GSQL schema and queries, the only place that talks to `pyTigerGraph`.
- `scripts/` — one-shot operational scripts (data download, TigerGraph setup/load/smoke test, reconciliation).
- `tests/` — unit tests against an in-memory backend; `tests/integration/` needs a live `TG_HOST`.
- `dashboard/` — Streamlit comparison dashboard.
- `docs/` — hackathon rules, literature scan, idea spec (with the LLM council's verdict), build status.
