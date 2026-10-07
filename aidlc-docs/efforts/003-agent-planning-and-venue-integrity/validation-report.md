# Effort 003 Validation Report

**Validated:** 2026-10-07  
**Branch:** `improvement-agentic-planner` in the isolated worktree  
**Graph:** live TigerGraph Savanna `OlympicsRAG`  
**LLM:** Groq `openai/gpt-oss-20b`, using the owner's private five-key same-organization pool. Credential values are excluded from logs, source, and results.

## Implementation

- `agrag/infobox.py` selects the Olympic Event infobox even when another infobox precedes it, reads the `dates` field fallback, and handles the source's Unicode/legacy encoded title delimiters.
- `agrag/tools.py` rejects empty venue/date evidence, bounds venue aliases, applies hard date eligibility and interval handling, and applies sport only when it is grounded in the question. The certificate records candidate provenance and ambiguity.
- `agrag/planner.py` validates a typed JSON action against an allowlist and the question. `AgenticPipeline` executes only validated graph tools, bounds planner calls, records selected tool and arguments, and keeps provider failures from triggering another generative fallback call. `--agent-mode template` preserves deterministic routing for comparison.
- `agrag/llm.py` applies a request deadline and rotates same-organization keys after rate-limit/authentication failures. Rotation health state is stored in ignored `data/.groq_key_state.json`; the `.env` itself is ignored and not included in branch output.
- Certificate metadata distinguishes unique support from an unresolved model choice. Report aggregation accepts explicit input paths and rejects duplicate pipeline/qid rows. The dashboard shows current planner/template results separately from the historical RAG/GraphRAG comparison.
- `scripts/tg_refresh_events.py` updates Event vertices and links without rebuilding Chunk vertices or embeddings.

## Live graph and reconciliation

- Parsed corpus: **2,187 Event IDs**; live TigerGraph: **2,187 Event IDs**; missing: **0**; unexpected: **0**.
- `Q942805` is present with sport Tennis, venue Olympic Tennis Centre, date `15 to 22 August 2004`, and gold `Li TingSun Tiantian`.
- Chunk vertices and embeddings were not rebuilt.
- Local reconciliation: **98/100 answer matches, 100/100 evidence checks**.
- TigerGraph reconciliation: **98/100 answer matches, 100/100 evidence checks**.
- Both reconciliations have the same two mismatch/LLM-needed questions: `pub-060` and `pub-099`.

Artifacts: `data/reconciliation-local-planner.json` and `data/reconciliation-tigergraph-planner.json`.

## Current agentic public comparison

Both 100-question runs used the same refreshed TigerGraph Event graph and Groq model.

| Metric | Template mode | LLM planner mode |
|---|---:|---:|
| Answer matches | 99/100 (99%) | 98/100 (98%) |
| Mean tokens | 14.27 | 1,148.46 |
| Median tokens | 0 | 817.5 |
| Mean latency | 414.16 ms | 1,768.87 ms |
| Median latency | 276.5 ms | 1,383 ms |
| LLM-selected graph tool actions | 0 | 86 |
| Certificate pass rate | 98% | 98% |
| Unverified ties | 2 | 2 |

Planner per-class matches: lookup 19/19; multi-hop 26/28; temporal 22/22; aggregation 21/21; superlative 10/10. Planner status counts: 86 selected, 11 ungrounded action, 3 invalid JSON. Certificate counts: 83 `pass`, 15 `pass_with_fallback`, and 2 `unverified`.

The planner demonstrates model-selected tool use but does **not** improve measured answer accuracy or efficiency over template mode in this run. Template mode makes model selections on both unresolved questions: it misses `pub-060` and happens to match `pub-099`; both certificates remain `unverified`. Planner mode abstains on both unresolved cases. For `pub-060`, three exact venue/date Events remain and the question names no sport, so neither the tie nor its certificate may be presented as uniquely resolved.

Raw outputs: `results/agentic_planner_public.jsonl` and `results/agentic_template_public.jsonl`. Mode summaries: `results/summary_agentic_planner.json` and `results/summary_agentic_template.json`.

## Hidden run

The final planner run contains **50/50 structurally valid rows**. It has 49 non-null answers; this count is descriptive only and is not an accuracy measure. Gold labels are unavailable, so **no hidden accuracy is reported**. Mean usage is 935.6 tokens and 1,441.3 ms per question (median 776 tokens and 1,051 ms). Certificate counts are 45 `pass`, 3 `pass_with_fallback`, 1 `fail`, and 1 `unverified`. Planner status counts are 46 selected, 2 provider errors, 1 ungrounded action, and 1 invalid JSON; fallback behavior is preserved in the trace.

For `eval-001`, the trace selects `resolve_multi_hop` with the grounded Olympic Tennis Centre/date/year arguments. The deterministic query returns `Li TingSun Tiantian` from `Q942805`, and the certificate is `pass` with one eligible Event. This validates the specific source/evidence path, not hidden-set answer accuracy.

Raw output: `results/agentic_planner_hidden.jsonl`.

## Tests and commands

The fresh full run `python -m pytest -q` exited successfully after the live graph refresh. It covers **120 offline tests plus four live TigerGraph integration tests**. One integration fixture was updated: `Q1183979` is now correctly an Event because of the parser repair, so the non-Event vertex-type regression now uses film document `Q1520721`.

Key checks performed:

```powershell
python -m pytest -q -m "not integration"
python -m pytest -q
python scripts/tg_refresh_events.py
python scripts/reconcile.py --backend local --out data/reconciliation-local-planner.json
python scripts/reconcile.py --backend tigergraph --out data/reconciliation-tigergraph-planner.json
python -m agrag.eval.report --public results/agentic_planner_public.jsonl --out results/summary_agentic_planner.json
python -m agrag.eval.report --public results/agentic_template_public.jsonl --out results/summary_agentic_template.json
```

RAG and GraphRAG were not rerun after the Event refresh. Their existing 25%/43% results are labeled historical and are not a same-snapshot comparison with the October measurements. No code was committed, merged, or pushed; the isolated branch is preserved for review.
