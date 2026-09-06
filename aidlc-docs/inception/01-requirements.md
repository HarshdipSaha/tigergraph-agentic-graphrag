# Requirements Baseline

Source of truth for external requirements: `docs/hackathon-brief.md` (organiser rules) and `docs/idea-spec.md` (the chosen approach and its council-mandated changes). This document distills both into a single baseline for future efforts to check against.

## External requirements (from the hackathon)

1. Build three question-answering pipelines over the provided Olympic-events corpus on TigerGraph: **RAG**, **GraphRAG**, and **Agentic GraphRAG**.
2. Benchmark all three on **accuracy, completeness, and token efficiency**, per-question and per-question-type.
3. The agentic pipeline must include: an agent harness (state, tools, context, evidence, stopping criteria), an orchestrator that decides the next action (not a fixed sequence), and specialised agents for entity linking, graph traversal, similarity search, document retrieval, aggregation, multi-hop reasoning, and evidence evaluation.
4. Deliverables: working system, GitHub repo, architecture diagram, demo video, metrics dashboard, raw outputs for 50 hidden questions (answers + tokens + agentic trace).
5. Judging weights: investigation accuracy 30%, evidence quality & explainability 15%, agentic effectiveness & efficiency 15%, engineering/code quality 15%, innovation 15%, presentation 10%.
6. Round 2 (top 15 only, out of scope for this baseline): extend to reasoning over evolving/conflicting/uncertain facts.

## Internal requirement this project adds on top

**The differentiator is provable completeness, not just correctness.** Every agentic answer must ship a machine-checkable Investigation Certificate proving the evidence set is as complete as the question class requires:

- `existential` (lookup, multi_hop): exact-match resolution is certified.
- `chained` (temporal): the PREV/NEXT hop sequence actually walked is certified.
- `exhaustive` (aggregation, superlative): the evidence set is certified equal to the graph's own structural `COUNT` for the exact filter predicate.

This is a self-imposed requirement (not asked for by the organisers) chosen because the literature scan found no existing work treats retrieval completeness as separate from answer correctness, and the dataset's clean infobox structure makes it achievable without any model training.

## Non-requirements (explicitly out of scope for this baseline)

- Round 2 (reasoning over time) — design hooks are noted (`docs/idea-spec.md` §7) but not built; revisit only if the team advances.
- A general-purpose question router — the router is documented everywhere as a benchmark-scoped heuristic over this hackathon's five known templates, not a claim of generality.
- Training any model (RL-based routing, fine-tuning) — deliberately avoided; see `docs/research-scan-agentic-graphrag.md` for why the training-free path was chosen over the RL-routing papers it's positioned against.

## Acceptance signal for this baseline

Reconciliation (`scripts/reconcile.py`) matching ≥99/100 on the public set, run against both the local parser and the live TigerGraph graph, is the load-bearing acceptance check — not a generic "tests pass." See `docs/idea-spec.md` §6 for why reconciliation is mandatory-first rather than an afterthought.
