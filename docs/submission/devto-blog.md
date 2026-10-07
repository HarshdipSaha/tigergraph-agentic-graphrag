---
title: "My agent now chooses its own graph tools. Same accuracy, 70x the tokens."
published: false
description: "A measured comparison of LLM-selected graph tools, deterministic routing, and evidence certificates on a TigerGraph Olympics benchmark."
tags: ai, rag, graphdatabase, python
cover_image:
---

I built an Agentic GraphRAG system around a simple problem: retrieval can show relevant documents, but it cannot prove that it showed every matching record. For counting questions, TigerGraph can return the full matching Event set and its structural bound. The certificate records that evidence so correctness and completeness remain separate claims.

The first implementation routed benchmark questions through fixed templates. After submission, a TigerGraph developer suggested letting the LLM choose the graph tools so the agentic behavior would be visible, and flagged a relaxed venue fallback that could select an event from a different sport at the same venue.

## The architecture change

Planner mode asks an LLM to choose one of five allowlisted tools and provide typed arguments. Python checks the action against the question, executes the graph tool, and evaluates its evidence. Invalid actions and provider errors stay visible in the trace. The model never writes an answer directly into a passing certificate.

For venue questions, the system requires a nonempty venue and supported date evidence. Sport may narrow the candidates only when the question actually names that sport. If a small tie of two or three events remains, the model picks one from the date-filtered set and the certificate says `unverified`, not `pass`.

I also fixed a parser gap: 25 Olympic pages had the relevant Olympic infobox after a secondary tournament infobox. After reparsing and refreshing the Event vertices and links, TigerGraph contains 2,187 Event IDs, exactly matching the parsed source set. Chunk vertices and embeddings were not rebuilt.

## Results on the refreshed graph

Both current agentic modes ran the same 100 public questions against the refreshed TigerGraph graph and the same Groq model:

| Mode | Answer match | Mean tokens | Mean latency | LLM-selected tool actions |
|---|---:|---:|---:|---:|
| Fixed template routing | 99/100 (99%) | 14.3 | 0.414 s | 0 |
| LLM action planner | 99/100 (99%) | 1,038.4 | 1.447 s | 87/100 |

The planner makes tool choice explicit at the same accuracy, but it costs roughly 70x the tokens and 3.5x the latency. That is the useful result to report: agentic tool selection makes the planning visible, and a deterministic route is still the cheaper way to answer these benchmark shapes. Template routing stays the default; the planner is opt-in.

Both modes miss only `pub-060`, a venue/date tie. There are three Events at ExCeL on 30 July 2012: fencing `Q1156695` and judo `Q1064016`, `Q1005551`, and the question gives no sport. Both modes pass small ties like this to the model, which matches `pub-099` and misses `pub-060`; both certificates remain `unverified`, because neither answer is uniquely supported by the question. An earlier planner run that abstained on these ties instead scored 98/100.

The older September RAG and GraphRAG figures (25% and 43%) were measured on the earlier Event graph. They have not been rerun after the parser repair and are not a same-snapshot comparison with the October planner numbers.

## What the certificate says

For exhaustive questions, TigerGraph returns the full population and structural count. Python checks that every required value was recovered before certifying the aggregation or superlative result. The October planner run matched all 21 public aggregation answers and all 10 superlative answers, with full evidence coverage across the public set.

The planner ran on all 50 hidden questions as well. Those labels are unavailable, so I report the raw traces, token use, and certificate status without claiming hidden accuracy. `eval-001` now uses source record `Q942805` and returns `Li TingSun Tiantian` with an exact venue/date evidence pass; the submitted run had answered it with a rowing event at the same venue, the exact miss the TigerGraph developer pointed out.

## Reproducibility

- Planner results: `results/agentic_planner_public.jsonl` and `results/agentic_planner_hidden.jsonl`
- Template comparison: `results/agentic_template_public.jsonl`
- Validation and commands: `aidlc-docs/efforts/003-agent-planning-and-venue-integrity/validation-report.md`
- Code: https://github.com/HarshdipSaha/tigergraph-agentic-graphrag
- Dashboard: https://investigation-certificates.streamlit.app/

Offline tests use an in-memory graph backend. Live TigerGraph integration requires the repository?s local environment configuration. The main finding is straightforward: graph structure makes completeness measurable; an LLM planner makes tool selection visible, but its cost and accuracy must still be measured against a deterministic route.
