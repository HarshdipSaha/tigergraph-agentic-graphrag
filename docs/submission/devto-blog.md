---
title: "My agent can choose a graph tool. The planner run still scored lower."
published: false
description: "A measured comparison of LLM-selected graph tools, deterministic routing, and evidence certificates on a TigerGraph Olympics benchmark."
tags: ai, rag, graphdatabase, python
cover_image:
---

I built an Agentic GraphRAG system around a simple problem: retrieval can show relevant documents, but it cannot prove that it showed every matching record. For counting questions, TigerGraph can return the full matching Event set and its structural bound. The certificate records that evidence so correctness and completeness remain separate claims.

The first implementation routed benchmark questions through fixed templates. Reviewer feedback asked for the LLM to choose the graph tools so the agentic behavior would be visible, and flagged a venue fallback that could select an event from the wrong day or sport.

## The architecture change

The default agentic mode now asks an LLM to choose one of five allowlisted tools and provide typed arguments. Python checks the action against the question, executes the graph tool, and evaluates its evidence. Invalid actions and provider errors stay visible in the trace. The model never writes an answer directly into a passing certificate.

For venue questions, the system requires a nonempty venue and supported date evidence. Sport may narrow the candidates only when the question actually names that sport. If several events remain, the answer stays unresolved and the certificate says `unverified`.

I also fixed a parser gap: 25 Olympic pages had the relevant Olympic infobox after a secondary tournament infobox. After reparsing and refreshing the Event vertices and links, TigerGraph contains 2,187 Event IDs, exactly matching the parsed source set. Chunk vertices and embeddings were not rebuilt.

## Results on the refreshed graph

Both current agentic modes ran the same 100 public questions against the refreshed TigerGraph graph and the same Groq model:

| Mode | Answer match | Mean tokens | Mean latency | LLM-selected tool actions |
|---|---:|---:|---:|---:|
| Fixed template routing | 99/100 (99%) | 14.3 | 0.414 s | 0 |
| LLM action planner | 98/100 (98%) | 1,148.5 | 1.769 s | 86/100 |

The planner makes tool choice explicit, but these measurements do not show an accuracy or efficiency gain. The template mode is one point more accurate and uses far fewer tokens. This is the useful result to report: agentic tool selection improved the architecture?s visible planning behavior, while the current planner prompt and validation loop have room to improve.

The planner?s two public abstentions are `pub-060` and `pub-099`, both unresolved venue/date ties. For `pub-060`, there are three Events at ExCeL on 30 July 2012: fencing `Q1156695` and judo `Q1064016`, `Q1005551`. The question gives no sport, so selecting one would add information that is not present in the question. Template mode emits model-selected answers for both ties. It misses `pub-060` and happens to match `pub-099`; both certificates remain `unverified`, because neither answer is uniquely supported by the question.

The older September RAG and GraphRAG figures (25% and 43%) were measured on the earlier Event graph. They have not been rerun after the parser repair and are not a same-snapshot comparison with the October planner numbers.

## What the certificate says

For exhaustive questions, TigerGraph returns the full population and structural count. Python checks that every required value was recovered before certifying the aggregation or superlative result. The October planner run matched all 21 public aggregation answers and all 10 superlative answers, with full evidence coverage across the public set.

The planner ran on all 50 hidden questions as well. Those labels are unavailable, so I report the raw traces, token use, and certificate status without claiming hidden accuracy. `eval-001` now uses source record `Q942805` and returns `Li TingSun Tiantian` with an exact venue/date evidence pass.

## Reproducibility

- Planner results: `results/agentic_planner_public.jsonl` and `results/agentic_planner_hidden.jsonl`
- Template comparison: `results/agentic_template_public.jsonl`
- Validation and commands: `aidlc-docs/efforts/003-agent-planning-and-venue-integrity/validation-report.md`
- Code: https://github.com/HarshdipSaha/tigergraph-agentic-graphrag
- Dashboard: https://investigation-certificates.streamlit.app/

Offline tests use an in-memory graph backend. Live TigerGraph integration requires the repository?s local environment configuration. The main finding is straightforward: graph structure makes completeness measurable; an LLM planner makes tool selection visible, but its cost and accuracy must still be measured against a deterministic route.
