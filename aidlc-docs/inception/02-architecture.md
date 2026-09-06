# Architecture Baseline

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

## Layering

1. **Pure logic layer** (`agrag/corpus.py`, `infobox.py`, `router.py`, `normalize.py`, `tools.py`, `certificate.py`) — no I/O dependency beyond reading local files. Fully unit-tested against fixtures, no TigerGraph or LLM required.
2. **Backend abstraction** (`agrag/backend.py` defines the `GraphBackend` protocol; `local_backend.py` and `graph/tg_backend.py` implement it). Every pipeline and the structural oracle depend only on the protocol, never on a concrete backend — this is what makes the LLM-free oracle and reconciliation runnable before TigerGraph even exists.
3. **LLM layer** (`agrag/llm.py`) — an `LLM` protocol with `GroqLLM` (default, free-tier, multi-key rotation), `AnthropicLLM` (alternative), and `FakeLLM` (tests). Pipelines depend only on the protocol.
4. **Pipelines** (`agrag/pipelines/`) — `RagPipeline`, `GraphRagPipeline` are fixed retrieval sequences (no branching, by design — this is what makes the three-way comparison honest). `AgenticPipeline` is the orchestrator: route → dispatch class-appropriate tool → evaluate evidence → LLM only for recovery/disambiguation → emit certificate.
5. **Evaluation layer** (`agrag/eval/`) — runner, scoring, report aggregation. Backend-agnostic (`--backend local|tigergraph`).

## Data flow for one agentic answer

1. `router.route(question)` — regex template match, returns `ParsedQuestion` with `completeness_class`.
2. Class-appropriate function from `agrag/tools.py` runs against the `GraphBackend` — deterministic, zero LLM tokens for the common case.
3. If the tool result says `needs_llm=True` (ambiguous tie, missing field), `AgenticPipeline` calls the LLM once for disambiguation or extraction — the only place tokens are spent in the agentic path for most questions.
4. A `Certificate` is assembled: completeness class, retrieval mode, the structural predicate used, the structural bound TigerGraph reports, the evidence set size actually inspected, and a `completeness_check` verdict (`pass` / `pass_with_llm_recovery` / `pass_with_fallback` / `fail` / `unverified`).

## Graph schema (TigerGraph, `agrag/graph/schema.gsql`)

Vertices: `Games`, `Sport`, `Venue`, `Athlete`, `Event` (the infobox as attributes), `Document`, `Chunk` (with a `VECTOR ATTRIBUTE emb(dimension=384, METRIC="COSINE")`).
Edges: `PART_OF`, `IN_SPORT`, `HELD_AT` (reverse `HOSTS`), `GOLD`, `PREV` (reverse `NEXT`), `DESCRIBED_BY`, `HAS_CHUNK`.

Seven installed queries (`agrag/graph/queries.gsql`) cover every structural need: filtered scans (`events_by_sport_games`, `events_at_games`), exact lookup (`event_by_title`, `events_by_venue`), hop traversal (`prev_event`, `event_neighborhood`), and vector search (`chunk_search`, using `vectorSearch()` with `SYNTAX v3`).

## Why this shape

The corpus's infobox structure means every one of the five question templates resolves to either an exact-match query, a bounded hop, or a filtered structural scan — never a fuzzy retrieval problem in disguise. The architecture makes that explicit rather than hiding it behind a general-purpose retrieval abstraction: `tools.py`'s five functions are a direct, auditable mapping from question template to TigerGraph query, which is also exactly what the Investigation Certificate reports back.
