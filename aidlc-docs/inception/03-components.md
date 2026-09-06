# Component Inventory

Reverse-engineered from the actual repository tree. Each entry is one clear responsibility (per this project's own deep-module convention: see `docs/superpowers/plans/2026-09-05-investigation-certificates.md` "File Structure" section for the original design intent, which the built code matches).

| Component | Responsibility |
|---|---|
| `agrag/corpus.py` | Load `data/corpus.jsonl` into `Doc` records. |
| `agrag/infobox.py` | Parse the infobox block out of a doc's text into `EventRecord`; resolve `PREV`/`NEXT` links by (sport, season, event name, year) matching, since raw year references don't always resolve directly. |
| `agrag/questions.py` | Load public (with gold answers) or hidden (without) question files into `Question` records. |
| `agrag/router.py` | Regex template matcher over the benchmark's five known question types → `ParsedQuestion` with slots and a `completeness_class`. Explicitly documented as benchmark-scoped, not general. |
| `agrag/normalize.py` | Answer string normalisation (accent-stripping, dash unification, case-folding) and exact/normalized/contains match flags. |
| `agrag/backend.py` | `GraphBackend` Protocol — the seam every pipeline and the oracle depend on instead of a concrete backend. |
| `agrag/local_backend.py` | In-memory `GraphBackend` implementation built directly from parsed `Doc`s — used by all unit tests, the reconciliation script's local pass, and local dev. |
| `agrag/graph/tg_backend.py` | `GraphBackend` implementation over a live TigerGraph connection via installed GSQL queries. |
| `agrag/graph/client.py` | `TigerGraphConnection` factory reading `.env` settings. |
| `agrag/graph/schema.gsql`, `queries.gsql` | The graph DDL and the 7 installed queries — the only GSQL in the repo. |
| `agrag/graph/load.py`, `scripts/tg_load.py` | Corpus → TigerGraph vertex/edge upserts, with embedding-cache and connect-first ordering (see incident log). |
| `agrag/chunk.py` | Paragraph-aware text chunking for the RAG/GraphRAG text-retrieval path. |
| `agrag/embed.py` | `Embedder` protocol; `SentenceTransformerEmbedder` (real, `all-MiniLM-L6-v2`) and `FakeEmbedder` (deterministic, tests). |
| `agrag/llm.py` | `LLM` protocol; `GroqLLM` (default, multi-key rotation), `AnthropicLLM` (alternative), `FakeLLM` (scripted, tests). |
| `agrag/tools.py` | The five deterministic, LLM-free retrieval/reasoning functions — one per question template — shared by both the oracle and the agentic orchestrator. |
| `agrag/certificate.py` | The `Certificate` and `TokenUsage` pydantic models — the schema of the project's core deliverable. |
| `agrag/oracle.py` | LLM-free dispatcher used only by `scripts/reconcile.py` to prove the graph agrees with the gold set before any agentic code runs. |
| `agrag/pipelines/base.py` | Shared `PipelineResult`, `Timer`, prompt text, and `event_facts` formatting used by all three pipelines. |
| `agrag/pipelines/rag.py` | Fixed-sequence baseline: embed → top-k chunks → one LLM call. |
| `agrag/pipelines/graphrag.py` | Fixed-sequence baseline: top-k chunks → one-hop graph neighbourhood expansion → one LLM call. |
| `agrag/pipelines/agentic.py` | The orchestrator: route → class-appropriate tool → LLM only for recovery → certificate. |
| `agrag/eval/run.py`, `score.py`, `report.py` | CLI runner producing `results/*.jsonl`, per-question scoring, and the aggregated `results/summary.json`. |
| `dashboard/app.py` | Streamlit comparison dashboard over `results/*.jsonl`. |
| `scripts/tg_setup.py`, `tg_smoke.py`, `tg_wait.py` | Operational scripts: DDL/query installation, connectivity smoke test, and polling for workspace resume after an auto-suspend. |
| `scripts/reconcile.py` | The council-mandated completeness check — runs the oracle against every public question and diffs against gold. |
| `scripts/download_data.py` | Reproduces the dataset download from the organiser's Google Drive folder. |

## Test inventory

62 tests total: 59 unit (one file per component above, all against `LocalBackend`/`FakeLLM`/`FakeEmbedder`) + 3 integration (`tests/integration/test_tigergraph.py`, gated on `TG_HOST` being set, exercising the real graph).
