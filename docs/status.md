# Status: TigerGraph Agentic GraphRAG Hackathon

Last updated: 2026-10-07

## Current implementation and validation (2026-10-07)

The `improvement-agentic-planner` branch is implemented in an isolated worktree. The default agentic mode now asks Groq to select a typed graph tool and grounded arguments; deterministic Python executes the tool and evaluates its evidence. The prior template route remains available as a comparison mode. Groq credentials are loaded from the private `.env`; the configured keys rotate within the user's organization when rate limits or authentication errors disable the active key. No credentials are stored in result artifacts.

The corpus parser now recognizes the secondary Olympic infoboxes. A live TigerGraph refresh updated Event vertices and links without rebuilding Chunk vertices or embeddings: **2,187 parsed Events, 2,187 live Event IDs, no missing or unexpected IDs**. Local and live reconciliation both report **98/100 answer matches and 100/100 evidence checks** on the refreshed corpus. This is a deliberate change from the former 99/100 because the source includes two questions whose venue/date evidence remains tied.

| Public run, 100 questions | Accuracy | Mean tokens | Mean latency | Planner tool choices | Unverified ties |
|---|---:|---:|---:|---:|---:|
| Template agentic | 99% | 14.3 | 0.414 s | 0 | 2 |
| LLM planner agentic | 98% | 1,148.5 | 1.769 s | 86 | 2 |

The planner demonstrates LLM-selected graph tools, but the measured run is one point less accurate and uses substantially more tokens and time than template mode. The planner abstains on both `pub-060` and `pub-099`; `pub-060` has three exact venue/date Events and the question names no sport, so the answer is correctly left unresolved. `eval-001` now resolves to `Q942805` (`Li TingSun Tiantian`) with an exact venue/date evidence pass. The hidden run has 50 structurally valid rows (49 non-null answers); its labels are unavailable, so hidden accuracy is not claimed. Mean use is 935.6 tokens and 1.441 s; the output contains fallback, provider-error, and unverified cases.

Current run artifacts: `results/agentic_planner_public.jsonl`, `results/agentic_planner_hidden.jsonl`, and `results/agentic_template_public.jsonl`; mode summaries: `results/summary_agentic_planner.json` and `results/summary_agentic_template.json`. RAG and GraphRAG public baselines remain historical September measurements and were not rerun after the Event graph refresh.

The full suite passed after the fixture update: 120 offline tests and four live TigerGraph integration tests. The former fixture ID `Q1183979` is now correctly an Event; the test uses film Document `Q1520721` to exercise the non-Event error path. See [`aidlc-docs/efforts/003-agent-planning-and-venue-integrity/validation-report.md`](../aidlc-docs/efforts/003-agent-planning-and-venue-integrity/validation-report.md) for commands and final verification.

## Historical September 6 snapshot (superseded by the current section below)

## Historical status at the time

- **All 17 plan tasks complete.** `docs/superpowers/plans/2026-09-05-investigation-certificates.md`. Repo, TigerGraph, all three pipelines, all six eval runs (3 pipelines × public + hidden), report, and dashboard are built and committed.
- **Full test suite: 65/65 passing** (61 unit + 4 live-TigerGraph integration tests).
- **TigerGraph Savanna: provisioned and loaded.** Workspace `MyWorkspace` (v4.2.5), graph `OlympicsRAG`, full schema + `Chunk.emb` vector attribute + all 7 queries. Loaded: 2,162 events, 23,081 chunks, 1,464 PREV edges. **Auto Resume is now enabled** on the workspace (user's deliberate choice, after being shown the cost/convenience tradeoff — see incident 1 below).
- **Reconciliation: 99/100 answers, 100/100 evidence sets correct**, confirmed twice — locally against the parsed corpus (`data/reconciliation-local.json`) and against the live TigerGraph graph (`data/reconciliation-tigergraph.json`). Both match exactly.
- **Repo pushed to GitHub (private):** https://github.com/HarshdipSaha/tigergraph-agentic-graphrag
- **AI-DLC inception baseline written:** `aidlc-docs/inception/` — full requirements/architecture/components/stack baseline plus a chronological timeline and a structured decisions/incidents log covering the entire build.
- **LLM provider: Groq**, free tier, model `openai/gpt-oss-20b`, with multi-key rotation across 4 separate free-tier accounts (`GROQ_API_KEY` comma-separated in `.env`). Chosen over Anthropic since this account has no Anthropic API access.

## Historical September public results (different Event graph snapshot)

| Pipeline | Accuracy | Avg tokens/answer | lookup | multi_hop | temporal | aggregation | superlative |
|---|---|---|---|---|---|---|---|
| RAG | 25% | 1,389 | 84% (16/19) | 11% (3/28) | 27% (6/22) | **0%** (0/21) | **0%** (0/10) |
| GraphRAG | 43% | 2,203 | 89% (17/19) | 32% (9/28) | 77% (17/22) | **0%** (0/21) | **0%** (0/10) |
| **Agentic** | **99%** | **18** | 100% | 96% (27/28) | 100% | 100% | 100% |

The one agentic miss (`pub-060`, a three-way date tie among fencing events) is exactly the case its own certificate flags as `pass_with_llm_recovery` rather than a deterministic `pass` — the certificate mechanism validating itself. Full breakdown: `results/summary.json`. Raw per-question outputs (all 100 public + 50 hidden, per pipeline, with full certificates for the agentic pipeline): `results/*_public.jsonl`, `results/*_hidden.jsonl`.

This is the intended finding, stated in `docs/idea-spec.md` before any code was written: RAG and GraphRAG both score 0% on `aggregation`/`superlative` (need 8–43 documents; no top-k similarity search retrieves an exhaustive set) regardless of how much one-hop graph expansion helps the other three question types.

## Historical infrastructure notes

1. **TigerGraph Savanna workspace auto-suspend.** Auto-suspends after 60 minutes with no database traffic. The corpus load's ~70-minute CPU-only embedding step generated no TigerGraph traffic and suspended the workspace mid-load (recurred at least twice more later, including across a session interruption). Fixed: `scripts/tg_load.py` connects *before* embedding (fail fast); the embedding matrix is cached to `data/embed_cache.pkl` (gitignored) so a retry skips the slow part. The user ultimately enabled **Auto Resume** in the workspace's Advanced Settings after being shown the tradeoff (convenience vs. any stray request silently spinning the workspace back up and consuming credits) — `scripts/tg_wait.py` remains available for polling through a suspend/resume cycle regardless.
2. **pyTigerGraph deprecated VERTEX<T> parameter format.** `{"ev": doc_id}` deprecated, silently retries over slower GET. Fixed: pass a 1-tuple, `{"ev": (doc_id,)}`.
3. **A non-Event doc crashed GraphRAG's hidden-set run.** `vector_search` embeds chunks from every corpus doc, including ~740 distractor (film/person) documents with no `Event` vertex. When one of those ranked in a question's top-k, `TigerGraphBackend.neighborhood()`/`prev_event()` let TigerGraph's "Failed to convert user vertex id" exception propagate and crash the whole run, instead of returning `None` like `LocalBackend` does for the same case. Fixed in `agrag/graph/tg_backend.py` (`_is_vertex_type_mismatch` catches it); regression test added in `tests/integration/test_tigergraph.py` using the real offending doc id (`Q1183979`). **This one produced a genuinely broken commit** (`GraphRAG pipeline hidden-set raw outputs (50 questions)` with an empty file, since the crash happened on the very first question) — superseded by a later commit with the real data once the fix landed; the broken commit was not rewritten/force-pushed, per policy against rewriting pushed history.
4. **Groq model retirement.** Originally-planned `llama-3.3-70b-versatile` was retired from Groq's catalog; discovered via a live 404. Switched to `openai/gpt-oss-20b` (checked via `client.models.list()`).
5. **Groq free-tier rate limits are two-layered.** A per-minute cap (~8,000 TPM) fixed with retry-with-backoff honoring Groq's stated wait. A separate **per-day cap** (~200,000 TPD, per model, per account) that a retry can't fix — hit on two different models in sequence during this build. Fixed two ways: (a) reduced RAG/GraphRAG context size (`k=8→4`, `k=6→3`, same-venue cap `5→2`) so a full run fits one day's budget on one model; (b) `GroqLLM` now rotates across multiple comma-separated `GROQ_API_KEY` values (4 keys from separate free accounts, supplied by the user) on a daily-cap error specifically, persisting rotation state to `data/.groq_key_state.json` (gitignored) across process runs.
6. **Windows/Git Bash stdout buffering.** `print()` output is fully block-buffered when redirected to a file, making a genuinely-still-running background job look silent/possibly-hung. Confirmed liveness via rising process CPU time (`Get-Process -Name python`) instead of trusting log content.

## Submission checklist recorded in the September snapshot

1. **Register on Unstop** if not already done (registration close is Sep 12–14 depending on source — see brief §2; register early regardless).
2. Architecture diagram export and a recorded demo video — the mermaid diagram is in `README.md`; a rendered image/video for the submission itself is not yet produced.
3. Watch Discord for the still-unpublished "Accuracy Evaluation Guide" and hidden-question submission format (brief §6, §12) — may change how results should be formatted for submission; re-check before the actual submission.
4. Round 2 ("reasoning over time") is out of scope until Round 1 ships and the team makes the top 15 — see spec §7 for the design hooks already left in place.

## Questions recorded in the September snapshot

- Exact submission format for the 50 hidden-question raw outputs is not specified anywhere yet.
- How Savanna credits are actually issued is unstated.
- Whether Round 1 submission happens on Unstop directly or via a separate form.

## Historical decision log

- **2026-09-05:** Chose "Investigation Certificates for Agentic GraphRAG" over no alternative — the council approved on the first pass (with mandatory changes), so no second idea was needed.
- **2026-09-05:** Chose Groq over Anthropic as the LLM provider — no Anthropic API access on this account.
- **2026-09-06:** User enabled Auto Resume on the TigerGraph workspace after an explicit cost-tradeoff discussion, prioritizing build velocity over tighter credit control.
- **2026-09-06:** User supplied 4 Groq API keys for rotation after hitting Groq's per-model daily token cap twice in one session.
