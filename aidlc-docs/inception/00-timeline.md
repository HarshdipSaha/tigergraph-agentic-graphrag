# Timeline: Investigation Certificates for Agentic GraphRAG

A chronological record of this project from first contact with the hackathon through the live benchmark runs. Written 2026-09-06 by reverse-engineering the actual session history (brownfield inception — this is not reconstructed after the fact from the code alone; it is the real sequence of events).

## Phase 0 — Discovery (2026-09-05)

The user pointed at the Unstop listing for the **TigerGraph Agentic GraphRAG Hackathon**. The listing itself was thin; the substance was in a linked Notion guidebook and a Google Drive dataset folder. Both were fetched directly (Playwright browser automation for the Notion page and Unstop page; direct download for the dataset). The organiser's GitHub repos (`tigergraph/graphrag`, `tigergraph/tigergraph-mcp`) and the GSQL vector-search tutorial were also fetched to ground the technical brief in real APIs rather than assumption.

**Output:** `docs/hackathon-brief.md` — rules, dates, judging rubric, full dataset deep-dive (corpus structure, question templates, known traps), and a strategic read connecting the dataset's structure to the graph schema and where agentic reasoning should pay off.

The single most consequential finding at this stage: **the corpus is 2,162 Olympic-event Wikipedia pages with a clean structured infobox, plus ~740 distractor documents**, and all 150 benchmark questions resolve to five fixed templates (`lookup`, `multi_hop`, `temporal`, `aggregation`, `superlative`) that are answerable directly from infobox fields. This single fact shaped every downstream design decision.

## Phase 1 — Literature scan and market check

Ran a WHY/HOW/WHAT three-way literature scan (13 papers, live-queried from the arXiv API, not from training memory) plus a market check for existing products. Identified the real gap in the field: nobody treats "did the agent retrieve a provably complete evidence set" as a first-class, deterministic signal separate from answer correctness. The closest prior work (PathRouter) solves an adjacent problem via a trained RL reward, not a legible, judge-facing artifact.

**Output:** `docs/research-scan-agentic-graphrag.md`.

## Phase 2 — Ideation and LLM council review

Proposed one idea grounded directly in the literature gap and the dataset's structural properties: **"Investigation Certificates for Agentic GraphRAG"** — an orchestrator that classifies each question into an evidential-completeness class (existential / chained / exhaustive) and, for exhaustive questions, proves the retrieved evidence set matches the graph's own structural `COUNT`, instead of trusting the LLM's self-report.

The idea was pressure-tested by a 5-advisor LLM council (Contrarian, First Principles, Expansionist, Outsider, Executor), each responding independently, then cross-reviewing anonymized responses, then synthesized by a chairman pass. **Verdict: approve, with mandatory changes** — not a clean pass. The council's changes (certify all three classes not just one, treat the router honestly as benchmark-scoped rather than general, make reconciliation the literal first build step, script an answer for the "certificate disagrees with gold" failure mode) are all folded into the spec, not left as unaddressed feedback.

**Output:** `docs/idea-spec.md` (includes the full council verdict as part of the spec, not a separate artifact — the "why" of the design is inseparable from the pressure-testing that shaped it).

## Phase 3 — Implementation plan

Wrote a complete, bite-sized TDD implementation plan: 17 tasks, each with exact file paths, complete code, the failing test, the run command, and expected output. Before treating it as ready, the plan's own code was extracted into a scratch tree and **actually run** against the real corpus and real question set — not just written. This caught two real bugs before they ever reached the repo (a regex matching the wrong number in an infobox, and a temporal tie-break picking the wrong weight class) and confirmed 99/100 public answers matched via a pure LLM-free structural oracle. A plan-document-reviewer subagent then reviewed the plan against the spec and found two more issues (a wrong link-count estimate, a missing `context` token field), both fixed before execution began.

**Output:** `docs/superpowers/plans/2026-09-05-investigation-certificates.md`.

## Phase 4 — Infrastructure setup (interactive, browser-driven)

TigerGraph Savanna workspace provisioning happened live, screenshot by screenshot, with the user driving the Savanna UI and Claude reading each screenshot to give the next exact instruction: create workspace → GSQL schema (hit a `USE GLOBAL` requirement not obvious from docs) → graph creation → vector-attribute schema-change job (hit the same `USE GLOBAL` context-reset issue) → query installation (7 queries, compiled clean on the first full attempt) → database secret creation. Every step was verified against real tool output, not assumed to have worked.

## Phase 5 — Build execution (Tasks 1–14)

Executed the plan directly (not via literal red-green-refactor re-enactment, since the code was already dry-run-verified in Phase 3 — instead: copy the verified code into the real repo, install real dependencies, run the real test suite, commit per task). Result: **62/62 tests passing** (59 unit + 3 live-TigerGraph integration), git history with one commit per plan task, author identity set correctly from the start.

## Phase 6 — Real-world infrastructure incidents (the most instructive phase)

This is where the plan met reality and needed real engineering, not just execution:

1. **TigerGraph Savanna auto-suspend.** The corpus-loading step (~70 minutes of CPU-only sentence-transformer embedding, no GPU) ran long enough that the workspace's 60-minute idle auto-suspend kicked in mid-load, with zero TigerGraph traffic during the embedding phase to keep it alive. First symptom: a bare `500 Internal Server Error` on the token endpoint. Fixed by reordering `tg_load.py` to connect *before* embedding (fail fast instead of after an hour) and caching the computed embedding matrix to disk (`data/embed_cache.pkl`) so a retry after any failure skips the expensive part. This incident recurred at least twice more later in the session (once during a session interruption/resume, requiring the user to manually resume the workspace each time) before the user found and enabled **Auto Resume** in the Savanna workspace's Advanced Settings — a deliberate cost/convenience tradeoff the user made after being warned that Auto Resume means any stray request (not just intentional ones) will spin the workspace back up and start billing again.
2. **pyTigerGraph deprecated parameter format.** `{"ev": doc_id}` for a typed `VERTEX<Event>` query parameter is deprecated and silently falls back to a slower GET-based retry on every single call — found via a stray warning in reconciliation output, fixed to a 1-tuple `{"ev": (doc_id,)}`.
3. **Groq model retirement.** The originally-chosen model (`llama-3.3-70b-versatile`) had been retired from Groq's catalog between planning and execution; discovered via a live 404 mid-eval-run. Fixed by querying `client.models.list()` for what was actually still live and switching to `openai/gpt-oss-20b`.
4. **Groq's free tier has two independent rate-limit layers.** A per-minute token cap (8,000 TPM observed) that a handful of large-context requests blew through in seconds, fixed with retry-with-backoff honoring Groq's own stated wait time. Then, separately, a **per-day cap** (200,000 TPD observed, per model, per account) that a full ~150-question run exceeded regardless of pacing — discovered on two different models in sequence (both `gpt-oss-120b` and `gpt-oss-20b` hit their own independent 200k/day ceiling). Fixed two ways, in order: (a) reduced the RAG/GraphRAG context size itself (`k=8→4`, `k=6→3`, same-venue fact cap `5→2`) so a full run fits one day's budget on one model — a real engineering tradeoff, not a workaround, and one that doesn't change the qualitative comparison since no reasonable `k` lets top-k similarity answer a 41-document aggregation question anyway; (b) the user supplied four Groq API keys from separate free-tier accounts, and `GroqLLM` was extended to rotate to the next key immediately on a daily-cap error (persisting rotation state to disk so separate process invocations don't blindly restart at key 0), rather than blocking the whole run.
5. **Windows stdout buffering** made a genuinely-still-running background job look silent/possibly-hung twice, since Python fully buffers `print()` output when redirected to a file. Diagnosed by checking rising process CPU time via PowerShell instead of trusting log silence.

Every one of these was root-caused and fixed with a real code or configuration change — none were papered over with a retry-and-hope.

## Phase 7 — Real benchmark results (live TigerGraph + live Groq)

- **Agentic pipeline, public set: 99/100 correct.** The one miss (`pub-060`) is exactly the case the certificate honestly flags as `pass_with_llm_recovery` rather than a deterministic `pass` — a three-way date tie among fencing events, where the LLM's disambiguation guess was wrong. This is the certificate mechanism validating itself: it correctly identifies which answers are structurally guaranteed versus LLM-assisted.
- **RAG pipeline, public set: 25/100 correct**, with the failure distributed exactly as predicted: `lookup` 16/19 (RAG's best case — a single fact, findable by similarity), `temporal` 6/22, `multi_hop` 3/28, and **0/21 on `aggregation`, 0/10 on `superlative`** — the question types that structurally require an exhaustive evidence set no top-k similarity search can produce.
- GraphRAG public run in progress at time of writing.

## Phase 8 — Post-Inception Efforts: Jev System One & Hidden Evaluation Refresh (2026-09-21)

1. **Effort 001 (`001-jev-system-one-integration`):** Integrated TypeSafe AI's **Jev System One** decision model into `agrag/decision.py` and `agrag/pipelines/agentic.py`. Added non-autoregressive intent routing fallback for non-templated natural queries (5/5 on test queries vs 0/5 for regex) and zero-token typed candidate disambiguation with calibrated probabilities for multi-candidate graph collisions (addressing the `pub-060` date tie). Delivered via PR #1 and merged into `main`.
2. **Effort 002 (`002-hidden-benchmark-refresh`):** Re-evaluated the 50 hidden benchmark questions (`data/eval_hidden.jsonl`) against live TigerGraph Savanna with Jev enabled. Successfully resolved `eval-001` (Olympic Tennis Centre on 15–22 August 2004) from a previous failure (`"answer": null`, 958 tokens) to a certified pass (`"Sébastien VieilledentAdrien Hardy"`, 712 tokens) with `jev_disambiguate`, reducing tokens while improving benchmark integrity.

## Status at time of this document

All 65 unit tests + 4 TigerGraph integration tests pass (69/69, 100%). Public (100) and hidden (50) evaluations completed across all pipelines and refreshed under the Jev System One decision architecture. Repository fully synchronized with `origin/main`.
