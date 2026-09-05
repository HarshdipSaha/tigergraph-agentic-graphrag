# Status: TigerGraph Agentic GraphRAG Hackathon

Last updated: 2026-09-05

## Where things stand

- **Research phase: done.** Hackathon rules, dataset, and stack fully scoped (`docs/hackathon-brief.md`). Literature scan across 13 papers + market check done (`docs/research-scan-agentic-graphrag.md`).
- **Idea: chosen and council-approved with changes.** "Investigation Certificates for Agentic GraphRAG" (`docs/idea-spec.md`). An LLM council (5 advisors, cross-review, synthesis) reviewed it before any code was written; verdict was approve-with-changes, not a clean pass — see spec §3 and §6 for what changed and why.
- **Implementation plan: written, reviewed, and executed.** `docs/superpowers/plans/2026-09-05-investigation-certificates.md` (17 TDD tasks, complete code). Tasks 1–14 are done and committed; Tasks 15–17 are in progress (this session).
- **TigerGraph Savanna: provisioned and loaded.** Workspace `MyWorkspace` (v4.2.5), graph `OlympicsRAG` with the full schema, `Chunk.emb` vector attribute, all 7 queries installed. Loaded: 2,162 events, 23,081 chunks, 1,464 PREV edges. Reconciliation against the live graph: **99/100 answers, 100/100 evidence sets correct** (`data/reconciliation-tigergraph.json`) — matches the local reconciliation exactly.
- **Repo pushed to GitHub (private):** https://github.com/HarshdipSaha/tigergraph-agentic-graphrag
- **Full test suite: 62/62 passing** (59 unit + 3 live-TigerGraph integration tests).
- **LLM provider: Groq**, free tier. Chosen over Anthropic since this account has no Anthropic API access.

## Real infra incidents hit and fixed during the build (worth knowing before re-running)

1. **TigerGraph Savanna workspace auto-suspend.** The workspace auto-suspends after 60 minutes with no database traffic. The corpus load's embedding step (sentence-transformers on CPU, no GPU) took ~70 minutes on its own, during which nothing touched TigerGraph — the workspace suspended mid-run and the load failed with a 500 on the token endpoint. Fixed two ways: (a) `scripts/tg_load.py` now connects to TigerGraph *first*, before the slow embedding step, so a dead workspace fails fast instead of after an hour; (b) the computed embedding matrix is now cached to `data/embed_cache.pkl` (gitignored — large binary), so a retry after any failure skips the expensive part entirely. If you hit a 500/502 on any TigerGraph call, check the workspace status in the Savanna UI and resume it manually — resuming takes 1–3 minutes for all backend services to come back (watch for 502 → success, not 500 → success).
2. **pyTigerGraph deprecated VERTEX<T> parameter format.** `{"ev": doc_id}` for a typed vertex parameter is deprecated and silently falls back to a slower GET-based retry on every single call. Fixed in `agrag/graph/tg_backend.py`: pass a 1-tuple, `{"ev": (doc_id,)}`.
3. **Groq model retirement.** The model this repo originally defaulted to (`llama-3.3-70b-versatile`) has been retired from Groq's catalog since it was chosen; it now 404s. Current default is `openai/gpt-oss-20b`.
4. **Groq free-tier rate limits are two-layered and easy to hit with a RAG-style pipeline.** There is a *per-minute* token cap (observed: 8,000 TPM for `openai/gpt-oss-120b`) that a handful of large-context requests (RAG/GraphRAG send ~2,000–3,500 tokens of retrieved context per call) blows through in seconds — fixed with retry-with-backoff in `GroqLLM.complete()` that honors Groq's stated wait time. There is *also* a **per-day cap** (observed: 200,000 TPD for `openai/gpt-oss-120b` on this account) that a retry loop cannot fix — hitting it mid-run forced a switch to `openai/gpt-oss-20b`, which has its own separate quota bucket. If a full benchmark run (RAG + GraphRAG × 150 questions, each ~2,000–3,500 tokens) exhausts `gpt-oss-20b`'s daily budget too, the next fallback is `qwen/qwen3.8-27b` or `groq/compound` (both available on this account) — check `client.models.list()` for what's currently live, since Groq's free-tier catalog changes over time.
5. **Windows/Git Bash stdout buffering.** A Python script's `print()` output is fully block-buffered when redirected to a file (not flushed line-by-line), so tailing a redirected log while a script runs shows nothing until the buffer flushes or the process exits — this made a genuinely-still-running background job look silent/possibly-hung twice. Confirmed liveness instead via rising process CPU time (`Get-Process -Name python | Select CPU`), not log content, when a long-running job goes quiet.

## Real results so far (live TigerGraph + Groq, public set, n=100)

| Pipeline | Correct | Notes |
|---|---|---|
| Agentic | 99/100 | The one miss (`pub-060`) is exactly the LLM-disambiguation case the certificate flags as `pass_with_llm_recovery` rather than a deterministic `pass` — a three-way date tie among fencing events at ExCeL, where the LLM guessed wrong. This is the certificate mechanism working as designed: it's honest about which answers are LLM-assisted vs. structurally verified. |
| RAG | in progress | First full run (gpt-oss-120b) hit the daily quota at question 59/100 with only 7/59 correct so far — confirms the spec's prediction that plain RAG struggles badly on `aggregation`/`superlative` (needs 8–43 docs, gets top-8 chunks). Re-running in full on `gpt-oss-20b` for a clean, consistent 100-question result. |
| GraphRAG | not yet run | Queued after RAG completes. |

Full numbers, per-qtype breakdown, and the RAG-vs-agentic token/coverage comparison will be finalized in `README.md` and `results/summary.json` once all six runs (3 pipelines × public + hidden) complete.

## Next actions, in order

1. Finish RAG public run (in progress) → GraphRAG public → hidden set for all three pipelines → `python -m agrag.eval.report` → fill in README's results table → `streamlit run dashboard/app.py` to sanity-check the dashboard renders.
2. Commit and push remaining results as each run completes.
3. **Register on Unstop** if not already done (registration close is Sep 12–14 depending on source — see brief §2; register early regardless).
4. Watch Discord for the still-unpublished "Accuracy Evaluation Guide" and hidden-question submission format (brief §6, §12) — these may change how the certificate/dashboard should be scored or formatted; re-check before finalizing the dashboard.
5. Round 2 ("reasoning over time") is out of scope until Round 1 ships and the team makes the top 15 — see spec §7 for the design hooks already left in place (validity intervals on PREV/NEXT, source authority).

## Open blockers / unknowns (carried over from the brief)

- Exact submission format for the 50 hidden-question raw outputs is not specified anywhere yet.
- How Savanna credits are actually issued is unstated.
- Whether Round 1 submission happens on Unstop directly or via a separate form.

## Decision log

- **2026-09-05:** Chose "Investigation Certificates for Agentic GraphRAG" over no alternative — the council approved on the first pass (with mandatory changes), so no second idea was needed. Changes required by the council are already folded into `docs/idea-spec.md`.
- **2026-09-05:** Chose Groq over Anthropic as the LLM provider — no Anthropic API access on this account, and Groq's free tier needs no card. Traded away a smoother experience (see incidents above) for zero cost.
