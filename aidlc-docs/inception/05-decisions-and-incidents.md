# Key Decisions and Incidents (quick-reference table)

Narrative version with full context: `00-timeline.md` §Phase 6. This is the scannable index.

## Design decisions

| Decision | Alternative considered | Why this one |
|---|---|---|
| Investigation Certificates as the core idea | Generic adaptive-complexity router (Adaptive-RAG style) | Council + literature scan found no prior work treats completeness as a first-class signal separate from correctness; a router alone was already published territory. |
| Training-free heuristic router | RL-trained routing (Graph-R1 / GraphRAG-Router / PathRouter pattern) | No time or infra for an RL training loop in the build window; also more honest about what a benchmark-scoped tool actually is. |
| Groq (free tier) over Anthropic | Anthropic Claude | No Anthropic API key on this account; Groq needs no card. Traded a smoother experience for zero cost — see incidents below for what that cost in engineering time. |
| RAG `k=4`, GraphRAG `k=3`, same-venue cap `2` (down from 8/6/5) | Keep original larger k | Original settings produced ~2,000–5,000 tokens/request, exceeding Groq's free-tier daily cap before a full run finished. Smaller k is still a realistic RAG deployment and doesn't change the qualitative result — no k under ~40 lets top-k retrieval answer a 41-document aggregation question anyway. |
| Multi-key rotation in `GroqLLM` | Wait out the daily cap / use one key and stop | User supplied 4 keys from separate free accounts; rotating on a daily-cap error (not a per-minute one) keeps a benchmark run moving without waiting a full day. |

## Infrastructure incidents (root cause → fix)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `500 Internal Server Error` on the TigerGraph token endpoint, recurring several times through the session | Savanna workspace auto-suspends after 60 minutes idle; the ~70-minute CPU-only embedding step generated zero TigerGraph traffic, so it suspended mid-load (and again later during a session gap) | `tg_load.py` connects *before* embedding (fail fast); embeddings cached to `data/embed_cache.pkl` (skip the slow part on retry); user ultimately enabled **Auto Resume** in the Savanna workspace's Advanced Settings, accepting the cost/convenience tradeoff after being shown it |
| 2 | Every `prev_event`/`event_neighborhood` call logged a deprecation warning and silently retried over GET | `pyTigerGraph` deprecates plain-value `VERTEX<T>` query parameters (`{"ev": doc_id}`) | Pass a 1-tuple: `{"ev": (doc_id,)}` |
| 3 | `404 model_not_found` for `llama-3.3-70b-versatile` | Model retired from Groq's catalog between planning and execution | Queried `client.models.list()` for what's actually live; switched to `openai/gpt-oss-20b` |
| 4a | `429` "tokens per minute" errors within seconds of starting a run | Free-tier TPM cap (~8,000) blown through by a handful of large-context requests | Retry-with-backoff in `GroqLLM.complete()`, honoring Groq's own stated wait time |
| 4b | `429` "tokens per day" errors partway through a ~100-question run, on two different models in turn | Free-tier TPD cap (~200,000, per model, per account) — retrying doesn't help a daily cap | Reduced context size (see design decisions) + multi-key rotation on daily-cap errors specifically (not per-minute ones) |
| 5 | A background job looked silent/possibly-hung for many minutes | Python fully buffers `print()` output when stdout is redirected to a file (not a TTY) | Confirmed liveness via rising process CPU time (`Get-Process -Name python`) instead of trusting log content |

## What this means for future efforts against this baseline

- Any change to `agrag/pipelines/rag.py` or `graphrag.py`'s `k` should be re-checked against the daily token budget math in incident 4b before assuming a full run will complete unattended.
- If TigerGraph calls start failing with 500/502, check Auto Resume status and workspace state before assuming a code bug — this has happened repeatedly and is an infra state issue, not a regression.
- `GroqLLM`'s rotation state (`data/.groq_key_state.json`, gitignored) persists across process runs; delete it if you want a clean start from key 0.
