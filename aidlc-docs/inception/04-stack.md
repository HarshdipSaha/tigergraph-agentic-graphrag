# Tech Stack Baseline

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | |
| Graph database | TigerGraph Savanna (cloud), v4.2.5 | Organiser-provided; free tier; ≥4.2 needed for vector attributes. Workspace `MyWorkspace`, graph `OlympicsRAG`. |
| Graph client | `pyTigerGraph` ≥1.8 | Official Python client; used for both DDL execution and query calls. |
| LLM (default) | Groq, `openai/gpt-oss-20b` | Free tier, no card required — chosen because this account has no Anthropic API access. Model chosen live after the originally-planned `llama-3.3-70b-versatile` was found retired from Groq's catalog. |
| LLM (alternative) | Anthropic (any Claude model via `AGRAG_MODEL`) | Implemented behind the same `LLM` protocol for anyone who has API access; not the default path taken in this build. |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2`, 384-dim, cosine | CPU-only in this environment (no GPU); embedding ~23k chunks took roughly an hour, which is itself the root cause of the auto-suspend incident (see decisions log). |
| Validation | `pydantic` v2 | `Certificate`/`TokenUsage` schema. |
| Testing | `pytest` | 62 tests, `pytest.ini_options` in `pyproject.toml`, `integration` marker gates live-TigerGraph tests. |
| Dashboard | `streamlit` + `pandas` | Comparison dashboard over `results/*.jsonl`. |
| Config | `python-dotenv` | `.env` for secrets (TigerGraph host/graph/secret, Groq keys), never committed. |
| VCS / hosting | git + GitHub (private repo) | `HarshdipSaha/tigergraph-agentic-graphrag`, pushed with `gh repo create`. |

## Deliberately not used

- No RL training pipeline (the literature scan's cited routing papers — Graph-R1, GraphRAG-Router, PathRouter — all require one; this project's router is a training-free regex heuristic by design, see `docs/idea-spec.md` §4.2).
- No vector database other than TigerGraph's own `vectorSearch()` — kept the stack to one database rather than adding a separate vector store.
- No agent framework (LangGraph, CrewAI, etc.) — the orchestrator in `agrag/pipelines/agentic.py` is plain Python; the problem space (5 known question templates) didn't need one.
