# AI-DLC Registry

Derived view — rebuilt from per-effort state files; the filesystem under `aidlc-docs/` is the source of truth.

## Inception Baseline

| Status | Path | Notes |
|---|---|---|
| Complete | [`aidlc-docs/inception/`](inception/) | Brownfield inception covering the full build from hackathon discovery through live benchmark results and TigerGraph Savanna setup. |

## Efforts

| Effort | Reference | Status | Completed At | Notes |
|---|---|---|---|---|
| `001` | [`001-jev-system-one-integration`](efforts/001-jev-system-one-integration/effort-state.md) | `complete` | 2026-09-21 | TypeSafe AI Jev System One decision model integration: non-template routing fallback, zero-token candidate disambiguation, 69/69 passing tests, merged via PR #1. |
| `002` | [`002-hidden-benchmark-refresh`](efforts/002-hidden-benchmark-refresh/effort-state.md) | `complete` | 2026-09-21 | Re-evaluation of 50 hidden questions with Jev; its reported `eval-001` pass was later found unsupported by source Event data (see Effort 003). |
| `003` | [`003-agent-planning-and-venue-integrity`](efforts/003-agent-planning-and-venue-integrity/effort-state.md) | `complete` | 2026-10-07 | LLM tool planning, secondary-infobox ingestion, evidence-safe venue/date filtering, live graph refresh, and measured comparison; see validation report. |
