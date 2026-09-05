# Status: TigerGraph Agentic GraphRAG Hackathon

Last updated: 2026-09-05

## Where things stand

- **Research phase: done.** Hackathon rules, dataset, and stack fully scoped (`docs/hackathon-brief.md`). Literature scan across 13 papers + market check done (`docs/research-scan-agentic-graphrag.md`).
- **Idea: chosen and council-approved with changes.** "Investigation Certificates for Agentic GraphRAG" (`docs/idea-spec.md`). An LLM council (5 advisors, cross-review, synthesis) reviewed it before any code was written; verdict was approve-with-changes, not a clean pass — see spec §3 and §6 for what changed and why.
- **Implementation plan: written and reviewed.** `docs/superpowers/plans/2026-09-05-investigation-certificates.md` (17 TDD tasks, complete code).
- **TigerGraph Savanna: provisioned.** Workspace `MyWorkspace` (v4.2.5), graph `OlympicsRAG` created with the full schema, `Chunk.emb` vector attribute, and all 7 queries installed and compiling clean. Secret created; `.env` holds `TG_HOST`/`TG_GRAPH`/`TG_SECRET`.
- **LLM provider: Groq** (free tier, `llama-3.3-70b-versatile`), chosen over Anthropic since this account has no Anthropic API access. `agrag/llm.py` implements both behind one `LLM` protocol; `AGRAG_LLM_PROVIDER` in `.env` switches.
- **Build phase: in progress, Tasks 1–13 done.** Repo is `git init`ed (author Harshdip Saha), package scaffolded, and Tasks 1–13 of the plan committed: corpus loader, infobox parser, question router, normalisation, `GraphBackend`/`LocalBackend`, chunking/embedding, LLM wrappers, deterministic tools, certificate model, structural oracle, and all three pipelines (RAG/GraphRAG/Agentic). **59/59 unit tests pass in the real repo** (not just the dry run).
- **Local reconciliation: done, real corpus.** `data/reconciliation-local.json`: 99/100 answers match, 100/100 evidence sets correct (only `pub-060`, a three-way date tie at ExCeL, needs the LLM disambiguation path — expected, not a bug).
- **TigerGraph load: in progress.** `scripts/tg_load.py` is embedding ~20k chunks with `all-MiniLM-L6-v2` on CPU (no GPU available) and upserting into `OlympicsRAG`. Running far longer than expected (60+ minutes) but confirmed still actively computing via rising process CPU time, not hung. Once it finishes: run `scripts/reconcile.py --backend tigergraph`, then Tasks 15–17 (eval runs on all three pipelines, dashboard, README finalization).

## What exists in this repo right now

| Path | Contents |
|---|---|
| `docs/hackathon-brief.md` | Full rules, dates, judging rubric, dataset deep-dive, strategic read |
| `docs/research-scan-agentic-graphrag.md` | WHY/HOW/WHAT scan of 13 papers + market scan, cross-paper synthesis |
| `docs/idea-spec.md` | The chosen idea, council verdict, system design, build plan, risk table, rubric alignment |
| `data/corpus.jsonl` | 2,951 documents (22MB), downloaded from the organiser's Google Drive |
| `data/eval_public.jsonl` | 100 questions with answers + gold doc IDs |
| `data/eval_hidden.jsonl` | 50 questions, no answers (submit raw outputs) |
| `data/dataset-README.md` | Organiser's original dataset README |

## Next actions, in order

1. **Register on Unstop** if not already done (registration close is Sep 12–14 depending on source — see brief §2; register early regardless).
2. **Provision TigerGraph Savanna** (tgcloud.io) — this is unstarted and is flagged in the spec as a real time risk (new platform for this builder, separate from the data-modeling work).
3. **Ingestion:** parse `data/corpus.jsonl` infoboxes into a graph schema (Games, Sport, Event, Venue, Athlete, NOC nodes; `PART_OF`/`HELD_AT`/`IN_SPORT`/`GOLD`/`PREV`/`NEXT` edges). See brief §10 for the proposed schema and known parsing traps (missing dates, non-numeric competitor counts, venue ambiguity across 317 venue+games pairs).
4. **Reconciliation pass (mandatory, before any agent/orchestrator code):** for all 100 public questions, compute each question's structural bound in TigerGraph and diff against `gold_doc_ids`. Fix ingestion bugs this surfaces. Commit the reconciliation result. This is spec §6 — do not skip ahead to pipeline code before this is done and logged here.
5. Build the three pipelines (RAG, GraphRAG, Agentic) per spec §5's week-by-week plan.
6. Metrics dashboard, demo video, architecture diagram.
7. Watch Discord for the still-unpublished "Accuracy Evaluation Guide" and hidden-question submission format (brief §6, §12) — these may change how the certificate/dashboard should be scored or formatted; re-check before finalizing the dashboard.

## Open blockers / unknowns (carried over from the brief)

- Exact submission format for the 50 hidden-question raw outputs is not specified anywhere yet.
- How Savanna credits are actually issued is unstated.
- Whether Round 1 submission happens on Unstop directly or via a separate form.

## Decision log

- **2026-09-05:** Chose "Investigation Certificates for Agentic GraphRAG" over no alternative — the council approved on the first pass (with mandatory changes), so no second idea was needed. Changes required by the council are already folded into `docs/idea-spec.md`; nothing here is stale relative to that file as of this date.
