# Idea Spec: Investigation Certificates for Agentic GraphRAG

Status: **Approved by LLM council, with mandatory changes (see §6).** Not yet built. Companion documents: `docs/hackathon-brief.md` (rules/dates/dataset), `docs/research-scan-agentic-graphrag.md` (literature), `docs/status.md` (build tracking).

## 1. One-line pitch

Every answer the agentic pipeline gives ships a machine-checkable **Investigation Certificate**: a JSON record proving, using the graph's own structure, that the evidence behind the answer is as complete as the question requires — not just that the final answer looks right.

## 2. The gap this fills

Per the literature scan (`docs/research-scan-agentic-graphrag.md`), nobody treats "did the agent retrieve a provably complete evidence set" as a first-class, deterministic, judge-legible signal distinct from answer correctness:

- RL-routing papers (Graph-R1, GraphRAG-Router, PathRouter) optimize a trained reward function. PathRouter's "answer-path reward aliasing" fix scores evidence-path overlap, but only inside a training loop — invisible to an end user or judge, and requires an RL pipeline this team will not build in 3 weeks.
- T-GRAG and Graphiti solve staleness/conflict (which fact is current), not exhaustiveness (did we find every matching entity).
- Adaptive-RAG's classifier is trained on auto-collected labels; this hackathon has no time or need to train one, because the benchmark hands you the question type (`qtype`) and the corpus's own filter predicates (sport, games, venue) are exact-match, not fuzzy.

The dataset makes this gap closeable without any training: `aggregation` and `superlative` questions (31% of public, 50% of hidden) require 8–43 gold documents each, i.e. an exhaustive scan, and TigerGraph can report the *exact* structural size of that answer set via a `COUNT`/traversal query before any LLM call happens. That structural bound is something plain RAG and even naive GraphRAG cannot compute — vector top-k has no concept of "have I seen everything."

## 3. What the council changed about the original pitch

An LLM council (Contrarian, First Principles, Expansionist, Outsider, Executor — 5 independent advisors, then cross-review, then synthesis) reviewed this idea before it was written up. Verdict: **approve, with changes.** Full reasoning below; changes are binding on this spec.

**Where the council agreed:**
- The Executor's response was unanimously rated strongest across all 5 peer reviews: it turned the idea into a concrete build plan, correctly identified gold-set verification (not the certificate mechanism) as the real bottleneck, and gave the sharpest actionable risk mitigation.
- The Expansionist's response ("this is a certification primitive for regulated industries, pitch it as infrastructure") was unanimously flagged as the weakest: it never engaged with feasibility or the shared risk every other advisor caught.
- The single most load-bearing risk, raised independently by four of five advisors: **the certificate's structural bound might not match the benchmark's actual gold-doc count** (infobox parsing gaps, duplicate/mis-tagged entities, disambiguation errors in the corpus-to-graph ingestion). If that ever diverges, the headline "provably complete" claim breaks in front of judges, on the 30%-weighted investigation-accuracy criterion.
- The classifier that routes questions into completeness classes is, honestly, a template match on the benchmark's own 5 known `qtype` values wearing a "classifier" costume. Say so plainly rather than framing it as a general contribution — a technical judge will ask "what happens on a 6th question type" and there is no good answer in 3 weeks.

**Where the council clashed:** ambitious framing (pitch this as reusable audit infrastructure for any agentic RAG system) vs. modest framing (this is a benchmark-scoped heuristic that happens to be honest about its own limits). Resolution used in this spec: **modest framing for every technical claim and judge Q&A answer; the ambitious framing is allowed only as a one-line closing remark in the demo**, not as a claim anyone has to defend under questioning.

**Blind spots the peer-review round caught that no single advisor saw alone:**
- The original pitch only gave a completeness certificate to the *exhaustive* class (aggregation/superlative) — leaving `existential` (lookup/multi_hop) and `chained` (temporal) with no completeness story at all, i.e. no differentiation on roughly 69% of the public question set. **Fixed in §4 below: every class gets a certificate.**
- Nobody had a concrete reconciliation procedure for the graph-count-vs-gold divergence risk beyond "validate it." **Fixed in §6: reconciliation is now the literal first deliverable, before any agent code.**
- Nobody scripted what to say if a certificate and the gold answer disagree live in front of judges. **Fixed in §6.**
- Nobody priced in the risk of learning TigerGraph Savanna itself (a new platform for this builder) as separate from the data-modeling work. **Fixed in §7 (risks).**

## 4. System design (post-council)

### 4.1 Evidential-completeness classes (all three now certified, not just one)

| Class | Maps to qtype | What "complete" means | Retrieval mode | Certificate content |
|---|---|---|---|---|
| **Existential** | `lookup`, `multi_hop` | Exactly the right single entity was found and matched (venue+date, or event title) | Targeted graph traversal / exact-match lookup, not vector top-k | Match key used (e.g. venue+date), candidate count at that key (should be 1; if >1, flag ambiguity), doc id resolved |
| **Chained** | `temporal` | Every hop in the `prev`/`next` chain was actually traversed and exists; no missing link was silently skipped | Bounded hop-following (start event → `prev` → target Games' event) | Hop sequence taken, whether every hop resolved to an existing node, fallback used if a hop was missing |
| **Exhaustive** | `aggregation`, `superlative` | The full candidate set matching the filter predicate (sport + games) was enumerated, not sampled | Deterministic GSQL `COUNT`/traversal query over the exact predicate | The predicate used, structural bound (`COUNT` from the graph), size of the evidence set the agent actually inspected, pass/fail equality check |

This directly answers peer-review finding #1 (only 1 of 3 classes had a story) — now all three do, each with a class-appropriate notion of "complete," not one generic template.

### 4.2 Orchestrator (kept deliberately simple, per council's "don't oversell" note)

The orchestrator is explicitly documented as **a benchmark-scoped heuristic dispatcher, not a general-purpose learned router.** It reads the question, extracts the filter fields it needs by simple pattern matching against known templates (this benchmark hands you 5 fixed templates — pretending otherwise would be dishonest), assigns a completeness class, and dispatches to the matching retrieval mode. Where the write-up or demo discusses "agentic effectiveness," the honest claim is: *the system spends exhaustive-search cost only where the question actually requires it, and can prove that it did.* That is a real, defensible, rubric-aligned claim; "we invented a general query router" is not, and should not be claimed.

### 4.3 Investigation Certificate schema (v0, all classes)

```json
{
  "qid": "pub-045",
  "qtype": "aggregation",
  "completeness_class": "exhaustive",
  "retrieval_mode": "gsql_count",
  "predicate": {"sport": "Athletics", "games": "2004 Summer"},
  "structural_bound": 41,
  "evidence_set_size": 41,
  "completeness_check": "pass",
  "docs_inspected": ["Q...", "Q...", "..."],
  "tokens": {"context": 0, "input": 812, "output": 64, "total": 876},
  "latency_ms": 340,
  "stop_reason": "structural_bound_met"
}
```

For `existential`/`chained` classes, `structural_bound` and `evidence_set_size` are both 1 (or the hop count), and `completeness_check` verifies exact-match resolution rather than a count equality.

### 4.4 Three-way benchmark integration

The certificate is emitted for the **agentic** pipeline only; the RAG and vanilla-GraphRAG pipelines are run unmodified and will simply show, in the same dashboard, incomplete evidence sets on `aggregation`/`superlative` questions (e.g. RAG's top-8 similarity hits covering 8 of 41 matching events). The three-way comparison the hackathon requires becomes the certificate's own proof: put RAG's and GraphRAG's retrieved-doc counts next to the agentic pipeline's `structural_bound`, per question, in the metrics dashboard.

## 5. Why this is feasible in ~3 weeks (Executor's plan, adopted as-is)

1. **Days 1–3:** Ingest `data/corpus.jsonl` infoboxes into TigerGraph (Games, Sport, Event, Venue, Athlete, NOC nodes; `PART_OF`, `HELD_AT`, `IN_SPORT`, `GOLD`, `PREV`/`NEXT` edges). Get `COUNT`-with-filter-predicate GSQL queries working on the exact predicates the benchmark will test.
2. **Days 3–5 (the real bottleneck, per council):** **Reconciliation pass, not agent code.** For all 100 questions in `data/eval_public.jsonl`, run the structural query implied by each question and diff the result against `gold_doc_ids`. Fix every ingestion/schema bug this surfaces before writing any agent logic. This step did not exist in the original pitch and is now mandatory — see §6.
3. **Week 2:** Build the three pipelines (RAG, GraphRAG, Agentic). Wire the Investigation Certificate as a logging side-effect of the orchestrator's dispatch decision — it is not a separate subsystem, it is nearly free once the exhaustive-query path exists (needed anyway to grade aggregation/superlative questions).
4. **Week 3:** Metrics dashboard (per-qtype, per-pipeline: accuracy, completeness, tokens, certificate pass/fail rate), demo video, architecture diagram, repo cleanup. **Test the classifier/router against all 150 questions before recording any demo footage** — a misrouted aggregation question would make the certificate publicly prove its own failure, which is a feature during judging but a landmine if it happens live and unexplained.

## 6. Mandatory pre-build step: reconciliation, and the scripted disagreement answer

Before writing orchestrator or agent code:

1. Load the corpus into TigerGraph.
2. For every public question, compute the structural bound TigerGraph reports for its filter predicate.
3. Diff against `gold_doc_ids` count in `data/eval_public.jsonl`.
4. Any mismatch is a data/schema bug — fix it now, log it in `docs/status.md`.
5. Record the final reconciliation result (e.g. "100/100 match" or "97/100 match, 3 known gaps: <list>") as a committed artifact (`data/reconciliation-report.json` or similar) that ships with the submission.

**Scripted answer if a judge finds a live disagreement:** "Here is our reconciliation log from build time — we validated the graph's structural count against all 100 public gold answers before demoing. If you've found a new case, it's most likely an infobox field we didn't parse (see `docs/hackathon-brief.md` §7's noted traps: missing dates, non-numeric competitor counts, venue ambiguity) — here's how we'd extend ingestion to cover it." Never claim the certificate is infallible; claim it is validated against a known, disclosed set, with known limitations.

## 7. Risks (updated per council peer-review findings)

| Risk | Source | Mitigation |
|---|---|---|
| Graph COUNT diverges from gold count | 4/5 advisors, independently | §6 reconciliation pass before any agent code |
| Classifier is template-matching dressed as a classifier | First Principles, Outsider | Document it honestly as a benchmark-scoped heuristic (§4.2); don't claim generality |
| Only exhaustive class had a certificate story | Cross-review finding | §4.1 now certifies all three classes |
| TigerGraph Savanna is a new platform for this builder — learning curve is separate from data-modeling time | Cross-review finding | Budget day 1 as platform ramp-up, not schema design; use TigerGraph MCP + GSQL vector-search tutorial (see `docs/hackathon-brief.md` §6) to move faster |
| No scripted answer for live certificate/gold disagreement | Cross-review finding | §6 scripted answer |
| Ambiguous questions that don't cleanly fit one of the 3 classes | Cross-review finding | Default to `existential` (cheapest, safest) and log a `"classification_confidence": "low"` field in the certificate rather than silently guessing |
| Overselling novelty vs. RL-routing papers under judge questioning | First Principles, Executor | Literature positioning is a one-slide footnote, not a defended claim (§4.2) |

## 8. Judging-rubric alignment (explicit, since no advisor scored this numerically)

| Criterion | Weight | How this idea earns it |
|---|---|---|
| Investigation accuracy | 30% | Reconciliation pass (§6) directly protects this; exhaustive-class certificates catch undercounting before it reaches the scored answer |
| Evidence quality & explainability | 15% | The certificate IS the explainability artifact — a structural, falsifiable citation, not an LLM-narrated one |
| Agentic effectiveness & efficiency | 15% | Cheap classes get cheap retrieval; exhaustive class only pays exhaustive-search cost when the question needs it, and proves it needed it |
| Engineering, code quality | 15% | Certificate schema (§4.3) is a concrete, testable artifact; reconciliation report is a committed test fixture |
| Innovation | 15% | The completeness-vs-correctness distinction, demoed via a falsifiable certificate, is the differentiator — framed modestly, not against the RL papers |
| Presentation | 10% | Dashboard's side-by-side "RAG found 8/41, agentic pipeline proved 41/41" is a one-slide, non-technical-judge-legible visual |
