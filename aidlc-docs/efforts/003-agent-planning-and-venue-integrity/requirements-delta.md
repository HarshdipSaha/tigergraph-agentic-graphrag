# Requirements Delta: 003-agent-planning-and-venue-integrity

> **Baseline:** `aidlc-docs/inception/01-requirements.md`  
> **Source of change:** reviewer feedback supplied by the user on 2026-10-07

## Reviewer's intent and the actual defect

The reviewer wants the agent to select investigation tools, and wants venue resolution constrained by date and sport. The current five regex patterns already choose the tool for every templated benchmark question. Jev is called only for an unmatched question or an ambiguous candidate selection, and the unmatched path classifies intent but still runs GraphRAG. The pipeline therefore offers little visible planning.

The venue issue has two layers. In `agrag/tools.py`, relaxed substring matching accepts records whose venue is empty, because the empty string is a substring of every venue name. More fundamentally, `agrag/infobox.py` accepts an Olympic event infobox only when it starts the document. The 2004 tennis document `Q942805` has a tennis tournament infobox first and then an Olympic event infobox with **venue `Olympic Tennis Centre`, date `15 to 22 August 2004`, and gold `Li TingSun Tiantian`**. It is present as a Document but absent as an Event in the current graph. `eval-001` consequently chooses rowing record `Q735286`, whose venue and date are both empty. The hidden set has no gold labels, so the older validation report cannot establish that this rowing answer is correct.

## New / changed requirements

| ID | Requirement | Acceptance signal |
|---|---|---|
| RQ-003-01 | For the agentic mode, the generative LLM must choose a named tool and supply its arguments from the question. The five existing graph tools remain the execution layer. | A trace records `planner_select` with selected tool/arguments for a representative question of each of the five classes; the question-file `qtype` is not shown to the planner. |
| RQ-003-02 | The orchestrator must validate each proposed action, execute only an allowlisted tool, inspect its result, and permit a bounded follow-up action when evidence is insufficient. | Invalid JSON, unknown tool, missing slot, repeated action, and planner timeout have deterministic outcomes; no model text executes as code or becomes a certified answer. |
| RQ-003-03 | The parser must find the Olympic event infobox even when it is the second infobox, and handle `date` and `dates` without changing canonical answer strings. | `Q942805` parses into an Event with the exact corpus venue/date/gold; the 25 previously omitted Olympic-infobox documents are audited; local and TigerGraph event views agree after refresh. |
| RQ-003-04 | Venue fallback must reject empty candidate venues, require a defensible venue relationship, filter by Games and date, and apply a sport constraint only when the question or a documented venue cue actually supplies one. | `eval-001` selects `Q942805` after graph refresh; unrelated rowing/cycling records cannot enter the candidate set; missing-date candidates cannot win a date-specific query. |
| RQ-003-05 | Certificates must distinguish a uniquely supported entity from a model choice among unresolved candidates. Jev confidence is a decision signal, not proof of uniqueness. | A same-venue/same-date tie is marked `unverified` or `fail`, with candidate IDs and the reason, even if a model supplies an answer; `pub-060` is never labeled a structurally proved pass. |
| RQ-003-06 | Exhaustive aggregation/superlative answers must retain their complete graph-derived population and exact canonical answer formatting. | Public reconciliation does not regress on aggregation/superlative; actual tool and predicate are recorded in each certificate. |
| RQ-003-07 | The submitted result files, metrics, architecture diagram, README, and any statement about `eval-001` must reflect the final code and live TigerGraph state. | Public/hidden agentic evaluations are rerun into reviewed outputs; trace shows planning; no stale `99%`, `18 tokens`, or `eval-001` correctness claim is presented as a new measurement. |

## Limits that must remain explicit

- `pub-060` asks only for the event at ExCeL on 30 July 2012. The corpus has at least three exact venue/date matches across judo and fencing, and the question gives no sport. Date-plus-sport filtering cannot make this question unique without adding information that is not in the question. A model may choose, but that is an uncertain choice, not a certificate of correctness.
- The hidden file contains no answer labels. Hidden-set completeness and parser checks are useful, but they are not hidden-set accuracy measurements.
- The current TigerGraph GSQL query returns the sport/Games Event set; `aggregation_scan` and `superlative_scan` calculate count/max over that complete set in Python. Preserve that behavior under the deadline and describe it accurately. No new schema or GSQL query is required for this effort.
