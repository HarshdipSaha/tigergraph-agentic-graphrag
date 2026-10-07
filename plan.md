# Agent Planning and Venue Integrity Implementation Plan

> **For agentic workers:** Implement the checked steps in order. Use subagent-driven development when subagents are available, or execute the plan with review checkpoints in a single session. The AI-DLC planning record is [`aidlc-docs/efforts/003-agent-planning-and-venue-integrity/`](aidlc-docs/efforts/003-agent-planning-and-venue-integrity/effort-state.md).

**Goal:** Let the LLM select the existing graph tools for agentic questions, repair the missing Olympic Event records, and make venue/date/sport resolution evidence-safe while preserving complete aggregation and canonical answers.

**Architecture:** Parse every Olympic infobox into the Event graph, refresh Event vertices and edges without rebuilding vectors, then put a small validated LLM action planner in front of the five existing tools. The executor and certificate evaluator remain deterministic. A model choice among unresolved candidates is recorded as uncertain rather than certified as a unique match.

**Tech stack:** Python 3.12, Pydantic 2, Groq through the existing `LLM` protocol, optional Jev decision model, TigerGraph GSQL queries, pytest.

**Execution status:** implementation, live validation, evaluations, and supporting documentation are complete on `improvement-agentic-planner`. See the Effort 003 [validation report](aidlc-docs/efforts/003-agent-planning-and-venue-integrity/validation-report.md). Commit steps are intentionally left for the owner.

---

## 0. Read this before changing code

The current path is `router.route()` → a fixed tool branch in `AgenticPipeline.answer()` → optional Jev/LLM recovery → `Certificate`. For benchmark-shaped questions the LLM does not plan. The unmatched-question Jev branch classifies a category but still runs GraphRAG, so it does not use the category to execute the corresponding graph tool.

The hidden question `eval-001` asks about **Olympic Tennis Centre, 15 to 22 August 2004**. The source document `Q942805` contains exactly that Olympic venue/date and gold `Li TingSun Tiantian`, but it starts with `[Infobox tennis tournament event]` and has `[Infobox Olympic event]` second. `event_from_doc()` accepts only the first header. The Event graph consequently lacks `Q942805`; the current relaxed venue code admits empty venues (`"" in venue`), and the saved result chose rowing `Q735286`, whose venue and date are empty. The hidden file has no gold label, so treat the existing “correct” claim as unsupported. A local audit found **25** title-parseable documents with this second-infobox pattern. The current parser yields **2,162 Event records** from 2,951 documents; 81 parsed Events have blank venues and 911 have blank dates.

`pub-060` is a different limit. ExCeL Exhibition Centre on 30 July 2012 has at least three exact-date Event records: fencing `Q1156695` (public gold), judo `Q1064016`, and judo `Q1005551`. The question does not name a sport. Date filtering excludes the currently chosen wrong-date `Q932145`, but no valid sport filter can make the three-event set unique. Do not promise 100/100 public accuracy or a clean certificate for that question.

**Implementation rules**

- Preserve the existing RAG and GraphRAG pipelines as comparison baselines.
- Keep the five existing tool functions; wrap them rather than replacing graph logic with LLM-generated answers.
- Never show `Question.qtype`, `answer`, or `gold_doc_ids` to the planner. Gold is for evaluation only.
- Return canonical `EventRecord.gold`, `title`, or numeric results. Never take an answer string from planner prose.
- Do not rewrite `results/*` until the new code and live graph have been validated. Use new output paths during comparison.
- No schema migration or vector re-embedding is needed. The installed graph queries already expose Event attributes.
- The five configured Groq keys are under one organization. Use key rotation for project/key-specific authentication or quota failures; keep the organization-wide limit as a hard ceiling and do not retry in a way that bypasses it.

## 1. File map

| File | Responsibility / change |
|---|---|
| `agrag/infobox.py` | Find the Olympic infobox after another leading infobox; support `date` and `dates`. |
| `tests/test_infobox.py` | Fixture and full-corpus regression for secondary Olympic infoboxes. |
| `scripts/tg_refresh_events.py` (new) | Reparse source docs and upsert only Event-related vertices/edges. Do not read `embed_cache.pkl`. |
| `agrag/tools.py` | Safe venue aliases, hard date eligibility, optional grounded sport, explicit candidate state. |
| `tests/test_tools.py` | Blank venue, exact venue, range dates, shared-venue tie, sport filtering. |
| `agrag/planner.py` (new) | Tool catalog, typed JSON action, validation, compact prompt and observations. |
| `tests/test_planner.py` (new) | Valid/invalid actions, slot provenance, FakeLLM response handling. |
| `agrag/pipelines/agentic.py` | Planner-first bounded loop, tool adapter, existing recovery, final certificate. |
| `tests/test_pipeline_agentic.py` | End-to-end offline action trace, fallback, tokens, ambiguity, all five classes. |
| `agrag/certificate.py` | Optional planning and venue-resolution provenance fields. |
| `agrag/eval/run.py` | Explicit agent mode and distinct output paths for fair before/after runs. |
| `agrag/eval/report.py` | Accept explicit result paths so template/planner files cannot be silently mixed. |
| `dashboard/app.py`, `scripts/make_results_chart.py` | Read only the reviewed final result set and regenerate the benchmark graphic. |
| `README.md`, `docs/status.md`, `docs/demo/*`, `docs/submission*`, `results/*` | Update only from measured final behavior; correct stale claims in demo and submission copy. |

For task commits, stage only files touched by that task. The workspace already contains unrelated untracked submission/spec files; leave them alone unless a later documentation task explicitly updates them.

## 2. Task A — Parse secondary Olympic infoboxes

**Test first**

- [x] In `tests/test_infobox.py`, add an inline `Doc` with a tennis-tournament infobox followed by an Olympic-event infobox. Assert `event_from_doc()` returns the Olympic `venue`, `date_text`, `gold`, `sport`, `games`, and `doc_id`.
- [x] Add a `dates:`-only variant, asserting `date_text` is populated. Add a first-infobox control and a film/no-Olympic-infobox control.
- [x] If `data/corpus.jsonl` is present, assert `Q942805` parses with venue `Olympic Tennis Centre`, date `15 to 22 August 2004`, and gold `Li TingSun Tiantian`. Count the 25 formerly omitted pages before changing the parser; after the change, verify each becomes an Event. Keep the test skippable if the ignored corpus is unavailable in CI.
- [ ] Run `python -m pytest tests/test_infobox.py -q`. The new tests must fail before implementation.

**Implementation sketch** in `agrag/infobox.py`:

```python
def parse_olympic_infobox(text: str) -> dict[str, str]:
    """Select the Olympic block; preserve parse_infobox() for first-block callers."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == EVENT_HEADER:
            header, fields = parse_infobox("\n".join(lines[i:]))
            if header == EVENT_HEADER:
                return fields
    return {}


def event_from_doc(doc: Doc) -> Optional[EventRecord]:
    ib = parse_olympic_infobox(doc.text)
    parsed = parse_title(doc.title)
    if not ib or parsed is None:
        return None
    # Keep the existing EventRecord construction below this guard.
    # Replace only: date_text=ib.get("date") or ib.get("dates", "")
```

`parse_infobox()` is used by `agrag/graph/load.py` to label Document kinds, so do not change its existing first-block meaning accidentally. Confirm that the secondary block parser stops at the next non-indented line; the delegated `parse_infobox()` already does this.

- [x] Run `python -m pytest tests/test_infobox.py -q`; expect pass.
- [x] Run `python scripts/reconcile.py --backend local --out data/reconciliation-local-planner-preflight.json`; inspect new mismatches, especially tennis-related ones. This is a check, not the final accuracy measurement.
- [ ] Commit parser and parser tests as one focused change if using commits.

## 3. Task B — Refresh only the live Event graph

The existing `scripts/tg_load.py` restores **events and links** from `data/embed_cache.pkl`. Running it unchanged after Task A would keep the old Event list. Add `scripts/tg_refresh_events.py` to reparse documents while retaining the existing Document/Chunk/vector data:

```python
from agrag.corpus import load_docs
from agrag.graph.client import connect
from agrag.graph.load import load_events
from agrag.infobox import events_from_docs, resolve_links


def main() -> None:
    docs = load_docs("data/corpus.jsonl")
    events = events_from_docs(docs)
    prev, _ = resolve_links(events)
    conn = connect()
    if conn.getVertexCount("Document") == 0:
        raise RuntimeError("Load the corpus Documents before refreshing Events")
    before = conn.getVertexCount("Event")
    load_events(conn, events, prev)
    after = conn.getVertexCount("Event")
    print(f"Event count: {before} -> {after}; parsed={len(events)}")
    if after != len(events):
        raise RuntimeError("Live Event count differs from parsed source count")


if __name__ == "__main__":
    main()
```

`load_events()` upserts Event, Games, Sport, Venue and Athlete vertices plus relevant edges; it does not regenerate chunks or embeddings. Upsert means repeat runs should be safe. Verify the installed `event_by_title` query returns `Q942805` with the exact venue/date/gold after the refresh. For a live query, `TigerGraphBackend.from_settings().event_by_title("Tennis at the 2004 Summer Olympics – Women's doubles")` is sufficient. A graph connection failure is a deployment issue, not permission to fabricate a passing result.

- [x] Run `python scripts/tg_refresh_events.py`; confirm the printed count matches the newly parsed local Event count.
- [x] Run `python scripts/reconcile.py --backend tigergraph --out data/reconciliation-tigergraph-planner-preflight.json`; compare local/live mismatches.
- [x] Event ID verification found 2,187 parsed and live Events with no missing/unexpected IDs; the targeted refresh left Chunk embeddings untouched.
- [ ] Commit the refresh script after it is verified on the live graph.

## 4. Task C — Make venue matching safe

Extend `ToolResult` with **optional**, backward-compatible multi-hop provenance. Keep the existing `candidates` field as the **eligible** set passed to a model; add raw count and why selection was or was not unique:

```python
@dataclass
class ToolResult:
    # existing fields stay in their current order
    # ...
    raw_candidate_count: int = 0
    eligible_count: int = 0
    match_mode: str = "none"       # exact | alias | none
    selection_basis: str = "none"  # unique_exact | unique_alias | model_choice | none
    ambiguity_reason: str = ""
```

Do not use `date_score()` as the sole proof of eligibility. Put a small pure date parser in `agrag/tools.py` or a focused `agrag/date_match.py`. It needs these outcomes: `exact_range`, `same_day`, `day_in_range`, `overlap_uncertain`, `disjoint`, `unknown`. Normalize en/em dashes to hyphens, read a year from the date text or the Games year, and cover day-first/month-first points and ranges seen in the corpus. Never treat a blank event date as a positive match. For concatenated stage dates, consider every parsed date rather than only the first. If parsing cannot establish a relationship, return `unknown` and avoid a certified unique match.

A minimal interface keeps the resolver readable:

```python
from typing import Literal

DateRelation = Literal[
    "exact_range", "same_day", "day_in_range",
    "overlap_uncertain", "disjoint", "unknown",
]

def compare_dates(question_date: str, event_date: str, games_year: int) -> DateRelation:
    # Parse all supported date spans with datetime.date; reject impossible dates.
    # Exact normalized range or equal parsed endpoints -> exact_range.
    # Equal point dates -> same_day.
    # Question point inside event interval -> day_in_range.
    # Overlapping but non-equal ranges -> overlap_uncertain.
    # Known non-overlap -> disjoint; unparseable/blank -> unknown.
    ...
```

Resolver order:

```python
def resolve_multi_hop(b: GraphBackend, pq: ParsedQuestion) -> ToolResult:
    slots = pq.slots
    year = slots.get("year")
    if type(year) is not int:
        return ToolResult("resolve_multi_hop", None, [], 0, dict(slots),
                          needs_llm=False, notes=["missing valid Games year"])
    games = ([f"{year} {slots['season']}"] if slots.get("season")
             else [f"{year} Summer", f"{year} Winter"])
    # 1. Fetch exact venue Events in these Games.
    # 2. Only if none: bounded alias comparison over nonempty venue strings.
    # 3. Deduplicate by doc_id.
    # 4. Reject disjoint/unknown dates for a date-specific certified answer.
    # 5. Apply sport only when grounded; keep its source in predicate/notes.
    # 6. If exactly one eligible Event: return its canonical gold.
    # 7. Otherwise return no deterministic answer and the eligible candidate IDs.
```

The alias check must require `norm_key(query_venue)` **and** `norm_key(event.venue)` to be nonempty. Prefer token-boundary aliases or a small explicit venue alias map; a raw one-character substring is too broad. If `sport` is supplied by the planner, validate that it is explicitly named in question text outside the venue/date span or belongs to a reviewed venue-cue map. Ignore or reject an ungrounded sport rather than silently narrowing the graph. If no sport was supplied, preserve cross-sport candidates.

For a unique match set `selection_basis=unique_exact` or `unique_alias`, `evidence=[doc_id]`, `candidates` to the single eligible Event, and `eligible_count=1`. For a tie, set `answer=None`, `candidates` to **all** eligible Events, `evidence` to their IDs, and `ambiguity_reason`. Preserve `structural_bound` as the raw venue population size; certificate uniqueness depends on `eligible_count`, not `structural_bound == evidence_set_size` for an existential question. Migrate the old `tests/test_tools.py` assertion that a unique date result still has four `candidates`: assert `raw_candidate_count == 4`, `eligible_count == 1`, and one eligible candidate instead.

**Tests** in `tests/test_tools.py`:

- [x] `Q942805` exact venue/range resolves after parsing; `Q735286` and the two cycling empty-venue rows cannot match.
- [x] Exact day `30 July` excludes `29 July`, while the three ExCeL same-day candidates remain a tie.
- [x] An explicit sport at a shared venue narrows the set; no sport keeps the tie.
- [x] A blank or unparseable event date never wins a date-specific query. Two partially overlapping ranges stay uncertain.
- [x] Both LocalBackend and TigerGraphBackend return compatible Event fields for the same source IDs (integration check when live graph is available).
- [x] Run `python -m pytest tests/test_tools.py -q`, then the local public reconciliation again; inspect any changed answers instead of relaxing predicates to make a test green.

## 5. Task D — Add an LLM action planner

Use one `LLM.complete(system, user)` call per normal question, with a compact catalog for the five existing graph tools. The model outputs an **action**, never an answer. Start with JSON text because `GroqLLM`, `AnthropicLLM`, and `FakeLLM` currently share only the `complete()` protocol. A provider-specific structured output adapter can be considered later if plain JSON proves unreliable.

Example expected output:

```json
{"tool":"aggregation_scan","args":{"sport":"athletics","year":2004,"season":"Summer","threshold":41},"reason":"count events above the stated competitor threshold"}
```

In `agrag/planner.py`, make the allowlist and schema explicit:

```python
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

ToolName = Literal[
    "lookup_nations", "resolve_multi_hop", "temporal_chain",
    "aggregation_scan", "superlative_scan",
]

REQUIRED = {
    "lookup_nations": {"title": str},
    "resolve_multi_hop": {"venue": str, "date": str, "year": int},
    "temporal_chain": {"event_phrase": str, "year": int, "season": str},
    "aggregation_scan": {"sport": str, "year": int, "season": str, "threshold": int},
    "superlative_scan": {"sport": str, "year": int, "season": str},
}
OPTIONAL = {"resolve_multi_hop": {"season": str, "sport": str}}


class PlannerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: ToolName
    args: dict[str, Any]
    reason: str = Field(default="", max_length=160)


def validate_action(action: PlannerAction, question: str) -> PlannerAction:
    required = REQUIRED[action.tool]
    optional = OPTIONAL.get(action.tool, {})
    if set(action.args) - set(required) - set(optional):
        raise ValueError("unknown tool argument")
    for name, typ in required.items():
        if name not in action.args or type(action.args[name]) is not typ:
            raise ValueError(f"missing or invalid {name}")
    for name, value in action.args.items():
        if type(value) is not (required | optional)[name]:
            raise ValueError(f"invalid {name}")
        if isinstance(value, str) and (not value.strip() or len(value) > 240):
            raise ValueError(f"invalid length for {name}")
    if "season" in action.args and action.args["season"] not in {"Summer", "Winter"}:
        raise ValueError("invalid season")
    if "year" in action.args and not 1890 <= action.args["year"] <= 2100:
        raise ValueError("invalid year")
    if "threshold" in action.args and not 0 <= action.args["threshold"] <= 100_000:
        raise ValueError("invalid threshold")
    # Add sport provenance and consistency checks before returning.
    return action
```

Use `json.loads(resp.text)`, then `PlannerAction.model_validate(...)`, then `validate_action(...)`, then a **question/action consistency check**. Schema validation alone is insufficient: a valid `aggregation_scan` on a venue question could produce a complete graph count and a false `pass`. Define the shared `TOOL_TO_TEMPLATE` mapping in `agrag/planner.py` and import it into the agent adapter. Use `router.route(question)` only **after** the LLM chooses an action, as an independent guard when the question matches a known benchmark template. The regex must not select the normal-path tool. For a matched template, reject a different selected tool and compare every extracted argument with the planner arguments after safe normalization; a planner may omit an optional `season` that the question omits, but must not invent it. For a non-template question, require year/season/threshold to appear literally in the question and text arguments to be anchored in its normalized wording, or mark the action unverified and replan. A `sport` on a venue question is allowed only if it appears outside the venue/date wording or comes from a reviewed venue-cue mapping. Keep the grounding outcome in the trace and never certify a result whose requested predicate cannot be tied to the question.

```python
def validate_grounding(action: PlannerAction, question: str) -> None:
    parsed = route(question)  # consistency guard after selection, not the planner
    if parsed.template is not None:
        if TOOL_TO_TEMPLATE[action.tool] != parsed.template:
            raise ValueError("selected tool contradicts question intent")
        for key, expected in parsed.slots.items():
            actual = action.args.get(key)
            if expected is None:
                if actual is not None:
                    raise ValueError(f"ungrounded {key}")
            elif type(expected) is int:
                if type(actual) is not int or actual != expected:
                    raise ValueError(f"wrong {key}")
            elif normalize(str(actual)) != normalize(str(expected)):
                raise ValueError(f"wrong {key}")
    else:
        qnorm = normalize(question)
        for key, value in action.args.items():
            if key == "sport" and action.tool == "resolve_multi_hop":
                continue  # check outside venue/date spans below
            if normalize(str(value)) not in qnorm:
                raise ValueError(f"ungrounded {key}")
    sport = action.args.get("sport")
    if action.tool == "resolve_multi_hop" and sport:
        outside_venue_date = normalize(question)
        for key in ("venue", "date"):
            outside_venue_date = outside_venue_date.replace(
                normalize(action.args[key]), ""
            )
        if normalize(sport) not in outside_venue_date:
            raise ValueError("ungrounded venue-question sport")
```

This is the conservative first version: a paraphrased non-template argument that cannot be grounded by a literal span is rejected and can use a labeled unverified fallback. The exhaustive tools' `sport` is checked like their other arguments; only the venue tool needs the special outside-venue/date check. If a reviewed venue-cue map is later added, allow only its explicit entries and record `sport_source=venue_cue`. This guard exposes planner mistakes through a `planner_action_rejected` trace step, followed by a bounded replan or labeled template fallback. Do not show the parsed template to the planner as its initial choice. Do not use `eval()`, dynamic imports, a Python expression returned by the model, or a free-form tool name. Build a short system prompt listing only the five tool signatures and this JSON shape. Include a few examples spanning lookup, multi-hop, temporal, and exhaustive questions. Keep retrieved prose and candidate gold values out of the planner prompt.

Bound every planner completion to **15 seconds** at both the SDK request layer and the retry loop. Extend `LLM.complete()` with an optional keyword-only `timeout_s`; use Groq's per-request `with_options(timeout=remaining)` and a monotonic deadline, and clamp every retry sleep to the remaining budget. Do not use a thread-pool timeout around a synchronous HTTP request because it cannot reliably cancel the request. Raise a dedicated timeout exception when the budget expires. Record `planner_timeout` in the trace; use the validated template fallback only when it is safe, otherwise return the accumulated evidence as unverified. Two planner calls therefore have a bounded combined API wait of 30 seconds per question. For key rotation, try the other configured keys when a key is invalid or a project-specific limit is hit; if all keys in the organization are exhausted or Groq reports the organization cap, stop with a clear rate-limit outcome rather than sleeping for an hour or claiming more quota.

Build an observation message for a second call only when the first tool cannot finish. Example:

```json
{"last_tool":"resolve_multi_hop","status":"ambiguous_candidates","candidate_count":3,"candidates":[{"id":"Q1156695","sport":"Fencing","date":"30 July"},{"id":"Q1064016","sport":"Judo","date":"30 July 2012"},{"id":"Q1005551","sport":"Judo","date":"30 July 2012"}]}
```

Do not include `gold` in that observation. Allow at most two planner responses and two graph tool executions per question; reject the same action repeated with identical arguments. A second action is useful when it adds a question-grounded refinement omitted from the first venue query, such as an explicitly named sport. When no distinct supported action exists, stop with the current evidence and an unverified outcome; do not make a ceremonial second call. Existing competitor-field recovery may inspect each unresolved document separately, but never mark an exhaustive scan complete while any required count remains unresolved. On malformed JSON, API error, or invalid slots, record the failure and use the existing regex route as an explicit fallback if it matches; otherwise return an unverified result or the existing GraphRAG fallback, clearly labeled. An already-known ambiguous result must not be handed to the old template path for a clean pass.

**Tests** in `tests/test_planner.py`:

- [x] One valid action for each tool; serializes to `ParsedQuestion` with the right completeness class.
- [x] Invalid JSON, an unknown tool, wrong types (`true` is not an integer year), missing threshold, extra argument, invalid season, and giant strings are rejected.
- [x] No `qtype`, gold answer, or `gold_doc_ids` appears in the prompt generated from a `Question`.
- [x] An ungrounded sport is rejected/ignored with a traceable reason; a grounded explicit sport is accepted.
- [x] A non-template aggregation/superlative action with an invented sport is rejected before it can receive an exhaustive pass.
- [x] A schema-valid but semantically wrong tool is rejected: `aggregation_scan` for a venue question must not yield a certified count. Wrong but well-typed venue/date/year/title/threshold arguments are also rejected.
- [x] A repeated action is stopped after the configured budget; model input/output tokens are counted even when the response is invalid.
- [x] Planner timeout stops within the configured budget, is recorded in trace, and follows the documented safe fallback. Groq key rotation tests cover a failed key and all keys exhausted without logging credentials.
- [x] Use `FakeLLM` to keep these tests offline. Run `python -m pytest tests/test_planner.py -q`.

## 6. Task E — Make the agent execute planner actions

Add a small adapter from a validated action to the existing `ParsedQuestion` and tool function. Do not route the question first and then ask the planner to echo the chosen template.

```python
# Define this mapping once in agrag/planner.py and import it here.
TOOL_TO_TEMPLATE = {
    "lookup_nations": "lookup", "resolve_multi_hop": "multi_hop",
    "temporal_chain": "temporal", "aggregation_scan": "aggregation",
    "superlative_scan": "superlative",
}
TOOL_FUNCTIONS = {
    "lookup_nations": lookup_nations, "resolve_multi_hop": resolve_multi_hop,
    "temporal_chain": temporal_chain, "aggregation_scan": aggregation_scan,
    "superlative_scan": superlative_scan,
}

def execute_action(backend: GraphBackend, question: str, action: PlannerAction) -> tuple[ParsedQuestion, ToolResult]:
    template = TOOL_TO_TEMPLATE[action.tool]
    slots = dict(action.args)
    if template == "multi_hop":
        slots.setdefault("season", None)
        slots.setdefault("sport", None)
    pq = ParsedQuestion(question, template, CLASS_OF[template], slots)
    return pq, TOOL_FUNCTIONS[action.tool](backend, pq)
```

Change `AgenticPipeline.answer()` so planner mode is the default agentic mode. Keep the old template implementation callable as a labeled fallback/baseline path. Migrate the old `FakeLLM([])` fixture cases in `tests/test_pipeline_agentic.py` and `tests/test_decision_jev.py` to pass `mode="template"` explicitly; add new `FakeLLM` action scripts for planner mode. The old ambiguous-venue test expects `pass_with_llm_recovery`; update it to `unverified` because a model choice is not structural proof. A useful skeleton is:

```python
@dataclass(frozen=True)
class EvidenceVerdict:
    check: str
    can_stop: bool
    reason: str

def evaluate_evidence(pq: ParsedQuestion, r: ToolResult) -> EvidenceVerdict:
    if pq.template == "multi_hop":
        if r.eligible_count == 1 and r.answer is not None:
            check = "pass" if r.match_mode == "exact" else "pass_with_fallback"
            return EvidenceVerdict(check, True, "unique_supported_event")
        reason = "ambiguous_candidates" if r.eligible_count > 1 else "no_supported_event"
        return EvidenceVerdict("unverified", False, reason)
    if pq.template in {"aggregation", "superlative"}:
        complete = (len(r.evidence) == r.structural_bound
                    and not r.unresolved and r.answer is not None)
        return EvidenceVerdict("pass" if complete else "fail", complete,
                               "structural_bound_met" if complete else "missing_values")
    if pq.template == "temporal":
        complete = r.answer is not None and len(r.hops) == 2
        check = "pass_with_fallback" if complete and r.fallback_used else "pass"
        return EvidenceVerdict(check if complete else "fail", complete,
                               "chain_complete" if complete else "chain_incomplete")
    complete = r.answer is not None and len(r.evidence) == 1
    return EvidenceVerdict("pass" if complete else "fail", complete,
                           "unique_supported_event" if complete else "lookup_unresolved")

def answer(self, q: Question) -> PipelineResult:
    if self.mode == "template":
        return self._answer_template(q)  # old behavior, clearly labeled
    steps: list[Step] = []
    tokens = TokenUsage()
    seen_actions: set[str] = set()
    observation = None
    last_result = None
    planner_failed = False
    for _ in range(2):
        action, planner_step = self.planner.select(q.question, observation)
        steps.append(planner_step)
        tokens = tokens + planner_step.tokens
        if action is None:  # malformed/API failure
            planner_failed = True
            break
        key = action.model_dump_json()
        if key in seen_actions:
            break
        seen_actions.add(key)
        pq, tool_result = execute_action(self.b, q.question, action)
        last_result = (pq, tool_result)
        steps.append(Step(tool=tool_result.tool, note="; ".join(tool_result.notes)))
        verdict = evaluate_evidence(pq, tool_result)
        if verdict.can_stop:
            return self._finish(q, pq, tool_result, verdict, steps, tokens)
        if not has_supported_refinement(q.question, tool_result):
            break
        observation = compact_observation(tool_result, verdict)
    if last_result is not None:
        return self._finish_unverified_or_model_choice(q, last_result, steps, tokens)
    if planner_failed:
        return self._recorded_template_fallback(q, steps, tokens)
    return self._empty_unverified_result(q, steps, tokens)
```

This skeleton does not replace `_extract_competitors()` or `_rescan_with()`; call them from the corresponding tool-result branch before evidence evaluation. `has_supported_refinement()` should return true only when the prior venue action omitted a sport that is explicitly present **outside** the venue/date spans and is one of the remaining candidates' sports. It should return false for `pub-060`, whose question contains no sport. That gives the planner a real second action for a supported refinement and stops other cases after one tool. Use the template fallback only after planner failure, not to turn an already-known ambiguity into an apparent pass. If a venue tie remains after the supported tool choices, Jev may make **one** selection among the eligible set; its selection is a hypothesis. Exclude `gold` from Jev and LLM selection prompts, send all eligible IDs when the list fits the decision input, and never silently truncate to `cands[:10]`. If selection comes from a model, return the chosen Event's canonical `gold` but keep certificate `unverified` and the eligible IDs in its evidence/provenance. If no candidate is supported, return `None` and `fail`/`unverified` as appropriate. Modify `_disambiguate()` to honor the one-decision budget: use Jev **or** the generative LLM for one selection attempt. The current Jev-failure → LLM fallback spends two decisions; do not keep it unchanged. Competitor-field extraction is a separate recovery operation and still counts toward total tokens.

The total answer latency must include planner calls, tool calls, Jev, and recovery. The total token count must include every LLM/Jev call, including invalid responses. A trace should let a reviewer follow: `planner_select → aggregation_scan → certificate`, or `planner_select → resolve_multi_hop → observation → planner_select → ...`. Classification comes from the **executed tool**; `q.qtype` is evaluation metadata and should never override it.

**Tests** in `tests/test_pipeline_agentic.py`:

- [x] Script `FakeLLM` to choose each of the five tools; assert the action really controls execution, answer, trace, and certificate class.
- [x] Force malformed planner output; assert the regex fallback is recorded and its token use included.
- [x] Force an invalid or repeated action; assert the loop stops and cannot emit a `pass` based on planner prose.
- [x] Force an ambiguous venue set and a Jev selection; assert the selected canonical answer may be emitted but `completeness_check == "unverified"`.
- [x] Force a scan with missing competitor values; assert existing recovery runs and unresolved values prevent a passing exhaustive certificate.
- [x] Run `python -m pytest tests/test_pipeline_agentic.py tests/test_decision_jev.py -q`.

## 7. Task F — Make the certificate say what happened

Keep existing fields for JSON compatibility. Add optional fields in `agrag/certificate.py` such as:

```python
class Certificate(BaseModel):
    # existing fields remain
    planner_mode: str = "template"          # planner | template | fallback
    selected_tool: str | None = None
    selection_basis: str | None = None       # unique_exact | unique_alias | model_choice | none
    raw_candidate_count: int | None = None
    eligible_candidate_ids: list[str] = Field(default_factory=list)
    constraint_sources: dict[str, str] = Field(default_factory=dict)
```

For existential venue queries:

| Condition | `completeness_check` | `stop_reason` |
|---|---|---|
| Unique exact venue + supported date (+ sport if supplied) | `pass` | `unique_supported_event` |
| Unique reviewed venue alias + supported date | `pass_with_fallback` | `unique_supported_alias` |
| Several eligible Events; Jev/LLM chooses one | `unverified` | `ambiguous_candidates` |
| No eligible Event / only blank dates or venues | `fail` or `unverified` | `no_supported_event` |
| Planner fails and GraphRAG supplies an answer | `unverified` | `graphrag_fallback` |

The current code labels any Jev-picked Event `pass_with_llm_recovery` and `structural_bound_met`; replace that behavior for venue ties. A `pass` should be computed from source predicates and eligible candidate count, never from model confidence. `docs_inspected` should list the actual eligible Events considered; record the chosen ID separately. For exhaustive questions, keep the complete sport/Games population and mark pass only when all required competitor values are known and the evidence set matches that population. Note that TigerGraph currently returns the complete Event set and **Python** computes the threshold count or max; do not claim GSQL performs the final aggregation unless it is changed and verified.

- [x] Add certificate tests for unique exact, unique alias, unresolved tie, failed search, and exhaustive scan.
- [x] Run `python -m pytest -q -m "not integration"`; expect the full offline suite to pass.

## 8. Task G — Evaluate both modes and publish measured results

Expose `--agent-mode planner|template` in `agrag/eval/run.py`, passing it through `build_pipeline()` to `AgenticPipeline`. Default the **agentic product mode** to `planner`, but keep `template` available for a fair comparison. Require an explicit `--out` path when comparing modes, so a trial cannot overwrite the existing submitted JSONL by accident. Rerun RAG and GraphRAG on the refreshed graph if the three-way comparison will claim a common graph state; their current files predate the Event refresh. Example commands after the live graph refresh:

```powershell
python -m agrag.eval.run --pipeline rag --backend tigergraph --questions data/eval_public.jsonl --out results/rag_refreshed_public.jsonl
python -m agrag.eval.run --pipeline graphrag --backend tigergraph --questions data/eval_public.jsonl --out results/graphrag_refreshed_public.jsonl
python -m agrag.eval.run --pipeline agentic --agent-mode template --backend tigergraph --questions data/eval_public.jsonl --out results/agentic_template_public.jsonl
python -m agrag.eval.run --pipeline agentic --agent-mode planner --backend tigergraph --questions data/eval_public.jsonl --out results/agentic_planner_public.jsonl
python -m agrag.eval.run --pipeline rag --backend tigergraph --questions data/eval_hidden.jsonl --out results/rag_refreshed_hidden.jsonl
python -m agrag.eval.run --pipeline graphrag --backend tigergraph --questions data/eval_hidden.jsonl --out results/graphrag_refreshed_hidden.jsonl
python -m agrag.eval.run --pipeline agentic --agent-mode planner --backend tigergraph --questions data/eval_hidden.jsonl --out results/agentic_planner_hidden.jsonl
```

Compare the two modes on **the same refreshed graph**. Inspect overall and per-class public accuracy, evidence coverage, certificate pass rate, planner action/fallback counts, median/mean tokens, and latency. Validate all 50 hidden JSONL rows structurally and manually inspect `eval-001`; hidden gold is unavailable, so do not report hidden accuracy. Inspect `pub-060` separately: date filter must exclude `Q932145`, while the same-day tie must remain visible. Preserve actual raw outputs with trace and token usage for submission.

The current `agrag/eval/report.py` globs `results/*_public.jsonl`. Once both new mode files exist, that glob mixes template and planner rows under the same `agentic` label and overwrites `results/summary.json`. Add a repeatable `--public` argument and an `--out` argument to the report command; make the comparison pass exactly one agentic mode with the RAG and GraphRAG baselines. Keep template and planner summaries in distinct files. Preserve `load_rows()` for the dashboard, but change its default from a glob to the three reviewed standard paths (`rag_public.jsonl`, `graphrag_public.jsonl`, `agentic_public.jsonl`). Reject duplicate `(pipeline, qid)` score rows so mixed modes cannot pass unnoticed.

```python
DEFAULT_PUBLIC = [
    "results/rag_public.jsonl",
    "results/graphrag_public.jsonl",
    "results/agentic_public.jsonl",
]

def load_rows(paths: list[str] | None = None) -> list[dict]:
    paths = DEFAULT_PUBLIC if paths is None else paths
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            rows.extend(json.loads(line)["score"] for line in stream if line.strip())
    pairs = [(row["pipeline"], row["qid"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate pipeline/qid: mixed evaluation modes")
    return rows

# In report.py main: parse repeatable --public and --out, then call
# summarize(load_rows(args.public)) and write only args.out.
```

Suggested non-negotiable checks before changing headline claims:

```powershell
python -m pytest -q -m "not integration"
python scripts/reconcile.py --backend local --out data/reconciliation-local-planner.json
python scripts/reconcile.py --backend tigergraph --out data/reconciliation-tigergraph-planner.json
python -m agrag.eval.report --public results/rag_refreshed_public.jsonl --public results/graphrag_refreshed_public.jsonl --public results/agentic_planner_public.jsonl --out results/summary_planner.json
git diff --check
```

Copy **reviewed** final outputs into the standard submission locations only after the comparison; then write the reviewed planner summary to `results/summary.json`. `scripts/make_results_chart.py` reads that fixed summary path, so rerun `python scripts/make_results_chart.py` **after** the summary replacement and inspect `docs/assets/results.png`. Confirm `dashboard/app.py` uses only the reviewed standard paths through the revised `load_rows()`; experimental files must not alter its charts. Do not silently combine old and new graph states. Update `README.md`, the architecture Mermaid/source image, dashboard labels, `docs/status.md`, `docs/submission-form-answers.md`, `docs/submission/devto-blog.md`, `docs/submission/tweet.txt`, `docs/demo/transcript.md`, and any demo slides or recording that still quote old accuracy, tokens, traces, or `eval-001` correctness. Preserve the historical 99%/18-token figures as historical measurements if useful; the planner's tokens and latency will be different. Record final results and commands in the Effort 003 validation report and move its state to `complete` only after live validation and documentation are finished.

## 9. Completion checklist

- [x] The 25 second-infobox Olympic documents are parsed as Events; `Q942805` exists locally and in TigerGraph with exact venue/date/gold.
- [x] The live graph was refreshed without rebuilding Chunk embeddings, and local/live Event counts and reconciliation agree.
- [x] Empty venue and missing date can never win a venue/date question; the sport constraint is grounded and recorded.
- [x] The normal agentic trace shows an LLM-selected **tool and arguments**, then a deterministic tool result and evidence evaluation.
- [x] Invalid planner output has a bounded, visible fallback; no model prose becomes a certified answer.
- [x] `eval-001` uses the tennis Event; `pub-060` does not claim structural uniqueness.
- [x] Exhaustive scans still inspect the complete sport/Games set and preserve canonical outputs.
- [x] Public and hidden runs were regenerated on the refreshed graph, with actual tokens, latency, trace, and certificates.
- [x] README, diagrams, dashboard, submission text, registry, and effort validation refer to those same outputs.

## Execution note (2026-10-07)

The full implementation checklist is verified against tests, live graph state, reconciliation, raw evaluations, and documentation. The test-before-implementation failure checkpoint is historical and cannot be evidenced retroactively; the tests pass against the final code. Optional commit steps remain unchecked because the isolated branch is preserved for owner review.
