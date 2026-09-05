"""Deterministic, LLM-free retrieval and reasoning steps.

Every function takes a GraphBackend and a ParsedQuestion and returns a ToolResult that
records the evidence it inspected, the structural bound the graph reported, and whether
an LLM is needed to finish. Both the oracle (reconciliation) and the agentic orchestrator
call these; the orchestrator adds LLM recovery on top.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from agrag.backend import GraphBackend
from agrag.infobox import EventRecord, norm_key, parse_title
from agrag.normalize import normalize
from agrag.router import ParsedQuestion

DATE_STOP = {"on", "to", "and", "at", "the", "of", "&"}
COMPETITORS_RE = re.compile(
    r"\b(\d{1,4})\s+(?:competitors|athletes|sailors|swimmers|boxers|fencers|skiers|riders|cyclists|players|wrestlers|judoka|shooters|rowers)\b",
    re.I,
)


@dataclass
class ToolResult:
    tool: str
    answer: Optional[str]
    evidence: list[str]                      # doc_ids actually inspected
    structural_bound: int                    # what the graph says exists for the predicate
    predicate: dict
    needs_llm: bool = False
    candidates: list[EventRecord] = field(default_factory=list)
    hops: list[str] = field(default_factory=list)
    fallback_used: bool = False
    recovered: dict[str, int] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _games(pq: ParsedQuestion) -> str:
    return f"{pq.slots['year']} {pq.slots['season']}"


# ---------------- existential: lookup ----------------
def lookup_nations(b: GraphBackend, pq: ParsedQuestion) -> ToolResult:
    title = pq.slots["title"]
    ev = b.event_by_title(title)
    if ev is None:
        parsed = parse_title(title)
        if parsed:
            games = f"{parsed[1]} {parsed[2]}"
            cands = [e for e in b.events_at_games(games) if normalize(e.title) == normalize(title)]
            if len(cands) == 1:
                ev = cands[0]
    if ev is None:
        return ToolResult("lookup_nations", None, [], 0, {"title": title}, notes=["title not found"])
    answer = str(ev.nations) if ev.nations is not None else None
    return ToolResult("lookup_nations", answer, [ev.doc_id], 1, {"title": title}, needs_llm=answer is None,
                      candidates=[ev])


# ---------------- existential: multi_hop (venue + date) ----------------
def _date_tokens(s: str) -> set[str]:
    s = normalize(s).replace("-", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    toks = {t for t in s.split() if t and t not in DATE_STOP}
    return {t for t in toks if not re.fullmatch(r"(19|20)\d{2}", t)}


def date_score(question_date: str, event_date: str) -> float:
    a, b = _date_tokens(question_date), _date_tokens(event_date)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def resolve_multi_hop(b: GraphBackend, pq: ParsedQuestion) -> ToolResult:
    venue, date, year, season = pq.slots["venue"], pq.slots["date"], pq.slots["year"], pq.slots["season"]
    predicate = {"venue": venue, "date": date, "year": year, "season": season}
    games_list = [f"{year} {season}"] if season else [f"{year} Summer", f"{year} Winter"]
    cands: list[EventRecord] = []
    fallback = False
    for g in games_list:
        cands.extend(b.events_by_venue(venue, g))
    if not cands:
        fallback = True
        v = norm_key(venue)
        for g in games_list:
            cands.extend(e for e in b.events_at_games(g) if v in norm_key(e.venue) or norm_key(e.venue) in v)
    if not cands:
        return ToolResult("resolve_multi_hop", None, [], 0, predicate, needs_llm=False, fallback_used=fallback,
                          notes=["no events at venue"])
    scored = sorted(((date_score(date, e.date_text), e) for e in cands), key=lambda t: -t[0])
    best_score, best = scored[0]
    tied = [e for s, e in scored if s == best_score]
    evidence = [e.doc_id for e in cands]
    if best_score > 0 and len(tied) == 1:
        return ToolResult("resolve_multi_hop", best.gold or None, [best.doc_id], len(cands), predicate,
                          candidates=cands, fallback_used=fallback)
    return ToolResult("resolve_multi_hop", None, evidence, len(cands), predicate, needs_llm=True, candidates=cands,
                      fallback_used=fallback, notes=[f"{len(tied)} candidates tied at date score {best_score:.2f}"])


# ---------------- chained: temporal ----------------
def _phrase_score(phrase: str, ev: EventRecord) -> float:
    a = set(normalize(phrase).replace("-", " ").split())
    b_ = set(normalize(f"{ev.sport} {ev.event_name}").replace("-", " ").split())
    return len(a & b_) / len(a | b_) if a and b_ else 0.0


def best_event_for_phrase(events: list[EventRecord], phrase: str, min_score: float = 0.0) -> Optional[EventRecord]:
    """Unique best token-overlap match, or None if nothing clears min_score or the top is tied."""
    if not events:
        return None
    scored = sorted(((_phrase_score(phrase, e), e) for e in events), key=lambda t: -t[0])
    top_score, top = scored[0]
    if top_score == 0 or top_score < min_score:
        return None
    tied = [e for s, e in scored if s == top_score]
    if len(tied) == 1:
        return top
    # Tie-break on raw substring: "men's 80 kg" is inside "men's 80 kg taekwondo", "men's +80 kg" is not.
    exact = [e for e in tied if e.event_name and e.event_name.lower() in phrase.lower()]
    return exact[0] if len(exact) == 1 else None


def temporal_chain(b: GraphBackend, pq: ParsedQuestion) -> ToolResult:
    phrase, season, year = pq.slots["event_phrase"], pq.slots["season"], pq.slots["year"]
    predicate = {"event_phrase": phrase, "season": season, "year": year}
    start = best_event_for_phrase(b.events_at_games(f"{year} {season}"), phrase)
    if start is None:
        return ToolResult("temporal_chain", None, [], 0, predicate, needs_llm=True, notes=["start event not resolved"])
    prev = b.prev_event(start.doc_id)
    if prev is not None:
        return ToolResult("temporal_chain", prev.gold or None, [start.doc_id, prev.doc_id], 2, predicate,
                          hops=[start.doc_id, prev.doc_id], candidates=[start, prev])
    # Fallback hop when the PREV edge is missing: look at the previous Games named in the infobox
    # (or year-4) for the same sport + event name. A high threshold (0.8) keeps this honest: it must
    # not pick "Women's sprint" for "Men's sprint" just because they share two of three words.
    fallback_year = start.prev_year or (year - 4)
    prev2 = best_event_for_phrase(b.events_at_games(f"{fallback_year} {season}"),
                                  f"{start.sport} {start.event_name}", min_score=0.8)
    if prev2 is not None:
        return ToolResult("temporal_chain", prev2.gold or None, [start.doc_id, prev2.doc_id], 2, predicate,
                          hops=[start.doc_id, prev2.doc_id], candidates=[start, prev2], fallback_used=True,
                          notes=["PREV edge missing; matched by title at previous Games"])
    return ToolResult("temporal_chain", None, [start.doc_id], 1, predicate, hops=[start.doc_id], fallback_used=True,
                      candidates=[start], needs_llm=True, notes=["PREV edge missing and no fallback match"])


# ---------------- exhaustive: aggregation / superlative ----------------
def prose_of(text: str) -> str:
    """Drop the infobox block (everything up to the first blank line). The infobox's `date:` line sits
    directly above `competitors:`, so searching the raw text would match the year as a count."""
    return text.split("\n\n", 1)[1] if text.startswith("[") and "\n\n" in text else text


def recover_competitors_from_text(text: str) -> Optional[int]:
    m = COMPETITORS_RE.search(prose_of(text))
    return int(m.group(1)) if m else None


def _scan(b: GraphBackend, pq: ParsedQuestion) -> tuple[list[EventRecord], dict[str, int], list[str]]:
    events = b.events_by_sport_games(pq.slots["sport"], _games(pq))
    recovered: dict[str, int] = {}
    unresolved: list[str] = []
    for e in events:
        if e.competitors is None:
            n = recover_competitors_from_text(b.doc_text(e.doc_id))
            if n is None:
                unresolved.append(e.doc_id)
            else:
                recovered[e.doc_id] = n
    return events, recovered, unresolved


def _competitors(e: EventRecord, recovered: dict[str, int]) -> Optional[int]:
    return e.competitors if e.competitors is not None else recovered.get(e.doc_id)


def aggregation_scan(b: GraphBackend, pq: ParsedQuestion) -> ToolResult:
    events, recovered, unresolved = _scan(b, pq)
    predicate = {"sport": pq.slots["sport"], "games": _games(pq), "threshold": pq.slots["threshold"]}
    count = sum(1 for e in events if (_competitors(e, recovered) or -1) > pq.slots["threshold"])
    return ToolResult("aggregation_scan", str(count), [e.doc_id for e in events], len(events), predicate,
                      needs_llm=bool(unresolved), candidates=events, recovered=recovered, unresolved=unresolved)


def superlative_scan(b: GraphBackend, pq: ParsedQuestion) -> ToolResult:
    events, recovered, unresolved = _scan(b, pq)
    predicate = {"sport": pq.slots["sport"], "games": _games(pq)}
    best: Optional[EventRecord] = None
    best_n = -1
    for e in events:
        n = _competitors(e, recovered)
        if n is not None and n > best_n:
            best, best_n = e, n
    return ToolResult("superlative_scan", best.title if best else None, [e.doc_id for e in events], len(events),
                      predicate, needs_llm=bool(unresolved), candidates=events, recovered=recovered, unresolved=unresolved)
