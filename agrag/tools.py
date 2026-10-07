"""Deterministic, LLM-free retrieval and reasoning steps.

Every function takes a GraphBackend and a ParsedQuestion and returns a ToolResult that
records the evidence it inspected, the structural bound the graph reported, and whether
an LLM is needed to finish. Both the oracle (reconciliation) and the agentic orchestrator
call these; the orchestrator adds LLM recovery on top.
"""
from __future__ import annotations

import calendar
import re
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

from agrag.backend import GraphBackend
from agrag.infobox import EventRecord, norm_key, parse_title
from agrag.normalize import normalize
from agrag.router import ParsedQuestion

DATE_STOP = {"on", "to", "and", "at", "the", "of", "&"}
DATE_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"
_MONTH_NAMES = {
    name.lower(): number
    for number in range(1, 13)
    for name in (calendar.month_name[number], calendar.month_abbr[number])
}
_MONTH_NAMES["sept"] = 9
_MONTH_RE = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
_YEAR_RE = r"\d{4}"
_MONTH_FIRST_RANGE_RE = re.compile(
    rf"(?P<m1>{_MONTH_RE})\s+(?P<d1>\d{{1,2}})(?!\d)(?:,?\s+(?P<y1>{_YEAR_RE}))?"
    rf"\s*(?:to|[{re.escape(DATE_DASHES)}-])\s*(?:(?P<m2>{_MONTH_RE})\s*)?"
    rf"(?P<d2>\d{{1,2}})(?!\d)(?:,?\s+(?P<y2>{_YEAR_RE}))?",
    re.I,
)
_DAY_FIRST_RANGE_RE = re.compile(
    rf"(?P<d1>\d{{1,2}})(?!\d)\s*(?P<m1>{_MONTH_RE})?(?:\s+(?P<y1>{_YEAR_RE}))?"
    rf"\s*(?:to|[{re.escape(DATE_DASHES)}-])\s*"
    rf"(?:(?P<d2>\d{{1,2}})(?!\d)\s*)?(?P<m2>{_MONTH_RE})(?:,?\s+(?P<y2>{_YEAR_RE}))?",
    re.I,
)
_DAY_LIST_RE = re.compile(
    rf"(?P<days>\d{{1,2}}(?!\d)(?:\s*(?:,|and|&)\s*(?:and\s*)?\d{{1,2}}(?!\d))*)\s+(?P<month>{_MONTH_RE})(?:,?\s+(?P<year>{_YEAR_RE}))?",
    re.I,
)
_MONTH_FIRST_POINT_RE = re.compile(
    rf"(?P<month>{_MONTH_RE})\s+(?P<day>\d{{1,2}})(?!\d)(?:,?\s+(?P<year>{_YEAR_RE}))?",
    re.I,
)
_MONTH_ONLY_RE = re.compile(rf"(?P<month>{_MONTH_RE})(?:\s+(?P<year>{_YEAR_RE}))?", re.I)
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
    # Appended defaults preserve the positional construction used by existing callers.
    raw_candidate_count: int = 0
    eligible_count: int = 0
    match_mode: str = "none"
    selection_basis: str = "none"
    ambiguity_reason: str = ""


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
DateSpan = tuple[date, date]


def _date_value(day: str, month: str, year: Optional[str], default_year: Optional[int]) -> Optional[date]:
    """Build a calendar date, returning None for absent years or invalid calendar values."""
    if not month:
        return None
    month_number = _MONTH_NAMES.get(month.lower())
    resolved_year = int(year) if year else default_year
    if month_number is None or resolved_year is None:
        return None
    try:
        return date(resolved_year, month_number, int(day))
    except (TypeError, ValueError):
        return None


def _normalize_date_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    for dash in DATE_DASHES:
        value = value.replace(dash, "-")
    return re.sub(r"\s+", " ", value).strip()


def parse_date_spans(value: str, default_year: Optional[int] = None) -> tuple[DateSpan, ...]:
    """Parse English point/range dates into inclusive spans.

    Supports day-first and month-first notation, omitted endpoint years/months, and
    comma/and-separated stage dates. An omitted year uses the enclosing Games year.
    Unparseable or blank input returns no spans, so it cannot become a venue match.
    """
    if not value or not value.strip():
        return ()
    remaining = _normalize_date_text(value)
    spans: list[DateSpan] = []

    def save_range(match: re.Match[str], first_month: str, second_month: str,
                   first_day: str, second_day: str, first_year: Optional[str],
                   second_year: Optional[str]) -> None:
        year_a = first_year or second_year
        start = _date_value(first_day, first_month or second_month, year_a, default_year)
        end = _date_value(second_day, second_month or first_month, second_year or year_a, default_year)
        if start is None or end is None:
            return
        if end < start and not first_year and not second_year and end.month < start.month:
            try:
                end = end.replace(year=end.year + 1)
            except ValueError:
                return
        if end >= start:
            spans.append((start, end))

    # Parse explicit ranges first, removing their text before looking for isolated dates.
    for pattern in (_MONTH_FIRST_RANGE_RE, _DAY_FIRST_RANGE_RE):
        matches = list(pattern.finditer(remaining))
        for match in matches:
            groups = match.groupdict()
            save_range(
                match,
                groups.get("m1") or "",
                groups.get("m2") or "",
                groups["d1"],
                groups["d2"] or groups["d1"],
                groups.get("y1"),
                groups.get("y2"),
            )
        for match in reversed(matches):
            remaining = remaining[:match.start()] + " " * (match.end() - match.start()) + remaining[match.end():]

    # `6, 8 and 10 August` represents distinct competition stages, not a continuous range.
    for match in list(_DAY_LIST_RE.finditer(remaining)):
        month, year = match["month"], match["year"]
        for day_text in re.findall(r"\d{1,2}", match["days"]):
            parsed = _date_value(day_text, month, year, default_year)
            if parsed:
                spans.append((parsed, parsed))
    matches = list(_DAY_LIST_RE.finditer(remaining))
    for match in reversed(matches):
        remaining = remaining[:match.start()] + " " * (match.end() - match.start()) + remaining[match.end():]

    # Month-first single points (e.g. `August 4, 2012`).
    for match in list(_MONTH_FIRST_POINT_RE.finditer(remaining)):
        parsed = _date_value(match["day"], match["month"], match["year"], default_year)
        if parsed:
            spans.append((parsed, parsed))
    matches = list(_MONTH_FIRST_POINT_RE.finditer(remaining))
    for match in reversed(matches):
        remaining = remaining[:match.start()] + " " * (match.end() - match.start()) + remaining[match.end():]

    # A month-only mention is a useful, conservative interval (e.g. `February 2018`).
    for match in list(_MONTH_ONLY_RE.finditer(remaining)):
        month = _MONTH_NAMES.get(match["month"].lower())
        year = int(match["year"]) if match["year"] else default_year
        if month is not None and year is not None:
            spans.append((date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])))

    # Deduplicate spans while retaining deterministic order.
    return tuple(dict.fromkeys(spans))


def date_match_status(question_date: str, event_date: str, default_year: Optional[int] = None) -> str:
    """Classify whether a question date safely identifies an event date.

    `exact_range`, `same_day`, and `day_in_range` are safe date matches. A partial
    intersection between longer ranges is deliberately `overlap_uncertain`.
    """
    question_spans = parse_date_spans(question_date, default_year)
    event_spans = parse_date_spans(event_date, default_year)
    if not question_spans or not event_spans:
        return "unknown"

    if question_spans == event_spans:
        if len(question_spans) > 1 or any(start < end for start, end in question_spans):
            return "exact_range"

    if (
        len(question_spans) == len(event_spans) == 1
        and question_spans[0][0] == question_spans[0][1]
        and question_spans == event_spans
    ):
        return "same_day"

    # A point within the opposite side's span is safe: the event occurred on the asked
    # day, or the single-day event falls within the asked range.
    for q_start, q_end in question_spans:
        for e_start, e_end in event_spans:
            if q_start == q_end and e_start <= q_start <= e_end:
                return "day_in_range"
            if e_start == e_end and q_start <= e_start <= q_end:
                return "day_in_range"

    if any(q_start <= e_end and e_start <= q_end
           for q_start, q_end in question_spans for e_start, e_end in event_spans):
        return "overlap_uncertain"
    return "disjoint"


def _date_tokens(s: str) -> set[str]:
    s = normalize(s).replace("-", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    toks = {t for t in s.split() if t and t not in DATE_STOP}
    return {t for t in toks if not re.fullmatch(r"\d{4}", t)}


def date_score(question_date: str, event_date: str) -> float:
    """Compatibility score retained for older callers; resolution uses date_match_status."""
    status = date_match_status(question_date, event_date, default_year=2000)
    return {
        "exact_range": 1.0,
        "same_day": 1.0,
        "day_in_range": 0.75,
        "overlap_uncertain": 0.25,
        "disjoint": 0.0,
        "unknown": 0.0,
    }[status]


def _venue_key(value: str) -> str:
    """Normalize a venue for exact comparison and token-bounded alias matching."""
    value = normalize(value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _venue_alias_match(requested: str, candidate: str) -> bool:
    requested_key, candidate_key = _venue_key(requested), _venue_key(candidate)
    if not requested_key or not candidate_key:
        return False
    generic_single_tokens = {"olympic", "centre", "center", "stadium", "arena", "venue"}
    if len(requested_key.split()) == 1 and requested_key in generic_single_tokens:
        return False
    if requested_key == candidate_key:
        return True
    return f" {requested_key} " in f" {candidate_key} " or f" {candidate_key} " in f" {requested_key} "


def _dedupe_events(events: list[EventRecord]) -> list[EventRecord]:
    by_id: dict[str, EventRecord] = {}
    for event in events:
        by_id.setdefault(event.doc_id, event)
    return list(by_id.values())


def resolve_multi_hop(b: GraphBackend, pq: ParsedQuestion) -> ToolResult:
    venue, date_text = pq.slots.get("venue", ""), pq.slots.get("date", "")
    year, season = pq.slots.get("year"), pq.slots.get("season")
    sport = pq.slots.get("sport")
    predicate = {"venue": venue, "date": date_text, "year": year, "season": season}
    if sport:
        predicate["sport"] = sport
    games_list = [f"{year} {season}"] if season else [f"{year} Summer", f"{year} Winter"]
    venue_key = _venue_key(venue)
    if not venue_key:
        return ToolResult("resolve_multi_hop", None, [], 0, predicate, notes=["venue is blank"],
                          raw_candidate_count=0, eligible_count=0, match_mode="none",
                          ambiguity_reason="blank_venue")

    exact: list[EventRecord] = []
    for games in games_list:
        exact.extend(e for e in b.events_by_venue(venue, games) if _venue_key(e.venue) == venue_key)
    exact = _dedupe_events(exact)
    fallback = False
    if exact:
        raw_candidates, match_mode = exact, "exact_venue"
    else:
        fallback = True
        aliases: list[EventRecord] = []
        for games in games_list:
            aliases.extend(e for e in b.events_at_games(games) if _venue_alias_match(venue, e.venue))
        raw_candidates, match_mode = _dedupe_events(aliases), "venue_alias"

    raw_count = len(raw_candidates)
    if not raw_candidates:
        return ToolResult("resolve_multi_hop", None, [], 0, predicate, needs_llm=False,
                          fallback_used=fallback, notes=["no events at venue"], raw_candidate_count=0,
                          eligible_count=0, match_mode=match_mode, ambiguity_reason="")

    # Planner-provided sport is an optional refinement. The caller is responsible for
    # supplying this slot only after grounding it in the question/evidence.
    sport_key = norm_key(sport) if sport else ""
    sport_candidates = [e for e in raw_candidates if not sport_key or norm_key(e.sport) == sport_key]
    statuses = {e.doc_id: date_match_status(date_text, e.date_text, default_year=year) for e in sport_candidates}
    safe_statuses = {"exact_range", "same_day", "day_in_range"}
    eligible = [e for e in sport_candidates if statuses[e.doc_id] in safe_statuses]
    # Preserve the strongest date evidence. An event scheduled on the exact queried
    # day/range outranks a competition whose longer interval merely contains it.
    date_priority = {"exact_range": 3, "same_day": 2, "day_in_range": 1}
    if eligible:
        strongest = max(date_priority[statuses[e.doc_id]] for e in eligible)
        eligible = [e for e in eligible if date_priority[statuses[e.doc_id]] == strongest]
    eligible = _dedupe_events(eligible)

    if len(eligible) == 1:
        best = eligible[0]
        status = statuses[best.doc_id]
        return ToolResult("resolve_multi_hop", best.gold or None, [best.doc_id], raw_count, predicate,
                          needs_llm=best.gold is None, candidates=eligible, fallback_used=fallback,
                          raw_candidate_count=raw_count, eligible_count=1, match_mode=match_mode,
                          selection_basis=f"unique_venue_and_{status}",
                          ambiguity_reason="missing_gold_answer" if best.gold is None else "")

    if len(eligible) > 1:
        ambiguity_reason = "multiple_eligible_candidates"
        note = f"{len(eligible)} candidates have a safe venue/date match"
    elif not sport_candidates:
        ambiguity_reason = "sport_filter_no_match"
        note = "no venue candidates match the grounded sport"
    elif all(status == "unknown" for status in statuses.values()):
        ambiguity_reason = "no_parseable_candidate_dates"
        note = "candidate dates are blank or unparseable"
    elif any(status == "overlap_uncertain" for status in statuses.values()):
        ambiguity_reason = "overlapping_ranges"
        note = "date ranges overlap but do not identify a day or exact range"
    else:
        ambiguity_reason = "no_date_match"
        note = "no candidate has a safe date match"
    # Only supported date-eligible records enter the evidence set. Raw rows rejected by
    # date/sport checks are summarized by raw_candidate_count and the ambiguity reason.
    evidence = [e.doc_id for e in eligible]
    return ToolResult("resolve_multi_hop", None, evidence, raw_count, predicate, needs_llm=True,
                      candidates=eligible, fallback_used=fallback, notes=[note], raw_candidate_count=raw_count,
                      eligible_count=len(eligible), match_mode=match_mode, ambiguity_reason=ambiguity_reason)


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
