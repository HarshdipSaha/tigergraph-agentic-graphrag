from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from agrag.corpus import Doc

EVENT_HEADER = "[Infobox Olympic event]"
TITLE_RE = re.compile(
    r"^(?P<sport>.+?) at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics"
    r"(?: (?:(?:[-\u2010-\u2015\u2212\ufffd])|(?:\u00e2\u20ac[\u201c\u2013\u2014])) (?P<event>.+))?$"
)
INT_RE = re.compile(r"^\d+$")


def parse_infobox(text: str) -> tuple[Optional[str], dict[str, str]]:
    """Return (header, fields) for the indented key: value block at the top of a doc."""
    if not text.startswith("["):
        return None, {}
    lines = text.split("\n")
    header = lines[0].strip()
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if not line.startswith("  "):
            break
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        fields[key.strip()] = value.strip()
    return header, fields


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    v = value.strip()
    return int(v) if INT_RE.match(v) else None


def parse_title(title: str) -> Optional[tuple[str, int, str, str]]:
    m = TITLE_RE.match(title)
    if not m:
        return None
    return m["sport"], int(m["year"]), m["season"], (m["event"] or "").strip()


@dataclass(frozen=True)
class EventRecord:
    doc_id: str
    title: str
    sport: str
    year: int
    season: str
    event_name: str
    venue: str
    date_text: str
    competitors: Optional[int]
    competitors_raw: str
    nations: Optional[int]
    gold: str
    gold_noc: str
    prev_year: Optional[int]
    next_year: Optional[int]
    url: str

    @property
    def games(self) -> str:
        return f"{self.year} {self.season}"


def event_from_doc(doc: Doc) -> Optional[EventRecord]:
    # Some source pages have a general tournament infobox before their
    # Olympic-specific event infobox. Keep parse_infobox()'s first-block
    # behavior for callers, but scan for the event block here.
    ib: Optional[dict[str, str]] = None
    lines = doc.text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != EVENT_HEADER:
            continue
        header, fields = parse_infobox("\n".join(lines[index:]))
        if header == EVENT_HEADER:
            ib = fields
            break
    if ib is None:
        return None
    parsed = parse_title(doc.title)
    if parsed is None:
        return None
    sport, year, season, event_name = parsed
    return EventRecord(
        doc_id=doc.doc_id,
        title=doc.title,
        sport=sport,
        year=year,
        season=season,
        event_name=event_name,
        venue=ib.get("venue", ""),
        date_text=ib.get("date") or ib.get("dates", ""),
        competitors=parse_int(ib.get("competitors")),
        competitors_raw=ib.get("competitors", ""),
        nations=parse_int(ib.get("nations")),
        gold=ib.get("gold", ""),
        gold_noc=ib.get("goldNOC", ""),
        prev_year=parse_int(ib.get("prev")),
        next_year=parse_int(ib.get("next")),
        url=doc.url,
    )


def events_from_docs(docs: Iterable[Doc]) -> list[EventRecord]:
    out = []
    for d in docs:
        ev = event_from_doc(d)
        if ev is not None:
            out.append(ev)
    return out


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def resolve_links(events: Iterable[EventRecord]) -> tuple[dict[str, str], dict[str, str]]:
    """Map doc_id -> doc_id for PREV and NEXT using (sport, season, event_name, year)."""
    events = list(events)
    index = {(norm_key(e.sport), e.season, norm_key(e.event_name), e.year): e.doc_id for e in events}
    prev: dict[str, str] = {}
    nxt: dict[str, str] = {}
    for e in events:
        if e.prev_year is not None:
            target = index.get((norm_key(e.sport), e.season, norm_key(e.event_name), e.prev_year))
            if target:
                prev[e.doc_id] = target
        if e.next_year is not None:
            target = index.get((norm_key(e.sport), e.season, norm_key(e.event_name), e.next_year))
            if target:
                nxt[e.doc_id] = target
    return prev, nxt
