"""Benchmark-scoped question router.

This is deliberately a regex template matcher over the five question templates in the
hackathon dataset (see docs/hackathon-brief.md §7). It is NOT a general query router and
must not be described as one (docs/idea-spec.md §4.2). Its job is to map a question to an
evidential-completeness class so the orchestrator can pick the cheapest sufficient tool.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Template = Literal["lookup", "multi_hop", "temporal", "aggregation", "superlative"]
CompletenessClass = Literal["existential", "chained", "exhaustive", "unknown"]

CLASS_OF: dict[str, CompletenessClass] = {
    "lookup": "existential",
    "multi_hop": "existential",
    "temporal": "chained",
    "aggregation": "exhaustive",
    "superlative": "exhaustive",
}

LOOKUP_RE = re.compile(r"^How many nations competed in (?P<title>.+?)\?$")
MULTI_HOP_RE = re.compile(
    r"^Who won the gold medal in the event held at (?P<venue>.+?) on (?P<date>.+?)"
    r"(?: at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics)?\?$"
)
TEMPORAL_RE = re.compile(
    r"^Who won the gold medal in the (?P<event_phrase>.+?) event at the (?P<season>Summer|Winter) Olympics "
    r"held immediately before (?P<year>\d{4})\?$"
)
AGG_RE = re.compile(
    r"^According to the provided corpus, how many (?P<sport>.+?) events at the (?P<year>\d{4}) "
    r"(?P<season>Summer|Winter) Olympics had more than (?P<threshold>\d+) competitors\?$"
)
SUP_RE = re.compile(
    r"^According to the provided corpus, which (?P<sport>.+?) event at the (?P<year>\d{4}) "
    r"(?P<season>Summer|Winter) Olympics had the highest number of competitors\?$"
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(frozen=True)
class ParsedQuestion:
    question: str
    template: Optional[Template]
    completeness_class: CompletenessClass
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: Literal["high", "low"] = "high"


def route(question: str) -> ParsedQuestion:
    q = question.strip()

    m = LOOKUP_RE.match(q)
    if m:
        return ParsedQuestion(q, "lookup", CLASS_OF["lookup"], {"title": m["title"].strip()})

    m = MULTI_HOP_RE.match(q)
    if m:
        year = int(m["year"]) if m["year"] else None
        if year is None:
            ym = YEAR_RE.search(m["date"])
            year = int(ym.group(0)) if ym else None
        return ParsedQuestion(
            q, "multi_hop", CLASS_OF["multi_hop"],
            {"venue": m["venue"].strip(), "date": m["date"].strip(), "year": year, "season": m["season"]},
        )

    m = TEMPORAL_RE.match(q)
    if m:
        return ParsedQuestion(
            q, "temporal", CLASS_OF["temporal"],
            {"event_phrase": m["event_phrase"].strip(), "season": m["season"], "year": int(m["year"])},
        )

    m = AGG_RE.match(q)
    if m:
        return ParsedQuestion(
            q, "aggregation", CLASS_OF["aggregation"],
            {"sport": m["sport"].strip(), "year": int(m["year"]), "season": m["season"], "threshold": int(m["threshold"])},
        )

    m = SUP_RE.match(q)
    if m:
        return ParsedQuestion(
            q, "superlative", CLASS_OF["superlative"],
            {"sport": m["sport"].strip(), "year": int(m["year"]), "season": m["season"]},
        )

    return ParsedQuestion(q, None, "unknown", {}, confidence="low")
