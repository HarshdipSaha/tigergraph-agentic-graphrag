from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from agrag.certificate import Certificate, TokenUsage
from agrag.infobox import EventRecord

ANSWER_SYSTEM = (
    "You answer questions about Olympic events using ONLY the provided context. "
    "Reply with the final answer string only: a person's name exactly as written in the context, "
    "a number, or an exact event title. No explanation, no punctuation after the answer. "
    "If the context is insufficient, reply exactly: UNKNOWN"
)


class PipelineResult(BaseModel):
    qid: str
    pipeline: str
    answer: Optional[str]
    docs_retrieved: list[str]
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    certificate: Optional[Certificate] = None


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = int((time.perf_counter() - self.t0) * 1000)


def event_facts(e: EventRecord) -> str:
    return (
        f"[{e.title}]\n  games: {e.games}\n  venue: {e.venue}\n  date: {e.date_text}\n"
        f"  competitors: {e.competitors_raw}\n  nations: {e.nations}\n  gold: {e.gold}\n"
        f"  prev: {e.prev_year}\n  next: {e.next_year}"
    )


def clean_answer(text: str) -> Optional[str]:
    t = text.strip().strip('"').strip()
    return None if not t or t.upper() == "UNKNOWN" else t
