from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, TypedDict

import numpy as np

from agrag.infobox import EventRecord


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class Neighborhood(TypedDict):
    event: EventRecord
    prev: Optional[EventRecord]
    next: Optional[EventRecord]
    same_venue: list[EventRecord]


class GraphBackend(Protocol):
    """Everything a pipeline may ask the graph. Implemented by LocalBackend and TigerGraphBackend."""

    def events_by_sport_games(self, sport: str, games: str) -> list[EventRecord]: ...
    def events_at_games(self, games: str) -> list[EventRecord]: ...
    def event_by_title(self, title: str) -> Optional[EventRecord]: ...
    def events_by_venue(self, venue: str, games: str) -> list[EventRecord]: ...
    def prev_event(self, doc_id: str) -> Optional[EventRecord]: ...
    def neighborhood(self, doc_id: str) -> Optional[Neighborhood]: ...
    def vector_search(self, query_vec: np.ndarray, k: int) -> list[ChunkHit]: ...
    def doc_text(self, doc_id: str) -> str: ...
