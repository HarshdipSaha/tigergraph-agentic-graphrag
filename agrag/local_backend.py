from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from agrag.backend import ChunkHit, Neighborhood
from agrag.chunk import Chunk, chunk_text
from agrag.corpus import Doc
from agrag.embed import Embedder
from agrag.infobox import EventRecord, events_from_docs, norm_key, resolve_links


class LocalBackend:
    def __init__(self, docs: Iterable[Doc], events: list[EventRecord], prev: dict[str, str], nxt: dict[str, str],
                 chunks: list[Chunk], matrix: np.ndarray, embedder: Embedder):
        self._docs = {d.doc_id: d for d in docs}
        self._events = {e.doc_id: e for e in events}
        self._prev, self._next = prev, nxt
        self._chunks, self._matrix = chunks, matrix
        self.embedder = embedder
        self._by_title = {e.title: e for e in events}

    @classmethod
    def from_docs(cls, docs: list[Doc], embedder: Embedder, max_chars: int = 1200) -> "LocalBackend":
        events = events_from_docs(docs)
        prev, nxt = resolve_links(events)
        chunks: list[Chunk] = []
        for d in docs:
            chunks.extend(chunk_text(d.doc_id, d.text, max_chars=max_chars))
        matrix = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, embedder.dim), dtype=np.float32)
        return cls(docs, events, prev, nxt, chunks, matrix, embedder)

    # ---- structural queries ----
    def events_by_sport_games(self, sport: str, games: str) -> list[EventRecord]:
        s = norm_key(sport)
        return [e for e in self._events.values() if norm_key(e.sport) == s and e.games == games]

    def events_at_games(self, games: str) -> list[EventRecord]:
        return [e for e in self._events.values() if e.games == games]

    def event_by_title(self, title: str) -> Optional[EventRecord]:
        return self._by_title.get(title)

    def events_by_venue(self, venue: str, games: str) -> list[EventRecord]:
        v = norm_key(venue)
        return [e for e in self._events.values() if e.games == games and norm_key(e.venue) == v]

    def prev_event(self, doc_id: str) -> Optional[EventRecord]:
        target = self._prev.get(doc_id)
        return self._events.get(target) if target else None

    def neighborhood(self, doc_id: str) -> Optional[Neighborhood]:
        ev = self._events.get(doc_id)
        if ev is None:
            return None
        same_venue = [e for e in self.events_by_venue(ev.venue, ev.games) if e.doc_id != doc_id] if ev.venue else []
        nxt = self._events.get(self._next.get(doc_id, ""))
        return {"event": ev, "prev": self.prev_event(doc_id), "next": nxt, "same_venue": same_venue}

    # ---- text / vector ----
    def vector_search(self, query_vec: np.ndarray, k: int) -> list[ChunkHit]:
        if len(self._chunks) == 0:
            return []
        scores = self._matrix @ query_vec.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [ChunkHit(self._chunks[i].chunk_id, self._chunks[i].doc_id, self._chunks[i].text, float(scores[i])) for i in top]

    def doc_text(self, doc_id: str) -> str:
        d = self._docs.get(doc_id)
        return d.text if d else ""

    # ---- for loading into TigerGraph ----
    @property
    def events(self) -> list[EventRecord]:
        return list(self._events.values())

    @property
    def links(self) -> tuple[dict[str, str], dict[str, str]]:
        return self._prev, self._next

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def docs(self) -> list[Doc]:
        return list(self._docs.values())
