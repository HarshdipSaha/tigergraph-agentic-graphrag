from __future__ import annotations

from typing import Optional

import numpy as np

from agrag.backend import ChunkHit, Neighborhood
from agrag.embed import Embedder, SentenceTransformerEmbedder
from agrag.graph.client import connect
from agrag.infobox import EventRecord


def _opt(v) -> Optional[int]:
    return None if v is None or int(v) < 0 else int(v)


def _is_vertex_type_mismatch(e: Exception) -> bool:
    """True when TigerGraph rejected a vertex id because it isn't an Event — e.g. a vector_search hit
    landed on a distractor Document (film/person) with no corresponding Event vertex. LocalBackend
    returns None/[] for this case (see local_backend.py); TigerGraphBackend must match that contract
    instead of letting the query exception propagate and crash the caller."""
    return "Failed to convert user vertex id" in str(e)


def event_from_vertex(v: dict) -> EventRecord:
    a = v["attributes"]
    year, season = a["games"].split(" ", 1)
    return EventRecord(
        doc_id=v["v_id"], title=a["title"], sport=a["sport"], year=int(year), season=season,
        event_name=a["event_name"], venue=a["venue"], date_text=a["date_text"], competitors=_opt(a["competitors"]),
        competitors_raw=a["competitors_raw"], nations=_opt(a["nations"]), gold=a["gold"], gold_noc=a["gold_noc"],
        prev_year=_opt(a["prev_year"]), next_year=_opt(a["next_year"]), url=a["url"],
    )


class TigerGraphBackend:
    def __init__(self, conn, embedder: Embedder):
        self.conn, self.embedder = conn, embedder
        self._text_cache: dict[str, str] = {}

    @classmethod
    def from_settings(cls, embedder: Embedder | None = None) -> "TigerGraphBackend":
        return cls(connect(), embedder or SentenceTransformerEmbedder())

    def _events(self, query: str, params: dict, key: str = "R") -> list[EventRecord]:
        try:
            res = self.conn.runInstalledQuery(query, params)
        except Exception as e:
            if _is_vertex_type_mismatch(e):
                return []
            raise
        block = next((b for b in res if key in b), {})
        return [event_from_vertex(v) for v in block.get(key, [])]

    def events_by_sport_games(self, sport: str, games: str) -> list[EventRecord]:
        return self._events("events_by_sport_games", {"sport": sport, "games": games})

    def events_at_games(self, games: str) -> list[EventRecord]:
        return self._events("events_at_games", {"games": games})

    def event_by_title(self, title: str) -> Optional[EventRecord]:
        evs = self._events("event_by_title", {"title": title})
        return evs[0] if evs else None

    def events_by_venue(self, venue: str, games: str) -> list[EventRecord]:
        return self._events("events_by_venue", {"venue": venue, "games": games})

    def prev_event(self, doc_id: str) -> Optional[EventRecord]:
        # VERTEX<Event> params take a 1-tuple; a plain string is deprecated and silently falls back
        # to a slower GET-based retry on every call.
        evs = self._events("prev_event", {"ev": (doc_id,)}, key="P")
        return evs[0] if evs else None

    def neighborhood(self, doc_id: str) -> Optional[Neighborhood]:
        try:
            res = self.conn.runInstalledQuery("event_neighborhood", {"ev": (doc_id,)})
        except Exception as e:
            if _is_vertex_type_mismatch(e):
                return None
            raise
        get = lambda k: next((b[k] for b in res if k in b), [])
        start = get("Start")
        if not start:
            return None
        prev, nxt = get("P"), get("N")
        ev = event_from_vertex(start[0])
        same_venue = [event_from_vertex(v) for v in get("SameVenue")]
        return {
            "event": ev,
            "prev": event_from_vertex(prev[0]) if prev else None,
            "next": event_from_vertex(nxt[0]) if nxt else None,
            "same_venue": [e for e in same_venue if e.games == ev.games and e.doc_id != ev.doc_id],
        }

    def vector_search(self, query_vec: np.ndarray, k: int) -> list[ChunkHit]:
        res = self.conn.runInstalledQuery("chunk_search", {"query_vector": [float(x) for x in query_vec], "k": k})
        verts = next((b["v"] for b in res if "v" in b), [])
        dist_block = next((b["@@distances"] for b in res if "@@distances" in b), {})
        dist: dict[str, float] = {}
        if isinstance(dist_block, dict):                 # MapAccum printed as {vertex_id: distance}
            dist = {str(k_): float(v_) for k_, v_ in dist_block.items()}
        elif isinstance(dist_block, list):               # or as [{"v_id":..., "value":...}]
            for d in dist_block:
                if isinstance(d, dict) and "v_id" in d:
                    dist[d["v_id"]] = float(d.get("value", 0.0))
        # smaller distance = better; negate so callers can sort descending like LocalBackend
        hits = [ChunkHit(v["v_id"], v["attributes"]["doc_id"], v["attributes"]["text"], -dist.get(v["v_id"], 0.0)) for v in verts]
        return sorted(hits, key=lambda h: -h.score)[:k]

    def doc_text(self, doc_id: str) -> str:
        """Rebuild the document text from its ordered chunks (cached per process)."""
        if doc_id in self._text_cache:
            return self._text_cache[doc_id]
        chunks = self.conn.getEdges("Document", doc_id, "HAS_CHUNK")
        ids = [e["to_id"] for e in chunks]
        verts = self.conn.getVerticesById("Chunk", ids) if ids else []
        verts = sorted(verts, key=lambda v: v["attributes"]["ordinal"])
        text = "\n\n".join(v["attributes"]["text"] for v in verts)
        self._text_cache[doc_id] = text
        return text
