from __future__ import annotations

from typing import Iterable

import numpy as np

from agrag.chunk import Chunk
from agrag.corpus import Doc
from agrag.infobox import EventRecord, parse_infobox

BATCH = 500


def _i(v):
    return -1 if v is None else int(v)


def _batches(items: list, n: int = BATCH) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def load_events(conn, events: list[EventRecord], prev: dict[str, str]) -> None:
    games = {(e.games, e.year, e.season) for e in events}
    conn.upsertVertices("Games", [(g, {"year": y, "season": s}) for g, y, s in games])
    conn.upsertVertices("Sport", [(s, {}) for s in {e.sport for e in events}])
    conn.upsertVertices("Venue", [(v, {}) for v in {e.venue for e in events if e.venue}])
    conn.upsertVertices("Athlete", [(a, {}) for a in {e.gold for e in events if e.gold}])
    for batch in _batches(events):
        conn.upsertVertices("Event", [(e.doc_id, {
            "title": e.title, "event_name": e.event_name, "sport": e.sport, "games": e.games, "venue": e.venue,
            "date_text": e.date_text, "competitors": _i(e.competitors), "competitors_raw": e.competitors_raw,
            "nations": _i(e.nations), "gold": e.gold, "gold_noc": e.gold_noc, "prev_year": _i(e.prev_year),
            "next_year": _i(e.next_year), "url": e.url}) for e in batch])
        conn.upsertEdges("Event", "PART_OF", "Games", [(e.doc_id, e.games, {}) for e in batch])
        conn.upsertEdges("Event", "IN_SPORT", "Sport", [(e.doc_id, e.sport, {}) for e in batch])
        conn.upsertEdges("Event", "HELD_AT", "Venue", [(e.doc_id, e.venue, {}) for e in batch if e.venue])
        conn.upsertEdges("Event", "GOLD", "Athlete", [(e.doc_id, e.gold, {}) for e in batch if e.gold])
        conn.upsertEdges("Event", "DESCRIBED_BY", "Document", [(e.doc_id, e.doc_id, {}) for e in batch])
    conn.upsertEdges("Event", "PREV", "Event", [(a, b, {}) for a, b in prev.items()])


def load_documents(conn, docs: list[Doc]) -> None:
    rows = []
    for d in docs:
        header, _ = parse_infobox(d.text)
        rows.append((d.doc_id, {"title": d.title, "kind": header or "none", "url": d.url}))
    for batch in _batches(rows):
        conn.upsertVertices("Document", batch)


def load_chunks(conn, chunks: list[Chunk], matrix: np.ndarray) -> None:
    for i in range(0, len(chunks), 200):
        batch = chunks[i:i + 200]
        conn.upsertVertices("Chunk", [(c.chunk_id, {
            "doc_id": c.doc_id, "ordinal": c.ordinal, "text": c.text,
            "emb": [float(x) for x in matrix[i + j]]}) for j, c in enumerate(batch)])
        conn.upsertEdges("Document", "HAS_CHUNK", "Chunk", [(c.doc_id, c.chunk_id, {}) for c in batch])
