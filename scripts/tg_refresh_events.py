"""Refresh TigerGraph's Event subgraph from the source corpus without re-embedding chunks."""
from __future__ import annotations

import sys
from pathlib import Path

# A globally installed editable copy can otherwise win when this file is invoked as
# `python scripts/tg_refresh_events.py` from an isolated worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrag.corpus import load_docs
from agrag.graph.client import connect
from agrag.graph.load import load_events
from agrag.infobox import events_from_docs, resolve_links


def main() -> None:
    docs = load_docs("data/corpus.jsonl")
    events = events_from_docs(docs)
    if not events:
        raise RuntimeError("No Event records parsed; refusing to modify TigerGraph")

    prev, _ = resolve_links(events)
    conn = connect()
    document_count = conn.getVertexCount("Document")
    if document_count != len(docs):
        raise RuntimeError(
            f"TigerGraph has {document_count} Documents but the source corpus has {len(docs)}; "
            "load/verify Documents before refreshing Events"
        )

    before = conn.getVertexCount("Event")
    load_events(conn, events, prev)
    # getVertexCount can briefly lag a batch upsert on the hosted workspace. Verify the
    # actual primary IDs once instead, including absence of stale Event vertices.
    vertices = conn.getVertices("Event", limit=len(events) + 1)
    actual_ids = {vertex.get("v_id") for vertex in vertices if vertex.get("v_id")}
    expected_ids = {event.doc_id for event in events}
    missing = expected_ids - actual_ids
    unexpected = actual_ids - expected_ids
    print(
        f"Event count: {before} -> {len(actual_ids)}; parsed={len(events)}; "
        f"missing={len(missing)}; unexpected={len(unexpected)}; chunks/embeddings unchanged"
    )
    if missing or unexpected:
        raise RuntimeError(
            "TigerGraph Event IDs do not match the parsed source corpus "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )


if __name__ == "__main__":
    main()
