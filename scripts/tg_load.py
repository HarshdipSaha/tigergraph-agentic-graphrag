"""Load data/corpus.jsonl into TigerGraph. ~2,951 docs, ~20k chunks; embedding is CPU-bound and can take
well over an hour without a GPU. Connects first so a dead/suspended workspace fails fast rather than after
the long embedding step, and caches the chunk embedding matrix to disk so a retry after a network failure
does not repeat the expensive part."""
import pickle
from pathlib import Path

import numpy as np

from agrag.corpus import load_docs
from agrag.embed import SentenceTransformerEmbedder
from agrag.graph.client import connect
from agrag.graph.load import load_chunks, load_documents, load_events
from agrag.local_backend import LocalBackend

CACHE = Path("data/embed_cache.pkl")

if __name__ == "__main__":
    conn = connect()
    print("Connected. Event count before load:", conn.getVertexCount("Event"))

    docs = load_docs("data/corpus.jsonl")

    if CACHE.exists():
        print("Loading cached embeddings from", CACHE)
        with open(CACHE, "rb") as f:
            cached = pickle.load(f)
        # Model instantiation is cheap (seconds); only the .encode() call over ~20k chunks is slow,
        # and that's exactly what the cache skips.
        local = LocalBackend(docs, cached["events"], cached["prev"], cached["next"], cached["chunks"],
                              cached["matrix"], embedder=SentenceTransformerEmbedder())
    else:
        print("No cache found, embedding from scratch (this is the slow part)...")
        local = LocalBackend.from_docs(docs, embedder=SentenceTransformerEmbedder())
        with open(CACHE, "wb") as f:
            pickle.dump({
                "events": local.events, "prev": local.links[0], "next": local.links[1],
                "chunks": local.chunks, "matrix": local.matrix,
            }, f)
        print("Cached embeddings to", CACHE)

    load_documents(conn, local.docs)
    prev, _ = local.links
    load_events(conn, local.events, prev)
    load_chunks(conn, local.chunks, local.matrix)
    print("Event:", conn.getVertexCount("Event"), "Chunk:", conn.getVertexCount("Chunk"), "PREV:", conn.getEdgeCount("PREV"))
