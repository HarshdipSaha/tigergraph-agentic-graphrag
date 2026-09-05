from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str


def chunk_text(doc_id: str, text: str, max_chars: int = 1200) -> list[Chunk]:
    """Greedy paragraph packing; a paragraph longer than max_chars is hard-split."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    buf = ""
    for p in paragraphs:
        while len(p) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(p[:max_chars])
            p = p[max_chars:]
        candidate = f"{buf}\n\n{p}" if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            pieces.append(buf)
            buf = p
    if buf:
        pieces.append(buf)
    if not pieces:
        pieces = [text]
    return [Chunk(chunk_id=f"{doc_id}#{i}", doc_id=doc_id, ordinal=i, text=t) for i, t in enumerate(pieces)]
