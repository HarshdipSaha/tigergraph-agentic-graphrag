from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    url: str
    approx_tokens: int
    text: str


def iter_docs(path: str | Path) -> Iterator[Doc]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Doc(
                doc_id=d["doc_id"],
                title=d["title"],
                url=d.get("url", ""),
                approx_tokens=int(d.get("approx_tokens", 0)),
                text=d["text"],
            )


def load_docs(path: str | Path) -> list[Doc]:
    return list(iter_docs(path))
