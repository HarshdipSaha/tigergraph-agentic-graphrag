from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    qtype: str
    answer: Optional[str]
    gold_doc_ids: tuple[str, ...]


def load_questions(path: str | Path) -> list[Question]:
    out: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ans = d.get("answer")
            if isinstance(ans, list):
                ans = ans[0] if ans else None
            out.append(
                Question(
                    qid=d["qid"],
                    question=d["question"],
                    qtype=d.get("qtype", ""),
                    answer=ans,
                    gold_doc_ids=tuple(d.get("gold_doc_ids", [])),
                )
            )
    return out
