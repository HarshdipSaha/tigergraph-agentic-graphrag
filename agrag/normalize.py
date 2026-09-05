from __future__ import annotations

import re
import unicodedata
from typing import Optional


def normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = s.lower()
    s = re.sub(r"[^\w\s'\-:/.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_flags(pred: Optional[str], gold: str) -> dict[str, bool]:
    if pred is None:
        return {"exact": False, "normalized": False, "contains": False}
    np, ng = normalize(pred), normalize(gold)
    return {
        "exact": pred.strip() == gold.strip(),
        "normalized": np == ng,
        "contains": ng != "" and ng in np,
    }
