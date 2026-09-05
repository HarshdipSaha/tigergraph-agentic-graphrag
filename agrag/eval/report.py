"""Aggregate results/*_public.jsonl into results/summary.json (per pipeline, per qtype)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

from agrag.eval.score import summarize


def load_rows(pattern: str = "results/*_public.jsonl") -> list[dict]:
    rows = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line)["score"] for line in f if line.strip())
    return rows


if __name__ == "__main__":
    rows = load_rows()
    summary = summarize(rows)
    Path("results/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for pipeline, by_type in summary.items():
        a = by_type["_all"]
        print(f"{pipeline:9s} n={a['n']:3d} acc={a['accuracy']:.2f} cov={a['avg_coverage']:.2f} tokens={a['avg_tokens']:.0f}")
