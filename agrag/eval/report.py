"""Aggregate explicitly selected public JSONL runs (never mix agentic modes silently)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from agrag.eval.score import summarize

DEFAULT_PUBLIC = [
    "results/rag_public.jsonl",
    "results/graphrag_public.jsonl",
    "results/agentic_public.jsonl",
]


def load_rows(paths: Sequence[str] | None = None) -> list[dict]:
    """Load score rows and fail if the same pipeline/question is present more than once."""
    selected = list(DEFAULT_PUBLIC if paths is None else paths)
    rows: list[dict] = []
    for path in selected:
        with open(path, encoding="utf-8") as stream:
            rows.extend(json.loads(line)["score"] for line in stream if line.strip())
    pairs = [(row["pipeline"], row["qid"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate pipeline/qid rows: selected files mix evaluation modes or repeat a run")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public", action="append", dest="public_paths", default=None,
                    help="public result JSONL to summarize; repeat once per pipeline/mode")
    ap.add_argument("--out", default="results/summary.json", help="summary JSON destination")
    args = ap.parse_args(argv)

    rows = load_rows(args.public_paths)
    summary = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for pipeline, by_type in summary.items():
        all_rows = by_type["_all"]
        print(f"{pipeline:9s} n={all_rows['n']:3d} acc={all_rows['accuracy']:.2f} "
              f"cov={all_rows['avg_coverage']:.2f} tokens={all_rows['avg_tokens']:.0f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
