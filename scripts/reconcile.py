"""Council-mandated reconciliation (docs/idea-spec.md §6).

For every public question: run the LLM-free oracle, compare the answer to gold, and compare the
evidence set to gold_doc_ids. Exhaustive classes require set equality; others require gold ⊆ evidence.

Usage:
  python scripts/reconcile.py --backend local      # validates parser + router, no TigerGraph needed
  python scripts/reconcile.py --backend tigergraph # validates the loaded graph (Task 14)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Prefer this checkout when the script is run inside an isolated Git worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrag.backend import GraphBackend
from agrag.normalize import match_flags
from agrag.questions import Question, load_questions
from agrag.oracle import oracle_answer
from agrag.router import route

EXHAUSTIVE = {"aggregation", "superlative"}


def reconcile(b: GraphBackend, questions: list[Question], backend_name: str) -> dict:
    rows = []
    for q in questions:
        r = oracle_answer(b, q)
        flags = match_flags(r.answer, q.answer or "")
        gold, ev = set(q.gold_doc_ids), set(r.evidence)
        if q.qtype in EXHAUSTIVE:
            evidence_ok = gold == ev
        else:
            evidence_ok = gold <= ev
        status = "ok" if (flags["normalized"] and evidence_ok) else "mismatch"
        rows.append({
            "qid": q.qid, "qtype": q.qtype, "template": route(q.question).template,
            "gold_answer": q.answer, "oracle_answer": r.answer, "answer_match": flags["normalized"],
            "gold_docs": sorted(gold), "evidence": sorted(ev), "structural_bound": r.structural_bound,
            "evidence_ok": evidence_ok, "missing_from_evidence": sorted(gold - ev), "extra_in_evidence": sorted(ev - gold),
            "needs_llm": r.needs_llm, "fallback_used": r.fallback_used, "notes": r.notes, "status": status,
        })
    summary = {
        "backend": backend_name,
        "questions": len(rows),
        "answer_match": sum(1 for r in rows if r["answer_match"]),
        "evidence_ok": sum(1 for r in rows if r["evidence_ok"]),
        "needs_llm": sum(1 for r in rows if r["needs_llm"]),
        "mismatch_qids": [r["qid"] for r in rows if r["status"] != "ok"],
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "tigergraph"], default="local")
    ap.add_argument("--questions", default="data/eval_public.jsonl")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.backend == "local":
        from agrag.corpus import load_docs
        from agrag.embed import FakeEmbedder
        from agrag.local_backend import LocalBackend

        backend: GraphBackend = LocalBackend.from_docs(load_docs(args.corpus), embedder=FakeEmbedder(dim=8))
    else:
        from agrag.graph.tg_backend import TigerGraphBackend

        backend = TigerGraphBackend.from_settings()

    report = reconcile(backend, load_questions(args.questions), args.backend)
    out = Path(args.out or f"data/reconciliation-{args.backend}.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    print(f"{s['backend']}: {s['answer_match']}/{s['questions']} answers match, {s['evidence_ok']}/{s['questions']} evidence ok, "
          f"{s['needs_llm']} need LLM; mismatches: {s['mismatch_qids']}")


if __name__ == "__main__":
    main()
