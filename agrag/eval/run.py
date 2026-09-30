"""Run one pipeline over a question file and write results JSONL.

  python -m agrag.eval.run --pipeline agentic --backend tigergraph --questions data/eval_public.jsonl
  python -m agrag.eval.run --pipeline rag --backend local --questions tests/fixtures/mini_public.jsonl --fake-llm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agrag.eval.score import score_result
from agrag.questions import load_questions


def build_backend(kind: str, corpus: str):
    if kind == "local":
        from agrag.corpus import load_docs
        from agrag.embed import SentenceTransformerEmbedder
        from agrag.local_backend import LocalBackend

        return LocalBackend.from_docs(load_docs(corpus), embedder=SentenceTransformerEmbedder())
    from agrag.graph.tg_backend import TigerGraphBackend

    return TigerGraphBackend.from_settings()


def build_pipeline(name: str, backend, llm, decision_model=None):
    from agrag.pipelines.agentic import AgenticPipeline
    from agrag.pipelines.graphrag import GraphRagPipeline
    from agrag.pipelines.rag import RagPipeline

    if name == "agentic":
        from agrag.decision import get_decision_model
        dm = decision_model if decision_model is not None else get_decision_model()
        return AgenticPipeline(backend, llm, decision_model=dm)
    classes = {"rag": RagPipeline, "graphrag": GraphRagPipeline}
    return classes[name](backend, llm)


def main() -> None:
    # Windows consoles default to cp1252; answers contain non-ASCII names (e.g. Süleymanoğlu).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["rag", "graphrag", "agentic"], required=True)
    ap.add_argument("--backend", choices=["local", "tigergraph"], default="tigergraph")
    ap.add_argument("--questions", default="data/eval_public.jsonl")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake-llm", action="store_true", help="script UNKNOWN answers; for smoke runs only")
    args = ap.parse_args()

    if args.fake_llm:
        from agrag.llm import FakeLLM
        llm = FakeLLM(["UNKNOWN"] * 10_000)
    else:
        from agrag.llm import build_llm
        llm = build_llm()   # honors AGRAG_LLM_PROVIDER (groq by default, or anthropic)

    backend = build_backend(args.backend, args.corpus)
    pipe = build_pipeline(args.pipeline, backend, llm)
    qs = load_questions(args.questions)[: args.limit]
    tag = Path(args.questions).stem.replace("eval_", "")
    out = Path(args.out or f"results/{args.pipeline}_{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            r = pipe.answer(q)
            row = {"question": q.question, "result": r.model_dump(), "score": score_result(q, r)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(qs)}] {q.qid} {q.qtype:12s} -> {r.answer!r} (gold {q.answer!r}) tokens={r.tokens.total}")
    print("wrote", out)


if __name__ == "__main__":
    main()
