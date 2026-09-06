from __future__ import annotations

from agrag.backend import GraphBackend
from agrag.certificate import TokenUsage
from agrag.llm import LLM
from agrag.pipelines.base import ANSWER_SYSTEM, PipelineResult, Timer, clean_answer
from agrag.questions import Question


class RagPipeline:
    """Fixed sequence: embed question -> top-k chunks -> one LLM call.

    k=4 (not 8): a full 150-question run at k=8 costs ~2,400 tokens/request in retrieved context alone,
    which exceeds Groq's free-tier 200k-tokens/day cap before the run finishes. k=4 is still a realistic
    RAG configuration and doesn't change the qualitative comparison — no k under ~40 lets top-k retrieval
    answer an aggregation question needing 41 documents."""

    name = "rag"

    def __init__(self, backend: GraphBackend, llm: LLM, k: int = 4):
        self.b, self.llm, self.k = backend, llm, k

    def answer(self, q: Question) -> PipelineResult:
        with Timer() as t:
            qvec = self.b.embedder.embed([q.question])[0]  # type: ignore[attr-defined]
            hits = self.b.vector_search(qvec, self.k)
            context = "\n\n---\n\n".join(f"(doc {h.doc_id})\n{h.text}" for h in hits)
            user = f"Context:\n{context}\n\nQuestion: {q.question}\nAnswer:"
            resp = self.llm.complete(ANSWER_SYSTEM, user)
        docs = list(dict.fromkeys(h.doc_id for h in hits))
        return PipelineResult(
            qid=q.qid, pipeline=self.name, answer=clean_answer(resp.text), docs_retrieved=docs,
            tokens=TokenUsage(input=resp.input_tokens, output=resp.output_tokens, context=len(context) // 4),
            latency_ms=t.ms,
            trace=[{"tool": "vector_search", "k": self.k, "doc_ids": docs}, {"tool": "llm_answer"}],
        )
