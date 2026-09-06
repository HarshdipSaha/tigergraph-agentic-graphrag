from __future__ import annotations

from agrag.backend import GraphBackend
from agrag.certificate import TokenUsage
from agrag.llm import LLM
from agrag.pipelines.base import ANSWER_SYSTEM, PipelineResult, Timer, clean_answer, event_facts
from agrag.questions import Question


class GraphRagPipeline:
    """Fixed sequence: top-k chunks -> 1-hop graph neighbourhood of every hit -> one LLM call. No branching.

    k=3 and a same-venue cap of 2 (not 6/5): each seed can otherwise contribute up to 8 fact blocks
    (event + prev + next + 5 same-venue), and some venues host 50+ events at one Games — at k=6 that
    blew past Groq's free-tier 200k-tokens/day cap partway through a 150-question run."""

    name = "graphrag"

    SAME_VENUE_CAP = 2

    def __init__(self, backend: GraphBackend, llm: LLM, k: int = 3):
        self.b, self.llm, self.k = backend, llm, k

    def answer(self, q: Question) -> PipelineResult:
        with Timer() as t:
            qvec = self.b.embedder.embed([q.question])[0]  # type: ignore[attr-defined]
            hits = self.b.vector_search(qvec, self.k)
            seed_docs = list(dict.fromkeys(h.doc_id for h in hits))
            facts: list[str] = []
            expanded: list[str] = []
            trace = [{"tool": "vector_search", "k": self.k, "doc_ids": seed_docs}]
            for doc_id in seed_docs:
                n = self.b.neighborhood(doc_id)
                if n is None:
                    continue
                facts.append(event_facts(n["event"]))
                for rel in ("prev", "next"):
                    if n[rel] is not None:
                        facts.append(f"({rel} Games) " + event_facts(n[rel]))
                        expanded.append(n[rel].doc_id)
                for e in n["same_venue"][: self.SAME_VENUE_CAP]:
                    facts.append("(same venue) " + event_facts(e))
                    expanded.append(e.doc_id)
                trace.append({"tool": "neighborhood", "doc_id": doc_id, "expanded": len(expanded)})
            chunks = "\n\n---\n\n".join(f"(doc {h.doc_id})\n{h.text}" for h in hits)
            context = "Structured facts:\n" + "\n\n".join(facts) + f"\n\nText chunks:\n{chunks}"
            user = f"{context}\n\nQuestion: {q.question}\nAnswer:"
            resp = self.llm.complete(ANSWER_SYSTEM, user)
            trace.append({"tool": "llm_answer"})
        return PipelineResult(
            qid=q.qid, pipeline=self.name, answer=clean_answer(resp.text),
            docs_retrieved=list(dict.fromkeys(seed_docs + expanded)),
            tokens=TokenUsage(input=resp.input_tokens, output=resp.output_tokens, context=len(context) // 4),
            latency_ms=t.ms, trace=trace,
        )
