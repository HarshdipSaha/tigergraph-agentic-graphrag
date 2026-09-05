"""LLM-free structural oracle. Used by scripts/reconcile.py to prove the graph agrees with the gold set."""
from __future__ import annotations

from agrag.backend import GraphBackend
from agrag.questions import Question
from agrag.router import route
from agrag.tools import (
    ToolResult,
    aggregation_scan,
    lookup_nations,
    resolve_multi_hop,
    superlative_scan,
    temporal_chain,
)

DISPATCH = {
    "lookup": lookup_nations,
    "multi_hop": resolve_multi_hop,
    "temporal": temporal_chain,
    "aggregation": aggregation_scan,
    "superlative": superlative_scan,
}


def oracle_answer(b: GraphBackend, q: Question) -> ToolResult:
    pq = route(q.question)
    if pq.template is None:
        return ToolResult("none", None, [], 0, {}, needs_llm=True, notes=["unrouted"])
    return DISPATCH[pq.template](b, pq)
