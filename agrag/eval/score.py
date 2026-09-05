from __future__ import annotations

from collections import defaultdict

from agrag.normalize import match_flags
from agrag.pipelines.base import PipelineResult
from agrag.questions import Question


def score_result(q: Question, r: PipelineResult) -> dict:
    flags = match_flags(r.answer, q.answer or "") if q.answer is not None else {"exact": None, "normalized": None, "contains": None}
    gold = set(q.gold_doc_ids)
    coverage = (len(gold & set(r.docs_retrieved)) / len(gold)) if gold else None
    cert = r.certificate
    return {
        "qid": q.qid, "qtype": q.qtype, "pipeline": r.pipeline, "answer": r.answer, "gold": q.answer,
        **flags, "coverage": coverage, "docs_retrieved": len(r.docs_retrieved), "gold_docs": len(gold),
        "tokens_input": r.tokens.input, "tokens_output": r.tokens.output, "tokens_context": r.tokens.context,
        "tokens_total": r.tokens.total, "latency_ms": r.latency_ms,
        "cert_class": cert.completeness_class if cert else None,
        "cert_check": cert.completeness_check if cert else None,
        "cert_pass": (cert.completeness_check.startswith("pass")) if cert else None,
        "cert_bound": cert.structural_bound if cert else None,
        "cert_evidence": cert.evidence_set_size if cert else None,
    }


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["pipeline"]][r["qtype"]].append(r)
        groups[r["pipeline"]]["_all"].append(r)
    out: dict = {}
    for pipeline, by_type in groups.items():
        out[pipeline] = {}
        for qtype, rs in by_type.items():
            scored = [r for r in rs if r["normalized"] is not None]
            cov = [r["coverage"] for r in rs if r.get("coverage") is not None]
            certs = [r["cert_pass"] for r in rs if r.get("cert_pass") is not None]
            out[pipeline][qtype] = {
                "n": len(rs),
                "accuracy": (sum(1 for r in scored if r["normalized"]) / len(scored)) if scored else None,
                "avg_coverage": (sum(cov) / len(cov)) if cov else None,
                "avg_tokens": sum(r["tokens_total"] for r in rs) / len(rs),
                "avg_latency_ms": sum(r["latency_ms"] for r in rs) / len(rs),
                "cert_pass_rate": (sum(1 for c in certs if c) / len(certs)) if certs else None,
            }
    return out
