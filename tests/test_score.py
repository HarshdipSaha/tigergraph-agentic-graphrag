from agrag.certificate import TokenUsage
from agrag.eval.score import score_result, summarize
from agrag.pipelines.base import PipelineResult
from agrag.questions import Question


def test_score_result_computes_match_and_coverage():
    q = Question("q1", "?", "aggregation", "5", ("A", "B", "C", "D"))
    r = PipelineResult(qid="q1", pipeline="rag", answer="5", docs_retrieved=["A", "B", "Z"], tokens=TokenUsage(input=10, output=2))
    s = score_result(q, r)
    assert s["normalized"] is True and s["coverage"] == 0.5 and s["tokens_total"] == 12
    assert s["qtype"] == "aggregation" and s["pipeline"] == "rag"


def test_summarize_groups_by_pipeline_and_qtype():
    rows = [
        {"pipeline": "rag", "qtype": "lookup", "normalized": True, "coverage": 1.0, "tokens_total": 10, "latency_ms": 5, "cert_pass": None},
        {"pipeline": "rag", "qtype": "lookup", "normalized": False, "coverage": 0.0, "tokens_total": 30, "latency_ms": 5, "cert_pass": None},
        {"pipeline": "agentic", "qtype": "lookup", "normalized": True, "coverage": 1.0, "tokens_total": 0, "latency_ms": 1, "cert_pass": True},
    ]
    s = summarize(rows)
    assert s["rag"]["lookup"]["n"] == 2 and s["rag"]["lookup"]["accuracy"] == 0.5 and s["rag"]["lookup"]["avg_tokens"] == 20
    assert s["agentic"]["lookup"]["cert_pass_rate"] == 1.0
    assert s["rag"]["_all"]["n"] == 2
