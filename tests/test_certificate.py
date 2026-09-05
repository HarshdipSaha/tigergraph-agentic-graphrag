import json

from agrag.certificate import Certificate, Step, TokenUsage


def test_certificate_serialises_with_all_required_fields():
    c = Certificate(
        qid="pub-045", qtype="aggregation", completeness_class="exhaustive", classification_confidence="high",
        retrieval_mode="structural_scan", predicate={"sport": "athletics", "games": "2004 Summer", "threshold": 41},
        structural_bound=41, evidence_set_size=41, completeness_check="pass", docs_inspected=["Q1"],
        steps=[Step(tool="aggregation_scan", note="41 events", tokens=TokenUsage())],
        tokens=TokenUsage(input=0, output=0), latency_ms=12, stop_reason="structural_bound_met",
    )
    d = json.loads(c.model_dump_json())
    assert d["completeness_check"] == "pass" and d["tokens"]["total"] == 0
    assert d["steps"][0]["tool"] == "aggregation_scan"


def test_certificate_rejects_bad_check_value():
    try:
        Certificate(qid="x", qtype="lookup", completeness_class="existential", classification_confidence="high",
                    retrieval_mode="m", predicate={}, structural_bound=1, evidence_set_size=1,
                    completeness_check="maybe", docs_inspected=[], steps=[], tokens=TokenUsage(), latency_ms=0,
                    stop_reason="s")
    except ValueError:
        return
    raise AssertionError("expected validation error")
