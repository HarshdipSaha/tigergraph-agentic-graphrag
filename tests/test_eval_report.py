import json

import pytest

from agrag.eval.report import DEFAULT_PUBLIC, load_rows, main


def _write(path, qid="pub-001", pipeline="agentic"):
    row = {
        "qid": qid,
        "qtype": "lookup",
        "pipeline": pipeline,
        "normalized": True,
        "coverage": 1.0,
        "tokens_total": 12,
        "latency_ms": 25,
        "cert_pass": True,
    }
    path.write_text(json.dumps({"score": row}) + "\n", encoding="utf-8")


def test_default_report_paths_are_explicit_and_not_a_glob():
    assert DEFAULT_PUBLIC == [
        "results/rag_public.jsonl",
        "results/graphrag_public.jsonl",
        "results/agentic_public.jsonl",
    ]


def test_load_rows_rejects_mixed_agentic_modes_for_same_question(tmp_path):
    template = tmp_path / "template.jsonl"
    planner = tmp_path / "planner.jsonl"
    _write(template)
    _write(planner)

    with pytest.raises(ValueError, match="duplicate pipeline/qid"):
        load_rows([str(template), str(planner)])


def test_report_cli_accepts_repeated_public_paths_and_explicit_output(tmp_path):
    rag = tmp_path / "rag.jsonl"
    agentic = tmp_path / "agentic.jsonl"
    out = tmp_path / "summary.json"
    _write(rag, pipeline="rag")
    _write(agentic)

    main(["--public", str(rag), "--public", str(agentic), "--out", str(out)])
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["rag"]["_all"]["n"] == 1
    assert summary["agentic"]["_all"]["n"] == 1
