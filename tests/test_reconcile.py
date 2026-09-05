import json

from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.local_backend import LocalBackend
from agrag.questions import load_questions
from scripts.reconcile import reconcile


def test_reconcile_reports_full_agreement_on_fixture(mini_corpus_path, mini_public_path):
    b = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    report = reconcile(b, load_questions(mini_public_path), backend_name="local")
    assert report["summary"]["questions"] == 5
    assert report["summary"]["answer_match"] == 5
    assert report["summary"]["evidence_ok"] == 5
    assert all(r["status"] == "ok" for r in report["rows"])
    json.dumps(report)  # serialisable
