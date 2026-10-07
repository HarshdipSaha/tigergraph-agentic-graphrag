import pytest

from agrag.corpus import load_docs
from agrag.decision import DecisionResult, JevDecisionModel, MockDecisionModel
from agrag.embed import FakeEmbedder
from agrag.llm import FakeLLM
from agrag.local_backend import LocalBackend
from agrag.pipelines.agentic import AgenticPipeline
from agrag.questions import Question


def test_mock_decision_model():
    mock = MockDecisionModel(default_choice="aggregation")
    res = mock.choice("state", "instructions", {"lookup": "desc", "aggregation": "desc"})
    assert res is not None
    assert res.decision == "aggregation"
    assert res.confidence == 1.0
    assert mock.noul("state", "instructions") == 0.5


def test_jev_decision_model_unconfigured():
    jev = JevDecisionModel(api_key="")
    assert not jev.is_configured
    assert jev.choice("state", "instructions", {"a": "1"}) is None
    assert jev.noul("state", "instructions") is None


def test_agentic_pipeline_uses_decision_model_disambiguation(mini_corpus_path):
    b = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    llm = FakeLLM([])
    mock_dec = MockDecisionModel(default_choice="Q3")
    pipe = AgenticPipeline(backend=b, llm=llm, decision_model=mock_dec, agent_mode="template")

    q = Question("t1", "Who won the gold medal in the event held at Alpensia Biathlon Centre on February 2018?", "multi_hop", None, ())
    r = pipe.answer(q)
    assert r.answer == "Martin Fourcade"
    assert llm.calls == 0  # Jev decision model was used instead of LLM!
    assert any(s.tool == "jev_disambiguate" for s in r.certificate.steps)


def test_agentic_pipeline_unrouted_decision_model(mini_corpus_path):
    b = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    llm = FakeLLM(["Tom Hanks"])
    mock_dec = MockDecisionModel(default_choice="lookup")
    pipe = AgenticPipeline(backend=b, llm=llm, decision_model=mock_dec, agent_mode="template")

    q = Question("t2", "Who starred in Forrest Gump?", "", None, ())
    r = pipe.answer(q)
    assert r.answer == "Tom Hanks"
    assert any(s.tool == "jev_route" for s in r.certificate.steps)
    assert r.certificate.completeness_class == "existential"
