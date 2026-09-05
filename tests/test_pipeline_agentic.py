from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.llm import FakeLLM
from agrag.local_backend import LocalBackend
from agrag.pipelines.agentic import AgenticPipeline
from agrag.questions import Question, load_questions


def make(mini_corpus_path, answers):
    b = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    llm = FakeLLM(answers)
    return AgenticPipeline(backend=b, llm=llm), llm


def test_exhaustive_question_uses_no_llm_and_certifies_completeness(mini_corpus_path, mini_public_path):
    p, llm = make(mini_corpus_path, [])
    q = [x for x in load_questions(mini_public_path) if x.qid == "mini-004"][0]
    r = p.answer(q)
    assert r.answer == "2" and llm.calls == 0 and r.tokens.total == 0
    c = r.certificate
    assert c.completeness_class == "exhaustive" and c.retrieval_mode == "structural_scan"
    assert c.structural_bound == 4 and c.evidence_set_size == 4
    assert c.completeness_check == "pass"          # Q5 recovered by regex, not by LLM
    assert c.stop_reason == "structural_bound_met"
    assert sorted(r.docs_retrieved) == ["Q1", "Q2", "Q3", "Q5"]


def test_existential_lookup_certificate(mini_corpus_path, mini_public_path):
    p, llm = make(mini_corpus_path, [])
    q = load_questions(mini_public_path)[0]
    r = p.answer(q)
    assert r.answer == "26" and r.certificate.completeness_class == "existential"
    assert r.certificate.structural_bound == 1 and r.certificate.evidence_set_size == 1
    assert r.certificate.completeness_check == "pass"


def test_multi_hop_ambiguous_uses_llm_for_disambiguation(mini_corpus_path):
    p, llm = make(mini_corpus_path, ["Q3"])
    q = Question("t1", "Who won the gold medal in the event held at Alpensia Biathlon Centre on February 2018?", "multi_hop", None, ())
    r = p.answer(q)
    assert llm.calls == 1 and r.answer == "Martin Fourcade"
    assert r.certificate.completeness_check == "pass_with_llm_recovery"
    assert r.certificate.structural_bound == 4 and r.tokens.total > 0
    assert any(s.tool == "llm_disambiguate" for s in r.certificate.steps)


def test_chained_temporal_certificate(mini_corpus_path, mini_public_path):
    p, llm = make(mini_corpus_path, [])
    q = [x for x in load_questions(mini_public_path) if x.qid == "mini-003"][0]
    r = p.answer(q)
    assert r.answer == "Anastasiya Kuzmina"
    assert r.certificate.completeness_class == "chained" and r.certificate.completeness_check == "pass"
    assert r.certificate.retrieval_mode == "hop_chain" and r.certificate.evidence_set_size == 2


def test_unrouted_question_falls_back_to_graphrag_unverified(mini_corpus_path):
    p, llm = make(mini_corpus_path, ["Tom Hanks"])
    q = Question("t2", "Who starred in Forrest Gump?", "", None, ())
    r = p.answer(q)
    assert r.answer == "Tom Hanks" and llm.calls == 1
    assert r.certificate.completeness_class == "unknown"
    assert r.certificate.classification_confidence == "low"
    assert r.certificate.completeness_check == "unverified"
