from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.llm import FakeLLM
from agrag.local_backend import LocalBackend
from agrag.pipelines.graphrag import GraphRagPipeline
from agrag.questions import load_questions


def test_graphrag_expands_neighborhood_and_includes_structured_facts(mini_corpus_path, mini_public_path):
    b = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    llm = FakeLLM(["Arnd Peiffer"])
    p = GraphRagPipeline(backend=b, llm=llm, k=3)
    q = load_questions(mini_public_path)[1]
    r = p.answer(q)
    assert r.pipeline == "graphrag" and r.answer == "Arnd Peiffer"
    user_prompt = llm.prompts[0][1]
    assert "competitors:" in user_prompt and "gold:" in user_prompt   # structured facts were injected
    assert any(t["tool"] == "neighborhood" for t in r.trace)
    assert set(r.docs_retrieved) >= set(h for h in r.trace[0]["doc_ids"])
