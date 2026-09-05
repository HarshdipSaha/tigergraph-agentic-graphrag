from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.llm import FakeLLM
from agrag.local_backend import LocalBackend
from agrag.pipelines.base import PipelineResult
from agrag.pipelines.rag import RagPipeline
from agrag.questions import load_questions


def test_rag_pipeline_retrieves_topk_and_asks_llm(mini_corpus_path, mini_public_path):
    b = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    llm = FakeLLM(["26"])
    p = RagPipeline(backend=b, llm=llm, k=3)
    q = load_questions(mini_public_path)[0]
    r = p.answer(q)
    assert isinstance(r, PipelineResult)
    assert r.pipeline == "rag" and r.answer == "26"
    assert len(r.docs_retrieved) <= 3 and r.tokens.total > 0
    assert llm.calls == 1 and q.question in llm.prompts[0][1]
    assert r.trace[0]["tool"] == "vector_search"
