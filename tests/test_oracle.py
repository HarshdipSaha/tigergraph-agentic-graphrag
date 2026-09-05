from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.local_backend import LocalBackend
from agrag.oracle import oracle_answer
from agrag.questions import load_questions


def test_oracle_answers_every_mini_question_without_llm(mini_corpus_path, mini_public_path):
    b = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    got = {q.qid: oracle_answer(b, q) for q in load_questions(mini_public_path)}
    assert got["mini-001"].answer == "26"
    assert got["mini-002"].answer == "Arnd Peiffer"
    assert got["mini-003"].answer == "Anastasiya Kuzmina"
    assert got["mini-004"].answer == "2"
    assert got["mini-005"].answer == "Biathlon at the 2018 Winter Olympics – Women's sprint"
    assert got["mini-004"].structural_bound == 4
