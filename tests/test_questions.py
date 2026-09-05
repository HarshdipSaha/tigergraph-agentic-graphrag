from agrag.questions import Question, load_questions


def test_load_public_questions(mini_public_path):
    qs = load_questions(mini_public_path)
    assert len(qs) == 5
    q = qs[0]
    assert isinstance(q, Question)
    assert q.qid == "mini-001" and q.qtype == "lookup"
    assert q.answer == "26"
    assert q.gold_doc_ids == ("Q7",)


def test_hidden_questions_have_no_answer(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"qid":"eval-001","question":"Who?","qtype":"multi_hop"}\n', encoding="utf-8")
    q = load_questions(p)[0]
    assert q.answer is None and q.gold_doc_ids == ()
