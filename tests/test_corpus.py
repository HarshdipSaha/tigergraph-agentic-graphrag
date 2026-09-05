from agrag.corpus import Doc, load_docs


def test_load_docs_reads_all_records(mini_corpus_path):
    docs = load_docs(mini_corpus_path)
    assert len(docs) == 7
    assert isinstance(docs[0], Doc)
    assert docs[0].doc_id == "Q1"
    assert docs[0].title.startswith("Biathlon at the 2018 Winter Olympics")
    assert docs[0].text.startswith("[Infobox Olympic event]")


def test_load_docs_skips_blank_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"doc_id":"A","title":"t","url":"","approx_tokens":1,"text":"x"}\n\n', encoding="utf-8")
    assert [d.doc_id for d in load_docs(p)] == ["A"]
