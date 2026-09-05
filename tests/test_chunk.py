from agrag.chunk import chunk_text


def test_chunk_text_splits_on_paragraphs_with_max_chars():
    text = "para one. " * 50 + "\n\n" + "para two. " * 50 + "\n\n" + "para three. " * 50
    chunks = chunk_text("D1", text, max_chars=600)
    assert all(len(c.text) <= 600 for c in chunks)
    assert [c.chunk_id for c in chunks][:2] == ["D1#0", "D1#1"]
    assert all(c.doc_id == "D1" for c in chunks)
    assert "".join(c.text for c in chunks).count("para one") == 50


def test_chunk_text_single_short_doc():
    chunks = chunk_text("D2", "short", max_chars=600)
    assert len(chunks) == 1 and chunks[0].text == "short"
