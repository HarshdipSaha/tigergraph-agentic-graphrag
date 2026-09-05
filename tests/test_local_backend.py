from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.local_backend import LocalBackend


def make_backend(mini_corpus_path):
    return LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=16))


def test_events_by_sport_games_is_case_insensitive(mini_corpus_path):
    b = make_backend(mini_corpus_path)
    evs = b.events_by_sport_games("biathlon", "2018 Winter")
    assert sorted(e.doc_id for e in evs) == ["Q1", "Q2", "Q3", "Q5"]
    assert b.events_by_sport_games("Biathlon", "1900 Summer") == []


def test_event_by_title_and_events_at_games(mini_corpus_path):
    b = make_backend(mini_corpus_path)
    ev = b.event_by_title("Sailing at the 2016 Summer Olympics – Women's RS:X")
    assert ev is not None and ev.nations == 26
    assert b.event_by_title("nope") is None
    assert len(b.events_at_games("2018 Winter")) == 4


def test_events_by_venue_exact_match(mini_corpus_path):
    b = make_backend(mini_corpus_path)
    evs = b.events_by_venue("Alpensia Biathlon Centre", "2018 Winter")
    assert sorted(e.doc_id for e in evs) == ["Q1", "Q2", "Q3", "Q5"]


def test_prev_event_uses_resolved_links(mini_corpus_path):
    b = make_backend(mini_corpus_path)
    assert b.prev_event("Q1").doc_id == "Q4"
    assert b.prev_event("Q2") is None


def test_neighborhood_returns_structured_facts(mini_corpus_path):
    b = make_backend(mini_corpus_path)
    n = b.neighborhood("Q1")
    assert n["event"].doc_id == "Q1"
    assert n["prev"].doc_id == "Q4" and n["next"] is None
    assert n["same_venue"] and all(e.doc_id != "Q1" for e in n["same_venue"])


def test_vector_search_returns_chunk_hits_with_doc_ids(mini_corpus_path):
    b = make_backend(mini_corpus_path)
    q = b.embedder.embed(["anything"])[0]
    hits = b.vector_search(q, k=3)
    assert len(hits) == 3
    assert all(h.doc_id and h.text and h.chunk_id for h in hits)
    assert hits[0].score >= hits[-1].score


def test_doc_text(mini_corpus_path):
    b = make_backend(mini_corpus_path)
    assert "Forrest Gump" in b.doc_text("Q6")
    assert b.doc_text("missing") == ""
