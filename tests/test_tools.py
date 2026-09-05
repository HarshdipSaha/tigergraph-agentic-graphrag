from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.local_backend import LocalBackend
from agrag.router import route
from agrag.tools import (
    aggregation_scan,
    date_score,
    lookup_nations,
    recover_competitors_from_text,
    resolve_multi_hop,
    superlative_scan,
    temporal_chain,
)


def backend(mini_corpus_path):
    return LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))


def test_lookup_nations(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = lookup_nations(b, route("How many nations competed in Sailing at the 2016 Summer Olympics – Women's RS:X?"))
    assert r.answer == "26" and r.evidence == ["Q7"] and r.structural_bound == 1 and r.needs_llm is False


def test_lookup_unknown_title(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = lookup_nations(b, route("How many nations competed in Curling at the 2018 Winter Olympics – Mixed doubles?"))
    assert r.answer is None and r.structural_bound == 0


def test_date_score_prefers_exact_day_month():
    assert date_score("11 February 2018", "11 February 2018") > date_score("11 February 2018", "10 February 2018")
    assert date_score("11–19 August", "11–19 August") == 1.0
    assert date_score("20 September 1988", "") == 0.0


def test_resolve_multi_hop_unique_by_date(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = resolve_multi_hop(b, route("Who won the gold medal in the event held at Alpensia Biathlon Centre on 11 February 2018?"))
    assert r.answer == "Arnd Peiffer" and r.evidence == ["Q2"]
    assert r.structural_bound == 4        # four events at that venue in 2018 Winter
    assert r.needs_llm is False
    assert r.candidates and len(r.candidates) == 4


def test_resolve_multi_hop_ambiguous_flags_llm(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = resolve_multi_hop(b, route("Who won the gold medal in the event held at Alpensia Biathlon Centre on February 2018?"))
    assert r.answer is None and r.needs_llm is True and len(r.candidates) == 4


def test_temporal_chain_follows_prev_edge(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = temporal_chain(b, route("Who won the gold medal in the women's sprint biathlon event at the Winter Olympics held immediately before 2018?"))
    assert r.answer == "Anastasiya Kuzmina"
    assert r.evidence == ["Q1", "Q4"] and r.hops == ["Q1", "Q4"] and r.fallback_used is False


def test_temporal_chain_missing_prev_uses_fallback(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = temporal_chain(b, route("Who won the gold medal in the men's sprint biathlon event at the Winter Olympics held immediately before 2018?"))
    assert r.answer is None and r.hops == ["Q2"] and r.fallback_used is True


def test_recover_competitors_from_text():
    assert recover_competitors_from_text("The relay was held. 72 competitors from 18 nations took part.") == 72
    assert recover_competitors_from_text("no numbers here") is None


def test_recover_competitors_ignores_infobox_year_line():
    text = "[Infobox Olympic event]\n  date: 22 February 2018\n  competitors: \n\n72 competitors took part."
    assert recover_competitors_from_text(text) == 72


def test_best_event_for_phrase_breaks_weight_class_ties_by_substring():
    from agrag.infobox import EventRecord
    from agrag.tools import best_event_for_phrase

    def ev(doc_id, name):
        return EventRecord(doc_id, f"Taekwondo at the 2016 Summer Olympics – {name}", "Taekwondo", 2016, "Summer",
                           name, "", "", None, "", None, "", "", None, None, "")
    events = [ev("A", "Men's 80 kg"), ev("B", "Men's +80 kg")]
    assert best_event_for_phrase(events, "men's 80 kg taekwondo").doc_id == "A"
    assert best_event_for_phrase(events, "men's +80 kg taekwondo").doc_id == "B"


def test_aggregation_scan_counts_and_recovers(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = aggregation_scan(b, route("According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?"))
    assert r.structural_bound == 4 and sorted(r.evidence) == ["Q1", "Q2", "Q3", "Q5"]
    assert r.answer == "2"                       # 87 and 86 > 73; 30 and 72(recovered) are not
    assert r.recovered == {"Q5": 72} and r.unresolved == []


def test_superlative_scan(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = superlative_scan(b, route("According to the provided corpus, which biathlon event at the 2018 Winter Olympics had the highest number of competitors?"))
    assert r.answer == "Biathlon at the 2018 Winter Olympics – Women's sprint"
    assert r.structural_bound == 4
