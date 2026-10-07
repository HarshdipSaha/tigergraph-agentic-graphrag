from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.local_backend import LocalBackend
from agrag.router import route
from agrag.tools import (
    aggregation_scan,
    date_score,
    date_match_status,
    parse_date_spans,
    lookup_nations,
    recover_competitors_from_text,
    resolve_multi_hop,
    superlative_scan,
    temporal_chain,
)
from agrag.infobox import EventRecord
from agrag.router import ParsedQuestion


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


def test_date_match_status_parses_ranges_and_missing_years():
    assert date_match_status("18 August 2004", "15 to 22 August 2004") == "day_in_range"
    assert date_match_status("August 18, 2004", "August 15-22, 2004") == "day_in_range"
    assert date_match_status("11–19 August", "11 to 19 August", default_year=2012) == "exact_range"
    assert date_match_status("11 February 2018", "10 February 2018") == "disjoint"
    assert date_match_status("30 July 2012", "", default_year=2012) == "unknown"
    assert date_match_status("8 August 2012", "6, 8, and 10 August", default_year=2012) == "day_in_range"
    schedule = "22 September 2000 (heats)25 September 2000 (final)"
    assert date_match_status(schedule, schedule, default_year=2000) == "exact_range"
    assert len(parse_date_spans("6, 8, and 10 August", default_year=2012)) == 3


def _event(doc_id, sport, venue, date_text, gold="Winner", year=2012, season="Summer"):
    return EventRecord(doc_id, f"{sport} at the {year} {season} Olympics – {doc_id}", sport, year, season,
                       doc_id, venue, date_text, None, "", None, gold, "", None, None, "")


class _FakeVenueBackend:
    def __init__(self, events):
        self.events = events

    def events_by_venue(self, venue, games):
        return [e for e in self.events if e.games == games and e.venue.casefold() == venue.casefold()]

    def events_at_games(self, games):
        return [e for e in self.events if e.games == games]


def _multi_hop(venue, date, sport=None):
    pq = route(f"Who won the gold medal in the event held at {venue} on {date} at the 2012 Summer Olympics?")
    if sport is not None:
        pq = ParsedQuestion(pq.question, pq.template, pq.completeness_class, {**pq.slots, "sport": sport})
    return pq


def test_same_day_venue_candidates_remain_ambiguous_and_exclude_other_day():
    events = [
        _event("Q1156695", "Fencing", "ExCeL Exhibition Centre", "30 July", "Yana Shemyakina"),
        _event("Q1064016", "Judo", "ExCeL Exhibition Centre", "30 July 2012", "Aiko"),
        _event("Q1005551", "Judo", "ExCeL Exhibition Centre", "30 July 2012", "Masashi"),
        _event("Q31844", "Boxing", "ExCeL Exhibition Centre", "29 July to 12 August 2012", "Boxing"),
        _event("Q932145", "Judo", "ExCeL Exhibition Centre", "29 July 2012", "Tadahiro"),
    ]
    r = resolve_multi_hop(_FakeVenueBackend(events), _multi_hop("ExCeL Exhibition Centre", "30 July"))
    assert r.answer is None and r.needs_llm is True
    assert {e.doc_id for e in r.candidates} == {"Q1156695", "Q1064016", "Q1005551"}
    assert "Q932145" not in [e.doc_id for e in r.candidates]
    assert r.raw_candidate_count == 5 and r.eligible_count == 3
    assert r.match_mode == "exact_venue" and r.ambiguity_reason == "multiple_eligible_candidates"


def test_grounded_sport_slot_filters_candidate_sports():
    events = [
        _event("fencing", "Fencing", "ExCeL Exhibition Centre", "30 July", "Yana Shemyakina"),
        _event("judo", "Judo", "ExCeL Exhibition Centre", "30 July", "Judo winner"),
    ]
    r = resolve_multi_hop(_FakeVenueBackend(events), _multi_hop("ExCeL Exhibition Centre", "30 July", sport="Fencing"))
    assert r.answer == "Yana Shemyakina"
    assert [e.doc_id for e in r.candidates] == ["fencing"]
    assert r.raw_candidate_count == 2 and r.eligible_count == 1


def test_duplicate_candidate_doc_ids_are_counted_once():
    event = _event("same-doc", "Sailing", "Test Venue", "30 July")
    r = resolve_multi_hop(_FakeVenueBackend([event, event]), _multi_hop("Test Venue", "30 July"))
    assert r.answer == "Winner" and r.raw_candidate_count == 1 and r.eligible_count == 1


def test_unique_event_inside_date_range_is_selected():
    event = EventRecord("Q942805", "Tennis at the 2004 Summer Olympics – Women's doubles", "Tennis", 2004,
                        "Summer", "Women's doubles", "Olympic Tennis Centre", "15 to 22 August 2004", None,
                        "", None, "Li Ting", "", None, None, "")
    pq = route("Who won the gold medal in the event held at Olympic Tennis Centre on 18 August 2004 at the 2004 Summer Olympics?")
    # The year in the date and the Games predicate can differ in shape, so use a matching 2004 fake backend.
    class TennisBackend(_FakeVenueBackend):
        def events_by_venue(self, venue, games):
            return [event] if games == "2004 Summer" and venue == event.venue else []

        def events_at_games(self, games):
            return [event] if games == "2004 Summer" else []

    r = resolve_multi_hop(TennisBackend([event]), pq)
    assert r.answer == "Li Ting" and r.eligible_count == 1
    assert r.selection_basis == "unique_venue_and_day_in_range"


def test_exact_day_outweighs_a_competition_interval_containing_that_day():
    events = [
        _event("exact", "Weightlifting", "Competition Hall", "23 September 2000", year=2012),
        _event("interval", "Boxing", "Competition Hall", "16 September to 30 September 2000", year=2012),
    ]
    r = resolve_multi_hop(_FakeVenueBackend(events), _multi_hop("Competition Hall", "23 September 2000"))
    assert r.answer == "Winner"
    assert [event.doc_id for event in r.candidates] == ["exact"]
    assert r.raw_candidate_count == 2 and r.eligible_count == 1


def test_blank_venues_and_unparseable_dates_never_match():
    event = _event("blank", "Sailing", "", "30 July", "Unsafe")
    r = resolve_multi_hop(_FakeVenueBackend([event]), _multi_hop("", "30 July"))
    assert r.answer is None and r.raw_candidate_count == 0 and r.candidates == []

    event = _event("bad-date", "Sailing", "Test Venue", "date unknown", "Unsafe")
    r = resolve_multi_hop(_FakeVenueBackend([event]), _multi_hop("Test Venue", "30 July"))
    assert r.answer is None and r.eligible_count == 0 and r.candidates == []
    assert r.ambiguity_reason == "no_parseable_candidate_dates"


def test_venue_alias_matching_is_token_bounded_and_reports_mode():
    event = _event("alias", "Biathlon", "Alpensia Biathlon Centre", "11 February 2018", year=2018, season="Winter")
    pq = route("Who won the gold medal in the event held at Alpensia on 11 February 2018?")
    pq = ParsedQuestion(pq.question, pq.template, pq.completeness_class,
                        {**pq.slots, "year": 2018, "season": "Winter"})
    r = resolve_multi_hop(_FakeVenueBackend([event]), pq)
    assert r.answer == "Winner" and r.match_mode == "venue_alias"

    pq_bad = ParsedQuestion(pq.question, pq.template, pq.completeness_class,
                            {**pq.slots, "venue": "alpen", "year": 2018, "season": "Winter"})
    r_bad = resolve_multi_hop(_FakeVenueBackend([event]), pq_bad)
    assert r_bad.answer is None and r_bad.raw_candidate_count == 0


def test_date_overlap_is_uncertain_and_not_eligible():
    assert date_match_status("15–20 August 2012", "18–23 August 2012") == "overlap_uncertain"
    assert date_match_status("August 2012", "18 August 2012") == "day_in_range"


def test_resolve_multi_hop_unique_by_date(mini_corpus_path):
    b = backend(mini_corpus_path)
    r = resolve_multi_hop(b, route("Who won the gold medal in the event held at Alpensia Biathlon Centre on 11 February 2018?"))
    assert r.answer == "Arnd Peiffer" and r.evidence == ["Q2"]
    assert r.structural_bound == 4        # four events at that venue in 2018 Winter
    assert r.needs_llm is False
    assert r.candidates and [e.doc_id for e in r.candidates] == ["Q2"]


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
