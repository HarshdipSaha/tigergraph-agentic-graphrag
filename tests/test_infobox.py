from pathlib import Path

import pytest

from agrag.corpus import Doc, load_docs
from agrag.infobox import (
    EventRecord,
    event_from_doc,
    events_from_docs,
    parse_infobox,
    parse_int,
    parse_title,
    resolve_links,
)


def test_parse_infobox_reads_header_and_fields():
    text = "[Infobox Olympic event]\n  event: Men's sprint\n  competitors: 86\n\nProse here."
    header, fields = parse_infobox(text)
    assert header == "[Infobox Olympic event]"
    assert fields == {"event": "Men's sprint", "competitors": "86"}


def test_parse_infobox_without_header():
    assert parse_infobox("Plain prose") == (None, {})


def test_parse_int_handles_non_numeric():
    assert parse_int("86") == 86
    assert parse_int("") is None
    assert parse_int("23 teams") is None
    assert parse_int(None) is None


def test_parse_title_handles_en_dash_and_hyphenated_sport():
    t = parse_title("Cross-country skiing at the 2010 Winter Olympics – Men's 15 kilometre freestyle")
    assert t == ("Cross-country skiing", 2010, "Winter", "Men's 15 kilometre freestyle")
    assert parse_title("Forrest Gump") is None


def test_event_from_doc_builds_record(mini_corpus_path):
    docs = {d.doc_id: d for d in load_docs(mini_corpus_path)}
    ev = event_from_doc(docs["Q1"])
    assert isinstance(ev, EventRecord)
    assert ev.sport == "Biathlon" and ev.year == 2018 and ev.season == "Winter"
    assert ev.games == "2018 Winter"
    assert ev.event_name == "Women's sprint"
    assert ev.venue == "Alpensia Biathlon Centre"
    assert ev.competitors == 87 and ev.nations == 27
    assert ev.gold == "Laura Dahlmeier" and ev.prev_year == 2014 and ev.next_year == 2022
    assert event_from_doc(docs["Q6"]) is None  # film


def test_event_from_doc_finds_olympic_infobox_after_tennis_tournament_infobox():
    doc = Doc(
        doc_id="QTEST-TENNIS",
        title="Tennis at the 2004 Summer Olympics",
        url="https://example.test/tennis",
        approx_tokens=0,
        text=(
            "[Infobox tennis tournament event]\n"
            "  name: 2004 Summer Olympics tennis\n"
            "  venue: Athens Olympic Tennis Centre\n"
            "\n"
            "[Infobox Olympic event]\n"
            "  venue: Olympic Tennis Centre\n"
            "  date: 15 to 22 August 2004\n"
            "  gold: Li TingSun Tiantian\n"
            "\n"
            "The Olympic tennis event."
        ),
    )

    # The public helper retains its established first-infobox contract.
    header, fields = parse_infobox(doc.text)
    assert header == "[Infobox tennis tournament event]"
    assert fields == {
        "name": "2004 Summer Olympics tennis",
        "venue": "Athens Olympic Tennis Centre",
    }

    event = event_from_doc(doc)
    assert isinstance(event, EventRecord)
    assert event.venue == "Olympic Tennis Centre"
    assert event.date_text == "15 to 22 August 2004"
    assert event.gold == "Li TingSun Tiantian"
    assert event.sport == "Tennis"
    assert event.games == "2004 Summer"
    assert event.doc_id == "QTEST-TENNIS"


def test_event_from_doc_reads_dates_field_when_date_is_absent():
    doc = Doc(
        doc_id="QTEST-DATES",
        title="Tennis at the 2004 Summer Olympics",
        url="https://example.test/tennis-dates",
        approx_tokens=0,
        text=(
            "[Infobox tennis tournament event]\n"
            "  name: tournament metadata\n"
            "\n"
            "[Infobox Olympic event]\n"
            "  venue: Olympic Tennis Centre\n"
            "  dates: 15 to 22 August 2004\n"
            "  gold: Li TingSun Tiantian\n"
        ),
    )

    event = event_from_doc(doc)
    assert event is not None
    assert event.date_text == "15 to 22 August 2004"


def test_event_from_doc_rejects_tennis_only_and_non_olympic_documents():
    tennis_only = Doc(
        doc_id="QTEST-TENNIS-ONLY",
        title="Tennis at the 2004 Summer Olympics",
        url="",
        approx_tokens=0,
        text="[Infobox tennis tournament event]\n  venue: Olympic Tennis Centre\n",
    )
    film = Doc(
        doc_id="QTEST-FILM",
        title="Forrest Gump",
        url="",
        approx_tokens=0,
        text="[Infobox film]\n  director: Robert Zemeckis\n",
    )

    assert event_from_doc(tennis_only) is None
    assert event_from_doc(film) is None


@pytest.mark.skipif(
    not Path("data/corpus.jsonl").exists(),
    reason="the full source corpus is optional and may be absent in CI",
)
def test_q942805_secondary_infobox_corpus_regression():
    docs = {doc.doc_id: doc for doc in load_docs("data/corpus.jsonl")}
    event = event_from_doc(docs["Q942805"])

    assert event is not None
    assert event.venue == "Olympic Tennis Centre"
    assert event.date_text == "15 to 22 August 2004"
    assert event.gold == "Li TingSun Tiantian"
    assert event.sport == "Tennis"
    assert event.games == "2004 Summer"
    assert event.doc_id == "Q942805"


def test_empty_competitors_becomes_none_but_raw_kept(mini_corpus_path):
    docs = {d.doc_id: d for d in load_docs(mini_corpus_path)}
    ev = event_from_doc(docs["Q5"])
    assert ev.competitors is None
    assert ev.competitors_raw == ""


def test_resolve_links_follows_prev_year_to_same_event(mini_corpus_path):
    events = events_from_docs(load_docs(mini_corpus_path))
    prev, nxt = resolve_links(events)
    assert prev["Q1"] == "Q4"   # 2018 women's sprint -> 2014 women's sprint
    assert nxt["Q4"] == "Q1"
    assert "Q2" not in prev      # 2014 men's sprint is not in the corpus
