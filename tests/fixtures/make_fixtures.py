"""Regenerate mini_corpus.jsonl and mini_public.jsonl. Run: python tests/fixtures/make_fixtures.py"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def event_doc(doc_id, sport, year, season, event_name, venue, date, competitors, nations, gold, gold_noc, prev, nxt, prose):
    title = f"{sport} at the {year} {season} Olympics – {event_name}"
    lines = [
        "[Infobox Olympic event]",
        f"  event: {event_name}",
        f"  games: {year} {season}",
        f"  venue: {venue}",
        f"  date: {date}",
        f"  competitors: {competitors}",
        f"  nations: {nations}",
        f"  gold: {gold}",
        f"  goldNOC: {gold_noc}",
        f"  prev: {prev}",
        f"  next: {nxt}",
        "",
        prose,
    ]
    text = "\n".join(lines)
    return {
        "doc_id": doc_id,
        "title": title,
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "wikidata_qid": doc_id,
        "wikipedia_pageid": 0,
        "approx_tokens": len(text.split()),
        "text": text,
    }


def film_doc(doc_id, title, prose):
    text = "\n".join(["[Infobox film]", f"  name: {title}", "  released: 1994", "", prose])
    return {"doc_id": doc_id, "title": title, "url": "", "wikidata_qid": doc_id, "wikipedia_pageid": 0,
            "approx_tokens": len(text.split()), "text": text}


DOCS = [
    event_doc("Q1", "Biathlon", 2018, "Winter", "Women's sprint", "Alpensia Biathlon Centre", "10 February 2018", 87, 27,
              "Laura Dahlmeier", "GER", 2014, 2022,
              "The women's sprint biathlon competition at the 2018 Winter Olympics was held on 10 February 2018 at the Alpensia Biathlon Centre. Laura Dahlmeier of Germany won the gold medal."),
    event_doc("Q2", "Biathlon", 2018, "Winter", "Men's sprint", "Alpensia Biathlon Centre", "11 February 2018", 86, 30,
              "Arnd Peiffer", "GER", 2014, 2022,
              "The men's sprint biathlon competition at the 2018 Winter Olympics was held on 11 February 2018. Arnd Peiffer won gold."),
    event_doc("Q3", "Biathlon", 2018, "Winter", "Men's mass start", "Alpensia Biathlon Centre", "18 February 2018", 30, 14,
              "Martin Fourcade", "FRA", 2014, 2022,
              "The men's mass start was held on 18 February 2018. Martin Fourcade won."),
    event_doc("Q4", "Biathlon", 2014, "Winter", "Women's sprint", "Laura Biathlon & Ski Complex", "9 February 2014", 83, 27,
              "Anastasiya Kuzmina", "SVK", 2010, 2018,
              "The women's sprint at the 2014 Winter Olympics took place on 9 February 2014. Anastasiya Kuzmina won gold."),
    event_doc("Q5", "Biathlon", 2018, "Winter", "Women's relay", "Alpensia Biathlon Centre", "22 February 2018", "", 18,
              "Belarus", "BLR", 2014, 2022,
              "The women's relay was held on 22 February 2018. 72 competitors from 18 nations took part. Belarus won the gold medal."),
    event_doc("Q7", "Sailing", 2016, "Summer", "Women's RS:X", "Marina da Glória", "8–14 August 2016", 26, 26,
              "Charline Picon", "FRA", 2012, 2020,
              "The women's RS:X sailing event at the 2016 Summer Olympics was held at Marina da Glória. 26 sailors from 26 nations competed. Charline Picon won gold."),
    film_doc("Q6", "Forrest Gump", "Forrest Gump is a 1994 American comedy-drama film directed by Robert Zemeckis. It won six Academy Awards."),
]

QUESTIONS = [
    {"qid": "mini-001", "question": "How many nations competed in Sailing at the 2016 Summer Olympics – Women's RS:X?",
     "qtype": "lookup", "gold_doc_ids": ["Q7"], "answer": ["26"]},
    {"qid": "mini-002", "question": "Who won the gold medal in the event held at Alpensia Biathlon Centre on 11 February 2018?",
     "qtype": "multi_hop", "gold_doc_ids": ["Q2"], "answer": ["Arnd Peiffer"]},
    {"qid": "mini-003", "question": "Who won the gold medal in the women's sprint biathlon event at the Winter Olympics held immediately before 2018?",
     "qtype": "temporal", "gold_doc_ids": ["Q1", "Q4"], "answer": ["Anastasiya Kuzmina"]},
    {"qid": "mini-004", "question": "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
     "qtype": "aggregation", "gold_doc_ids": ["Q1", "Q2", "Q3", "Q5"], "answer": ["2"]},
    {"qid": "mini-005", "question": "According to the provided corpus, which biathlon event at the 2018 Winter Olympics had the highest number of competitors?",
     "qtype": "superlative", "gold_doc_ids": ["Q1", "Q2", "Q3", "Q5"], "answer": ["Biathlon at the 2018 Winter Olympics – Women's sprint"]},
]


def main() -> None:
    with open(HERE / "mini_corpus.jsonl", "w", encoding="utf-8") as f:
        for d in DOCS:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(HERE / "mini_public.jsonl", "w", encoding="utf-8") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
