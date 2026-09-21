"""Benchmark Jev (TypeSafe AI) System One decision model against the Agentic GraphRAG pipeline.

Demonstrates:
1. Intent routing on natural language queries where rigid regex returns 'unknown'.
2. Disambiguation evaluation on multi-hop venue/date ties (pub-060).
3. Latency and token efficiency comparison.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agrag.decision import JevDecisionModel
from agrag.questions import load_questions
from agrag.router import route


def benchmark_non_template_routing(jev: JevDecisionModel):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Natural Language / Non-Template Intent Routing")
    print("=" * 60)

    test_queries = [
        ("How many countries participated in the 1996 Summer Olympics?", "lookup"),
        ("Tell me who took the gold medal at ExCeL on 30 July 2012", "multi_hop"),
        ("Which athlete won the 100m sprint right before the 2012 Games?", "temporal"),
        ("Find the number of wrestling events in 2004 with more than 20 competitors", "aggregation"),
        ("What was the largest event by participant count in 2008?", "superlative"),
    ]

    criteria = {
        "lookup": "Single fact lookup such as counting participating nations in an Olympic Games",
        "multi_hop": "Cross-reference who won an event held at a specific venue on a specific date",
        "temporal": "Event held immediately before or after in time",
        "aggregation": "Count or filter total events in a sport matching a threshold of competitors",
        "superlative": "Identify the event in a sport with the highest or lowest number of competitors",
    }

    regex_success = 0
    jev_success = 0
    total = len(test_queries)

    for q, expected in test_queries:
        pq = route(q)
        regex_matched = pq.template == expected
        if regex_matched:
            regex_success += 1

        res = jev.choice(f"Question: {q}", "Classify this Olympic query into the single most appropriate evidential category", criteria)
        jev_matched = res.decision == expected if res else False
        if jev_matched:
            jev_success += 1

        print(f"\nQuestion: \"{q}\"")
        print(f"  Expected   : {expected}")
        print(f"  Regex Route: {pq.template} [{'PASS' if regex_matched else 'FAIL -> falls back to unverified RAG'}]")
        if res:
            print(f"  Jev Route  : {res.decision} (conf={res.confidence:.2f}, latency={res.latency_ms}ms) [{'PASS' if jev_matched else 'FAIL'}]")

    print("\n" + "-" * 60)
    print(f"Regex Routing Accuracy on Natural Language: {regex_success}/{total} ({regex_success/total*100:.0f}%)")
    print(f"Jev System One Routing Accuracy           : {jev_success}/{total} ({jev_success/total*100:.0f}%)")
    print("-" * 60)


def benchmark_pub_060(jev: JevDecisionModel):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Disambiguation on pub-060 (3-way Venue/Date Tie)")
    print("=" * 60)

    pub_questions = [q for q in load_questions("data/eval_public.jsonl") if q.qid == "pub-060"]
    if not pub_questions:
        print("pub-060 not found in data/eval_public.jsonl")
        return
    q = pub_questions[0]

    tied_events = [
        {"doc_id": "Q1064016", "title": "Judo Women's 57 kg", "gold": "Kaori Matsumoto", "date": "30 July 2012", "venue": "ExCeL Exhibition Centre"},
        {"doc_id": "Q1005551", "title": "Judo Men's 73 kg", "gold": "Mansur Isaev", "date": "30 July 2012", "venue": "ExCeL Exhibition Centre"},
        {"doc_id": "Q1156695", "title": "Fencing Women's épée", "gold": "Yana Shemyakina", "date": "30 July", "venue": "ExCeL Exhibition Centre"},
    ]

    criteria = {
        e["doc_id"]: f"{e['title']} (Venue: {e['venue']}, Date: {e['date']}, Gold: {e['gold']})"
        for e in tied_events
    }
    state = f"Question: {q.question}\n\nCandidate Events:\n" + "\n".join(
        f"- {e['doc_id']}: {e['title']} | Date in infobox: '{e['date']}' | Venue: '{e['venue']}' | Gold: {e['gold']}"
        for e in tied_events
    )
    instructions = "Select the candidate Olympic event that matches the venue and date in the question. Prefer exact date string matches if there are ties."

    res = jev.choice(state, instructions, criteria, question_key="candidate_selection")

    print(f"Question: \"{q.question}\"")
    print(f"Benchmark Gold Answer: {q.answer}")
    print("\nTied candidates on 30 July 2012 at ExCeL:")
    for e in tied_events:
        print(f"  - {e['doc_id']}: {e['title']} -> Gold: {e['gold']} (Date field: '{e['date']}')")

    if res:
        print(f"\nJev Decision: {res.decision} (Latency: {res.latency_ms}ms, Confidence: {res.confidence:.2f})")
        print(f"Jev Calibrated Probabilities: {res.probabilities}")
        chosen_event = next((e for e in tied_events if e["doc_id"] == res.decision), None)
        if chosen_event:
            print(f"Chosen Winner: {chosen_event['gold']}")
            print(f"Matches Gold: {chosen_event['gold'] in q.answer}")


def main():
    jev = JevDecisionModel()
    if not jev.is_configured:
        print("Error: JEV_API_KEY is not configured in .env or environment.")
        return

    print(f"Running Jev Benchmark with model: {jev.model}")
    benchmark_non_template_routing(jev)
    benchmark_pub_060(jev)


if __name__ == "__main__":
    main()
