from agrag.questions import load_questions
from agrag.router import ParsedQuestion, route


def test_lookup_template():
    p = route("How many nations competed in Sailing at the 2016 Summer Olympics – Women's RS:X?")
    assert p.template == "lookup" and p.completeness_class == "existential"
    assert p.slots["title"] == "Sailing at the 2016 Summer Olympics – Women's RS:X"


def test_multi_hop_template_year_inside_date():
    p = route("Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988?")
    assert p.template == "multi_hop" and p.completeness_class == "existential"
    assert p.slots["venue"] == "Olympic Weightlifting Gymnasium"
    assert p.slots["date"] == "20 September 1988"
    assert p.slots["year"] == 1988 and p.slots["season"] is None


def test_multi_hop_template_games_suffix():
    p = route("Who won the gold medal in the event held at Riocentro – Pavilion 4 on 11–19 August at the 2016 Summer Olympics?")
    assert p.slots["venue"] == "Riocentro – Pavilion 4"
    assert p.slots["date"] == "11–19 August"
    assert p.slots["year"] == 2016 and p.slots["season"] == "Summer"


def test_temporal_template():
    p = route("Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?")
    assert p.template == "temporal" and p.completeness_class == "chained"
    assert p.slots["event_phrase"] == "men's 20 kilometres walk athletics"
    assert p.slots["season"] == "Summer" and p.slots["year"] == 2016


def test_aggregation_template():
    p = route("According to the provided corpus, how many cross-country skiing events at the 2010 Winter Olympics had more than 62 competitors?")
    assert p.template == "aggregation" and p.completeness_class == "exhaustive"
    assert p.slots == {"sport": "cross-country skiing", "year": 2010, "season": "Winter", "threshold": 62}


def test_superlative_template():
    p = route("According to the provided corpus, which sailing event at the 2004 Summer Olympics had the highest number of competitors?")
    assert p.template == "superlative" and p.completeness_class == "exhaustive"
    assert p.slots == {"sport": "sailing", "year": 2004, "season": "Summer"}


def test_unknown_question():
    p = route("What is the capital of France?")
    assert p.template is None and p.completeness_class == "unknown" and p.confidence == "low"


def test_router_agrees_with_qtype_on_real_public_set():
    qs = load_questions("data/eval_public.jsonl")
    mismatches = [(q.qid, q.qtype, route(q.question).template) for q in qs if route(q.question).template != q.qtype]
    assert mismatches == []
