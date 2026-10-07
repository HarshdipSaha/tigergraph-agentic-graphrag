import json

from agrag.corpus import load_docs
from agrag.embed import FakeEmbedder
from agrag.llm import FakeLLM, LLMResponse
from agrag.local_backend import LocalBackend
from agrag.pipelines.agentic import AgenticPipeline
from agrag.questions import Question, load_questions
from agrag.router import route
from agrag.corpus import Doc


def make(mini_corpus_path, answers):
    b = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    llm = FakeLLM(answers)
    return AgenticPipeline(backend=b, llm=llm, agent_mode="template"), llm


def _action(tool, args):
    return json.dumps({"tool": tool, "args": args, "reason": "matches question"})


def _planner_tool_args(question):
    parsed = route(question)
    tool = {
        "lookup": "lookup_nations",
        "multi_hop": "resolve_multi_hop",
        "temporal": "temporal_chain",
        "aggregation": "aggregation_scan",
        "superlative": "superlative_scan",
    }[parsed.template]
    return tool, {key: value for key, value in parsed.slots.items() if value is not None}


def _venue_docs():
    return [
        Doc(
            doc_id="QROW",
            title="Rowing at the 1988 Summer Olympics â€“ Men's single sculls",
            url="",
            approx_tokens=10,
            text="[Infobox Olympic event]\n  event: Men's single sculls\n  games: 1988 Summer\n  venue: Lake Venue\n  date: 20 September 1988\n  gold: Rowing Winner\n",
        ),
        Doc(
            doc_id="QSAIL",
            title="Sailing at the 1988 Summer Olympics â€“ Men's dinghy",
            url="",
            approx_tokens=10,
            text="[Infobox Olympic event]\n  event: Men's dinghy\n  games: 1988 Summer\n  venue: Lake Venue\n  date: 20 September 1988\n  gold: Sailing Winner\n",
        ),
    ]


class ScriptedLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.prompts = []

    def complete(self, system, user, *, timeout_s=None):
        self.calls += 1
        self.prompts.append((system, user))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return LLMResponse(answer, len(system.split()) + len(user.split()), len(answer.split()))


def test_exhaustive_question_uses_no_llm_and_certifies_completeness(mini_corpus_path, mini_public_path):
    p, llm = make(mini_corpus_path, [])
    q = [x for x in load_questions(mini_public_path) if x.qid == "mini-004"][0]
    r = p.answer(q)
    assert r.answer == "2" and llm.calls == 0 and r.tokens.total == 0
    c = r.certificate
    assert c.completeness_class == "exhaustive" and c.retrieval_mode == "structural_scan"
    assert c.structural_bound == 4 and c.evidence_set_size == 4
    assert c.completeness_check == "pass"          # Q5 recovered by regex, not by LLM
    assert c.stop_reason == "structural_bound_met"
    assert sorted(r.docs_retrieved) == ["Q1", "Q2", "Q3", "Q5"]


def test_existential_lookup_certificate(mini_corpus_path, mini_public_path):
    p, llm = make(mini_corpus_path, [])
    q = load_questions(mini_public_path)[0]
    r = p.answer(q)
    assert r.answer == "26" and r.certificate.completeness_class == "existential"
    assert r.certificate.structural_bound == 1 and r.certificate.evidence_set_size == 1
    assert r.certificate.completeness_check == "pass"


def test_multi_hop_ambiguous_uses_llm_for_disambiguation(mini_corpus_path):
    p, llm = make(mini_corpus_path, ["Q3"])
    q = Question("t1", "Who won the gold medal in the event held at Alpensia Biathlon Centre on February 2018?", "multi_hop", None, ())
    r = p.answer(q)
    assert llm.calls == 1 and r.answer == "Martin Fourcade"
    assert r.certificate.completeness_check == "unverified"
    assert r.certificate.structural_bound == 4 and r.tokens.total > 0
    assert any(s.tool == "llm_disambiguate" for s in r.certificate.steps)


def test_chained_temporal_certificate(mini_corpus_path, mini_public_path):
    p, llm = make(mini_corpus_path, [])
    q = [x for x in load_questions(mini_public_path) if x.qid == "mini-003"][0]
    r = p.answer(q)
    assert r.answer == "Anastasiya Kuzmina"
    assert r.certificate.completeness_class == "chained" and r.certificate.completeness_check == "pass"
    assert r.certificate.retrieval_mode == "hop_chain" and r.certificate.evidence_set_size == 2


def test_unrouted_question_falls_back_to_graphrag_unverified(mini_corpus_path):
    p, llm = make(mini_corpus_path, ["Tom Hanks"])
    q = Question("t2", "Who starred in Forrest Gump?", "", None, ())
    r = p.answer(q)
    assert r.answer == "Tom Hanks" and llm.calls == 1
    assert r.certificate.completeness_class == "unknown"
    assert r.certificate.classification_confidence == "low"
    assert r.certificate.completeness_check == "unverified"


def test_planner_executes_all_five_existing_graph_tools(mini_corpus_path, mini_public_path):
    docs = load_docs(mini_corpus_path)
    backend = LocalBackend.from_docs(docs, embedder=FakeEmbedder(dim=8))
    for q in load_questions(mini_public_path):
        tool, args = _planner_tool_args(q.question)
        llm = ScriptedLLM([_action(tool, args)])
        result = AgenticPipeline(backend, llm).answer(q)
        assert result.answer == q.answer
        assert result.certificate.planning_mode == "planner"
        assert result.certificate.selected_tool == tool
        assert result.certificate.completeness_check in {"pass", "pass_with_fallback", "pass_with_llm_recovery"}
        assert result.tokens.total > 0 and llm.calls == 1


def test_planner_retries_invalid_json_with_sanitized_error_observation(mini_corpus_path):
    backend = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    q = load_questions("tests/fixtures/mini_public.jsonl")[0]
    tool, args = _planner_tool_args(q.question)
    action = _action(tool, args)
    llm = ScriptedLLM(["not-json", action])
    result = AgenticPipeline(backend, llm).answer(q)

    observation = json.loads(llm.prompts[1][1])["observation"]
    assert observation == {"type": "planner_error", "code": "invalid_json"}
    assert result.answer == "26" and llm.calls == 2
    assert result.tokens.total == sum(
        len(system.split()) + len(user.split()) + len(text.split())
        for (system, user), text in zip(llm.prompts, ["not-json", action])
    )


def test_planner_replans_only_for_a_question_supported_sport_refinement(mini_corpus_path):
    backend = LocalBackend.from_docs(_venue_docs(), embedder=FakeEmbedder(dim=8))
    initial = _action("resolve_multi_hop", {"venue": "Lake Venue", "date": "20 September 1988", "year": 1988})
    refined = _action(
        "resolve_multi_hop",
        {"venue": "Lake Venue", "date": "20 September 1988", "year": 1988, "sport": "Rowing"},
    )
    question = "Who won the gold medal in rowing at the event held at Lake Venue on 20 September 1988?"
    llm = ScriptedLLM([initial, refined])
    result = AgenticPipeline(backend, llm).answer(Question("planner-refine", question, "", None, ()))

    assert result.answer == "Rowing Winner"
    assert result.docs_retrieved == ["QROW"]
    assert llm.calls == 2
    observation = json.loads(llm.prompts[1][1])["observation"]
    assert [item["id"] for item in observation["candidates"]] == ["QROW", "QSAIL"]
    assert "Rowing Winner" not in llm.prompts[1][1]
    assert "gold" not in json.dumps(observation).lower()


def test_planner_preserves_unsupported_venue_tie_without_spending_second_call(mini_corpus_path):
    backend = LocalBackend.from_docs(_venue_docs(), embedder=FakeEmbedder(dim=8))
    question = "Who won the gold medal in the event held at Lake Venue on 20 September 1988?"
    action = _action("resolve_multi_hop", {"venue": "Lake Venue", "date": "20 September 1988", "year": 1988})
    llm = ScriptedLLM([action])
    result = AgenticPipeline(backend, llm).answer(Question("planner-tie", question, "", None, ()))

    assert result.answer is None and result.certificate.completeness_check == "unverified"
    assert result.certificate.docs_inspected == ["QROW", "QSAIL"]
    assert llm.calls == 1


def test_planner_does_not_repeat_a_tie_when_the_first_action_already_used_sport(mini_corpus_path):
    docs = _venue_docs() + [
        Doc(
            doc_id="QROW2",
            title="Rowing at the 1988 Summer Olympics â€“ Women's single sculls",
            url="",
            approx_tokens=10,
            text="[Infobox Olympic event]\n  event: Women's single sculls\n  games: 1988 Summer\n  venue: Lake Venue\n  date: 20 September 1988\n  gold: Another Rowing Winner\n",
        )
    ]
    backend = LocalBackend.from_docs(docs, embedder=FakeEmbedder(dim=8))
    action = _action(
        "resolve_multi_hop",
        {"venue": "Lake Venue", "date": "20 September 1988", "year": 1988, "sport": "Rowing"},
    )
    question = "Who won the gold medal in rowing at the event held at Lake Venue on 20 September 1988?"
    llm = ScriptedLLM([action])
    result = AgenticPipeline(backend, llm).answer(Question("planner-repeat-tie", question, "", None, ()))

    assert result.answer is None and result.certificate.completeness_check == "unverified"
    assert result.docs_retrieved == ["QROW", "QROW2"]
    assert llm.calls == 1


def test_planner_provider_failure_does_not_issue_graph_rag_request(mini_corpus_path):
    backend = LocalBackend.from_docs(load_docs(mini_corpus_path), embedder=FakeEmbedder(dim=8))
    llm = ScriptedLLM([RuntimeError("secret credential and provider details")])
    question = "Who starred in Forrest Gump?"
    result = AgenticPipeline(backend, llm).answer(Question("planner-provider-error", question, "", None, ()))

    assert result.answer is None and llm.calls == 1
    assert result.certificate.planner_status == "provider_error"
    assert "secret" not in str(result.trace).lower()
    assert "suppressed after planner provider failure" in str(result.trace)
