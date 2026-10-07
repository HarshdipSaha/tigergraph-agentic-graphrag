import json

import pytest
from pydantic import ValidationError

from agrag.certificate import TokenUsage
from agrag.infobox import EventRecord
from agrag.llm import LLMResponse
from agrag.planner import (
    Planner,
    PlannerAction,
    action_to_parsed_question,
    build_candidate_observation,
    build_user_prompt,
    validate_action,
    validate_grounding,
)
from agrag.questions import Question


class FakeLLM:
    """Small offline LLM stub that records planner calls and enforces the timeout API."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.prompts = []
        self.timeouts = []

    def complete(self, system, user, *, timeout_s=None):
        self.calls += 1
        self.prompts.append((system, user))
        self.timeouts.append(timeout_s)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, LLMResponse):
            return answer
        return LLMResponse(
            text=answer,
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(answer.split()),
        )


def encoded(tool, args, reason="grounded action"):
    return json.dumps({"tool": tool, "args": args, "reason": reason})


@pytest.mark.parametrize(
    ("question", "tool", "args", "template", "completeness"),
    [
        (
            "How many nations competed in Sailing at the 2016 Summer Olympics?",
            "lookup_nations",
            {"title": "Sailing at the 2016 Summer Olympics"},
            "lookup",
            "existential",
        ),
        (
            "Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988 at the 1988 Summer Olympics?",
            "resolve_multi_hop",
            {
                "venue": "Olympic Weightlifting Gymnasium",
                "date": "20 September 1988",
                "year": 1988,
                "season": "Summer",
            },
            "multi_hop",
            "existential",
        ),
        (
            "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?",
            "temporal_chain",
            {"event_phrase": "men's 20 kilometres walk athletics", "season": "Summer", "year": 2016},
            "temporal",
            "chained",
        ),
        (
            "According to the provided corpus, how many cross-country skiing events at the 2010 Winter Olympics had more than 62 competitors?",
            "aggregation_scan",
            {"sport": "cross-country skiing", "year": 2010, "season": "Winter", "threshold": 62},
            "aggregation",
            "exhaustive",
        ),
        (
            "According to the provided corpus, which sailing event at the 2004 Summer Olympics had the highest number of competitors?",
            "superlative_scan",
            {"sport": "sailing", "year": 2004, "season": "Summer"},
            "superlative",
            "exhaustive",
        ),
    ],
)
def test_each_graph_tool_action_maps_to_parsed_question(
    question, tool, args, template, completeness
):
    llm = FakeLLM([encoded(tool, args)])
    result = Planner(llm).start(question).select()

    assert result.status == "selected"
    assert result.action.tool == tool
    assert result.parsed_question.template == template
    assert result.parsed_question.completeness_class == completeness
    assert result.parsed_question.slots == args | ({"season": None} if tool == "resolve_multi_hop" and "season" not in args else {})
    assert f"{tool} args=" in result.step.note
    assert "question grounding passed" in result.step.note
    assert llm.calls == 1
    assert llm.timeouts == [15]
    assert isinstance(result.tokens, TokenUsage)
    assert result.tokens.input > 0 and result.tokens.output > 0


@pytest.mark.parametrize(
    ("text", "error_code"),
    [
        ("not json", "invalid_json"),
        (encoded("not_a_tool", {}), "invalid_action"),
        (encoded("aggregation_scan", {"sport": "sailing", "year": True, "season": "Summer", "threshold": 2}), "invalid_action"),
        (encoded("aggregation_scan", {"sport": "sailing", "year": 2004, "season": "Summer"}), "invalid_action"),
        (encoded("lookup_nations", {"title": "Sailing at the 2016 Summer Olympics", "surprise": 1}), "invalid_action"),
        (encoded("superlative_scan", {"sport": "sailing", "year": 2004, "season": "Autumn"}), "invalid_action"),
        (encoded("aggregation_scan", {"sport": "sailing", "year": 2004, "season": "Summer", "threshold": 100001}), "invalid_action"),
        (encoded("lookup_nations", {"title": "x" * 241}), "invalid_action"),
        (encoded("lookup_nations", {"title": "Sailing at the 2016 Summer Olympics"}, reason="r" * 161), "invalid_action"),
    ],
)
def test_malformed_or_invalid_actions_are_rejected_and_usage_is_kept(text, error_code):
    question = "How many nations competed in Sailing at the 2016 Summer Olympics?"
    result = Planner(FakeLLM([text])).start(question).select()

    assert result.action is None
    assert result.status == "rejected"
    assert result.error_code == error_code
    assert result.tokens.input > 0
    assert result.tokens.output > 0
    assert result.step.note == error_code


def test_strict_action_model_forbids_extra_top_level_fields_and_non_string_reason():
    with pytest.raises(ValidationError):
        PlannerAction.model_validate(
            {"tool": "lookup_nations", "args": {"title": "x"}, "extra": "forbidden"}
        )
    with pytest.raises(ValidationError):
        PlannerAction.model_validate(
            {"tool": "lookup_nations", "args": {"title": "x"}, "reason": 12}
        )


@pytest.mark.parametrize(
    ("action", "question"),
    [
        (
            {"tool": "aggregation_scan", "args": {"sport": "rowing", "year": 1988, "season": "Summer", "threshold": 1}},
            "Who won the gold medal in the event held at Lake Venue on 20 September 1988?",
        ),
        (
            {"tool": "lookup_nations", "args": {"title": "Swimming at the 2016 Summer Olympics"}},
            "How many nations competed in Sailing at the 2016 Summer Olympics?",
        ),
        (
            {"tool": "resolve_multi_hop", "args": {"venue": "Lake Venue", "date": "21 September 1988", "year": 1988}},
            "Who won the gold medal in the event held at Lake Venue on 20 September 1988?",
        ),
        (
            {"tool": "resolve_multi_hop", "args": {"venue": "Lake Venue", "date": "20 September 1988", "year": 1984}},
            "Who won the gold medal in the event held at Lake Venue on 20 September 1988?",
        ),
        (
            {"tool": "aggregation_scan", "args": {"sport": "sailing", "year": 2004, "season": "Summer", "threshold": 63}},
            "According to the provided corpus, how many sailing events at the 2004 Summer Olympics had more than 62 competitors?",
        ),
    ],
)
def test_semantic_or_slot_mismatches_are_rejected(action, question):
    with pytest.raises(ValueError):
        validate_grounding(validate_action(PlannerAction.model_validate(action)), question)


def test_venue_sport_requires_literal_grounding_outside_venue_and_date():
    question = "Who won the gold medal in the event held at Lake Rowing Center on 20 September 1988?"
    action = PlannerAction.model_validate(
        {
            "tool": "resolve_multi_hop",
            "args": {"venue": "Lake Rowing Center", "date": "20 September 1988", "year": 1988, "sport": "rowing"},
        }
    )
    with pytest.raises(ValueError, match="grounded"):
        validate_grounding(action, question)

    explicit_question = "Who won the gold medal for rowing at Lake Venue on 20 September 1988?"
    grounded = PlannerAction.model_validate(
        {
            "tool": "resolve_multi_hop",
            "args": {"venue": "Lake Venue", "date": "20 September 1988", "year": 1988, "sport": "rowing"},
        }
    )
    validate_grounding(grounded, explicit_question)
    parsed = action_to_parsed_question(grounded, explicit_question)
    assert parsed.slots["sport"] == "rowing"


@pytest.mark.parametrize(
    ("tool", "args", "question"),
    [
        (
            "aggregation_scan",
            {"sport": "archery", "year": 2004, "season": "Summer", "threshold": 20},
            "Count rowing events in Summer 2004 with more than 20 competitors.",
        ),
        (
            "superlative_scan",
            {"sport": "fencing", "year": 2012, "season": "Summer"},
            "Which rowing event at Summer 2012 had the most competitors?",
        ),
    ],
)
def test_non_template_exhaustive_actions_need_literal_sport_grounding(tool, args, question):
    with pytest.raises(ValueError):
        validate_grounding(PlannerAction.model_validate({"tool": tool, "args": args}), question)


def test_question_prompt_contains_only_question_and_never_question_metadata():
    secret_answer = "secret-gold-answer"
    question = Question(
        qid="private-qid",
        question="How many nations competed in Sailing at the 2016 Summer Olympics?",
        qtype="private-qtype",
        answer=secret_answer,
        gold_doc_ids=("Q_PRIVATE_999",),
    )
    llm = FakeLLM([encoded("lookup_nations", {"title": "Sailing at the 2016 Summer Olympics"})])
    result = Planner(llm).start(question.question).select()
    system, user = llm.prompts[0]

    assert result.status == "selected"
    for secret in (question.qid, question.qtype, secret_answer, *question.gold_doc_ids):
        assert secret not in system + user
    assert question.question in user
    with pytest.raises(TypeError):
        Planner(llm).start(question)


def test_non_template_action_is_accepted_only_when_arguments_are_literal_spans():
    question = "Count archery events in Summer 2004 with more than 20 competitors."
    good = PlannerAction.model_validate(
        {"tool": "aggregation_scan", "args": {"sport": "archery", "year": 2004, "season": "Summer", "threshold": 20}}
    )
    validate_grounding(good, question)
    parsed = action_to_parsed_question(good, question)
    assert parsed.template == "aggregation"
    assert parsed.completeness_class == "exhaustive"
    assert parsed.confidence == "low"

    bad = PlannerAction.model_validate(
        {"tool": "aggregation_scan", "args": {"sport": "rowing", "year": 2004, "season": "Summer", "threshold": 20}}
    )
    with pytest.raises(ValueError):
        validate_grounding(bad, question)


def test_candidate_observation_uses_allowlist_and_omits_gold_and_answer():
    event = EventRecord(
        doc_id="Q123",
        title="Rowing at the 2012 Summer Olympics – Men's single sculls",
        sport="Rowing",
        year=2012,
        season="Summer",
        event_name="Men's single sculls",
        venue="Lake Venue",
        date_text="30 July 2012",
        competitors=40,
        competitors_raw="40",
        nations=25,
        gold="SECRET_GOLD",
        gold_noc="SECRET_ANSWER",
        prev_year=None,
        next_year=None,
        url="https://private.example/doc",
    )
    result = type(
        "ToolResultStub",
        (),
        {"tool": "resolve_multi_hop", "candidates": [event], "needs_llm": True, "unresolved": []},
    )()
    observation = build_candidate_observation(result)
    encoded_observation = json.dumps(observation)

    assert observation["status"] == "ambiguous_candidates"
    assert observation["candidates"][0]["id"] == "Q123"
    assert "SECRET_GOLD" not in encoded_observation
    assert "SECRET_ANSWER" not in encoded_observation
    assert "gold" not in encoded_observation.lower()
    assert "answer" not in encoded_observation.lower()
    assert "url" not in encoded_observation.lower()


def test_planner_error_observation_accepts_only_stable_grounding_details():
    prompt = build_user_prompt(
        "Who won the event?",
        {"type": "planner_error", "code": "ungrounded_action", "detail": "wrong event_phrase"},
    )
    assert json.loads(prompt)["observation"] == {
        "type": "planner_error", "code": "ungrounded_action", "detail": "wrong event_phrase"
    }
    with pytest.raises(ValueError, match="detail"):
        build_user_prompt(
            "Who won the event?",
            {"type": "planner_error", "code": "ungrounded_action", "detail": "provider key leaked"},
        )


def test_optional_observation_replan_is_bounded_and_repeated_action_rejected():
    question = "Who won the gold medal in the event held at Lake Venue on 20 September 1988?"
    action = encoded(
        "resolve_multi_hop",
        {"venue": "Lake Venue", "date": "20 September 1988", "year": 1988},
    )
    llm = FakeLLM([action, action])
    session = Planner(llm).start(question)
    first = session.select()
    second = session.select(
        observation={
            "last_tool": "resolve_multi_hop",
            "status": "ambiguous_candidates",
            "candidate_count": 1,
            "candidates": [{"id": "Q123", "sport": "Rowing", "date": "20 September 1988"}],
            "gold": "must be stripped",
            "answer": "must be stripped",
        }
    )
    third = session.select(observation={"last_tool": "resolve_multi_hop", "status": "incomplete"})

    assert first.status == "selected"
    assert second.status == "rejected"
    assert second.error_code == "repeated_action"
    assert second.tokens.output > 0
    assert third.error_code == "response_budget_exhausted"
    assert session.responses == 2
    assert session.total_tokens.input == first.tokens.input + second.tokens.input
    assert llm.calls == 2
    observation_text = json.dumps(json.loads(llm.prompts[1][1])["observation"]).lower()
    assert "must be stripped" not in observation_text
    assert "gold" not in observation_text
    assert "answer" not in observation_text


@pytest.mark.parametrize(
    ("exc", "error_code"),
    [
        (TimeoutError("secret provider detail"), "planner_timeout"),
        (RuntimeError("secret api key detail"), "provider_error"),
    ],
)
def test_timeout_and_provider_errors_are_bounded_and_do_not_echo_exception(exc, error_code):
    llm = FakeLLM([exc])
    result = Planner(llm).start("What happened at Lake Venue?").select()

    assert result.status == "failed"
    assert result.error_code == error_code
    assert exc.args[0] not in result.step.note
    assert result.tokens == TokenUsage()
    assert llm.timeouts == [15]
