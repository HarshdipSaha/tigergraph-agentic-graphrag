"""Strict, question-grounded action planning for the five graph tools.

The planner chooses a tool and arguments; it never generates an answer. The graph tools,
answer selection, and certificate evaluation remain in the deterministic executor.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agrag.certificate import Step, TokenUsage
from agrag.normalize import normalize
from agrag.router import CLASS_OF, ParsedQuestion, Template, route
from agrag.tools import (
    aggregation_scan,
    lookup_nations,
    resolve_multi_hop,
    superlative_scan,
    temporal_chain,
)

ToolName = Literal[
    "lookup_nations",
    "resolve_multi_hop",
    "temporal_chain",
    "aggregation_scan",
    "superlative_scan",
]

TOOL_TO_TEMPLATE: dict[str, Template] = {
    "lookup_nations": "lookup",
    "resolve_multi_hop": "multi_hop",
    "temporal_chain": "temporal",
    "aggregation_scan": "aggregation",
    "superlative_scan": "superlative",
}
TOOL_FUNCTIONS = {
    "lookup_nations": lookup_nations,
    "resolve_multi_hop": resolve_multi_hop,
    "temporal_chain": temporal_chain,
    "aggregation_scan": aggregation_scan,
    "superlative_scan": superlative_scan,
}

REQUIRED: dict[str, dict[str, type]] = {
    "lookup_nations": {"title": str},
    "resolve_multi_hop": {"venue": str, "date": str, "year": int},
    "temporal_chain": {"event_phrase": str, "year": int, "season": str},
    "aggregation_scan": {"sport": str, "year": int, "season": str, "threshold": int},
    "superlative_scan": {"sport": str, "year": int, "season": str},
}
OPTIONAL: dict[str, dict[str, type]] = {
    "resolve_multi_hop": {"season": str, "sport": str},
}

MAX_ARGUMENT_TEXT = 240
MAX_QUESTION_LENGTH = 4_000
MAX_OBSERVATION_CANDIDATES = 100
PLANNER_TIMEOUT_SECONDS = 15
MAX_PLANNER_RESPONSES = 2

SYSTEM_PROMPT = """Choose one graph action for the supplied question. Return one JSON object only:
{"tool":"...","args":{...},"reason":"brief explanation"}. Never answer the question.
Use only these tools and exact argument names:
- lookup_nations(title: string)
- resolve_multi_hop(venue: string, date: string, year: integer; optional season: Summer|Winter, sport: string)
- temporal_chain(event_phrase: string, year: integer, season: Summer|Winter)
- aggregation_scan(sport: string, year: integer, season: Summer|Winter, threshold: integer)
- superlative_scan(sport: string, year: integer, season: Summer|Winter)
Ground every argument in the question or in an explicitly supplied candidate observation. Do not invent a sport.
If an observation supplies candidates, select a different grounded query only when it resolves an ambiguity.
If an observation has type planner_error, correct the rejected action and return one valid JSON object.
Copy venue/date wording from the question. Include season only when the question states Summer or Winter.
Examples:
Question: How many nations competed in Sailing at the 2016 Summer Olympics?
{"tool":"lookup_nations","args":{"title":"Sailing at the 2016 Summer Olympics"},"reason":"exact event title"}
Question: Who won the gold medal in the event held at Lake Venue on 20 September 1988?
{"tool":"resolve_multi_hop","args":{"venue":"Lake Venue","date":"20 September 1988","year":1988},"reason":"venue and date lookup"}
Question: Who won the gold medal in the men's walk event at the Summer Olympics held immediately before 2016?
{"tool":"temporal_chain","args":{"event_phrase":"men's walk","season":"Summer","year":2016},"reason":"previous Games event"}
Question: How many sailing events at the 2004 Summer Olympics had more than 20 competitors?
{"tool":"aggregation_scan","args":{"sport":"sailing","year":2004,"season":"Summer","threshold":20},"reason":"count matching events"}"""


class PlannerAction(BaseModel):
    """The only model output accepted by the action planner."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tool: ToolName
    args: dict[str, Any]
    reason: str = Field(default="", max_length=160)


PlannerStatus = Literal["selected", "rejected", "failed", "limited"]


@dataclass(frozen=True)
class PlannerResult:
    action: Optional[PlannerAction]
    parsed_question: Optional[ParsedQuestion]
    status: PlannerStatus
    error_code: Optional[str]
    tokens: TokenUsage
    step: Step
    response_number: int
    failure_detail: Optional[str] = None


def _exact_text_match(actual: Any, expected: Any) -> bool:
    return type(actual) is str and type(expected) is str and normalize(actual) == normalize(expected)


def validate_action(action: PlannerAction, question: str | None = None) -> PlannerAction:
    """Validate exact argument names, runtime types, and conservative value bounds."""
    if not isinstance(action, PlannerAction):
        raise TypeError("action must be a PlannerAction")
    if question is not None and type(question) is not str:
        raise TypeError("question must be a string")

    required = REQUIRED[action.tool]
    optional = OPTIONAL.get(action.tool, {})
    allowed = required | optional
    keys = set(action.args)
    if keys - set(allowed):
        raise ValueError("unknown tool argument")
    if set(required) - keys:
        raise ValueError("missing required tool argument")

    for name, value in action.args.items():
        expected_type = allowed[name]
        # bool is a subclass of int in Python; exact type equality intentionally rejects it.
        if type(value) is not expected_type:
            raise ValueError(f"invalid type for {name}")
        if expected_type is str and (not value.strip() or len(value) > MAX_ARGUMENT_TEXT):
            raise ValueError(f"invalid length for {name}")

    if "season" in action.args and action.args["season"] not in {"Summer", "Winter"}:
        raise ValueError("invalid season")
    if "year" in action.args and not 1890 <= action.args["year"] <= 2100:
        raise ValueError("invalid year")
    if "threshold" in action.args and not 0 <= action.args["threshold"] <= 100_000:
        raise ValueError("invalid threshold")
    return action


def _contains_literal(question: str, literal: str) -> bool:
    """True when normalized text contains a literal phrase with token boundaries."""
    text, phrase = normalize(question), normalize(literal)
    if not phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _remove_literal_once(text: str, literal: str) -> str:
    phrase = normalize(literal)
    if not phrase:
        return text
    return re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", text, count=1)


def _outside_venue_date(question: str, venue: str, date: str) -> str:
    remaining = normalize(question)
    # Remove the longer span first so an overlapping date/venue phrase cannot obscure the other.
    for phrase in sorted((venue, date), key=lambda value: len(normalize(value)), reverse=True):
        remaining = _remove_literal_once(remaining, phrase)
    return remaining


def validate_grounding(action: PlannerAction, question: str) -> None:
    """Check that the selected tool and every argument are supported by the question.

    The benchmark regex is consulted only after an action exists, as a consistency guard.
    Non-template questions use literal spans and numeric tokens; this deliberately rejects
    paraphrased slot values that cannot be tied back to the user's wording.
    """
    if type(question) is not str:
        raise TypeError("question must be a string")
    parsed = route(question)
    selected_template = TOOL_TO_TEMPLATE[action.tool]

    if parsed.template is not None:
        if selected_template != parsed.template:
            raise ValueError("selected tool contradicts question intent")
        for key, expected in parsed.slots.items():
            actual = action.args.get(key)
            if expected is None:
                if actual is not None:
                    raise ValueError(f"ungrounded {key}")
            elif type(expected) is int:
                if type(actual) is not int or actual != expected:
                    raise ValueError(f"wrong {key}")
            elif not _exact_text_match(actual, expected):
                raise ValueError(f"wrong {key}")
    else:
        for key, value in action.args.items():
            if key == "sport" and action.tool == "resolve_multi_hop":
                continue
            if type(value) is str:
                grounded = _contains_literal(question, value)
            elif type(value) is int:
                grounded = re.search(rf"(?<!\d){value}(?!\d)", question) is not None
            else:
                grounded = False
            if not grounded:
                raise ValueError(f"ungrounded {key}")

    if action.tool == "resolve_multi_hop" and "sport" in action.args:
        outside = _outside_venue_date(question, action.args["venue"], action.args["date"])
        if not _contains_literal(outside, action.args["sport"]):
            raise ValueError("ungrounded venue-question sport")


def action_to_parsed_question(action: PlannerAction, question: str) -> ParsedQuestion:
    """Convert a validated model action into the existing deterministic tool input type."""
    validate_action(action, question)
    validate_grounding(action, question)
    template = TOOL_TO_TEMPLATE[action.tool]
    slots = dict(action.args)
    if action.tool == "resolve_multi_hop":
        # resolve_multi_hop expects this key even if the question supplied only a dated venue.
        slots.setdefault("season", None)
    confidence = "high" if route(question).template is not None else "low"
    return ParsedQuestion(
        question=question.strip(),
        template=template,
        completeness_class=CLASS_OF[template],
        slots=slots,
        confidence=confidence,
    )


def build_user_prompt(question: str, observation: Mapping[str, Any] | None = None) -> str:
    """Construct the user message from a question string and a sanitized observation only."""
    if type(question) is not str:
        raise TypeError("planner accepts a question string only")
    if not question.strip() or len(question) > MAX_QUESTION_LENGTH:
        raise ValueError("invalid question length")
    payload: dict[str, Any] = {"question": question}
    if observation is not None:
        payload["observation"] = _sanitize_observation(observation)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _sanitize_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only candidate facts the planner needs; unknown keys (including answer/gold) vanish."""
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    if observation.get("type") == "planner_error":
        code = observation.get("code")
        allowed_errors = {"invalid_json", "ungrounded_action", "invalid_action"}
        if code not in allowed_errors:
            raise ValueError("invalid planner error observation")
        out = {"type": "planner_error", "code": code}
        detail = observation.get("detail")
        if detail is not None:
            allowed_details = {
                "selected tool contradicts question intent",
                "wrong title", "wrong venue", "wrong date", "wrong year", "wrong season", "wrong event_phrase",
                "wrong sport", "wrong threshold", "ungrounded title", "ungrounded venue",
                "ungrounded date", "ungrounded year", "ungrounded season", "ungrounded sport",
                "ungrounded event_phrase",
                "ungrounded threshold", "ungrounded venue-question sport",
            }
            if detail not in allowed_details:
                raise ValueError("invalid planner error detail")
            out["detail"] = detail
        return out
    last_tool = observation.get("last_tool")
    status = observation.get("status")
    if last_tool not in TOOL_TO_TEMPLATE:
        raise ValueError("invalid observation tool")
    allowed_status = {"ambiguous_candidates", "unresolved_fields", "no_candidates", "incomplete"}
    if status not in allowed_status:
        raise ValueError("invalid observation status")
    raw_candidates = observation.get("candidates", [])
    if type(raw_candidates) is not list or len(raw_candidates) > MAX_OBSERVATION_CANDIDATES:
        raise ValueError("invalid observation candidates")
    candidate_count = observation.get("candidate_count", len(raw_candidates))
    if type(candidate_count) is not int or candidate_count < len(raw_candidates):
        raise ValueError("invalid observation count")

    candidates: list[dict[str, str]] = []
    allowed_candidate_fields = ("id", "sport", "date", "venue")
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("invalid candidate observation")
        clean: dict[str, str] = {}
        for field_name in allowed_candidate_fields:
            value = candidate.get(field_name)
            if value is not None:
                if type(value) is not str or len(value) > MAX_ARGUMENT_TEXT:
                    raise ValueError("invalid candidate field")
                clean[field_name] = value
        if not clean.get("id"):
            raise ValueError("candidate id is required")
        candidates.append(clean)

    return {
        "last_tool": last_tool,
        "status": status,
        "candidate_count": candidate_count,
        "candidates": candidates,
    }


def has_supported_sport_refinement(
    question: str, venue: str, date: str, candidates: list[Any]
) -> bool:
    """Whether the question explicitly names a sport represented in the eligible tie.

    The venue and date are removed before checking so a venue such as "Olympic Tennis
    Centre" cannot accidentally act as a sport cue. This predicate gates the second
    planner call; candidate facts alone never authorize narrowing the requested query.
    """
    if type(question) is not str or not candidates:
        return False
    outside = _outside_venue_date(question, venue, date)
    return any(
        type(getattr(candidate, "sport", None)) is str
        and _contains_literal(outside, candidate.sport)
        for candidate in candidates
    )


def build_candidate_observation(result: Any, status: str | None = None) -> dict[str, Any]:
    """Build a safe second-turn observation from a graph ToolResult.

    Only doc id, sport, date, and venue are read from each event. Gold medals, answers,
    competitor data, and document text are never inspected or serialized.
    """
    if not hasattr(result, "tool") or not hasattr(result, "candidates"):
        raise TypeError("expected a graph tool result")
    candidates = []
    for event in result.candidates:
        candidates.append(
            {
                "id": event.doc_id,
                "sport": event.sport,
                "date": event.date_text,
                "venue": event.venue,
            }
        )
    if status is None:
        if getattr(result, "unresolved", None):
            status = "unresolved_fields"
        elif getattr(result, "needs_llm", False) and candidates:
            status = "ambiguous_candidates"
        elif not candidates:
            status = "no_candidates"
        else:
            status = "incomplete"
    observation = {
        "last_tool": result.tool,
        "status": status,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    return _sanitize_observation(observation)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


class Planner:
    """Factory for isolated, bounded per-question planning sessions."""

    def __init__(self, llm: Any, *, timeout_s: int = PLANNER_TIMEOUT_SECONDS):
        if type(timeout_s) is not int or timeout_s <= 0 or timeout_s > PLANNER_TIMEOUT_SECONDS:
            raise ValueError("planner timeout must be in (0, 15]")
        self.llm = llm
        self.timeout_s = timeout_s

    def start(self, question: str) -> "PlannerSession":
        return PlannerSession(self, question)


class PlannerSession:
    """A single question with at most one evidence observation and one replan."""

    def __init__(self, planner: Planner, question: str):
        # Reject Question objects and all metadata-bearing wrappers at the boundary.
        if type(question) is not str:
            raise TypeError("planner accepts a question string only")
        if not question.strip() or len(question) > MAX_QUESTION_LENGTH:
            raise ValueError("invalid question length")
        self.planner = planner
        self.question = question
        self.responses = 0
        self.total_tokens = TokenUsage()
        self._seen_actions: set[str] = set()

    def select(self, observation: Mapping[str, Any] | None = None) -> PlannerResult:
        if self.responses >= MAX_PLANNER_RESPONSES:
            return self._local_result("limited", "response_budget_exhausted", self.responses + 1)
        if self.responses == 0 and observation is not None:
            return self._local_result("rejected", "unexpected_observation", self.responses + 1)
        if self.responses > 0 and observation is None:
            return self._local_result("rejected", "observation_required", self.responses + 1)

        # Count the attempt before calling the provider: failures still consume the bounded
        # response budget, so a retry cannot turn provider failure into unlimited calls.
        self.responses += 1
        response_number = self.responses
        try:
            user_prompt = build_user_prompt(self.question, observation)
        except (TypeError, ValueError):
            return self._local_result("rejected", "invalid_observation", response_number)

        started = time.monotonic()
        try:
            response = self.planner.llm.complete(
                SYSTEM_PROMPT,
                user_prompt,
                timeout_s=self.planner.timeout_s,
            )
        except Exception as exc:
            # Provider messages can contain request details, URLs, or credentials. Keep only
            # a stable category in the trace and result.
            elapsed_ms = int((time.monotonic() - started) * 1000)
            is_timeout = isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
            return self._failure(
                "planner_timeout" if is_timeout else "provider_error",
                response_number,
                elapsed_ms,
            )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        usage = TokenUsage(
            input=max(0, int(getattr(response, "input_tokens", 0))),
            output=max(0, int(getattr(response, "output_tokens", 0))),
        )
        self.total_tokens = self.total_tokens + usage

        try:
            parsed_json = json.loads(response.text, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, TypeError, ValueError):
            return self._failure("invalid_json", response_number, elapsed_ms, usage)
        try:
            if type(parsed_json) is not dict:
                raise ValueError("response must be a JSON object")
            action = PlannerAction.model_validate(parsed_json)
            validate_action(action, self.question)
        except (ValidationError, TypeError, ValueError, KeyError):
            return self._failure("invalid_action", response_number, elapsed_ms, usage)

        try:
            parsed_question = action_to_parsed_question(action, self.question)
        except (TypeError, ValueError, KeyError) as exc:
            detail = str(exc) if type(exc) is ValueError else None
            return self._failure("ungrounded_action", response_number, elapsed_ms, usage, detail)

        signature = json.dumps(
            {"tool": action.tool, "args": action.args},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if signature in self._seen_actions:
            return self._failure("repeated_action", response_number, elapsed_ms, usage)
        self._seen_actions.add(signature)

        rendered_args = json.dumps(action.args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        note = f"selected {action.tool} args={rendered_args}; question grounding passed"
        step = Step(tool="planner_select", note=note, tokens=usage, latency_ms=elapsed_ms)
        return PlannerResult(
            action=action,
            parsed_question=parsed_question,
            status="selected",
            error_code=None,
            tokens=usage,
            step=step,
            response_number=response_number,
        )

    def _local_result(self, status: PlannerStatus, code: str, response_number: int) -> PlannerResult:
        step = Step(tool=self._step_tool(code), note=code, tokens=TokenUsage(), latency_ms=0)
        return PlannerResult(
            action=None,
            parsed_question=None,
            status=status,
            error_code=code,
            tokens=TokenUsage(),
            step=step,
            response_number=response_number,
        )

    def _failure(
        self,
        code: str,
        response_number: int,
        elapsed_ms: int,
        usage: TokenUsage | None = None,
        detail: str | None = None,
    ) -> PlannerResult:
        usage = usage or TokenUsage()
        step = Step(
            tool=self._step_tool(code),
            note=f"{code}: {detail}" if detail else code,
            tokens=usage,
            latency_ms=elapsed_ms,
        )
        return PlannerResult(
            action=None,
            parsed_question=None,
            status="failed" if code in {"planner_timeout", "provider_error"} else "rejected",
            error_code=code,
            tokens=usage,
            step=step,
            response_number=response_number,
            failure_detail=detail,
        )

    @staticmethod
    def _step_tool(code: str) -> str:
        if code in {"invalid_json", "invalid_action", "ungrounded_action", "repeated_action"}:
            return "planner_action_rejected"
        if code == "planner_timeout":
            return "planner_timeout"
        if code == "provider_error":
            return "planner_provider_error"
        return f"planner_{code}"
