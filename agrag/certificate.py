from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

CompletenessCheck = Literal["pass", "pass_with_llm_recovery", "pass_with_fallback", "fail", "unverified"]


class TokenUsage(BaseModel):
    """input/output are what the API billed (from response.usage). context is an estimate (chars/4) of the
    retrieved-context share of the prompt, reported separately because the organisers ask for it."""

    input: int = 0
    output: int = 0
    context: int = 0

    @computed_field  # type: ignore[misc]
    @property
    def total(self) -> int:
        return self.input + self.output

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(input=self.input + other.input, output=self.output + other.output,
                          context=self.context + other.context)


class Step(BaseModel):
    tool: str
    note: str = ""
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0


class Certificate(BaseModel):
    """Per-answer, machine-checkable record of what the agent did and whether its evidence is complete."""

    qid: str
    qtype: str
    completeness_class: Literal["existential", "chained", "exhaustive", "unknown"]
    classification_confidence: Literal["high", "low"]
    retrieval_mode: str
    predicate: dict[str, Any]
    structural_bound: int
    evidence_set_size: int
    completeness_check: CompletenessCheck
    docs_inspected: list[str]
    steps: list[Step]
    tokens: TokenUsage
    latency_ms: int
    stop_reason: str
    planning_mode: Literal["template", "planner"] = "template"
    selected_tool: str | None = None
    planner_status: str = "not_used"
    planner_grounding: str = "not_applicable"
    selected_doc_id: str | None = None
    venue_resolution: dict[str, Any] = Field(default_factory=dict)
