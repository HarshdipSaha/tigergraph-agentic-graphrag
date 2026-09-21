"""Agentic GraphRAG orchestrator.

Route -> dispatch class-appropriate tool -> evaluate evidence -> LLM only for recovery or
disambiguation -> emit an Investigation Certificate. See docs/idea-spec.md §4.
"""
from __future__ import annotations

import re
from typing import Optional

from agrag.backend import GraphBackend
from agrag.certificate import Certificate, Step, TokenUsage
from agrag.decision import DecisionModel, get_decision_model
from agrag.infobox import EventRecord
from agrag.llm import LLM
from agrag.pipelines.base import PipelineResult, Timer, clean_answer, event_facts
from agrag.pipelines.graphrag import GraphRagPipeline
from agrag.questions import Question
from agrag.router import CLASS_OF, ParsedQuestion, Template, route
from agrag.tools import (
    ToolResult,
    aggregation_scan,
    lookup_nations,
    resolve_multi_hop,
    superlative_scan,
    temporal_chain,
)

DISAMBIGUATE_SYSTEM = (
    "You are given a question and a numbered list of candidate Olympic events with their infobox facts. "
    "Reply with the doc id (e.g. Q123) of the single event that matches the question's venue and date. "
    "Reply with the doc id only."
)
EXTRACT_SYSTEM = (
    "Extract the total number of competitors in this Olympic event from the text. "
    "Reply with an integer only, or UNKNOWN if the text does not state it."
)
RETRIEVAL_MODE = {"lookup": "exact_match", "multi_hop": "venue_date_traversal", "temporal": "hop_chain",
                  "aggregation": "structural_scan", "superlative": "structural_scan"}


class AgenticPipeline:
    name = "agentic"

    def __init__(self, backend: GraphBackend, llm: LLM, decision_model: Optional[DecisionModel] = None):
        self.b, self.llm = backend, llm
        self.decision_model = decision_model
        self._fallback = GraphRagPipeline(backend, llm)

    # ---------- Decision & LLM helpers (where tokens / decisions are spent) ----------
    def _disambiguate(self, q: Question, cands: list[EventRecord]) -> tuple[Optional[EventRecord], Step]:
        # Fast path: Jev System One non-autoregressive decision model
        if self.decision_model and getattr(self.decision_model, "is_configured", False):
            criteria = {
                e.doc_id: f"{e.title} (Date: '{e.date_text}', Venue: '{e.venue}', Gold: {e.gold})"
                for e in cands[:10]
            }
            state = (
                f"Question: {q.question}\n\n"
                "Candidate Events:\n"
                + "\n".join(f"- {e.doc_id}: {e.title} | Date in infobox: '{e.date_text}' | Venue: '{e.venue}' | Gold: {e.gold}" for e in cands[:10])
            )
            instructions = (
                "Select the candidate Olympic event that most precisely and specifically matches "
                "the venue and date in the question. Prefer exact date string matches if there are ties."
            )
            dec = self.decision_model.choice(state, instructions, criteria, question_key="event_selection")
            if dec and dec.decision in criteria:
                chosen = next((e for e in cands if e.doc_id == dec.decision), None)
                if chosen:
                    return chosen, Step(
                        tool="jev_disambiguate",
                        note=f"Jev ({dec.model}) {len(cands)} candidates -> {chosen.doc_id} (conf={dec.confidence:.2f})",
                        tokens=TokenUsage(input=dec.input_tokens, output=dec.output_tokens),
                        latency_ms=dec.latency_ms,
                    )

        # Fallback to general generative LLM
        listing = "\n\n".join(f"{i+1}. doc {e.doc_id}\n{event_facts(e)}" for i, e in enumerate(cands))
        with Timer() as t:
            resp = self.llm.complete(DISAMBIGUATE_SYSTEM, f"Question: {q.question}\n\nCandidates:\n{listing}")
        m = re.search(r"Q\d+", resp.text)
        chosen = next((e for e in cands if m and e.doc_id == m.group(0)), None)
        return chosen, Step(tool="llm_disambiguate", note=f"{len(cands)} candidates -> {chosen.doc_id if chosen else 'none'}",
                            tokens=TokenUsage(input=resp.input_tokens, output=resp.output_tokens), latency_ms=t.ms)

    def _extract_competitors(self, doc_id: str) -> tuple[Optional[int], Step]:
        text = self.b.doc_text(doc_id)[:6000]
        with Timer() as t:
            resp = self.llm.complete(EXTRACT_SYSTEM, text)
        m = re.search(r"\d+", resp.text)
        n = int(m.group(0)) if m else None
        return n, Step(tool="llm_extract_competitors", note=f"{doc_id} -> {n}",
                       tokens=TokenUsage(input=resp.input_tokens, output=resp.output_tokens), latency_ms=t.ms)

    # ---------- orchestration ----------
    def answer(self, q: Question) -> PipelineResult:
        pq = route(q.question)
        if pq.template is None:
            return self._unrouted(q, pq)
        with Timer() as t:
            steps: list[Step] = []
            tokens = TokenUsage()
            answer: Optional[str]
            if pq.template == "lookup":
                r = lookup_nations(self.b, pq)
                answer, check = r.answer, ("pass" if r.answer else "fail")
                steps.append(Step(tool=r.tool, note="; ".join(r.notes) or f"bound={r.structural_bound}"))
            elif pq.template == "multi_hop":
                r = resolve_multi_hop(self.b, pq)
                steps.append(Step(tool=r.tool, note=f"{len(r.candidates)} candidates" + (" (relaxed venue match)" if r.fallback_used else "")))
                answer, check = r.answer, "pass"
                if r.needs_llm and r.candidates:
                    chosen, step = self._disambiguate(q, r.candidates)
                    steps.append(step)
                    tokens = tokens + step.tokens
                    answer = chosen.gold if chosen else None
                    check = "pass_with_llm_recovery" if chosen else "fail"
                    r.evidence = [chosen.doc_id] if chosen else r.evidence
                elif r.fallback_used and answer:
                    check = "pass_with_fallback"
                elif answer is None:
                    check = "fail"
            elif pq.template == "temporal":
                r = temporal_chain(self.b, pq)
                steps.append(Step(tool=r.tool, note=f"hops={r.hops}" + ("; fallback" if r.fallback_used else "")))
                answer = r.answer
                check = "fail" if answer is None else ("pass_with_fallback" if r.fallback_used else "pass")
            else:  # aggregation / superlative
                scan = aggregation_scan if pq.template == "aggregation" else superlative_scan
                r = scan(self.b, pq)   # regex recovery of missing competitor counts already applied inside
                steps.append(Step(tool=r.tool, note=f"bound={r.structural_bound}; regex-recovered={list(r.recovered)}"))
                check = "pass"
                if r.unresolved:   # only docs where neither infobox nor regex gave a count reach the LLM
                    for doc_id in list(r.unresolved):
                        n, step = self._extract_competitors(doc_id)
                        steps.append(step)
                        tokens = tokens + step.tokens
                        if n is not None:
                            r.recovered[doc_id] = n
                    r = self._rescan_with(r, pq)
                    check = "pass_with_llm_recovery" if not r.unresolved else "fail"
                answer = r.answer
            stop_reason = "structural_bound_met" if check.startswith("pass") else "evidence_incomplete"
        cert = Certificate(
            qid=q.qid, qtype=q.qtype or pq.template, completeness_class=pq.completeness_class,
            classification_confidence=pq.confidence, retrieval_mode=RETRIEVAL_MODE[pq.template], predicate=r.predicate,
            structural_bound=r.structural_bound, evidence_set_size=len(r.evidence), completeness_check=check,
            docs_inspected=list(r.evidence), steps=steps, tokens=tokens, latency_ms=t.ms, stop_reason=stop_reason,
        )
        return PipelineResult(qid=q.qid, pipeline=self.name, answer=answer, docs_retrieved=list(r.evidence),
                              tokens=tokens, latency_ms=t.ms, trace=[s.model_dump() for s in steps], certificate=cert)

    def _rescan_with(self, r: ToolResult, pq: ParsedQuestion) -> ToolResult:
        """Recompute the exhaustive answer using r.recovered (which now includes LLM-extracted values)."""
        recovered = dict(r.recovered)
        events = r.candidates
        def n_of(e: EventRecord) -> Optional[int]:
            return e.competitors if e.competitors is not None else recovered.get(e.doc_id)
        if pq.template == "aggregation":
            r.answer = str(sum(1 for e in events if (n_of(e) or -1) > pq.slots["threshold"]))
        else:
            best = max((e for e in events if n_of(e) is not None), key=lambda e: n_of(e), default=None)
            r.answer = best.title if best else None
        r.unresolved = [e.doc_id for e in events if n_of(e) is None]
        return r

    def _unrouted(self, q: Question, pq: ParsedQuestion) -> PipelineResult:
        jev_step: Optional[Step] = None
        detected_template = None
        if self.decision_model and getattr(self.decision_model, "is_configured", False):
            criteria = {
                "lookup": "Single fact lookup such as counting participating nations in an Olympic Games",
                "multi_hop": "Cross-reference who won an event held at a specific venue on a specific date",
                "temporal": "Event held immediately before or after in time",
                "aggregation": "Count or filter total events in a sport matching a threshold of competitors",
                "superlative": "Identify the event in a sport with the highest or lowest number of competitors",
            }
            dec = self.decision_model.choice(
                f"Question: {q.question}",
                "Classify this Olympic query into the single most appropriate evidential category",
                criteria,
                question_key="inquiry_class",
            )
            if dec and dec.decision in criteria:
                detected_template = dec.decision
                jev_step = Step(
                    tool="jev_route",
                    note=f"Jev System One ({dec.model}) classified intent -> {detected_template} (conf={dec.confidence:.2f})",
                    tokens=TokenUsage(input=dec.input_tokens, output=dec.output_tokens),
                    latency_ms=dec.latency_ms,
                )

        base = self._fallback.answer(q)
        steps = [Step(tool="graphrag_fallback", note="question did not match regex template", tokens=base.tokens)]
        if jev_step:
            steps.insert(0, jev_step)
        tokens = base.tokens + (jev_step.tokens if jev_step else TokenUsage())
        latency_ms = base.latency_ms + (jev_step.latency_ms if jev_step else 0)

        cert = Certificate(
            qid=q.qid,
            qtype=q.qtype or (detected_template or "unknown"),
            completeness_class=CLASS_OF.get(detected_template, "unknown") if detected_template else "unknown",
            classification_confidence="high" if (jev_step and "conf=1.00" in jev_step.note) else "low",
            retrieval_mode="graphrag_fallback" if not detected_template else f"jev_{detected_template}_fallback",
            predicate={},
            structural_bound=0,
            evidence_set_size=len(base.docs_retrieved),
            completeness_check="unverified",
            docs_inspected=base.docs_retrieved,
            steps=steps,
            tokens=tokens,
            latency_ms=latency_ms,
            stop_reason="fallback_single_pass",
        )
        return PipelineResult(
            qid=q.qid,
            pipeline=self.name,
            answer=base.answer,
            docs_retrieved=base.docs_retrieved,
            tokens=tokens,
            latency_ms=latency_ms,
            trace=[s.model_dump() for s in steps],
            certificate=cert,
        )
