"""Agentic GraphRAG orchestrator.

Route -> dispatch class-appropriate tool -> evaluate evidence -> LLM only for recovery or
disambiguation -> emit an Investigation Certificate. See docs/idea-spec.md §4.
"""
from __future__ import annotations

import re
import json
from typing import Optional

from agrag.backend import GraphBackend
from agrag.certificate import Certificate, Step, TokenUsage
from agrag.decision import DecisionModel, get_decision_model
from agrag.infobox import EventRecord
from agrag.llm import LLM
from agrag.pipelines.base import PipelineResult, Timer
from agrag.pipelines.graphrag import GraphRagPipeline
from agrag.planner import (
    Planner,
    TOOL_FUNCTIONS,
    TOOL_TO_TEMPLATE,
    build_candidate_observation,
    has_supported_sport_refinement,
)
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

    def __init__(self, backend: GraphBackend, llm: LLM, decision_model: Optional[DecisionModel] = None,
                 agent_mode: str = "planner"):
        if agent_mode not in {"planner", "template"}:
            raise ValueError("agent_mode must be 'planner' or 'template'")
        self.b, self.llm = backend, llm
        self.decision_model = decision_model
        self.agent_mode = agent_mode
        self.planner = Planner(llm)
        self._fallback = GraphRagPipeline(backend, llm)

    # ---------- Decision & LLM helpers (where tokens / decisions are spent) ----------
    def _disambiguate(self, q: Question, cands: list[EventRecord]) -> tuple[Optional[EventRecord], Step]:
        # Fast path: Jev System One non-autoregressive decision model
        if self.decision_model and getattr(self.decision_model, "is_configured", False):
            criteria = {
                e.doc_id: f"{e.title} (Sport: '{e.sport}', Date: '{e.date_text}', Venue: '{e.venue}')"
                for e in cands
            }
            state = (
                f"Question: {q.question}\n\n"
                "Candidate Events:\n"
                + "\n".join(f"- {e.doc_id}: {e.title} | Sport: '{e.sport}' | Date: '{e.date_text}' | Venue: '{e.venue}'" for e in cands)
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

        # Use one decision source per question: if Jev is configured, do not spend a
        # second decision request after it declines to choose.
        if self.decision_model and getattr(self.decision_model, "is_configured", False):
            return None, Step(tool="jev_disambiguate", note=f"{len(cands)} candidates -> none")

        # Use the generative LLM only when Jev is not configured.
        listing = "\n\n".join(
            f"{i+1}. doc {e.doc_id}\nTitle: {e.title}\nSport: {e.sport}\nDate: {e.date_text}\nVenue: {e.venue}"
            for i, e in enumerate(cands)
        )
        with Timer() as t:
            resp = self.llm.complete(
                DISAMBIGUATE_SYSTEM, f"Question: {q.question}\n\nCandidates:\n{listing}", timeout_s=15.0
            )
        m = re.search(r"Q\d+", resp.text)
        chosen = next((e for e in cands if m and e.doc_id == m.group(0)), None)
        return chosen, Step(tool="llm_disambiguate", note=f"{len(cands)} candidates -> {chosen.doc_id if chosen else 'none'}",
                            tokens=TokenUsage(input=resp.input_tokens, output=resp.output_tokens), latency_ms=t.ms)

    def _extract_competitors(self, doc_id: str) -> tuple[Optional[int], Step]:
        text = self.b.doc_text(doc_id)[:6000]
        with Timer() as t:
            try:
                resp = self.llm.complete(EXTRACT_SYSTEM, text, timeout_s=15.0)
            except Exception as error:
                return None, Step(
                    tool="llm_extract_competitors",
                    note=f"{doc_id} extraction unavailable ({type(error).__name__})",
                    latency_ms=t.ms,
                )
        m = re.search(r"\d+", resp.text)
        n = int(m.group(0)) if m else None
        return n, Step(tool="llm_extract_competitors", note=f"{doc_id} -> {n}",
                       tokens=TokenUsage(input=resp.input_tokens, output=resp.output_tokens), latency_ms=t.ms)

    # ---------- orchestration ----------
    def answer(self, q: Question) -> PipelineResult:
        if self.agent_mode == "planner":
            return self._answer_planner(q)
        return self._answer_template(q)

    def _answer_template(self, q: Question) -> PipelineResult:
        pq = route(q.question)
        if pq.template is None:
            return self._unrouted(q, pq)
        with Timer() as t:
            steps: list[Step] = []
            tokens = TokenUsage()
            answer: Optional[str]
            selected_doc_id: Optional[str] = None
            if pq.template == "lookup":
                r = lookup_nations(self.b, pq)
                answer, check = r.answer, ("pass" if r.answer else "fail")
                steps.append(Step(tool=r.tool, note="; ".join(r.notes) or f"bound={r.structural_bound}"))
            elif pq.template == "multi_hop":
                r = resolve_multi_hop(self.b, pq)
                steps.append(Step(tool=r.tool, note=f"{len(r.candidates)} candidates" + (" (relaxed venue match)" if r.fallback_used else "")))
                answer, check = r.answer, "pass" if r.answer is not None else "fail"
                selected_doc_id = r.evidence[0] if r.answer is not None and len(r.evidence) == 1 else None
                if r.needs_llm and r.candidates:
                    chosen, step = self._disambiguate(q, r.candidates)
                    steps.append(step)
                    tokens = tokens + step.tokens
                    answer = chosen.gold if chosen else None
                    selected_doc_id = chosen.doc_id if chosen else None
                    r.selection_basis = "model_choice" if chosen else "none"
                    check = "unverified"
                elif answer is not None and r.match_mode == "venue_alias":
                    check = "pass_with_fallback"
                elif answer is None and getattr(r, "ambiguity_reason", ""):
                    check = "unverified"
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
            planning_mode="template", selected_tool=r.tool, planner_status="not_used",
            planner_grounding="not_applicable", selected_doc_id=selected_doc_id if pq.template == "multi_hop" else None,
            venue_resolution={
                "raw_candidate_count": getattr(r, "raw_candidate_count", 0),
                "eligible_count": getattr(r, "eligible_count", 0),
                "match_mode": getattr(r, "match_mode", "none"),
                "selection_basis": getattr(r, "selection_basis", "none"),
                "ambiguity_reason": getattr(r, "ambiguity_reason", ""),
            } if pq.template == "multi_hop" else {},
        )
        return PipelineResult(qid=q.qid, pipeline=self.name, answer=answer, docs_retrieved=list(r.evidence),
                              tokens=tokens, latency_ms=t.ms, trace=[s.model_dump() for s in steps], certificate=cert)

    def _answer_planner(self, q: Question) -> PipelineResult:
        """Ask the LLM to choose a graph tool, then execute only the validated action."""
        with Timer() as timer:
            session = self.planner.start(q.question)
            steps: list[Step] = []
            attempted = 0
            executed: set[str] = set()
            planner_status = "not_started"
            planner_grounding = "unverified"
            result = None
            parsed: Optional[ParsedQuestion] = None
            tool_result: Optional[ToolResult] = None
            selected_tool: Optional[str] = None

            while attempted < 2:
                observation = None
                if attempted == 1 and result is not None and result.parsed_question is None:
                    if result.error_code not in {"invalid_json", "ungrounded_action", "invalid_action"}:
                        break
                    observation = {
                        "type": "planner_error",
                        "code": result.error_code,
                        "detail": result.failure_detail,
                    }
                elif attempted == 1 and tool_result is not None:
                    if (
                        tool_result.tool != "resolve_multi_hop"
                        or parsed is None
                        or parsed.slots.get("sport") is not None
                        or not has_supported_sport_refinement(
                            q.question,
                            parsed.slots.get("venue", ""),
                            parsed.slots.get("date", ""),
                            tool_result.candidates,
                        )
                    ):
                        break
                    observation = build_candidate_observation(tool_result, status="ambiguous_candidates")

                result = session.select(observation=observation)
                attempted += 1
                planner_status = result.error_code or result.status
                if result.step is not None:
                    steps.append(result.step)
                if result.parsed_question is None:
                    continue

                parsed = result.parsed_question
                action = result.action
                signature = json.dumps({"tool": action.tool, "args": action.args}, sort_keys=True, ensure_ascii=False)
                if signature in executed:
                    planner_status = "repeated_action"
                    steps.append(Step(tool="planner_action_rejected", note="same tool and arguments already executed"))
                    break
                executed.add(signature)
                selected_tool = action.tool
                planner_grounding = "verified"
                tool_fn = TOOL_FUNCTIONS[action.tool]
                tool_result = tool_fn(self.b, parsed)
                steps.append(Step(
                    tool=action.tool,
                    note=f"answer={tool_result.answer!r}; evidence={tool_result.evidence}; bound={tool_result.structural_bound}",
                ))

                # Replan only when the original question explicitly names a sport among
                # the eligible candidates. Candidate facts by themselves cannot narrow it.
                if (
                    attempted == 1
                    and tool_result.tool == "resolve_multi_hop"
                    and parsed.slots.get("sport") is None
                    and has_supported_sport_refinement(
                        q.question,
                        parsed.slots.get("venue", ""),
                        parsed.slots.get("date", ""),
                        tool_result.candidates,
                    )
                ):
                    continue
                break

            total_tokens = session.total_tokens
            fallback_used = parsed is None or tool_result is None
            selected_doc_id: Optional[str] = None
            if fallback_used:
                # A recognized benchmark shape may use its deterministic tool as a visible fallback.
                # This route runs only after the LLM had a chance to select its action.
                fallback_pq = route(q.question)
                if fallback_pq.template is not None:
                    selected_tool = next(
                        name for name, template in TOOL_TO_TEMPLATE.items() if template == fallback_pq.template
                    )
                    tool_result = TOOL_FUNCTIONS[selected_tool](self.b, fallback_pq)
                    parsed = fallback_pq
                    planner_grounding = "template_fallback"
                    planner_status = planner_status or "planner_failed"
                    steps.append(Step(tool="planner_template_fallback", note=f"{planner_status} -> {selected_tool}"))
                else:
                    provider_failed = planner_status in {"planner_timeout", "provider_error"}
                    return self._planner_unrouted(
                        q, steps, total_tokens, timer.ms, planner_status,
                        allow_graphrag=not provider_failed,
                    )

            assert parsed is not None and tool_result is not None
            if parsed.template == "lookup":
                answer = tool_result.answer
                check = "pass" if answer is not None else "fail"
            elif parsed.template == "multi_hop":
                answer = tool_result.answer
                selected_doc_id = tool_result.evidence[0] if answer is not None and len(tool_result.evidence) == 1 else None
                if tool_result.match_mode == "venue_alias" and answer is not None:
                    check = "pass_with_fallback"
                elif answer is not None:
                    check = "pass"
                elif tool_result.eligible_count > 1 or tool_result.ambiguity_reason:
                    check = "unverified"
                else:
                    check = "fail"
            elif parsed.template == "temporal":
                answer = tool_result.answer
                check = "fail" if answer is None else ("pass_with_fallback" if tool_result.fallback_used else "pass")
            else:
                used_llm_recovery = False
                if tool_result.unresolved:
                    for doc_id in list(tool_result.unresolved):
                        count, step = self._extract_competitors(doc_id)
                        steps.append(step)
                        total_tokens = total_tokens + step.tokens
                        used_llm_recovery = used_llm_recovery or step.tokens.total > 0
                        if count is not None:
                            tool_result.recovered[doc_id] = count
                    tool_result = self._rescan_with(tool_result, parsed)
                answer = tool_result.answer
                check = "fail" if tool_result.unresolved else (
                    "pass_with_llm_recovery" if used_llm_recovery else "pass"
                )

            if fallback_used and check == "pass":
                check = "pass_with_fallback"
            stop_reason = "planner_action_complete" if check.startswith("pass") else (
                "ambiguous_candidates" if check == "unverified" else "evidence_incomplete"
            )
            venue_resolution = {
                "raw_candidate_count": tool_result.raw_candidate_count,
                "eligible_count": tool_result.eligible_count,
                "match_mode": tool_result.match_mode,
                "selection_basis": tool_result.selection_basis,
                "ambiguity_reason": tool_result.ambiguity_reason,
            } if parsed.template == "multi_hop" else {}
            cert = Certificate(
                qid=q.qid, qtype=parsed.template, completeness_class=parsed.completeness_class,
                classification_confidence=parsed.confidence,
                retrieval_mode=RETRIEVAL_MODE[parsed.template], predicate=tool_result.predicate,
                structural_bound=tool_result.structural_bound, evidence_set_size=len(tool_result.evidence),
                completeness_check=check, docs_inspected=list(tool_result.evidence), steps=steps,
                tokens=total_tokens, latency_ms=timer.ms, stop_reason=stop_reason,
                planning_mode="planner", selected_tool=selected_tool, planner_status=planner_status,
                planner_grounding=planner_grounding, selected_doc_id=selected_doc_id,
                venue_resolution=venue_resolution,
            )
            return PipelineResult(
                qid=q.qid, pipeline=self.name, answer=answer, docs_retrieved=list(tool_result.evidence),
                tokens=total_tokens, latency_ms=timer.ms, trace=[step.model_dump() for step in steps], certificate=cert,
            )

    def _planner_unrouted(self, q: Question, planner_steps: list[Step], planner_tokens: TokenUsage,
                          planner_latency_ms: int, planner_status: str,
                          allow_graphrag: bool = True) -> PipelineResult:
        """Use GraphRAG after a non-provider planner failure; avoid calls after provider errors."""
        with Timer() as timer:
            if allow_graphrag:
                try:
                    base = self._fallback.answer(q)
                    fallback_step = Step(tool="graphrag_fallback", note="planner produced no grounded graph action",
                                         tokens=base.tokens, latency_ms=base.latency_ms)
                except Exception as error:
                    base = None
                    fallback_step = Step(tool="graphrag_fallback", note=f"unavailable ({type(error).__name__})")
            else:
                base = None
                fallback_step = Step(tool="graphrag_fallback", note="suppressed after planner provider failure")
        tokens = planner_tokens + (base.tokens if base is not None else TokenUsage())
        docs = base.docs_retrieved if base is not None else []
        steps = [*planner_steps, fallback_step]
        cert = Certificate(
            qid=q.qid, qtype=q.qtype or "unknown", completeness_class="unknown", classification_confidence="low",
            retrieval_mode="graphrag_fallback" if allow_graphrag else "planner_unavailable",
            predicate={}, structural_bound=0, evidence_set_size=len(docs),
            completeness_check="unverified", docs_inspected=docs, steps=steps, tokens=tokens,
            latency_ms=planner_latency_ms + timer.ms, stop_reason="planner_failed_unrouted_fallback",
            planning_mode="planner", planner_status=planner_status, planner_grounding="unverified",
        )
        return PipelineResult(
            qid=q.qid, pipeline=self.name, answer=base.answer if base is not None else None, docs_retrieved=docs,
            tokens=tokens, latency_ms=planner_latency_ms + timer.ms,
            trace=[step.model_dump() for step in steps], certificate=cert,
        )

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
