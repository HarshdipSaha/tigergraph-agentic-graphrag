# Requirements Delta: 001-jev-system-one-integration

> **Effort:** 001-jev-system-one-integration  
> **Baseline:** `aidlc-docs/inception/01-requirements.md`  

---

## 1. Context

The Inception baseline established that the agentic pipeline routes queries by template matching against five known question shapes (`lookup`, `multi_hop`, `temporal`, `aggregation`, `superlative`). While highly accurate on conforming benchmark questions, this heuristic had two major deficiencies:
1. Questions phrased in natural language outside exact regex syntax defaulted to `unknown`, falling back to unverified vector RAG.
2. Multi-hop questions with multiple candidate records sharing a venue and date (e.g. `pub-060`) fell back to conversational LLM completion, which was prone to hallucination, wasted generation tokens, and lacked calibrated confidence.

---

## 2. Requirements Delta

### RQ-DELTA-01: Decision Model Abstraction
- The system MUST define a `DecisionModel` protocol supporting typed decision primitives:
  - `choice(state, instructions, criteria, question_key) -> Optional[DecisionResult]`
  - `noul(state, instructions, question_key) -> Optional[float]`
- Implementations MUST include `JevDecisionModel` (calling the TypeSafe AI System One API) and `MockDecisionModel` (deterministic offline execution for CI/testing).

### RQ-DELTA-02: Non-Template Intent Classification Fallback
- When `router.route()` fails to match a question to a template, the `AgenticPipeline._unrouted()` method MUST invoke the decision model to classify query intent into one of the five evidential classes.
- If classification succeeds, the certificate's `completeness_class` and `retrieval_mode` MUST be updated accordingly, preserving structural scan guarantees.

### RQ-DELTA-03: Zero-Token Candidate Disambiguation
- When graph traversal returns multiple candidate events (`len(candidates) > 1`) in `_disambiguate()`, the pipeline MUST query `DecisionModel.choice()` with candidate document IDs as discrete criteria.
- The pipeline MUST fall back to conversational LLM completion only if Jev is unconfigured or returns `None`.

### RQ-DELTA-04: Certificate Telemetry & Calibrated Confidence
- Decision model calls MUST emit step entries into `Certificate.steps`:
  - `tool="jev_route"` or `tool="jev_disambiguate"`
  - Model name, decision selected, confidence score, and token usage.

### RQ-DELTA-05: Hermetic Testability & Zero Regressions
- Unit test suite MUST run completely offline without requiring live Jev API keys using `MockDecisionModel`.
- All pre-existing 62 tests MUST continue to pass without regression.
