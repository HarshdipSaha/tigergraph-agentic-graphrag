# Requirements Delta: 002-hidden-benchmark-refresh

> **Effort:** 002-hidden-benchmark-refresh  
> **Baseline:** `aidlc-docs/inception/01-requirements.md`  

---

## 1. Context & Architectural Inquiry

In Effort 001, TypeSafe AI's Jev System One decision model was merged into `main`. The existing hidden benchmark evaluation trace (`results/agentic_hidden.jsonl`, generated 2026-09-06) was evaluated on the **previous baseline architecture** (rigid regex router with Groq conversational LLM fallback).

### Architectural Analysis: "Do we need to re-run on the new architecture?"

**Yes, re-running is necessary and beneficial for the following reasons:**
1. **Empirical Grounding:** In AI-DLC and rigorous hackathon benchmarks, committing benchmark traces that do not reflect the code currently on `main` causes evaluation drift and audit discrepancies.
2. **Resolution of Disambiguation Collisions:** Under the previous architecture, multi-hop queries with candidate ties (such as `eval-001`: Olympic Tennis Centre on 15–22 August 2004 with 3 candidates) fell back to `llm_disambiguate`, resulting in `"answer": null`, 958 wasted tokens, and `"completeness_check": "fail"`.
3. **Verification of Jev Integration:** With Jev System One enabled, `eval-001` queries `JevDecisionModel.choice()`, identifying `Q735286` (`Sébastien VieilledentAdrien Hardy`), passing the certificate verification check (`pass_with_llm_recovery`), and reducing token consumption.

---

## 2. Requirements Delta

### RQ-DELTA-01: Hidden Evaluation Execution
- The agentic pipeline MUST be executed against all 50 questions in `data/eval_hidden.jsonl` using the live TigerGraph Savanna backend (`--backend tigergraph`).

### RQ-DELTA-02: Certificate Telemetry Verification
- The output traces in `results/agentic_hidden.jsonl` MUST record active `jev_disambiguate` or `jev_route` tool invocations where applicable.

### RQ-DELTA-03: Result File Integrity
- All 50 records in `results/agentic_hidden.jsonl` MUST be valid JSON lines containing question, pipeline result, certificate, and scoring structures without schema corruption.
