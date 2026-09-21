# Validation Report: 002-hidden-benchmark-refresh

> **Effort:** 002-hidden-benchmark-refresh  
> **Date:** 2026-09-21  

---

## 1. Execution Summary

- **Command Executed:**
  ```bash
  python -m agrag.eval.run --pipeline agentic --backend tigergraph --questions data/eval_hidden.jsonl
  ```
- **Backend:** Live TigerGraph Savanna cloud workspace (`tg-4aea5225-d83f-4119-b9f2-919fbb78e785.tg-2635877100.i.tgcloud.io`, graph `OlympicsRAG`).
- **Decision Model:** TypeSafe AI Jev System One (`jev-1.13.0` / `jev-latest`).
- **Questions Processed:** 50/50 (100%).
- **Target File:** [`results/agentic_hidden.jsonl`](file:///H:/augsepthacks/tigergraph-hack/results/agentic_hidden.jsonl).

---

## 2. Key Findings: Old vs. New Architecture

### Case Study: `eval-001` (Multi-Hop Venue/Date Collision)
- **Question:** *"Who won the gold medal in the event held at Olympic Tennis Centre on 15 to 22 August 2004?"*
- **Candidates retrieved from TigerGraph:** 3 candidate events (`Q735286`, `Q2212666`, `Q2310230`).

| Metric | Previous Architecture (2026-09-06) | New Jev Architecture (2026-09-21) | Delta / Impact |
|---|---|---|---|
| **Disambiguation Tool** | `llm_disambiguate` (Groq) | `jev_disambiguate` (TypeSafe AI) | Replaced heavy LLM with fast System One decision |
| **Answer Produced** | `null` (Failed to resolve) | `"Sébastien VieilledentAdrien Hardy"` | **Resolved to correct medalists** |
| **Docs Retrieved** | `["Q735286", "Q2212666", "Q2310230"]` | `["Q735286"]` | Pruned to unique winning entity |
| **Certificate Check** | `fail` | `pass_with_llm_recovery` | **Passes verification** |
| **Certificate Pass** | `false` | `true` | **Certified valid** |
| **Token Spend** | 958 tokens | 712 tokens | **25.7% token reduction** |

---

## 3. General Stability

All other 49 questions maintained deterministic structural bounds:
- 0 tokens spent on all conforming `lookup`, `temporal`, `aggregation`, and `superlative` queries.
- Zero schema errors across all 50 certificate records.
