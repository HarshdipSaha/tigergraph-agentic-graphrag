# Validation Report: 001-jev-system-one-integration

> **Effort:** 001-jev-system-one-integration  
> **Date:** 2026-09-21  

---

## 1. Unit & Integration Test Suite

- **Offline Unit Tests:** 65 passed in 2.97s (`python -m pytest --ignore=tests/integration`)
  - New test file: [`tests/test_decision_jev.py`](file:///H:/augsepthacks/tigergraph-hack/tests/test_decision_jev.py) (4 tests: Mock decision model, unconfigured fallback, agentic pipeline disambiguation, unrouted intent classification).
- **TigerGraph Savanna Integration Tests:** 4 passed in 3.51s (`python -m pytest tests/integration/test_tigergraph.py`).
- **Total Passing Tests:** 69 / 69 (100%).

---

## 2. Jev Benchmark (`scripts/benchmark_jev.py`)

Live evaluation against TypeSafe AI System One endpoint (`jev-latest`):

### Experiment 1: Non-Template Natural Language Routing
| Query | Regex Router | Jev Route | Confidence |
|---|---|---|---|
| *How many countries participated in the 1996 Summer Olympics?* | None (0%) | `lookup` | 1.00 |
| *Tell me who took the gold medal at ExCeL on 30 July 2012* | None (0%) | `multi_hop` | 0.98 |
| *Which athlete won the 100m sprint right before the 2012 Games?* | None (0%) | `temporal` | 0.95 |
| *Find the number of wrestling events in 2004 with more than 20 competitors* | None (0%) | `aggregation` | 1.00 |
| *What was the largest event by participant count in 2008?* | None (0%) | `superlative` | 0.99 |

**Accuracy:** Regex = 0/5 (0%), Jev System One = 5/5 (100%).

### Experiment 2: Disambiguation on `pub-060`
Evaluated 3-way date/venue collision at ExCeL on 30 July 2012:
- Jev Calibrated Probabilities: `{'Q1156695': 0.14, 'Q1005551': 0.24, 'Q1064016': 0.62}`
- Latency: 2,285 ms, zero output tokens generated.
