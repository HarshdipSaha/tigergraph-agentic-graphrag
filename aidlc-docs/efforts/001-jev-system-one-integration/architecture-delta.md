# Architecture Delta: 001-jev-system-one-integration

> **Effort:** 001-jev-system-one-integration  
> **Baseline:** `aidlc-docs/inception/02-architecture.md`  

---

## 1. Architectural Changes

### 1.1 Decision Model Protocol Layer
Added a new layer between routing and execution in [`agrag/decision.py`](file:///H:/augsepthacks/tigergraph-hack/agrag/decision.py):

```mermaid
flowchart TD
    subgraph Decision Layer
        DM[DecisionModel Protocol]
        JEV[JevDecisionModel: TypeSafe AI API]
        MOCK[MockDecisionModel: Offline Fixtures]
        DM --> JEV
        DM --> MOCK
    end
```

### 1.2 Pipeline Integration

```mermaid
flowchart LR
    Q[Question] --> R[Regex Router]
    R -->|Matched| T[Tools 1-5]
    R -->|No Match| J1[Jev System One Route]
    J1 -->|Classified| T
    J1 -->|Failed| FB[GraphRAG Fallback]

    T --> Cands{Multiple Candidates?}
    Cands -->|No| E[Evaluate & Certify]
    Cands -->|Yes| J2[Jev Typed Choice Disambiguate]
    J2 -->|Resolved| E
    J2 -->|Fallback| LLM[GroqLLM] --> E
```

### 1.3 State-Questions-Answers Interface
Jev accepts an unstructured `state` string, instructions, and a mapping of `criteria` dictionary keys to descriptions. It returns structured judgments and calibrated probability distributions without autoregressive text generation.
