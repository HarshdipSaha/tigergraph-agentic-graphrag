# Jev (TypeSafe AI) — Technical Notes & Integration Guide

> **Last verified:** 21 September 2026  
> **Purpose:** Practical reference for evaluating and integrating **Jev**, TypeSafe AI's first **System One** model, especially for the TigerGraph Agentic GraphRAG hackathon.

---

## 1. What is Jev?

**Jev is a decision model from TypeSafe AI designed to make structured judgments inside software rather than generate conversational text.**

The core interaction is:

```text
application state + typed questions
            ↓
           Jev
            ↓
typed decisions + probabilities/confidence
```

TypeSafe describes Jev as a "frontier-intelligence function call": the application supplies unstructured state and Jev returns typed probabilistic decisions. Unlike a normal LLM, Jev is not intended to generate free-form prose.  
Source: TypeSafe AI, *Introducing System One Models & Jev*  
https://typesafe.ai/blog/introducing-system-one-models-and-jev

The official TypeSafe site currently describes Jev as a System One model for automation, with typed decisions and calibrated probabilities that software can use to decide when to act autonomously or request review.  
Source: TypeSafe AI  
https://typesafe.ai/

---

## 2. Why the "System One" name?

TypeSafe's terminology is inspired by the distinction between fast, intuitive judgments and slower, deliberate reasoning.

For software engineering, the important distinction is not the psychology analogy. It is the **interface**:

- A normal LLM is optimized to generate text.
- Jev is optimized to answer **bounded decisions** that code can consume directly.
- Your program owns the workflow, thresholds, arithmetic, branching, retries, tool calls, and side effects.

This makes Jev a better fit for small decisions such as:

```text
Which route should this request take?
Should this item be escalated?
Is this evidence sufficient to continue?
Which candidate should be selected?
How urgent is this?
```

Rather than:

```text
Write the final answer.
Perform a long chain-of-thought investigation.
Generate a report.
```

Source:
https://vercel.com/i/what-is-jev

---

# 3. Jev's basic interface

The general pattern is:

```text
STATE
  +
QUESTIONS
  ↓
JEV
  ↓
ANSWERS
```

The state contains whatever information is relevant to the decision.

The questions define the **bounded judgment** the model needs to make.

The application then interprets the answer.

Example conceptually:

```python
state = """
Question:
Who originally won the men's 56 kg weightlifting event
at the 1988 Olympics?

Evidence found:
- Infobox says Oksen Mirzoyan
- Prose says Mitko Grablev originally won
- Prose mentions disqualification
"""

question = {
    "type": "choice",
    "instructions": "Which investigation action should be taken next?",
    "choices": {
        "inspect_claim_history": "...",
        "search_documents": "...",
        "graph_traversal": "...",
        "stop": "..."
    }
}
```

Jev returns a typed decision that your program can branch on.

---

# 4. The three Jev decision primitives

Jev's interface is centered around three kinds of judgments:

## 4.1 Choice

Use **Choice** when the application must select one option from a bounded set.

Typical examples:

```text
Which tool should run next?
Which question type is this?
Which candidate entity is the best match?
Which workflow branch should execute?
```

Conceptually:

```text
choice:
    graph_traversal
    vector_search
    document_search
    claim_history
    stop
```

The response can include the selected option together with probabilities/confidence.

**Best fit:** routing, classification, tool selection, intent decisions.

---

## 4.2 Score

Use **Score** when you need an ordered rating rather than a single category.

Typical examples:

```text
How risky is this?
How relevant is this evidence?
How strong is this candidate match?
How urgent is this?
```

A score is useful when code wants to compare or threshold a graded judgment.

**Best fit:** prioritization, ranking-related decisions, quality/risk/severity judgments.

---

## 4.3 Noul

Use **Noul** for a yes/no judgment represented probabilistically.

Typical examples:

```text
Does the evidence support this claim?
Should the agent continue investigating?
Is there a contradiction?
Should this case be escalated?
Is this answer ready to return?
```

Conceptually:

```text
YES = 0.91
NO  = 0.09
```

Your code then decides what to do with that probability:

```python
if probability >= 0.90:
    take_action()
else:
    use_fallback()
```

The important point is that **the threshold belongs to your application**, not to Jev.

---

# 5. Jev vs. a normal LLM

| Property | Normal LLM | Jev |
|---|---|---|
| Main output | Generated text | Typed decisions |
| Typical use | Reasoning, explanation, generation | Routing, gating, scoring, verification |
| Interface | Prompt → text | State + questions → typed answers |
| Application parsing | Usually required | Much less parsing because answer type is declared |
| Free-form generation | Yes | No |
| Probabilities for bounded decisions | Not normally the primary interface | Central to the interface |
| Workflow control | LLM proposes; code interprets | Code owns control flow |
| Best role | System 2 / deep reasoning / answer generation | System 1 / low-level decisions |

This does **not** mean Jev replaces LLMs.

The intended architecture is often:

```text
Jev
 ↓
decide which expensive operation to run
 ↓
LLM
 ↓
perform deep reasoning / generation
```

TypeSafe's launch materials explicitly position Jev as something that can work alongside existing models rather than as a universal replacement for generative LLMs.

Sources:

- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://vercel.com/i/what-is-jev

---

# 6. Why Jev can be useful for agents

Agent systems repeatedly make small decisions:

```text
Which tool?
Which sub-agent?
Continue or stop?
Retry?
Ask the user?
Escalate?
Which candidate?
Which route?
```

A conventional approach is:

```text
LLM
 ↓
generate tool-call text
 ↓
parse it
 ↓
validate it
 ↓
execute
```

A Jev-oriented approach is closer to:

```text
application state
 ↓
Jev
 ↓
typed decision
 ↓
code
 ↓
execute
```

That can reduce the amount of text-generation overhead for decisions that do not actually require free-form prose.

Vercel's Jev announcement specifically lists:

- choosing the next tool or subagent
- deciding whether to continue, retry, ask the user, or stop

as example use cases.

Source:
https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway

---

# 7. Jev is NOT a reasoning loop

A common integration mistake would be:

```text
Question
 ↓
Jev
 ↓
Jev thinks
 ↓
Jev thinks again
 ↓
Jev writes answer
```

That is not the intended programming model.

A better model is:

```text
State
 ↓
Jev decides ONE bounded property
 ↓
Your code executes an action
 ↓
State changes
 ↓
Jev decides another bounded property
```

Therefore:

```text
Jev = decision primitive
Code = control-flow engine
LLM = deep reasoning/generation when needed
```

---

# 8. Jev in the TigerGraph Agentic GraphRAG architecture

## 8.1 Existing architecture

The current system already has:

```text
User Question
     ↓
router.py
     ↓
ParsedQuestion
     ↓
agentic.py
     ↓
tools.py
     ↓
TigerGraph
     ↓
Evidence
     ↓
Investigation Certificate
     ↓
Final LLM Answer
```

The project intentionally keeps deterministic retrieval/reasoning in `tools.py`, keeps the graph implementation behind the `GraphBackend` abstraction, and keeps the benchmark router deliberately simple.

**Do not replace this architecture.**

---

## 8.2 Recommended Jev placement

Add Jev as a **thin decision sidecar** in the agentic pipeline:

```text
                         USER QUESTION
                              │
                              ▼
                     EXISTING ROUTER
                              │
                              ▼
                      ParsedQuestion
                              │
                              ▼
                       AGENTIC ENGINE
                              │
                 ┌────────────┴────────────┐
                 │                         │
             confident                 uncertain
                 │                         │
                 │                         ▼
                 │                       JEV
                 │                         │
                 └────────────┬────────────┘
                              ▼
                        DETERMINISTIC
                            TOOLS
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
            GSQL/Graph    Vector Search   Claim History
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                         TigerGraph
                              │
                              ▼
                           Evidence
                              │
                              ▼
                   Investigation Certificate
                              │
                              ▼
                         Final LLM
```

---

# 9. The most useful Jev integration points for this project

## 9.1 Uncertain routing

Keep the existing regex/template router first.

Only invoke Jev when the deterministic router cannot confidently classify the question.

```text
regex router
   │
   ├── matched → existing path
   │
   └── unmatched → Jev
```

Example:

```text
Question:
"Which athlete was the gold medallist before the ruling changed
the result?"

Existing router:
unknown

Jev:
history / original / current / as_of
```

This preserves the existing high-performing path.

---

## 9.2 Tool selection

Suppose the agent has:

```text
Known entity: Q2071129
Evidence found:
- current gold
- prose mentions multiple disqualifications
- 2024 reallocation mentioned
```

Available tools:

```text
graph_traversal
vector_search
claim_history
document_search
stop
```

Jev can answer:

```text
claim_history
```

Your code then invokes the deterministic claim-history tool.

---

## 9.3 Continue / stop gating

Jev can answer:

```text
Does the current state contain enough evidence to proceed?
```

Your code applies the threshold.

Important:

**For objectively enumerable questions, TigerGraph should remain the authority for completeness.**

For example:

```text
structural_bound = 41
evidence_set_size = 41
```

The certificate can deterministically say:

```text
completeness_check = pass
```

Jev should not replace this structural proof.

---

## 9.4 Evidence relevance / prioritization

Suppose retrieval produces 20 candidate documents.

Jev can help identify which candidates deserve deeper investigation:

```text
relevant to current claim?
high / medium / low
```

or score each candidate.

The application can then keep the strongest subset for expensive LLM reasoning.

This should be benchmarked carefully rather than assumed to improve accuracy.

---

# 10. Jev + Investigation Certificates

Your Investigation Certificate is fundamentally a **deterministic evidence/completeness artifact**.

That should remain the source of truth.

A useful separation is:

```text
Jev
 ↓
"What should the agent do next?"
```

versus

```text
TigerGraph / deterministic resolver
 ↓
"Did we actually retrieve everything required?"
```

Therefore a certificate might conceptually contain:

```json
{
  "completeness_class": "exhaustive",
  "structural_bound": 41,
  "evidence_set_size": 41,
  "completeness_check": "pass",

  "decision_trace": {
    "provider": "jev",
    "decision": "claim_history",
    "confidence": 0.94
  }
}
```

The important distinction is:

```text
Jev decision
    ≠
proof of correctness
```

Jev helped the agent navigate.

The graph and deterministic resolver established the evidence condition.

---

# 11. Jev for Round 2 "Reasoning Over Time"

Round 2 introduces temporal modes such as:

```text
original
as_of
current
history
```

Your existing Round 2 design already has a temporal resolver that can enumerate claims, apply validity windows, rank authority, and expose uncertainty without using an LLM in the resolver.

A very clean Jev insertion is:

```text
Question
   ↓
deterministic regex parser
   ↓
matched?
   │
   ├── yes → existing path
   │
   └── no → Jev Choice
                │
                ├── original
                ├── as_of
                ├── current
                └── history
                         ↓
                 deterministic
                 temporal resolver
```

This keeps the resolver LLM-free while using Jev only where a bounded semantic decision is useful.

---

# 12. Example: Jev selecting a retrieval strategy

State:

```text
Question:
"Who originally won the men's 56 kg event at the 1988 Olympics?"

Entity:
Q25239321

Known evidence:
- infobox gold = Oksen Mirzoyan
- prose mentions Mitko Grablev originally won
- prose mentions disqualification
- ruling year unknown

Available actions:
1. inspect_claim_history
2. vector_search
3. graph_traversal
4. final_answer
```

Choice question:

```text
Which action should the agent execute next?

criteria:
- inspect_claim_history:
  Use when the evidence indicates multiple versions of one fact.
- vector_search:
  Use when the relevant event/entity is not established.
- graph_traversal:
  Use when a known entity requires structural traversal.
- final_answer:
  Use only when sufficient evidence is already assembled.
```

Expected architecture:

```text
Jev → inspect_claim_history
       ↓
Claim resolver
       ↓
All claims enumerated
       ↓
Supersession chain
       ↓
Certificate
       ↓
Final LLM answer
```

---

# 13. Example: Jev as a stop/continue gate

State:

```text
question = ...
evidence = ...
missing_fields = ...
previous_actions = ...
```

Noul:

```text
Is another retrieval/investigation step required before answering?
```

Example output:

```text
YES = 0.88
NO  = 0.12
```

Application policy:

```python
if yes_probability >= 0.85:
    investigate_more()
else:
    attempt_answer()
```

The threshold should be chosen from validation data.

Do not choose a threshold because it "looks good" in the demo.

---

# 14. Shadow-mode integration — safest way to test it

Because the current system already performs well, the lowest-risk experiment is:

```text
Current orchestrator
      │
      ├──────────────► actual action
      │
      └──────────────► Jev action
                              │
                              ▼
                         comparison log
```

Record:

```json
{
  "existing_action": "claim_history",
  "jev_action": "claim_history",
  "agreement": true,
  "jev_confidence": 0.94,
  "latency_ms": 120
}
```

Then benchmark Jev against the current routing policy.

Only allow Jev to control the live path if it demonstrates useful agreement/performance.

---

# 15. Recommended code architecture

Add one abstraction:

```text
agrag/
├── router.py
├── tools.py
├── certificate.py
├── llm.py
├── decision.py          # new
│
└── pipelines/
    └── agentic.py
```

Conceptual interface:

```python
from typing import Protocol


class DecisionModel(Protocol):
    def choice(self, state: str, question: dict):
        ...

    def score(self, state: str, question: dict):
        ...

    def noul(self, state: str, question: dict):
        ...
```

Implementations:

```text
DecisionModel
├── NoOpDecisionModel
├── FakeDecisionModel
└── JevDecisionModel
```

This keeps the rest of your application independent of the provider.

---

# 16. Suggested environment variables

Keep credentials outside source control.

Example:

```text
JEV_API_KEY=
JEV_MODEL=jev-latest
JEV_ENABLED=false
JEV_SHADOW_MODE=true
JEV_TIMEOUT_MS=...
```

Recommended progression:

```text
Stage 1:
JEV_ENABLED=false

Stage 2:
JEV_SHADOW_MODE=true

Stage 3:
benchmark results

Stage 4:
enable Jev only on selected branches
```

---

# 17. API/access model

Jev is a hosted/API-based service rather than a local model with downloadable weights.

A current TypeSafe-oriented developer guide says:

- an API key is required
- there is no local downloadable model
- Jev is exposed through an API
- output is typed and probabilities/confidence are returned

Source:
https://jevtypesafe.org/docs/jev-sdk/

**Important:** `jevtypesafe.org` is not the primary TypeSafe AI corporate domain. Treat that site as a developer/community reference and verify the current endpoint, fields, and access process against TypeSafe's official documentation before production use.

Official site:
https://typesafe.ai/

---

# 18. Vercel AI Gateway

Jev is currently exposed through **Vercel AI Gateway**.

Vercel's Jev model page shows the model identifier:

```text
typesafe-ai/jev
```

and demonstrates it through AI SDK's `experimental_evaluate` API.

Source:
https://vercel.com/ai-gateway/models/jev

Example pattern shown by Vercel:

```typescript
import { experimental_evaluate as evaluate } from "ai";

const result = await evaluate({
  model: "typesafe-ai/jev",
  state: "...",
  questions: {
    refunded: {
      type: "boolean",
      instructions: "Was a refund issued?"
    }
  }
});
```

The exact API shape should be checked against the current provider documentation before coding.

---

# 19. Pricing and performance claims

The numbers around Jev have changed as the service has launched and different gateways/pages currently surface slightly different pricing figures.

Current surfaced sources include:

- TypeSafe's own site with a very low per-token price
- Vercel AI Gateway with a Jev-specific price page
- TypeSafe's launch post with benchmark claims
- Vercel's launch announcement with benchmark claims

**Do not hard-code a price in the hackathon write-up until you verify it on the day you submit.**

Sources:

TypeSafe:
https://typesafe.ai/

Vercel model page:
https://vercel.com/ai-gateway/models/jev

TypeSafe launch post:
https://typesafe.ai/blog/introducing-system-one-models-and-jev

Vercel launch announcement:
https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway

TypeSafe currently markets Jev as substantially faster/cheaper for the decision workloads it targets, but those are vendor-reported benchmark claims and should be presented as such unless independently reproduced.

---

# 20. Current availability

TypeSafe announced Jev publicly in **September 2026** and describes it as an **early-access** product.

Source:
https://typesafe.ai/blog/introducing-system-one-models-and-jev

This matters for hackathon planning:

```text
Do not make Jev a hard dependency.
```

Use a fallback.

Recommended:

```text
Jev available
    ↓
use Jev

Jev unavailable
    ↓
existing LLM/deterministic path
```

---

# 21. Official TypeSafe Agent Skill

TypeSafe maintains an official GitHub repository containing agent skills for building with its System One API.

Repository:

https://github.com/typesafe-ai/skills

The repository is described as:

> Agent skills for building with TypeSafe: typed decisions and probabilities from System One models.

It is useful for:

- designing Jev questions
- understanding TypeSafe workflows
- discovering current documentation/patterns
- composing typed judgments

Source:
https://github.com/typesafe-ai/skills

---

# 22. Useful official / first-party links

## TypeSafe AI

https://typesafe.ai/

Main company/product site.

## Jev launch announcement

https://typesafe.ai/blog/introducing-system-one-models-and-jev

Best source for:
- what Jev is
- System One framing
- product motivation
- TypeSafe's reported benchmark claims

## TypeSafe Agent Skills

https://github.com/typesafe-ai/skills

Official code/skills repository for building with System One.

## Vercel Jev model page

https://vercel.com/ai-gateway/models/jev

Useful for:
- current gateway integration
- model identifier
- AI SDK example
- current surfaced pricing

## Vercel Jev announcement

https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway

Useful for:
- workflow examples
- routing/agent-loop use cases
- launch context

## Vercel explainer

https://vercel.com/i/what-is-jev

Useful for:
- high-level explanation
- bounded decision model
- application-controlled routing

---

# 23. Documentation links

The current TypeSafe documentation ecosystem is referenced by the official TypeSafe skills repository and community tooling.

Useful starting points:

https://docs.typesafe.ai/

Potentially useful documentation paths include:

```text
https://docs.typesafe.ai/primitives
https://docs.typesafe.ai/primitives/score
https://docs.typesafe.ai/confidence
https://docs.typesafe.ai/patterns
https://docs.typesafe.ai/patterns/intent-routing
https://docs.typesafe.ai/model-jaggedness/jev-1.13
```

The exact contents/API should be treated as live documentation because Jev is newly launched.

---

# 24. Community / third-party resources

These are **not official TypeSafe sources**, but can be useful for experimentation.

## Community Jev notes

https://github.com/codaaiteam/jev-typesafe-ai

Unofficial examples and notes covering:
- API concepts
- Choice / Score / Noul
- examples
- links to official resources

## Community Jev quickstart

https://github.com/codaaiteam/jev-ai

Unofficial quickstart and FAQ.

## Community Jev CLI / skill

https://github.com/okooo5km/jev

A third-party CLI/agent-skill project around Jev.

## Community Jev skill for coding agents

https://github.com/dbreunig/building-with-jev-skill

Useful for question design and integration patterns, but it is not the TypeSafe-owned source of truth.

## Community curated list

https://github.com/yibie/awesome-jev

Curated third-party projects and integrations.

---

# 25. What makes a good Jev question?

A useful principle is:

> Ask Jev one bounded judgment that a knowledgeable person could answer quickly from the supplied state.

Good:

```text
Should the agent inspect claim history?
```

Good:

```text
Which retrieval tool is appropriate next?
```

Good:

```text
Does the evidence indicate a temporal conflict?
```

Bad:

```text
Analyze the whole investigation and decide what to do.
```

Bad:

```text
Solve the user's question completely.
```

Bad:

```text
Count all documents and calculate the final aggregate.
```

The last class of work belongs in deterministic code whenever possible.

A third-party Jev skill explicitly emphasizes the same design rule: keep each question narrow, keep arithmetic/date comparisons in code, and compose multiple small decisions programmatically.

Reference:
https://github.com/dbreunig/building-with-jev-skill

---

# 26. What Jev should NOT do in your GraphRAG system

## Do not use Jev as the final answer generator

Use your current LLM.

## Do not let Jev replace TigerGraph structural proof

If TigerGraph can compute:

```text
COUNT = 41
```

and your evidence set has:

```text
41
```

that is stronger than asking another AI model "does this look complete?"

## Do not move deterministic temporal resolution into Jev

Your temporal resolver can remain deterministic.

## Do not make Jev mandatory

Your current system should still work if Jev is unavailable.

## Do not replace the whole router

The existing benchmark-specific router can stay the primary path.

---

# 27. Best Jev role for this hackathon

The cleanest division is:

```text
                 ┌──────────────────┐
                 │      LLM         │
                 │ Deep reasoning   │
                 │ Final generation │
                 └────────┬─────────┘
                          │
                   expensive work
                          │
                          ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│    Jev      │────►│ Orchestrator │────►│ TigerGraph +  │
│ decisions   │     │              │     │ deterministic │
└─────────────┘     └──────┬───────┘     │ tools         │
                           │             └───────┬───────┘
                           │                     │
                           └─────────────────────┘
                                         │
                                         ▼
                                  Investigation
                                    Certificate
```

### Recommended responsibilities

| Component | Responsibility |
|---|---|
| Existing router | Known benchmark/template routing |
| **Jev** | Bounded uncertain decisions |
| Agentic orchestrator | Control flow |
| TigerGraph | Structural graph evidence |
| Deterministic tools | Retrieval/resolution/counting |
| LLM | Deep reasoning + final answer |
| Certificate | Machine-checkable evidence/completeness proof |

---

# 28. Recommended first experiment

Do not start by rewriting `agentic.py`.

Run this experiment first:

### Experiment A — Shadow routing

For every question:

```text
Existing router → decision A
Jev             → decision B
```

Measure:

```text
agreement %
wrong decision %
low-confidence %
latency
input tokens
cost
```

### Experiment B — Jev stop gate

Compare:

```text
existing stop policy
vs.
Jev stop policy
```

Measure:

```text
answer accuracy
evidence completeness
average retrieval steps
token usage
latency
```

### Experiment C — Tool routing

Compare:

```text
LLM tool router
vs.
Jev tool router
vs.
deterministic current router
```

Use the same question set and same tools.

This gives you evidence rather than a marketing claim.

---

# 29. Recommended integration sequence

```text
Step 1
Build DecisionModel abstraction.

Step 2
Implement FakeDecisionModel.

Step 3
Add JevDecisionModel.

Step 4
Run Jev in shadow mode.

Step 5
Compare Jev decisions with current policy.

Step 6
Enable Jev only on uncertainty branches.

Step 7
Benchmark accuracy / latency / tokens / cost.

Step 8
Only then decide whether Jev appears in the critical path.
```

---

# 30. One-line mental model

The easiest way to remember Jev is:

```text
LLM  = "write / reason"
Jev  = "judge / choose"
Code = "act"
Graph = "prove / retrieve"
```

For your Agentic GraphRAG project:

```text
Jev decides what to do.
TigerGraph determines what the graph says.
Deterministic code determines whether the evidence is complete.
The LLM performs deep reasoning and writes the answer.
The Investigation Certificate records the proof.
```

That is the architecture in which Jev adds value without undermining the existing system.

---

# 31. Source hierarchy for this document

When there is a conflict between sources, use this order:

1. **TypeSafe AI official site**
   - https://typesafe.ai/

2. **TypeSafe AI official GitHub**
   - https://github.com/typesafe-ai/skills

3. **Vercel's Jev integration pages**
   - https://vercel.com/ai-gateway/models/jev
   - https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway

4. **Other community documentation**
   - useful for examples and experiments
   - not authoritative for API behavior

---

# 32. Important caveat

Jev is a very new product as of September 2026.

That means:

- APIs may change.
- Model identifiers may change.
- Pricing may change.
- Access/availability may change.
- Third-party integrations may change.
- Benchmark claims should be treated as vendor-reported until independently reproduced.

For a hackathon submission, **verify the live TypeSafe/Vercel documentation immediately before recording the final demo and writing exact pricing/performance figures.**

