# Three-Way Scan (WHY / HOW / WHAT): Agentic GraphRAG Literature

Mode: `deep-research` / `three-way-scan` (ARS suite). Compiled 2026-09-05 alongside `docs/hackathon-brief.md`. Retrieval via the arXiv Atom API (live queries, not from training memory) plus targeted web search for market/product context. 13 papers screened, 10 shortlisted below; 3 more surfaced in market search and folded into WHAT.

---

## Per-paper WHY / HOW / WHAT

### Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG
Source: arXiv | Year: 2025 | Link: https://arxiv.org/abs/2501.09136

- WHY: Static RAG can't do multi-step reasoning or adapt its workflow; LLMs need autonomous control over retrieval to handle complex tasks.
- HOW: Taxonomizes Agentic RAG by agent cardinality, control structure, autonomy, and knowledge representation; surveys design patterns (reflection, planning, tool use, multi-agent collaboration).
- WHAT: A map of the field, not a system. Names evaluation, coordination, memory management, efficiency, and governance as open problems — i.e., nobody has a settled way to measure whether the agentic layer is worth its cost.

### When to use Graphs in RAG: A Comprehensive Analysis for GraphRAG (→ GraphRAG-Bench)
Source: arXiv | Year: 2025 | Link: https://arxiv.org/abs/2506.05690

- WHY: GraphRAG is assumed to beat vanilla RAG, but field reports say it frequently doesn't. The paper asks the same question this hackathon asks: when does structure actually help?
- HOW: Builds GraphRAG-Bench, a benchmark spanning fact retrieval, complex reasoning, summarization, and creative generation, with holistic pipeline evaluation (construction → retrieval → generation), not just final-answer accuracy.
- WHAT: Graphs help on deep multi-hop reasoning and hierarchical retrieval, not on simple fact lookup — direct evidence that a one-size pipeline is wrong, and that per-question-type evaluation (which this hackathon's own qtype field enables) is the right lens.

### Do We Still Need GraphRAG? Benchmarking RAG and GraphRAG for Agentic Search Systems
Source: arXiv | Year: 2026 | Link: https://arxiv.org/abs/2604.09666

- WHY: Agentic (multi-round) search already improves plain RAG by adding implicit structure through interaction. If agentic search can substitute for explicit graph structure, GraphRAG's cost may not be worth it.
- HOW: RAGSearch, a unified benchmark standardizing LLM backbone, retrieval budget, and inference protocol across dense-RAG and GraphRAG backends under agentic (training-free and RL-trained) search; reports accuracy, offline preprocessing cost, online efficiency, and stability.
- WHAT: Agentic search narrows the RAG-to-GraphRAG gap substantially, especially with RL. But GraphRAG remains ahead specifically on complex multi-hop reasoning and is more stable once its offline construction cost is amortized. This is the single most load-bearing finding for the hackathon: it says graphs earn their keep exactly on the aggregation/superlative-style questions this dataset is full of, not on everything.

### Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning
Source: arXiv | Year: 2025 | Link: https://arxiv.org/abs/2507.21892

- WHY: Chunk-based RAG lacks structural semantics; static one-shot GraphRAG has high construction cost and fixed retrieval. Neither adapts mid-investigation.
- HOW: Lightweight knowledge-hypergraph construction + models retrieval as multi-turn agent-environment interaction, trained end-to-end with RL and a process reward.
- WHAT: Beats traditional GraphRAG and RL-enhanced RAG on accuracy, retrieval efficiency, and generation quality — but the entire gain is bought with an RL training loop, which is the expensive part to reproduce, not the agent design.

### GraphSearch: An Agentic Deep Searching Workflow for GraphRAG
Source: arXiv | Year: 2025 | Link: https://arxiv.org/abs/2509.22009

- WHY: Existing GraphRAG retrieval is shallow (misses critical evidence) and underuses the graph's own structure.
- HOW: A six-module agentic workflow with dual-channel retrieval — semantic queries over text chunks and relational queries over the graph in parallel — enabling multi-turn, iterative reasoning.
- WHAT: Consistently improves accuracy and generation quality over single-channel GraphRAG on six multi-hop benchmarks. Evidence that combining structural and semantic retrieval, not choosing one, is where gains live.

### T-GRAG: A Dynamic GraphRAG Framework for Resolving Temporal Conflicts and Redundancy
Source: arXiv | Year: 2025 | Link: https://arxiv.org/abs/2508.01680

- WHY: GraphRAG ignores time. Facts change; a static graph merges temporally distinct facts under one node, causing temporal ambiguity and stale answers.
- HOW: Time-stamped, evolving knowledge-graph generator; decomposes temporal queries into sub-queries; a three-layer interactive retriever progressively filters temporal subgraphs; a source-text extractor cuts noise.
- WHAT: Beats prior RAG/GraphRAG baselines on a new corporate-annual-report benchmark (Time-LongQA) under temporal constraints. Directly maps to Round 2 ("reasoning over time"), but was built and evaluated on a different domain (financial reports, not sports records) — the mechanism, not the benchmark, transfers.

### Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity
Source: arXiv | Year: 2024 | Link: https://arxiv.org/abs/2403.14403

- WHY: No single retrieval strategy is right for all queries — simple ones waste compute on iterative retrieval, complex ones fail with single-shot retrieval.
- HOW: A small trained classifier predicts query complexity (from auto-collected labels) and routes to no-retrieval / single-step / iterative pipelines accordingly.
- WHAT: Established the "route by predicted complexity" pattern with a trained classifier — the precedent every later routing paper (GraphRAG-Router, PathRouter) builds on. Requires a labeled training pass, which a 3-week hackathon can substitute with a cheap structural heuristic instead (see idea spec).

### GraphRAG-Router: Learning Cost-Efficient Routing over GraphRAGs and LLMs with RL
Source: arXiv | Year: 2026 | Link: https://arxiv.org/abs/2604.16401

- WHY: One-size-fits-all GraphRAG pipelines waste large-LLM calls on easy queries.
- HOW: Hierarchical routing across heterogeneous GraphRAG backends and generator LLMs, warmed up with supervised fine-tuning then optimized with two-stage RL using a curriculum cost-aware reward.
- WHAT: Cuts large-LLM overuse ~30% while holding accuracy, across six QA benchmarks. Same lesson as Graph-R1: real gains, real RL infrastructure cost.

### When Knowledge Is Not Free: Cost-Aware Evidence Selection in RAG
Source: arXiv | Year: 2026 | Link: https://arxiv.org/abs/2606.02245

- WHY: RAG research assumes evidence is free; real evidence has access-cost tiers (paywalled, licensed, restricted).
- HOW: Introduces cost-aware RAG with an explicit evidence-access budget; studies both static selectors and agentic controllers that decide when to retrieve, which cost tier, and when to stop.
- WHAT: No fixed selector dominates; bigger budgets don't reliably help. Agentic cost-aware stopping shows promise but is "highly model- and task-dependent" — a named, unsolved reliability gap in exactly the "when does the agent know it's done" question this hackathon's rubric scores under "agentic effectiveness and efficiency."

### PathRouter: Aligning Rewards with Retrieval Quality in Agentic GraphRAG
Source: arXiv | Year: 2026 | Link: https://arxiv.org/abs/2606.16409

- WHY: Outcome-only RL for agentic GraphRAG suffers "answer-path reward aliasing" — the agent gets a correct answer via a shortcut, not real evidence, and the reward can't tell the difference.
- HOW: Path-aware RL reward that scores trajectories on both answer correctness and evidence-path overlap with a gold path; frozen teacher gives token-level guidance on evidence-poor trajectories.
- WHAT: Improves both answer F1 and evidence-path overlap over a strong RL baseline. This is the closest existing work to "prove the answer came from real evidence, not a shortcut" — but it solves the problem via a trained reward signal, not a deterministic, judge-legible certificate.

---

## Market / adjacent-product scan (not academic papers, folded into WHAT)

- **Graphiti** (getzep, OSS): gives facts validity windows — old facts are invalidated, not deleted, when information changes — so queries can ask "what's true now" vs "what was true then." Real production answer to T-GRAG's problem, at the product layer rather than the research layer.
- **"Rethinking Agentic RAG"** (arXiv 2605.27123): argues agentic RAG should let the LLM express retrieval intent as logical/structured queries over a lightweight inverted-index backend rather than embeddings — cuts hallucination and serving cost, but is backend-substitution, not a completeness argument.
- **SSRN survey, "A Survey of Agentic GraphRAG"**: confirms the field currently frames "agentic GraphRAG" as GraphRAG + Agentic RAG combined to fix each other's failure modes — consistent with this hackathon's own framing.
- No product or paper found (arXiv or web) that ships a **deterministic, judge-legible completeness certificate** distinct from answer correctness. PathRouter gets closest but via RL reward, not a user-facing artifact; Graphiti and T-GRAG solve temporal staleness, not exhaustiveness-proving for counting/superlative queries.

---

## Cross-paper synthesis

**Common WHY.** Every paper converges on one problem: a fixed retrieval strategy (one-shot RAG, or one-shot GraphRAG) cannot match the true diversity of question difficulty. Some questions need one fact, some need a multi-hop chain, some need an exhaustive enumeration, some need to resolve which of several conflicting facts is current. Treating all of these the same way either wastes compute or silently under-retrieves.

**Divergent HOW.** Two competing engineering philosophies:
1. **Learn the routing/reward function** (Adaptive-RAG's trained classifier; Graph-R1, GraphRAG-Router, PathRouter's RL-trained agents). Strong empirical gains, but requires a training pipeline — infeasible to build and validate from scratch in a 3-week hackathon.
2. **Make the structure do the work directly** (GraphSearch's dual-channel retrieval; T-GRAG's time-stamped subgraphs; Graphiti's validity windows). Training-free, deterministic, and exactly what a graph database like TigerGraph is built to answer cheaply via GSQL.

**Strongest WHAT.** "Do We Still Need GraphRAG?" gives the cleanest empirical answer to the hackathon's own headline question: agentic search narrows the RAG/GraphRAG gap in general, but graphs still win specifically on complex multi-hop reasoning, and the win holds once construction cost is amortized. Translated to this hackathon's dataset: expect RAG to be competitive on `lookup`, and to lose clearly on `aggregation`/`superlative`, where 8–43 gold docs must all be found, not just the top few similar ones.

**Unresolved global gap.** Nobody — not the RL-routing papers, not T-GRAG, not the cost-aware RAG paper — treats **"did the agent retrieve a provably complete evidence set"** as a first-class, deterministic, judge-legible output distinct from "was the final answer correct." PathRouter gets closest, but proves path quality via a trained reward signal invisible to an end user; Cost-Aware RAG explicitly flags agentic stopping behavior as unverified and task-dependent. On a benchmark like this hackathon's — where the graph itself can compute the exact size of the correct answer set via `COUNT` — that gap is closeable without any RL training, which is what makes it a legitimate 3-week hackathon idea rather than a paper-scale undertaking. See `docs/idea-spec.md` for the concrete design, and note its scope is deliberately narrower than the pitch above: it is a benchmark-scoped completeness certificate, not a general-purpose router (see council verdict in that document for why).
