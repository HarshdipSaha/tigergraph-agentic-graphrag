# Effort 003: Agent Planning and Venue Integrity

**Status:** Complete on the isolated `improvement-agentic-planner` branch (2026-10-07). Implementation, live Event refresh, evaluations, and supporting documentation are complete and validated. The branch remains uncommitted and unmerged for owner review.

## Outcome

- Default agentic mode asks the LLM to select one of five typed graph tools; Python validates the action, executes the graph query, and evaluates evidence. Template mode remains available for comparison.
- Groq key rotation uses the five configured same-organization credentials without exposing them in code or outputs. Non-secret key health/rotation state is stored under ignored `data/.groq_key_state.json`.
- Parser and live Event refresh cover 2,187 source Events. The TigerGraph Event ID set matches the corpus exactly; chunks and embeddings were not rebuilt.
- Local and live reconciliation each report 98/100 public answer matches and 100/100 evidence checks. `pub-060` and `pub-099` remain visibly unresolved ties.
- Public planner: 98/100, 1,148.5 mean tokens, 1.769 s mean latency, and 86 selected graph-tool actions. Public template: 99/100, 14.3 mean tokens, 0.414 s mean latency. The planner makes tool choice explicit but does not improve measured accuracy or cost in this comparison.
- Hidden planner output has 50 structurally valid rows and 49 non-null answers. Hidden labels are unavailable; no hidden answer accuracy is claimed. `eval-001` returns `Li TingSun Tiantian` from `Q942805` with an exact venue/date evidence pass.
- Full tests passed: 120 offline tests plus four live TigerGraph integration tests. The integration regression fixture changed because the parser repair correctly made former distractor `Q1183979` an Event.

## Artifacts

- [Requirements delta](requirements-delta.md)
- [Implemented architecture and measured comparison](architecture-delta.md)
- [Standalone implementation plan](../../../plan.md)
- [Validation report](validation-report.md)
- [Audit trail](audit.md)
