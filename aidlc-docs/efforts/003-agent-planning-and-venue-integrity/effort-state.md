# Effort 003: Agent Planning and Venue Integrity

**Status:** Complete and merged into `main` (2026-10-07). Implements post-submission feedback from a TigerGraph developer. Implementation, live Event refresh, evaluations, and supporting documentation are complete and validated. The submitted headline results are unchanged.

## Outcome

- Planner mode (`--agent-mode planner`) asks the LLM to select one of five typed graph tools; Python validates the action, executes the graph query, and evaluates evidence. Template mode stays the CLI default so the submitted headline run remains reproducible.
- Groq key rotation uses the five configured same-organization credentials without exposing them in code or outputs. Non-secret key health/rotation state is stored under ignored `data/.groq_key_state.json`.
- Parser and live Event refresh cover 2,187 source Events. The TigerGraph Event ID set matches the corpus exactly; chunks and embeddings were not rebuilt.
- Local and live reconciliation each report 98/100 public answer matches and 100/100 evidence checks. `pub-060` and `pub-099` remain visibly unresolved ties; the pipelines resolve them by model choice with an `unverified` certificate.
- Public planner: 99/100, 1,038.4 mean tokens, 1.447 s mean latency, and 87 selected graph-tool actions. Public template: 99/100, 14.3 mean tokens, 0.414 s mean latency. The planner makes tool choice explicit at equal accuracy but much higher cost. (An earlier planner run that abstained on ties scored 98/100.)
- Hidden planner output has 50 structurally valid rows and 50 non-null answers. Hidden labels are unavailable; no hidden answer accuracy is claimed. `eval-001` returns `Li TingSun Tiantian` from `Q942805` with an exact venue/date evidence pass, fixing the submitted run's cross-sport answer; it is the only hidden answer that changes.
- Full tests passed: 120 offline tests plus four live TigerGraph integration tests. The integration regression fixture changed because the parser repair correctly made former distractor `Q1183979` an Event.

## Artifacts

- [Requirements delta](requirements-delta.md)
- [Implemented architecture and measured comparison](architecture-delta.md)
- [Standalone implementation plan](../../../plan.md)
- [Validation report](validation-report.md)
- [Audit trail](audit.md)
