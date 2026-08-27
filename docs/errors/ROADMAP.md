# Systematic Bug-Hunt Roadmap

Tracker for issue #61. The sequence follows the component flow in
`docs/moon/ARCHITECTURE.md` §§1–7 and command flows in §8. Each completed
component is checked for cross-boundary shape/type errors, invalid state
mutation, device handling, and silent routing/solver failures.

- [COMPLETED] CLI entry point and Hydra command dispatch (`main.py`, task routing) — normalized documented aliases `evaluation` → `eval` and `sim_hpo` → `hpo_sim`; regression coverage added.
- [COMPLETED] Typed configuration composition and validation (`logic/src/configs/`) — all canonical task groups compose through Hydra; no additional typed-config defect confirmed in this pass.
- [IN PROGRESS] Feature engines: train, evaluation, data generation, and simulation dispatch
- [PENDING] Routing environments, generators, and task objectives
- [PENDING] Model encoders, decoders, embeddings, and critic interfaces
- [PENDING] Policy and solver construction, including exact and heuristic routes
- [PENDING] RL training lifecycle, baselines, callbacks, and tracking
- [PENDING] Multi-day simulation state machine, day context, and result aggregation
- [PENDING] Data repositories, processors, and dataset builders
- [PENDING] Cross-cutting utilities, package imports, lint/type checks, and regression sweep

## Current pass

Configuration composition is complete. Trace each feature engine's config
extraction, factory call, and return contract next.
