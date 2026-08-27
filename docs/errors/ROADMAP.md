# Systematic Bug-Hunt Roadmap

Tracker for issue #61. The sequence follows the component flow in
`docs/moon/ARCHITECTURE.md` §§1–7 and command flows in §8. Each completed
component is checked for cross-boundary shape/type errors, invalid state
mutation, device handling, and silent routing/solver failures.

- [IN PROGRESS] CLI entry point and Hydra command dispatch (`main.py`, task routing)
- [PENDING] Typed configuration composition and validation (`logic/src/configs/`)
- [PENDING] Feature engines: train, evaluation, data generation, and simulation dispatch
- [PENDING] Routing environments, generators, and task objectives
- [PENDING] Model encoders, decoders, embeddings, and critic interfaces
- [PENDING] Policy and solver construction, including exact and heuristic routes
- [PENDING] RL training lifecycle, baselines, callbacks, and tracking
- [PENDING] Multi-day simulation state machine, day context, and result aggregation
- [PENDING] Data repositories, processors, and dataset builders
- [PENDING] Cross-cutting utilities, package imports, lint/type checks, and regression sweep

## Current pass

Start with the CLI/Hydra hand-off. Record confirmed defects with their
regression coverage before advancing to configuration composition.
