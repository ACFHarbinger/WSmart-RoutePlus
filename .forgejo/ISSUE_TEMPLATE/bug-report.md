---
name: Bug report
about: Create a bug report to help us improve WSmart+ Route
title: "[BUG] "
labels: 'bug'
assignees: ''
---
**Describe the bug**
A clear and concise description of what the issue is and what the expected behavior is.

**How To Reproduce**
Steps to reproduce the behavior, e.g.:
1. Config/scenario used (Hydra overrides, area, waste type, policy)
2. Command run (`python -m logic ...`, task name)
3. What you observed vs. expected (metric, log line, traceback)

**Evidence**
If applicable, attach the relevant `simulation_summary*.csv` rows, raw logs, or a
traceback. Per project convention every reported number should trace back to one
of these — see `AGENTS.md` for the data-integrity ground rules.

**Environment**
- OS: [e.g., Ubuntu 24.04]
- Python version / `uv` lock state:
- GPU (if relevant): [e.g., NVIDIA RTX 4070 12GB]
- Solver backend (if relevant): [e.g., Gurobi 11.0]

**Additional context**
Add any other context about the problem here.
