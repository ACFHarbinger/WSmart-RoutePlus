# Summary CSVs, 2026-10-19

Built with `logic/gen/build_simulation_summary.py docs/private/global/simulation/rerun-2026-10-19 --override outputs/simulations-cptp --override outputs/simulations-acohh`.

Differences from `rerun-2026-10-16` (every other row is identical):
- **BPC (120 rows: 60 at 30 days, 60 at 90 days)** comes from the rerun with the CPTP branch-and-cut pricer (`pricing_engine: cptp`), run 2026-10-18/19 from the pinned worktree at `6b772f0f9` (`outputs/simulations-cptp`).
- **ACO-HH (90 rows: 60 at 30 days, 30 at 90 days)** comes from the rerun with the mandatory-bin fix (`d846e0fa7`), run 2026-10-19/20 from the pinned worktree at `0b7f90fef` (`outputs/simulations-acohh`). No mandatory bin is skipped in any of the 90 runs (0 missed visits in 4,500 run-days).
- A control cell (30 days, Rio Maior 100 Empirical, LM-CF70, Fast-TSP) rerun at the pre-fix `64acefeb8` differs from the archived result too (2.548 against 2.319 kg/km): the packing step chooses the same bins until day 14, but ACO-HH's own tours differ from day 1. So the new ACO-HH rows reflect the current ACO-HH code (the #78 fixes as well as the mandatory-bin fix), not only the fix.
