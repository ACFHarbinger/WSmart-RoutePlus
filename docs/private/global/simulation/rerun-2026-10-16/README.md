# Summary CSVs, 2026-10-16

Built from `outputs/simulations` with `logic/gen/build_simulation_summary.py`.

Differences from `rerun-2026-10-13`:
- **PSOMA (60 rows at 30 days, 12 at 90 days)** comes from its rerun after the 2026-10-13 PSOMA fixes (`244b7479e`, `e97180567`), run in `~/Repositories/Doc/wsr-rerun84`. Its Figueira da Foz Empirical runs no longer fall below the exclusion threshold, so no cell is excluded.
- **Attention Model rows** were added (60 per horizon, `constructor` = `AM`). The Figueira da Foz 350 Empirical AM cells come from the model retrained on 2026-10-16.

Every other classical row is identical to `rerun-2026-10-13`.
