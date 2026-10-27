# Summary CSVs, 2026-10-26

Built with `logic/gen/build_simulation_summary.py docs/private/global/simulation/rerun-2026-10-26 --override outputs/simulations-cptp --override outputs/simulations-acohh --override outputs/simulations-swc-directed`.

Differences from `rerun-2026-10-19` (every other row is identical):
- **SWC-TCF (66 rows: 60 at 30 days, 6 at 90 days)** comes from the rerun with the directed two-commodity flow formulation (`formulation: directed`), run 2026-10-26 from the pinned release 1.0.2 tree (`753b8fe5e`) on 8 workers, one campaign at a time (`outputs/simulations-swc-directed`). The configuration is the paper's (350 s no-solution factor 9, the same instances and seed). The undirected runs stay in `outputs/simulations` and in `rerun-2026-10-19`.
