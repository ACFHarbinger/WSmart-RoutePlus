> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## I — Publication & Dissemination

> The MPVRPP manuscript, the tooling that keeps it honest, and the public-facing website. Added 2026-08-25, when rewriting the paper's Results section surfaced enough data-integrity work to justify its own track.

### I.1 — MPVRPP manuscript

- [x] Methodology completed: BPC, SANS, PG-CLNS and PSOMA written from `logic/src/policies/` and `bibliography/`; Look-Ahead described (it was a third of the experimental grid and went unnamed); CF70/CF90 and SL1/SL2 variants defined; the sentence that ended mid-clause finished (2026-08-25)
- [x] Results and the stale Data/Baselines subsections rewritten against `docs/private/global/simulation/simulation_summary{,_90d}.csv`. The previous Results described an Attention Model / gurobi / look-ahead study that nothing in the repo reproduces (2026-08-25)
- [x] Simulation Protocol subsection stating paired demand realisations, sensing noise, the overflow/loss distinction and the intended single-vehicle single-depot restriction (2026-08-25)
- [ ] `[Blocker]` Correct the operational-setting claim: all 36 archived
  configurations set `sim.n_vehicles: 0`, which current routing contracts treat
  as automatic/unlimited rather than one vehicle. Recover per-day route counts
  or rerun with an asserted positive fleet limit before retaining
  “single-vehicle” (shared audit RCP-001, 2026-08-28)
- [x] Conclusion and Future Work completed; Related Work now includes the multi-period profitable-routing gap and scopes NCO to framework capability rather than a benchmarked result (2026-08-25)
- [x] Final publication edit: concise introduction/contributions, corrected route formulation, evidence-calibrated Results prose, six-author block and PDF metadata, citation records checked against original publication pages (2026-08-25)
- [x] Final Experimental Evaluation and closing-section review: removed the unsupported CLS-dominance claim, restored integrity framing for raw granular figures, gathered Limitations/Future Work, corrected the exclusion and frontier arithmetic, and repaired the remaining BibTeX defects, including the Barnhart publication year (issue #50, 2026-08-27)
- [x] Conference-abstract reconciliation: preserved the abstract of record verbatim, separated registered NCO capability from the classical-only stored benchmark, defined the mandatory-selection dispatch referent, and scoped the unified baseline to the complete 30-day factorial design (issue #53, 2026-08-27)
- [x] Real side-by-side coordinate maps for Rio Maior–170 and Figueira da Foz–350 from the retained selected-bin maps over OpenStreetMap drive-network geometry; MDS layouts removed from the paper figure (2026-08-25)
- [x] Started the collaborative research/codebase/manuscript audit at
  `.agent/reports/shared/COMPREHENSIVE_REPORT.md`, with evidence statuses,
  claim-to-artifact mapping, and a prioritized amendment ledger (2026-08-28)
- [ ] `[Research]` Run a complete, replicated 90-day grid so cross-constructor and population-level horizon effects become estimable (see §I.3). Current paired values describe only policies selected on 30-day Pareto performance

### I.2 — Reproducible generation

- [x] `logic/gen/gen_paper_latex.py`: six tables and seven figures generated from the summary CSVs/raw logs into the paper's own tree, including balanced demand/network marginals and the simulation loop architecture (2026-08-25, 2026-08-27)
- [x] Simulation Protocol loop diagram (`simulation_loop.png`) generated programmatically via `fig_simulation_loop` in `logic/gen/gen_paper_latex.py` and included in `sec:protocol` (issue #54, 2026-08-27)
- [x] Network-map generation reuses tracked coordinate-derived analysis artifacts and fails explicitly when they are absent, avoiding geographic reconstruction from inconsistent distance-matrix copies (2026-08-25)
- [x] Report/deck generators revived from `archive/gen/` to `logic/gen/` and brought up to ruff (2026-08-25)
- [x] A `just paper` target that regenerates tables and figures and rebuilds the PDF in one step (2026-08-27)
- [ ] Wire the same degenerate-run exclusion into `gen_simulation_analysis.py` and the Studio's native `app/src/gen/` engine, so all four consumers apply one rule
- [ ] Port the LaTeX table path into the native §H engine, or decide explicitly that LaTeX stays Python-only

### I.3 — Data integrity

- [x] Degenerate-run detection on collected tonnage, with whole-cell exclusion so no constructor is averaged over a subset that flatters it (2026-08-25)
- [x] Root-cause and repair the duplicated/partial road-distance artifacts: parallel policy workers shared one non-atomic CSV writer. All correctly sized copies agreed within each network; atomic publication now prevents mixed output (issue #48, 2026-08-25)
- [ ] Root-cause the degenerate SWC-TCF runs at Figueira da Foz N=350 / Gamma-3. The stored logger erases `mandatory_nodes` whenever the returned tour is empty, so the current logs cannot distinguish a selection failure from solver infeasibility; instrument both values and the solver status in a targeted rerun (issue #41)
- [ ] `[Research]` Re-run CLS and Fast-TSP from identical stored constructor outputs and controlled seeds; the current matched-demand pairs differ in upstream collected-bin counts and cannot identify a causal improver effect
- [ ] `[Research]` Horizon-adaptive time budgets and constructor-specific timeout/fallback handling for long-horizon runs. This addresses an observed failure, not a hypothetical one
- [ ] Recover and version the 90-day carry-forward manifest. The tracked 174
  rows comprise 29 policies repeated across six scenarios, but neither the
  row-level scenario fronts nor the globally aggregated policy front reproduces
  the paper's literal “only Pareto-front configurations” statement (shared
  audit RCP-004, 2026-08-28)
- [ ] Preserve depot separators or list-of-routes in raw daily telemetry, plus
  route count, per-route payload, fleet/trip interpretation, and shift
  feasibility. The current logger removes internal zero separators, preventing
  reconstruction of the archived fleet usage (shared audit RCP-001,
  2026-08-28)
- [ ] Sweep for degenerate runs the tonnage rule may miss (issue #41, assigned to review)

### I.4 — Public website (`docs/website/`)

- [x] Visual identity and design system — semantic tokens, real light *and* dark states, a hero that is not three blurred orbs (issue #45)
- [x] Interactive policy-pipeline diagram covering all three stages (issue #46)
- [x] Animated bin selection with a draggable critical-fill threshold, making the efficiency/service trade-off legible (issue #46)
- [x] 3D/4D routing view — multi-period is space plus time; scrubbable 30-day routes from `assets/output/30days/**/log_*.json` (issue #46)
- [x] Results charts driven by generated JSON under `docs/website/public/data/`, never hand-copied numbers, with the degenerate-run exclusion applied (issue #46)
- [x] Website data generator `logic/gen/export_website_data.py` reuses the paper's integrity machinery (2026-08-25)

---
