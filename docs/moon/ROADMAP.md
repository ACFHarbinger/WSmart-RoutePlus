# WSmart+ Route — Moon Roadmap

> **Version**: 1.1  
> **Last Updated**: July 2026  
> **Status**: Living Document — updated each sprint  
> **Scope**: Logic layer (`logic/src/`), GUI layer (migrating from `gui/src/` PySide6 → Tauri), CI/CD, documentation

This document captures medium-to-long-horizon improvements for the WSmart+ Route framework across eight dimensions: Analytics & Interpretability, Architecture, Documentation, GUI/UX, New Features, Performance, the WSmart-Route Studio Tauri application, and the Analysis & Presentation Studio migration. Each item follows a **Pain → Options → Recommendation** structure with effort/impact tags.

Tags: `[Quick Win]` ≤ 1 day · `[Research]` involves novel work · `[Blocked]` depends on another item

---

## Anchor Index

Integration review, 2026-10-16: corrected gate YAML materialization and the
baseline optional-routing dispatch amendment are documented in
[`tools/gates/BASELINE.md`](../../tools/gates/BASELINE.md). The source-hashed
reference guard, dispatch regression and 2,124-test unit suite pass. Q-14
calibration, all paired cells and the archived time-capped comparisons are
still merge prerequisites; unit success does not replace those gates.

> Each section below now lives in its own file under [`docs/moon/roadmaps/`](roadmaps/), split out on 2026-08-27 so themes can be updated independently. This file keeps the index and the Cross-Cutting Themes table.

| Section                                                              | Topic                                                                  |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [§A — Analytics & Interpretability](roadmaps/analytics_interpretability.md) | Telemetry, attention maps, policy dashboards, HPO analytics |
| [§B — Architecture](roadmaps/architecture.md) | Test coverage, plugin system, logging, type safety, interfaces |
| [§C — Documentation](roadmaps/documentation.md) | API docs, architecture diagrams, Jupyter notebooks, CI docs pipeline |
| [§D — GUI / UX](roadmaps/gui_ux.md) | Route visualization, training progress, themes, session persistence |
| [§E — New Features](roadmaps/new_features.md) | Multi-problem benchmarking, REST API, LLM integration, export formats |
| [§F — Performance](roadmaps/performance.md) | Batched inference, GPU memory, test suite speed, simulation throughput |
| [§G — WSmart-Route Studio](roadmaps/studio.md) | Tauri 2.0 app: analytics, geospatial, ML introspection, launcher UIs |
| [§H — Analysis & Presentation Studio](roadmaps/presentation_studio.md) | Migration of `logic/gen/` report + deck generation into the Studio |
| [§I — Publication & Dissemination](roadmaps/publication_dissemination.md) | MPVRPP paper, reproducible LaTeX generation, public website |


---

## Simulator 1.0.0 refactor — 2026-10-15

- WP-01 (`refactor/v1-wp01-registry-lazy-facades`, #106): one `Registry` with name lookup and tag queries, lazy parent packages and generated manifests, `envs/problems.py` removed, RL expert configs renamed to `Vector*`, temporal defaults moved to `constants/temporal.py`. Codex reviews; Claude integrates. Not on `main`.
- WP-02 (`refactor/v1-wp02-release-manifest`, #107): `logic/release.yaml` is the single source of truth; `logic/package/build_release.py` builds the package from `main` (check / build / torch-free smoke), the release pyproject installs without torch (`wsmart-route[neural]` carries the training stack), and the acronym-driven export flow (`ci/packager.py`, `export_config.json`, `prune_codebase.py` + the 16 `remove_*.py` scripts, the `minimal-ptp` patch) is retired. P-25 lazy-fication extended to the RL core, vector facade, models/subnets re-export inits, the model loader and the imitation/builder expert maps. Not on `main`.
- Remaining packages: WP-03 config/rename leftovers, WP-07 ESPPRC envs, WP-08 modular simulator. See `.agent/tasks/simulator-v1-refactor-proposal-2026-10-15.md` §12.
- WP-03 (`refactor/v1-wp03-config-rename`, #108): Cursor landed P-13 / P-45 / P-54 — policy YAML is a plain mapping, packing leftovers (`pk_*`, `pack`, `PackingContext`, `SearchPhase.PACKING/ROUTING`, `routing_algorithms`) are renamed, and packing remaps are gone. Mistral still owns P-55 / P-56. Codex reviews; Claude integrates. Not on `main`.
- WP-03 (Cursor: `refactor/v1-wp03-config-rename`, #108): P-13 plain-mapping policy YAML with loud unknown-key failure, P-45 rename leftovers (`pk_*.yaml`, `pack`, `PackingContext`, `routing_algorithms`), P-54 packing field names (`threshold`).
- WP-03 (Mistral: `refactor/v1-wp03-usage-audit-hpo`, #108): P-55 corrected usage audit (`tools/audit_param_usage.py`, no-false-INERT three-pattern method; evidence appendix committed) and P-56 HPO job conversion + key-liveness CI gate (`tools/check_hpo_jobs.py`, 96 jobs converted, check in all four CI files).
- WP-06 (`refactor/v1-wp06-unreleased-migration`, #111): `tools/migrate_unreleased_params.py` moved 84 of 86 remaining params modules to the config layer in one scripted pass (merge/append/wire), rewrote every `XParams` reference, banned all 95 deleted symbols in `check_params_refs.py`, and kept both guards green (params-refs, HPO jobs). Hand-fix queue: `calm`, `hmlns` (multi-class params).
- WP-07 (`refactor/v1-wp07-espprc-envs`, #112): Gemini landed ESPPRC envs, unified `evaluate_tour`, and embedding checkpoint parity. Codex reviews; Claude integrates. Not on `main`.
- WP-09 (`refactor/v1-wp09-packing-numpy-torch`, #114): Cursor landed P-37 / RQ-codex-1 (NumPy spec + batched Torch adapters + parity), RQ-cursor-3 (EOQ last-minute split out of 1.0.0), and the P-05 follow-up that keeps `vectorized/manager.py` from pulling the unreleased HRL manager into the release. Stacked on WP-07 because WP-08 is not landed; day-loop torch-free remains WP-08. Codex reviews; Claude integrates. Not on `main`.
- Remaining packages: WP-08 modular simulator, then WP-10/WP-11/WP-13. See `.agent/tasks/simulator-v1-refactor-proposal-2026-10-15.md` §12.
- Remaining packages: WP-07 ESPPRC envs, WP-08 modular simulator. See `.agent/tasks/simulator-v1-refactor-proposal-2026-10-15.md` §12.
- WP-11 (`refactor/v1-wp11-solver-consolidation`, #116): one engine contract per solver family. Cursor finished Kimi's P-33 leftover — both BPC engines share `search_bpc_tree`; MS-BPC-SP is knapsack + that walk + Phase 3/5 set partitioning. P-30/P-31/P-34/P-35 already on the branch. Q-14 still waits for `sim chain done`. Codex reviews; Claude integrates. Not on `main`.
- WP-13 (`refactor/v1-wp13-docs-cleanup`, #118, Mistral pass): P-03 trimmer + relaxed checkers (Q-7b; 333 files, −589 lines), Q-22 generated API reference (`docs/api/`, `tools/gen_api_docs.py`), Q-21 `rl4co/` clone deleted, and the WP-02 release manifest fixed for WP-07's `envs.tasks.*` layout (check/build/smoke green on the merged chain). Follow-ups: the deferred CLI-job packages (`cloud`, `benchmark`, `test`) excluded from the manifest — they were kept-but-unstaged and would have crashed the release CLI jobs — and the trimmer no longer walks the pinned submodule. Claude's pass: the 127 pre-existing missing public docstrings, per-package trimmer review, and integration-order completions.
- Remaining: Claude's WP-13 close-out and integration of WP-08/09/10/11/12 in §12 order.
- WP-02 (`refactor/v1-wp02-release-manifest`, #107): `logic/release.yaml` is the single source of truth; `logic/package/build_release.py` builds the package from `main` (check / build / torch-free smoke), the release pyproject installs without torch (`wsmart-route[neural]` carries the training stack), and the acronym-driven export flow (`ci/packager.py`, `export_config.json`, `prune_codebase.py` + the 16 `remove_*.py` scripts, the `minimal-ptp` patch) is retired. P-25 lazy-fication extended to the RL core, vector facade, models/subnets re-export inits, the model loader and the imitation/builder expert maps. Not on `main`.
- Remaining packages: WP-03 config/rename leftovers, WP-07 ESPPRC envs, WP-08 modular simulator. See `.agent/tasks/simulator-v1-refactor-proposal-2026-10-15.md` §12.
## Simulator 1.0.0 Architecture Review — 2026-10-15

- Architectural review of the Simulator 1.0.0 refactor proposal (`.agent/tasks/simulator-v1-refactor-proposal-2026-10-15.md`, draft 8).
- Envs & Neural Stack review completed by Gemini:
  - Base environment consolidation on batched ESPPRC (`P-23`) with vectorized resource transitions.
  - Complete elimination of legacy unvectorized `BaseProblem.get_costs` across 6 neural caller sites (`P-20`).
  - Unified node and dynamic feature embedding module (`P-22`).
  - Standardized policy forward contract and lazy subnet loading (`P-24`, `P-25`).
  - Offline precomputed expert dataset generation (`P-51`) to maximize imitation training throughput.
- Reviewer questions `RQ-gemini-1` to `RQ-gemini-4` submitted for owner sign-off.

---

## Combinatorial Optimization survey — 2026-10-13

- Round-2 proposals by Codex, Cursor, Kimi, and Gemini are recorded in `.agent/cache/co-survey-round2-shared-2026-10-13.md`, adhering to the owner's field-wide scope, INFORMS format, and exclusion of simulator experiments.
- Gemini established the §5 neural operational taxonomy (L2C, L2P-O, L2P-M, L2I, Neuro-MIP), the §6.4 unified diffusion/consistency comparison box (owner #12), and the §6.6 four-tier LLM optimization taxonomy (owner #11).
- Pending consolidation: agree the cross-family structure, build the claim/source catalogue in the dedicated bibliography pass, and reconcile equations and repeated mechanisms before the INFORMS manuscript build. These remain proposed work, not implemented manuscript changes.

---

## Research-code follow-through — 2026-10-04

Round-5 implementation is integrated. The
[issue #61 tracker](../errors/ROADMAP.md#current-status--2026-10-04) records
commit evidence, reviewed limitations, and the remaining work.

- ✅ Integrated: ARCO simulator entry and generator import, old-SANS deadline,
  shared ILS/VNS operators, security utility fixes, revised DACT/NeuOpt/N2S
  decoders, and the strengthened certified-pricing regression.
- 🚧 Pending integration: the reviewed security source-validation patch,
  N2S action-contract documentation, and generator lint cleanup. Patch locations
  and checks are linked from the tracker.
- ❌ Figueira reruns blocked: resolve the bin-selection discrepancy and validate
  fill/matrix ordering before placing inputs and running those cells. Archived
  90-day logs are also needed for historical result comparisons.
- 🚧 Separate follow-up: deterministic worst-removal RNG for ILS/VNS. The
  completed extraction preserves the previous behavior.

This status update does not close #61, #83, or #90, change paper claims, or
certify GPU training or full paper equivalence.

---

## Cross-Cutting Themes

Several items across sections are tightly coupled and should be sequenced together:

| Cluster                        | Items                          | Rationale                                                                                |
| ------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------- |
| **Plugin System**              | §B.3, §B.6                     | Policy and env registration should share the same Hydra-based mechanism                  |
| **Async Worker Contract**      | §B.8, §D.5, §G.15              | Rust AsyncTask trait + Python BackgroundTask protocol are prerequisites for cancel UX    |
| **Route Visualization**        | §A.1, §D.1, §G.3               | All three need the same spatial renderer; deck.gl in §G.3 is the shared base             |
| **Docs Infrastructure**        | §C.1, §C.7                     | MkDocs setup is a prerequisite for the CI docs pipeline                                  |
| **Test Quality**               | §B.1, §F.3                     | Coverage uplift and test-suite speed are best addressed together                         |
| **Telemetry**                  | §A.3, §A.4                     | PolicyVizMixin and TrainingHealthCallback both feed the Studio analytics dashboard        |
| **Config System**              | §B.3, §B.6, §D.6, §G.13        | Plugin registry + Hydra `_target_` + Studio config editor all depend on a clean config schema |
| **Process Streaming**          | §G.9, §G.10, §G.11, §G.12, §G.15, §G.16 | All launchers share the same Rust→React stdout streaming infrastructure from §G.15 |
| **Streamlit Parity**           | §G.16, §G.17, §G.18            | These three phases are a 1:1 port of the three most-used Streamlit modes; complete before removing Streamlit dependency |
| **Document Authoring**         | §H.0–§H.8, §G.0, §G.1, §G.3, §G.6 | The Analysis & Presentation Studio reuses the Arrow/DuckDB pipeline, ECharts components, deck.gl maps and OLAP explorer; chart components should be built once and shared between dashboards (§G.1) and documents (§H.2) |

---

_This roadmap is a living document. Update item status inline (✅ Done, 🚧 In Progress, ❌ Blocked) and refresh the Effort × Impact matrices each quarter._
