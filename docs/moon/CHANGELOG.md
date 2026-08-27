# Changelog

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Gurobi](https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-50%25-green.svg)](https://coverage.readthedocs.io/)
[![CI](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

#### Hydra task aliases (`2026-08-27`)

- The documented `evaluation` and `sim_hpo` CLI aliases now select the real
  Hydra task groups (`eval` and `hpo_sim`) instead of composing a missing
  configuration group and failing before dispatch.

#### Train curriculum configuration (`2026-08-27`)

- Curriculum training now reads and injects graph settings under Hydra's
  task-scoped `train.env`, preventing the silent one-epoch fallback caused by
  looking for an uncomposed root `env` key.

#### TTOP feasibility and state-resume review (`2026-08-27`)

- TTOP now validates the full depot-to-depot trip, including a trailing or
  explicit return to the depot. Static scoring and the live action mask use a
  supplied road-distance matrix for travel time, with the same inclusive
  `1e-6` feasibility tolerance.
- TTOP defaults now live in the routing package rather than importing the
  simulation repository during environment construction, which removes an
  import cycle that made `TTOPGenerator` unusable.
- Resetting an initialized CVRPP/TTOP state preserves its remaining capacity
  and time resources, so resumed search keeps the original depot coordinate
  and resource accounting.
- TTOP now resolves to the CVRPP initial, context, state, and edge embedding
  components, allowing neural training and critic construction; the
  drift-report CLI also accepts `--problem ttop`.
- `TTOPEnv` had no `__init__` override, so it inherited `VRPPEnv`'s, which
  always builds a plain `VRPPGenerator`. `get_env("ttop",
  shift_hours=6.5, ...)` (including Hydra-composed overrides) silently
  swallowed the three temporal kwargs in `VRPPGenerator.**kwargs`; they
  never reached the environment and `_reset_instance` fell back to
  `get_default_temporal_params()` regardless of what was requested.
  Confirmed live before the fix (`env.generator` was a `VRPPGenerator`
  with no `shift_hours` attribute at all). Added
  `TTOPEnv.__init__`, building a `TTOPGenerator` and passing it through
  to `VRPPEnv.__init__` (which then skips its own generator construction
  since one is already supplied). Regression test added:
  `test_get_env_factory_builds_a_ttop_generator_not_a_plain_vrpp_one`.

### Added

#### TTOP multi-day simulator support and dual constraint enforcement (`2026-08-27`)

- **Simulation Problem Registration & Validation**: Registered `ttop` across `logic/src/constants/simulation.py`, `logic/src/configs/tasks/sim.py`, and `logic/src/pipeline/features/test/validation.py` / `engine.py`.
- **Temporal Parameter Loading & Simulation Context**: Automated retrieval of working shift duration (`shift_hours`), average vehicle speed (`avg_speed_kmh`), and per-bin service time (`service_time_h`) in `InitializingState`, forwarded across `SimulationDayContext` and `SearchContext`.
- **Dual Constraint Route Construction**: Extended `get_multi_tour` and `BaseRoutingPolicy` to enforce both vehicle waste capacity ($Q$) and shift time budget ($T_{\max}$) with depot returns.
- **Operational Metric & KPI Tracking**: Implemented additive tracking for `time_spent = (raw_km / avg_speed_kmh) + (ncol * service_time_h)` in `CollectAction`, `LogAction`, and summary tables in `logic/src/tracking/logging/modules/analysis.py`.
- **Unit Testing**: Added `logic/test/unit/pipeline/simulations/test_ttop_simulation.py` covering validation, dual constraint splitting, operational time calculation, and shift budget enforcement.

#### Grok joins in place of Opencode; continuous work queued for all three agents (`2026-08-27`)

- Grok replaces Opencode in the agent rotation. `git/messages/opencode_coauthor.msg`
  renamed to `grok_coauthor.msg`; `.agent/bus/AGENT_BUS.md`'s roster table
  updated with current assignments (it had gone stale — Agy's row still said
  "website visual design" while actually on TTOP simulator work). Opencode's
  prior website-interactive lane is left as historical record
  (`.agent/tasks/opencode-website-interactive.md`), not rewritten.
- Issue #60 (TTOP Hydra configs) reassigned Opencode → Grok, same scope.
- Queued a second issue behind each agent's current one, so all three can
  work continuously without needing a live check-in: #61 (Codex, after #58 —
  general bug/lint pass over `logic/src/`, following the existing but
  never-yet-run `.agent/skills/systematic-bug-hunt.md` convention, tracked
  via a new `docs/errors/ROADMAP.md`), #62 (Agy, after #59 — refresh the
  existing March-dated `docs/moon/review/{POLICY_IMPLEMENTATION_ANALYSIS,
  OPERATOR_IMPLEMENTATION_ANALYSIS,ACCEPTANCE_CRITERIA_ANALYSIS,
  OPERATOR_PROFIT_AWARE_FEEDBACK}.md` reports against current code and
  extend their coverage), #63 (Grok, after #60 — new
  `docs/moon/review/MODEL_IMPLEMENTATION_ANALYSIS.md`, no prior report
  exists for `logic/src/models/` vs. `bibliography/models/`). All four are
  the three codebase passes requested earlier in the session, now mapped
  1:1 onto the three agents instead of held back for a later batch.
- Fixed a real, pre-existing broken reference found while writing #61's
  brief: `README.md` and `.agent/skills/systematic-bug-hunt.md` both
  pointed at `docs/ARCHITECTURE.md`, which doesn't exist — the file is at
  `docs/moon/ARCHITECTURE.md`. Both repointed.

#### Temporal Team Orienteering Problem (TTOP) — foundation (`2026-08-27`)

- New problem type: TTOP = CVRPP's existing objective and per-trip vehicle
  capacity constraint, **plus** an independent per-trip time budget (travel
  time at `avg_speed_kmh` + `service_time_h` per bin). Both constraints hold
  simultaneously; neither replaces the other. Single-vehicle, multi-trip
  within a period (not a concurrent fleet — see
  `docs/moon/roadmaps/new_features.md` §E.8 for the future true-fleet
  variant, added for the heterogeneous-waste-stream scenario).
- `SimulationRepository.get_temporal_params()` (staticmethod, no args,
  mirrors `get_area_params`'s shape) returns `(shift_hours=7.0,
  avg_speed_kmh=35.0, service_time_h=0.025)`, plus a `load_temporal_params()`
  wrapper. `EnvironmentTag.TIME_BUDGET` added.
- `logic/src/envs/tasks/ttop.py`, `logic/src/envs/routing/ttop.py`,
  `logic/src/envs/generators/ttop.py` (`TTOP(CVRPP)`, `TTOPEnv(CVRPPEnv)`,
  `TTOPGenerator(VRPPGenerator)`), registered across `ENV_REGISTRY`,
  `GENERATOR_REGISTRY`, the `envs.problems` facade, and
  `utils.model.problem_factory.load_problem`. `logic/configs/envs/ttop.yaml`
  added, mirroring `vrpp.yaml`.
- KPI tracking made additive rather than fixed-schema: `cost_dict`/results
  dicts now carry a `"time"` key only when the problem tracks one (ttop),
  tolerated as absent everywhere else — the plug-and-play seam for future
  problem-specific KPIs.
- **Bug found and fixed, independent of TTOP**: `CVRPPEnv._step_instance`
  never called `super()._step_instance()` — `current_node`/`visited`/
  `tour_length`/`tour` never advanced past their reset values for any
  cvrpp rollout. Capacity tracking ran, but the agent's position and the
  episode's own state were frozen after the first action. Confirmed via
  direct before/after rollout test. Any committed cvrpp results predating
  this fix should be treated as suspect.
- Delegated the rest of TTOP: #58 (Codex — adversarial review of the RL-envs
  side + remaining training/eval pipeline gaps), #59 (Agy — wire the actual
  test simulator `pipeline/simulations`/`policies` to respect a time
  budget), #60 (Opencode — Hydra config tree completion, blocked in part on
  #59).

#### Models-vs-bibliography analysis started (#63) (`2026-08-27`)

- New `docs/moon/review/MODEL_IMPLEMENTATION_ANALYSIS.md`, templated on
  the policy report. First increment: paper↔path map (POMO is an RL
  trainer, not a `core/` model), AM 4/5, Pointer Network 4/5, POMO 4/5.
- First pass complete (15/15). Constructive/NAR/ACO family mostly 4/5.
  Improvement family weaker: DACT 3/5 (dual-aspect collapsed), NeuOpt 3/5
  (encoder ignores incumbent tour), N2S 2/5 (k-NN attention on `tsp_kopt`,
  not PDP). Sym-NCO 3/5: `problem_symmetricity_loss` exists but is never
  called from `shared_step` (`loss_ps` stays 0) — flagged for #61.

#### TTOP Hydra config tree (#60, training/eval side) (`2026-08-27`)

- `logic/configs/envs/ttop.yaml` rewritten to match the landed design
  (`TTOP = CVRPP capacity + independent time budget`). The previous header
  still described the discarded uncapacitated/VRPP-with-time-replacing-
  capacity draft, and the file had no `capacity:` key — `VRPPGenerator`
  would then default to `1.0` instead of CVRPP's `100.0`.
- `EnvConfig` now declares optional `shift_hours` / `avg_speed_kmh` /
  `service_time_h`. `ttop.yaml` lists them as `null` (Python default) so a
  Hydra CLI override such as `train.env.shift_hours=6.5` is a real key
  change rather than a struct-missing error.
- Task YAMLs document the Hydra *group swap* needed to actually load
  `ttop.yaml`: `envs@train.env=ttop`, `envs@eval.env=ttop` (plus the
  hardcoded `eval.env.name` / `eval.problem` lockstep), `envs@hpo.env=ttop`,
  `envs@meta_rl.env=ttop` (plus `meta_rl.env.name`), `envs@hpo_sim.env=ttop`,
  `gen_data data.problem=ttop`, `slurm` `problem=ttop`. `test_sim.yaml` and
  `logic/configs/policies/*.yaml` are untouched — blocked on #59.
- `tracking/*.yaml` have no per-problem metric lists; ttop's additive
  `"time"` KPI is code-side (`eval/engine.py`) and needs no tracking YAML.
- `docs/CONFIGURATION_GUIDE.md` env table and CLI examples now use the
  real group-override syntax (`envs@<task>.env=`) instead of the
  non-composed `envs=vrpp` form.
- After #59 landed: `test_sim.yaml` documents `sim.problem=ttop` as the
  switch (problem-level, not a per-policy "respects time budget" flag).
  Default stays `vrpp` so the paper's 30-day factorial is unchanged.
  Policy YAMLs were not given a new flag -- `BaseRoutingPolicy` already
  reads optional `shift_hours` / `avg_speed_kmh` / `service_time_h` and
  falls back to `load_temporal_params()`.

### Changed

#### Repository Documentation & Infrastructure Reorganization (`2026-08-27`)

- `markdown/` → `docs/moon/markdown/`, `review/` → `docs/moon/review/`,
  `reports/` → `docs/moon/reports/`, `mappings/` → `docs/moon/mappings/`
  (`git mv`, history preserved).
- `.agent/AGENTS.md` → `AGENTS.md` (repo root). `CLAUDE.md`/`GEMINI.md`
  pointer includes and every other real reference (`README.md`,
  `.agent/tasks/*.md`, `.agent/bus/AGENT_BUS.md`) repointed. The unrelated
  `docs/errors/ROADMAP.md` referenced from `.agent/skills/systematic-bug-hunt.md`
  is a different file and was left alone.
- `docs/moon/ROADMAP.md` (3128 lines) split by its existing §A–§I sections into
  `docs/moon/roadmaps/{analytics_interpretability,architecture,documentation,
  gui_ux,new_features,performance,studio,presentation_studio,
  publication_dissemination}.md`, mirroring the `Image-Toolkit` repo's
  `docs/moon/roadmaps/` pattern. Root `ROADMAP.md` keeps the header, an Anchor
  Index retargeted to the split files, and the Cross-Cutting Themes table.
- `public/` → `docs/private/`, and its nested `public/private/` subdir →
  `docs/private/html/`. Every real hardcoded reference to the old `public/...`
  paths was repointed: `logic/gen/{export_website_data,gen_paper_latex,
  gen_simulation_analysis,gen_presentation,gen_dataset_analysis}.py`, both
  copies of the analysis/presentation config JSONs (`logic/gen/json/` and
  `app/src/gen/config/`), the two generated markdown reports themselves, and
  the `.agent/` briefs. `docs/website/public/` is a separate, unrelated dir
  (the website's own Vite public assets) and was not touched. Historical
  `docs/moon/CHANGELOG.md` entries and dated `.agent/bus/*.md` journal entries
  citing the old paths were left untouched (append-only record).
- `git/CODE_OF_CONDUCT.md`, `git/CODEOWNERS` added (were missing).
- `.github/PULL_REQUEST_TEMPLATE.md` and `.github/ISSUE_TEMPLATE/{bug_report.md,
  feature_request.md,bug_agent.yml,feature_agent.yml}` added, adapted from the
  `Image-Toolkit` repo's templates to this repo's actual modules/labels.
- `.gitea/` and `.forgejo/` mirror-host dirs added (`workflows/{ci,docs,
  package-and-build}.yml` copied from `.github/workflows/`, GH-Actions-syntax
  compatible but unverified on a real runner — see #57; `sync-to-public.yml`
  deliberately not mirrored, it's GitHub-specific). `.gitlab/issue_templates/`
  and `.gitlab/merge_request_templates/` added; `.gitlab-ci.yml` deferred to
  #57 (different schema, needs a real GitLab runner to validate against).
- New `infra` label created for these and future infrastructure issues.
- `LICENSE.md` + `LICENSE.txt` merged into a single `LICENSE` file (Section A:
  AGPL-3.0, Section B: commercial terms), mirroring the `Image-Toolkit` repo's
  single-file license layout. `README.md` license links repointed.
- `git/README.md`, `git/config/project_labels.json` (reference snapshot of
  the actual label taxonomy), `git/messages/{claude,codex,gemini,opencode}_coauthor.msg`,
  and `git/hooks/install.sh` added. `Image-Toolkit`'s `git/scripts/` and
  `git/config/automation_rules.yaml` (a live Gemini-driven backlog-sync bot
  wired to secrets and a GitHub Actions workflow) were deliberately **not**
  ported — this repo's coordination mechanism is `.agent/bus/` + `AGENTS.md`,
  and copying that automation verbatim would be dead/misleading tooling. See
  `git/README.md` for the reasoning.

See #56 (closed, full change list) and #57 (open follow-up: mirror-host CI
runner verification).

### Added

#### Simulation Loop Architecture Diagram (`2026-08-27`)

- Added programmatic simulation loop figure generator `fig_simulation_loop` in `logic/gen/gen_paper_latex.py`, emitting `simulation_loop.png` into `Images/Results/Generated/`.
- Integrated `fig:sim_loop` into `Simulation Protocol` (`sec:protocol`) in `paper.tex`, illustrating the daily cycle: stochastic waste accumulation, observation asymmetry (noisy sensing $\tilde{\mathbf{f}}_t$ vs.\ exact ground-truth audit $\mathbf{f}_t$), modular three-stage policy pipeline from independent registries, physical execution, and multi-day state transition.

### Changed

#### Paper Abstract/Body Reconciliation (`2026-08-27`)

- Reconciled the manuscript body with the immutable conference abstract of
  record without altering the abstract or funding text. The Introduction now
  distinguishes the framework's registered classical/NCO adapter scope from
  the classical-only experiment and states the mandatory-selection result
  explicitly.
- Verified all stored benchmark rows against both summary CSVs and raw-log
  names: the 480 thirty-day runs and 174 selected ninety-day runs contain only
  the eight reported classical constructors. Related Work now states that the
  current results include no learned-solver observation, while retaining NCO as
  a valid framework topic and keyword.
- Defined mandatory selection as dynamic dispatch at the service-request level
  and scoped the unified baseline to the complete thirty-day factorial design.
  Future Work now calls for benchmarking the registered learned constructors
  under that same controlled protocol.
- Rebuilt the 34-page paper from source in an isolated directory. The build has
  no undefined citations or references, BibTeX/package/class warnings,
  duplicate destinations, overfull boxes, or LaTeX errors.

#### Paper Final Evidence and Citation Pass (`2026-08-27`)

- Reworked Experimental Evaluation, Discussion, and Conclusion prose to remove
  duplicated synthesis, causal language unsupported by the design, and the
  contradictory claim that CLS dominates Fast-TSP. The granular raw figures now
  carry an explicit integrity lead-in and no longer invite reconstruction of the
  filtered marginal results.
- Recomputed every retained closing-section quantity from the stored summary
  CSVs. Corrected whole-cell exclusion from 23 to 21 additional peer runs and
  replaced the claimed order-of-magnitude frontier increase with the supported
  roughly sevenfold increase in distance per avoided overflow.
- Gathered the study boundaries into a dedicated Limitations paragraph, moved
  the sensing caveat into Simulation Protocol, separated Future Work, restored
  LNCS `Sect.` references and en-dash compounds, and tightened the Discussion's
  network-geometry and horizon explanations from mechanisms to testable
  interpretations.
- Completed the incomplete PSOMA and ACO-HH publication records and corrected
  the Barnhart branch-and-price article from 1970 to its DOI-registered 1998
  publication metadata. The rebuilt 30-page PDF has no undefined
  citations or references, bibliography warnings, duplicate destinations,
  overfull boxes, or LaTeX errors.

#### Paper Figure Framing and Pareto Disambiguation (`2026-08-27`)

- Disambiguated constructor-level aggregate Pareto trade-off (`fig:pareto`) from the granular per-run policy trade-off (`fig:app-pareto`) in `paper.tex`, making explicit the difference between the integrity-filtered constructor means and the full-factorial scenario evaluation.
- Clarified captions on strategy trade-off and Empirical scenario heatmaps to explicitly document data-integrity exclusions and caveats.

#### Website Civic Design Theme (`2026-08-26`)

- Restored internal page component layouts while replacing all hardcoded colours with semantic token mappings from `tokens.css`, completing the cartographic and civic redesign across all pages.

### Changed

#### Simulation distance-matrix integrity (`2026-08-25`)

- Made per-run distance-matrix publication atomic. Parallel policy workers
  still share a scenario output path, but each now publishes one validated,
  complete square matrix through a same-directory temporary file and
  `os.replace`; workers can no longer concatenate or interleave CSV rows
- Repaired all 14 malformed 30-day matrix copies from the unanimously matching
  correctly sized artifacts for their network: one Rio Maior–100 copy, four
  Rio Maior–170 copies, and nine Figueira da Foz–350 copies. The directional
  road costs remain asymmetric by design
- Added regression coverage for shape rejection and concurrent publication

#### Empty-tour diagnostics (`2026-08-25`)

- Preserved the pre-solver mandatory-node set in daily logs when a constructor
  returns an empty tour. Previously that branch overwrote the set with `[]`,
  making selection failure and solver infeasibility indistinguishable in the
  stored evidence; issue #41's targeted rerun can now diagnose the boundary

#### MPVRPP paper tracked as a submodule (`2026-08-25`)

- Replaced the vendored `assets/papers/Simulation_Framework_for_the_MPVRP_with_Profits_in_Smart_Waste_Collection/`
  copy and its zip with a git submodule at
  `assets/papers/Simulation-Framework-for-the-MPVRP-with-Profits-in-Smart-Waste-Collection`
  pointing at the paper's own repo. The paper's edit history now lives with
  the paper instead of being duplicated and re-zipped inside this repo
- `logic/gen/json/paper_latex_config.json`'s `paper_dir` updated to the new
  path; `gen_paper_latex.py` re-verified idempotent against the submodule
  (all tables and figures regenerate byte-identical)

#### MPVRPP paper — final editorial, citation, and map pass (`2026-08-25`)

- Reworked the introduction, contributions, problem definition, NCO survey, and
  results commentary to remove repetition, synthetic contrasts, excessive
  parenthetical interruptions, and causal language unsupported by the design.
  The route model now has consistent per-bin capacities, explicit return-to-depot
  indices, a valid travel summation, and uniqueness constraints that exclude the
  repeated depot
- Restored the six-author/affiliation block from the matching Optimization 2026
  abstract, added PDF title/author metadata, and removed the generic dynamic-VRP
  illustration whose availability semantics did not match the paper's use
- Audited every cited key against its publication record. Corrected the Attention
  Model and Neural k-opt publication years, promoted the Joshi generalisation
  paper from its preprint record, completed the SIAM chapter metadata, normalised
  malformed DOI fields, and rewrote the multi-period citation context to match
  Zhang et al.'s scheduling formulation
- Replaced the distance-matrix embeddings with real coordinate maps for Rio
  Maior–170 and Figueira da Foz–350, composed reproducibly from the retained
  selected-bin/OSM-road artifacts under `public/figures/simulation/30d/`. The
  publication path no longer reconstructs geography from the disputed matrices
- The converged 22-page PDF has complete author metadata and no undefined
  references, bibliography warnings, duplicate destinations, or overfull boxes;
  the LaTeX generator is byte-idempotent and `logic/gen` remains ruff-clean

#### MPVRPP paper — Codex final editorial and data audit (`2026-08-25`)

- Rewrote the abstract, contribution statement, multi-period Related Work, improver interpretation, horizon caveat, scenario-effects discussion, and Conclusion into one consistent manuscript voice. Learned constructors are now explicitly framework capability/future work, not a result of the classical benchmark
- Corrected a missed balancing error in both the paper and website: demand-process and network-size marginals had retained unequal policy slices after the Gamma-3/$N=350$ exclusions. `balance_marginal` now supports scenario factors and includes every other policy stage in its slice key; a sixth generated LaTeX table reports the corrected like-for-like scenario marginals
- Downgraded the CLS-versus-Fast-TSP result from causal to descriptive. Although 224 configurations can be matched on constructor, selection, scenario, and demand realisation, upstream collected tonnage differs in 90 pairs and collected-bin count in 128; the stored experiment therefore does not isolate the improver stage. Also corrected the loss distribution: 21 of 22, not all 22, occur at $N=350$
- Removed stale hand-maintained registry totals from the paper figure. The website policy picker now reads literal registry decorator keys while continuing to count implementation files, so aliases and filenames no longer masquerade as configuration keys
- Website light/dark choice now initializes from the operating-system preference and persists in `localStorage`; the results view carries the same improver and scenario-balance caveats as the paper
- Eliminated duplicate PDF destinations from the current LaTeX/LLNCS combination with unique internal hypertext names

### Added

#### MPVRPP paper + abstract — build warnings cleared (`2026-08-27`)

- **`caption` "Unknown document class"** — not cosmetic. `subcaption` pulls in
  `caption`, which does not recognise `llncs` and therefore replaced the
  class's caption format with its own for *every* caption in the document:
  LNCS's bold label with period separator at `\small` ("**Fig. 1.**") had
  become a plain label with a colon at `\normalsize` ("Fig. 1:"). Dropped
  `subcaption` — it was used in one figure and nothing used `\subref` — and
  rebuilt that figure from minipages with hand-set (a)/(b) panel labels
- Caught only by building HEAD and the candidate *both from source* and
  rasterising the same page; the committed `paper.pdf` was stale and still
  showed the correct form, i.e. it failed in the reassuring direction. Do not
  use a committed PDF as a formatting baseline in this repo
- **`aliascnt` "The package is obsolete"** — raised by `llncs.cls` line 53,
  so the call is Springer's. Filtered with `silence` before `\documentclass`
  rather than patching a publisher's class. Not reproducible on this machine
  (TeX Live here ships the 2018 aliascnt, which does not warn), so the filter
  string was verified against a stub package emitting that exact message
- **Abstract document** (`Simulation-Framework-Abstract`): `geometry`
  over-specification in the h-direction (`paperwidth` + `inner` + `textwidth`
  + `outer` fixes it three times over; dropped the redundant `textwidth`) and
  `hyperref` draft mode (`[draft]` → `[hidelinks]`, which keeps the page
  identical but lets hyperref emit metadata and anchors). Both lines sit in
  the conference template's "do not modify" block, so the render was verified
  pixel-identical at 150 dpi before and after rather than assumed

#### MPVRPP paper — conference abstract restored as the version of record (`2026-08-27`)

- The paper's abstract is now the one presented at conference, reproduced
  verbatim from `assets/papers/Simulation-Framework-Abstract/main.tex` and
  verified byte-identical after whitespace normalisation (#55)
- A comment above `\begin{abstract}` records that it is the version of record
  and that where it and the body disagree, the body moves — the failure mode
  being a later pass quietly editing the abstract back toward the body
- Carried the conference keyword list across via LNCS `\keywords`; the paper
  had none before
- The swap inverts a load-bearing claim: the replaced abstract ended "Learned
  constructors share the framework interface but are not bench-marked here",
  which set up both Related Work's "the present experiment does not benchmark
  them" and Future Work's promise to exercise learned constructors against
  this baseline. The abstract of record instead lists NCO among the algorithms
  the paper adapts and benchmarks. Reconciliation delegated as #53, and it
  moves the body, not the abstract
- Delegated alongside it: #54 to Agy — a simulation-loop figure (`sec:protocol`
  is the paper's core contribution and is entirely prose, while the abstract of
  record makes the framework the headline contribution), plus the #51 figure
  items the caption pass did not reach (house style across the two image sets,
  colour-blind and greyscale safety, two text-only results)
- Board hygiene: #50, #51 and #52 had never been added to project 31 —
  `gh issue create --label` does not put anything on the board. Backfilled

#### MPVRPP paper — Related Work restructured around exact and heuristic methods (`2026-08-27`)

- Dropped `\subsection{Neural Combinatorial Optimization}`. It gave a whole
  heading to the one family this paper does not benchmark, while the two it
  actually exercises had none
- New untitled Related Work intro distinguishes the families by what each
  guarantees against what it costs — exact, heuristic/meta-heuristic,
  hyper-heuristic, matheuristic, machine-learning — and condenses NCO into it.
  Written so as *not* to repeat the Introduction's existing family list: the
  Introduction names them to motivate the paper, the Related Work intro
  distinguishes them
- New `\subsection{Exact Methods}` (compact vs decomposition lineages, and the
  three developments that made branch-price-and-cut practical for routing) and
  `\subsection{Heuristic Methods}` (trajectory / population / hyper-heuristic
  levels, then the waste-collection-specific heuristic literature)
- Ordering is intro → MPVRPP → Exact → Heuristic, putting the methods survey
  adjacent to the Methodology that implements it. Each new subsection
  forward-references `sec:constructors` rather than re-explaining ng-route
  pricing or Farkas pricing — Related Work cites the literature's positions,
  Methodology describes this paper's implementations
- Deleting the NCO subsection would have silently dropped eight references,
  which BibTeX does not warn about; all eight were carried into the condensed
  mention and the cited-key set was diffed to confirm no loss (25 → 33)
- Fixed `Ryan1981ANIP` in `mybibliography.bib`: its booktitle was jammed into
  the title in capitals, which BibTeX flagged the moment the entry was cited

#### MPVRPP paper — Discussion subsection, and the closing-sections work split (`2026-08-27`)

- New `\subsection{Discussion}` under Experimental Evaluation (#52). Written
  as mechanism and interpretation rather than as a synthesis of the results
  subsections, because the Conclusion already carries the headline result
  verbatim and a summarising Discussion would only duplicate it
- Its substantive additions: the three policy stages differ in *kind* and not
  merely in effect size (a trade-off curve, a robustness-and-cost story, and
  an experimental-design finding respectively); the remote depot as the
  geometric mechanism that predicts the selection ordering a priori, stated
  with a falsifiable bound on its own generality; the horizon result read the
  same way (efficiency is route geometry, overflow an accumulated per-day
  hazard); operating point over mechanism, given a sharply curved frontier;
  and the three integrity problems named once as one failure family — a
  filter correlated with the compared factor — instead of three cautions
- Every number reused from already-generated text, none computed in prose
- Delegated: #50 to Codex (prose review of Experimental Evaluation and
  Conclusion/Limitations/Future Work), #51 to Agy (charts/plots in both
  sections). Both issues carry the concrete defects already located, by line
- Recorded in the same pass: the Overleaf sync moved three appendix figures
  into the main body without their data-integrity lead-in, and introduced a
  body sentence claiming CLS "dominates Fast-TSP on basically almost all
  simulation scenarios" that contradicts §Route Improvers and #49
- The user's pending Overleaf working-tree edits were committed separately
  first (`63dc332` in the paper submodule) so no agent is credited with them

#### MPVRPP paper — appendix with presentation figures and CLS results table (`2026-08-25`)

- New `\appendix` in `paper.tex`: the Pareto front, strategy trade-off, and
  Empirical-only per-scenario heatmaps (30d), the policy×scenario overflow
  and efficiency heatmaps (90d, landscape), and the full CLS-only results
  table (30d, landscape) from the results presentation. Pulled from their
  generator source (`public/figures/simulation/`, and
  `gen_presentation.py`'s own `render_hier_table_image`), not screenshots
- Four of the six items predate the degenerate-run exclusion and still carry
  the four truncated SWC-TCF runs (`tab:excluded`) as raw values — kept
  as-is per explicit direction, with each caption stating exactly where the
  artefact appears (e.g. the 2,168-overflow LA/CLS/Gamma-3/Figueira-da-Foz
  cell) rather than silently reproducing an unqualified number the main text
  excludes everywhere else

#### MPVRPP paper — Methodology and Results rewritten against the real experiment (`assets/papers/Simulation_Framework_for_the_MPVRP_with_Profits_in_Smart_Waste_Collection/`)

- **The paper described two different studies.** Methodology committed to eight route constructors (ALNS, HGS, SANS, PG-CLNS, PSOMA, BPC, SWC-TCF, ACO-HH) x three selection strategies x two improvers; Results discussed an Attention Model / gurobi / look-ahead comparison on Gamma-1/2/3 at N=20…317 over 31/93/365 days that nothing in the repo reproduces. `public/global/simulation/simulation_summary{,_90d}.csv` holds exactly the design Methodology promised, so Results, the stale Data subsection and the incoherent Baselines subsection were rewritten from it
- **Methodology completed**: the four placeholder paragraphs (BPC, SANS, PG-CLNS, PSOMA) written from `logic/src/policies/` and `bibliography/`; the sentence that ended mid-clause finished; Look-Ahead described for the first time despite being a third of the experimental grid; the CF70/CF90 and SL1/SL2 variants used throughout the results finally defined; a new **Simulation Protocol** subsection stating the paired demand realisations, the sensing-noise gap, the overflow-flag vs. kg-lost distinction, and the single-vehicle/single-depot restriction the abstract had left implicit
- **Results rebuilt from data**: every table and figure `\input` from generator output instead of hand-maintained. Findings that survived the cleaned data — selection matters more than construction (48% efficiency range across selection variants vs. 24% across all constructors); the efficiency/service trade-off is *not* a single dial (efficiency falls monotonically, overflow risk does not); constructor overflow *medians* are 4.0 for seven of eight, so the spread in the means is tail behaviour; CLS beats Fast-TSP on 202 of 224 matched pairs but all but one of the 22 losses are at N=350 under PSOMA/SWC-TCF/HGS; BPC is *not* the most expensive method (three meta-heuristics cost more per run at N=350); overflows accumulate at a roughly constant rate across horizons rather than compounding
- **Data integrity is a section of the paper, not a silent filter**: the three degenerate SWC-TCF runs get their own table with the whole-cell exclusion justified, and the 90-day sample's conditioning on 30-day Pareto-front membership is disclosed before any 90-day number appears
- Bare `&` escaped in the Vidal entry of `mybibliography.bib`, which broke bibtex as soon as that entry was first cited

#### Paper LaTeX generator (`logic/gen/gen_paper_latex.py`)

- New sibling to `gen_simulation_analysis.py` / `gen_presentation.py`: same summary-CSV schema, theme and Jinja machinery, emitting `.tex` fragments (`jinja/paper_results_table.tex.j2`, `json/paper_latex_config.json`) plus the figures they reference into the paper's own `Images/Results/Generated/` tree. Five tables (constructors, strategies, paired improvers, paired horizons, excluded runs) and four figures (Pareto, strategy trade-off, per-pair improver delta, runtime scaling)
- **Degenerate-run detection**, centralised so the paper, the markdown reports and the website cannot disagree: flags runs whose collected tonnage falls >20% below their scenario-cell median. Justified rather than tuned — across 576 rows the shortfall distribution has median 0.00 and 99th percentile 0.057, then jumps to three rows above 0.20. Collection-day count is explicitly *not* used as a signal: it counts collection days, not elapsed days, and several of the best policies collect on 15 of 30 days by bundling well
- **Whole affected scenario cells are dropped, not just the offending rows.** All three degenerate runs are SWC-TCF's, so removing only them would average SWC-TCF over the scenarios it did not fail while averaging rivals over those too — the same selection bias the module refuses to accept in the 90-day data. Costs 21 further runs and restores a uniform n=57
- **Refuses to emit a cross-constructor 90-day aggregate.** Only 30-day Pareto-front policies were re-run at 90 days, so the 90-day sample is conditioned on the outcome variable; the horizon table pairs each configuration against itself instead, documented as a conservative estimate rather than an unbiased one
- Means reported beside medians throughout: removing one run in ~90 moved Service-Level (SL2) from 9.3 to 1.4 mean overflows

#### Report/deck generators revived from `archive/` into `logic/gen/`

- `gen_dataset_analysis.py`, `gen_simulation_analysis.py`, `gen_presentation.py`, `report_utils.py` and their `jinja/ json/ style/ js/ images/ svg/ links/ templates/` assets moved back out of `archive/gen/` to `logic/gen/`, alongside `gen_dist_matrix.py` / `export_for_studio.py` / `export_loss_landscape.py`. The scripts resolve assets from `__file__`, so only path strings in docstrings, `app/src/gen/*` provenance comments and `ReportStudio.GEN_SCRIPTS_DIR` needed rewriting
- `logic/` is linted where `archive/` was not: import blocks sorted, `zip(strict=True)` on the equal-length local-search point pairs, one dead colour constant dropped; `C901` waived per-file for the three generators in `pyproject.toml`

#### Multi-agent coordination (`.agent/`)

- `.agent/bus/` (index + dated daily log, modelled on the Image-Toolkit layout) and `.agent/tasks/` with a brief per agent: Codex as reviewer/co-lead, Agy on the website design system, Opencode on interactive/3D visualisation. GitHub issues #40–#47 track the same work

#### Interactive simulation website (`docs/website/` + `logic/gen/export_website_data.py`)

- **Website data generator** (`logic/gen/export_website_data.py`) emits four JSON documents into `docs/website/public/data/` so the site never hand-copies a number: `pipeline.json` (the policy configuration space enumerated from the plugin registries — 31 selection strategies, 8 construction families / 96 constructors, 30 improvers, with the benchmarked subset marked), `results.json` (the balanced 30-day aggregates, Pareto front, paired improver deltas, horizon pairs and constructor × scenario heatmap), `bin_fills.json` (real per-bin per-day fill levels plus recovered policy-independent daily increments), and `routes.json` (per-day tours with a 2-D layout by classical MDS of the real road-distance matrices). It imports and reuses `gen_paper_latex`'s `split_degenerate` / `drop_affected_cells` / `balance_marginal` / `improver_pairs` / `pareto_front`, so the website and the paper share one definition of "clean data" and one set of exclusions
- **A new `/simulation` route** with four interactive components: a three-stage policy-pipeline picker (V1); a Last-Minute bin-selection widget that re-simulates the rule on the recovered increments so dragging the CF threshold moves the selected set and the overflow count (V2); a 30-day route scrubber with a lazy-loaded three.js "routes stacked in time" view — x/y the MDS embedding, z the day — code-split so three.js stays out of the initial bundle, with a WebGL-free 2-D SVG fallback (V3); and results charts with the caveats built in (Pareto front, paired CLS-vs-Fast-TSP with the N=350 loss breakdown, constructor × scenario heatmap that refuses to pool Gamma-3 with Empirical) (V4)
- **Paper figure**: `fig_fill_trajectory` in `gen_paper_latex.py` renders the multi-period mechanism behind the paper's central claim — three bins' fill levels over the horizon under CF70 vs CF90 — and `paper.tex` gains `Fig.~fill`, the first figure that shows the fill-and-collect dynamic rather than a static aggregate
- The two counts in the brief/paper prose that do not match the filesystem (32 selection / 33 improvers / "seven families") are surfaced on the agent bus rather than silently "corrected": the generator emits the registry truth (31 / 30 / 8)

#### Analysis & Presentation Studio — native §H engine (`app/src/gen/`)

Full native port of the archived `gen` pipeline into the Studio — no Python in the loop:

- **§H.0 foundation**: gen images/SVGs bundled as typed assets; the five JSON configs + reference-links XML as typed TS config modules (themes merged with the mplstyle sheets); Rust `write_binary_file` / `list_files_recursive` / `load_npz_flat` commands
- **§H.1 data engine** (`gen/data/`): output-tree filename decoding (metadata-driven), horizon CSV loading, scenario detection + config narrowing, CF/SL variant aggregation, Pareto fronts, hierarchical results matrix + all markdown tables; dataset extended stats (median/variance/quartiles/IQR/fences/binned mode), Gaussian KDE, raw NPZ waste loading
- **§H.2 chart library** (`gen/charts/`): ECharts ports of every figure — Pareto scatter with step fronts + filled/open markers + symlog, KPI bars/combined grid with min–max whiskers + hatch decals, km violins, policy×scenario & scenario×constructor heatmaps (RdYlGn, symlog norm), strategy/improver bubbles with label dodging, constructor ranking, radar, improver delta, horizon comparison/delta/ranking, and the full dataset-analysis chart family; headless render-to-PNG (preview ≡ export)
- **§H.5 report builders** (`gen/report/`): both markdown reports as native TS templates with figure/table numbering, `<figure>` wrapping, TOC/scope construction, parse-output→CSV mode, and scatter-mode bin maps with stripped-decimal repair
- **§H.3/§H.4/§H.6 deck builder** (`gen/deck/`): 21-slide PPTX via pptxgenjs — cover/agenda/diagram/equation/figure slides, native-shape diagrams (pipeline chevrons, policy grid, taxonomy, DoE tree, simulator scene), seeded SVG illustrations (QA route, B&B tree, LS operators, knapsack, framework objective, metaheuristic family panels) with reference-link fetch mode, MathJax LaTeX equations (image embed + ported plain-text fallback), the results table as a **native PPTX table** (merged spans, global best highlighting, split partitioning), speaker-script DOCX (`docx`) and results XLSX (`exceljs`)
- **§H.7 (partial)**: Report Studio **Native/Legacy engine toggle** — native runs in-app with live progress log and artefact chips; legacy still spawns the archived scripts
- Data-engine parity verified against the Python reference (filename decoding, Pareto, aggregation, tables, numpy-style percentiles, KDE normalisation, equation fallback)

- **§H.5 interactive HTML exports** (`gen/report/interactiveHtml.ts`): self-contained single-file chart pages with the ECharts bundle inlined (no CDN, open offline) — pareto scatter with All-Points/Pareto-Front views, strategy bubble, policy×scenario heatmap with an Overflows/kg-km toggle, NPZ mean-vs-std scatter, per-city waste-distribution bars, and the city & network comparison grid; both report generators write them under the private dirs (per-horizon when multi-horizon) and the markdown gains the ported "Interactive version" links + "Interactive Charts" footer sections — replaces the archived Plotly-CDN HTML files and `pareto_buttons.js`

- **§H.6 HTML deck export** (`gen/deck/htmlDeck.ts`): the 21-slide deck as one self-contained offline HTML slideshow — keyboard/click navigation with hash-synced index and slide counter, `N` toggles a speaker-notes panel fed from the per-slide scripts, figures/logos/icons/equations embedded as data URLs, native-shape diagrams reproduced in HTML/CSS (chevron pipeline, policy grid, taxonomy, DoE tree, objective flow, bin/truck scene), and the results table as a real HTML table with merged header spans and best-cell highlighting; exposed as an "HTML slideshow" toggle on the native deck build

- **§H.6 PDF deck export** (`gen/deck/pdfExport.ts`): each HTML deck slide is rasterised off-screen with the real WebKit engine (html-to-image) at fixed 1600×900 geometry and assembled into a compressed 16:9 landscape PDF via jsPDF, written beside the PPTX — "PDF deck" toggle on the native build

- **§H.7 deck preview canvas** (`components/gen/DeckPreview.tsx`): "Preview deck" on the Report Studio presentation tab renders the deck in-app through the same slide builder as the HTML/PDF exports — shadow-root style isolation, true 16:9 geometry with fit-to-width scaling, arrow-key pagination, slide counter, speaker-notes strip, rebuild-from-settings

- **§H.5 self-contained HTML report export** (`gen/report/htmlReport.ts`): converts the generated markdown reports into standalone styled HTML documents (GFM tables, figure blocks) with every referenced figure inlined as a data URL — "HTML report" toggles on the native dataset/simulation tabs

- **§H.7 report preview** (`components/gen/ReportPreview.tsx`): after a native report completes, "Preview report" renders the written markdown through the shared HTML builder (figures inlined) in a sandboxed iframe — scrolling document preview with refresh

Remaining §H scope: document spec + override patch layer, native OMML equations, report PDF export, inspector/direct-manipulation editing.

#### Report Studio — §H interim launcher

- **Report Studio** Launch-section page (`report_studio` mode): generates the dataset / simulation analysis reports and the results presentation deck from the Studio — three tabs assembling the full CLI for the archived `gen_dataset_analysis.py`, `gen_simulation_analysis.py` (report + raw-output→CSV parse modes) and `gen_presentation.py` (PPTX / speaker-script DOCX / XLSX), with persisted forms (`useReportGenStore`), command preview, live log tail, and post-run artefact path chips (§H interim)

#### Archived Streamlit dashboard (`archive/streamlit_ui/`)

- The Streamlit UI removed in `31b1b19` recovered from git history (parent commit) and archived frozen/reference-only like `archive/gen/`: all 79 deleted files (`logic/src/ui/`, `logic/src/utils/ui/`, `logic/configs/ui/`, `dashboard_entry.py`, dashboard constants, `.streamlit/`, `tools/ui/`, UI unit tests, `remove_ui.py`) preserved at their original repo-relative paths, plus a README documenting provenance and layout; `archive/` stays excluded from ruff/pytest/packaging

#### Headless §H generation runner (`app/scripts/gen-headless.ts`)

- Runs the Studio's native `src/gen` pipeline (simulation analysis figures/markdown/interactive HTML + PPTX deck) **without the GUI**: vite-node + jsdom + node-canvas for ECharts rasterisation, resvg for SVG→PNG (injectable via new `setSvgRasterizer` hook), and a Node fs implementation of the Tauri `invoke` commands (`scripts/headless/tauri-core-shim.ts`); `npx vite-node --config scripts/headless/vite.config.ts scripts/gen-headless.ts -- [--report] [--deck] …`

#### Studio documentation (`app/docs/`)

- New in-app docs dir: `README.md` (index), `ARCHITECTURE.md` (process model, reorganised source layout, stores, Rust command surface, IPC conventions), `PAGES.md` (page-by-page guide), `GEN_PIPELINE.md` (native §H engine), `TESTING.md`, `DEVELOPMENT.md`

#### Studio test suites (`app/tests/`, `app/cypress/`)

- **Vitest** unit tests (Pareto front/step-line, symlog, chart log-scale helpers, process progress markers, run-path/URI resolution), component tests (KpiCard, StatusPill via Testing Library/jsdom), and integration tests (native §H.1 data engine pipeline: filename decoding → filter → aggregate → context → Pareto; global filter store ↔ chart transforms) — `npm test`, 70 tests
- **Cypress** e2e tests (shell boot, sidebar navigation with hash sync, `useHashSync` deep-links, hash filter restore, Ctrl+K command palette) running against `vite dev` with the Tauri v2 IPC internals stubbed — `npm run test:e2e`
- npm scripts: `test`, `test:unit`, `test:component`, `test:integration`, `test:coverage`, `cy:open`, `cy:run`, `test:e2e`

### Changed

- **Presentation & analysis deliverables now regenerated with the archived Python gen pipeline** (`archive/gen/`, user preference for its figures/tables/diagrams): `gen_simulation_analysis.py` rebuilt `public/figures/simulation` + `public/simulation_analysis.md`, `gen_presentation.py` rebuilt `assets/windows/wsmart_route_results.pptx` (21 slides); the native §H engine stays available in Report Studio and keeps parity fixes

### Fixed

- **Results table (slide 17) best-cell ties**: `compute_global_best` picked a single best column per row even when several tied on the optimal value (e.g. five constructors all at 1.0 overflows in the FFZ-350/Empirical/SL row — only one was highlighted). Now returns every column tied for best (within a small float tolerance) and the table renderer highlights all of them; fixed in both the PPTX table image and the Excel export's cell-fill logic.

### Changed

- **Q&A appendix rewritten for conceptual clarity** (`archive/gen/appendix_notes.md`): added a new "Concepts & Terminology" primer up front (local/neighborhood search, trajectory- vs. population-based search, evolutionary algorithms, memetic search, swarm intelligence, metaheuristic vs. hyper-heuristic, exact-method branching/cutting/pricing) and a "Big picture" framing paragraph ahead of every route constructor's technical detail; removed all code file paths, line numbers, and class/function names throughout (kept the substantive algorithmic/mathematical content, citations, and limitations) since those aren't useful for live Q&A.

### Added

- **Speaker-script Q&A appendix** (`archive/gen/appendix_notes.md`, ~9,000 words, appended to `assets/windows/wsmart_route_results.docx` after the per-slide scripts via `append_markdown_appendix`): detailed prep notes researched directly from `logic/src/policies/` and `bibliography/policies|operators` for every route constructor (BPC, SWC-TCF, HGS, ALNS, SANS, PG-CLNS, PSOMA, ACO-HH — the most in-depth subsection, covering exact formulations/algorithm loops/operator inventories/VRPP-profitable-subset handling/citations/limitations for each), the two route improvers (CLS, Fast-TSP) and core search operators (2-opt, Or-opt, swap/relocate, node exchange), additional depth on the Last-Minute and Service-Level mandatory-selection strategies plus a LA/LM/SL comparative-properties note, simulator behavior (daily state transition, capacity enforcement, stochastic waste-generation models, fairness/seeding mechanism), data-driven "Additional Result Conclusions" computed directly from the simulation CSVs (Pareto-front membership rates, execution-time spread, horizon-robustness ratios with an outlier caveat), and expanded, code-grounded limitations/future-work remarks. A small hand-rolled Markdown-to-Word renderer (`_add_markdown_paragraph`/`_split_inline_markdown`/`append_markdown_appendix`) handles headings, `**bold**`, and `` `code` `` spans — not a general CommonMark implementation, but sufficient for this content.

### Fixed

- **Presentation feedback fixes, round 7** (`archive/gen`; deck + speaker script regenerated): slide 7's equation gains a bold "Look-Ahead (LA):" header above its 6 steps (matching the existing "Service-Level (SL):"/"Last-Minute (LM):" labels); the slide's figure reverts from the LA flowchart back to the knapsack-equivalence illustration

- **Presentation feedback fixes, round 6** (`archive/gen`; deck now 20 slides, figures + deck + speaker script regenerated):
  - removed the KPI-bar-charts slide (former slide 13); the underlying `kpi_combined.png` generator is unchanged and still feeds the markdown report
  - Pareto front (slide 12) restyled: fronts are keyed by **region only** (not distribution) so RM-100/RM-170/FFZ-350 draw with the same darker colour + linestyle (solid purple / dashed bronze / dotted slate) in both the Empirical and Gamma-3 panels — previously a global scenario index gave the same region a different colour per distribution; the point palette (LA/LM/SL colours) is untouched and the front colours are deliberately distinct from it; legend collapses from 6 scenario entries to 3 region entries
  - **Look-Ahead selection**: read `logic/src/policies/mandatory_selection/selection_lookahead.py` and replaced slide 7's 2-line LA equation with the full 6-step derivation (flag $\mathcal{M}_0$ → check empty → simulate collection → compute next collection day $nc$ → check $nc\leq t$ → bundle bins overflowing before $nc$), matched 1:1 to `select_bins`; added a new native flowchart diagram (`lookahead_flow.png`, ports the reference flowchart) illustrating the same 6 phases with the branch/loop structure
  - **equations are now images, not OOXML**: replaced the pandoc→OMML `<m:oMath>` embedding (blank/uneditable in LibreOffice and the PowerPoint web app) with `render_equation_image`/`render_glossary_image` — transparent PNGs rendered via matplotlib mathtext (no LaTeX install needed), sized so points map 1:1 to the embedded picture; the VRPP (slide 3) and mandatory-selection (slide 7) equations both switched
  - added a **variable glossary** textbox below both equation slides (`render_glossary_image`, word-wrapped per-column, dynamically sized to its wrapped content so rows never overlap) explaining every symbol (e.g. $nc$, $\delta_{i,d}$, $\tau$, $\mathcal{A}_d$)
  - **speaker script regenerated to match the new deck** (`--speaker-script`, 20 slide scripts): slide 4's KPI list expanded from 5 to all 9 simulator metrics (overflows, bins collected, kg, km, kg/km, RL reward, profit, run execution time, days with a collection tour); Exact Methods (8) / Meta-/Hyper-Heuristics (9) / Route Improvers (10) scripts compressed to 2–3 terse bullets each; slide 12's script updated to describe the new region-styled fronts instead of the old per-scenario colouring

- **Presentation feedback fixes, round 5** (`archive/gen`; figures + deck regenerated): slide-8 BPC diagram loses the "Branch-and-Price-and-Cut" heading (single-line title) and renders larger (narrower/taller figure aspect); slide-11 map titles read "N Plastic Bins"; slide-18 best half-cells get a light-green background behind the bold green values

- **Presentation feedback fixes, round 4** (`archive/gen`; figures + deck regenerated):
  - slide 5: chevron captions reduced to one phrase each; selection caption now "is there a collection today?"
  - slide 7: knapsack "Today's Route (capacity Q)" label moved clear of the sack outline
  - slide 8: equations replaced by a new native **BPC price/cut phases** diagram (`bpc_phases.png`: column-generation loop between the restricted master problem and pricing subproblem, plus a cutting plane separating the fractional LP optimum) shown beside the existing B&B tree
  - slide 9: "Trajectory-Based"/"Population-Based" panel titles matched in size and stripped of the algorithm examples (shared axes geometry so both render equally)
  - slide 11: per-scenario maps now plot the **actual simulated bin selections** — RM-100/RM-170 from `selected_coordinates/coordinates{N}_plastic[riomaior].xlsx` (depot row dropped), FFZ-350 from the plastic bins of `out_info[figdafoz].csv` indexed by `bins_selection/graphs_350V_1N_plastic.json` — instead of falling back to the all-bins maps
  - slide 12: axis names/titles enlarged again, legend moved further below, panels enlarged via tightened margins
  - slide 13: y-labels and panel titles enlarged again
  - slide 15: every fontsize slightly increased (ticks, titles, side labels, colorbar)
  - slide 16: colorbar labels enlarged again
  - slide 18: all table cell values bold (best values remain bold green)

- **Presentation feedback fixes, round 3** (figure/table readability, `archive/gen`; figures + deck regenerated):
  - pareto scatter (slide 12), strategy bubble (14), improver bubble (17): axis-name and panel-title fontsizes increased
  - KPI combined (13): y-labels/titles increased; only the FFZ-350 x tick labels shrunk slightly so they no longer collide (RM labels keep their size)
  - per-scenario heatmaps (15): panels widened (3.8″→4.6″ each) + slightly smaller x tick labels so the LA/LM/SL × FTSP/CLS labels separate
  - 90-day policy×scenario heatmaps (16): colorbar labels ("Overflow Count", "KG / KM Efficiency") enlarged
  - radar figures (19): axis labels padded farther from the ring (pad 28→72) and the all-constructors legend moved right, clear of the radar
  - results table (18): overflow now stacks **above** kg/km in each cell (full cell width per value — the side-by-side layout overflowed), row-label band narrowed (27%→16% of width) in favour of constructor/data columns, table image spans the entire slide width, corner note/speaker script updated to TOP/BOTTOM

- **Presentation feedback fixes, round 2** (applied to `archive/gen/gen_presentation.py` / `gen_simulation_analysis.py`, with parity fixes in the native TS deck builder):
  - slide 4: the Hyper-Heuristics→Simulator connector sources at the Hyper-Heuristics box again (the TS port had shifted its bounding box onto the Exact/Meta rows, leaving Hyper-Heuristics arrowless)
  - slide 4 taxonomy: full algorithm names + acronyms ported to the Python deck; items middle-anchored and Meta-Heuristics boxes taller (0.45" on a 0.49" pitch) so PG-CLNS/PSOMA no longer overflow their borders
  - slide 5: tour arrows now use the reference deck's exact connector bounding boxes **and flips** (the archived port had dropped flipH/flipV, scrambling the directions) — vehicle → 50 → 60 → 87 → 30 → 80 → vehicle, in orange; bin percentages white; the reference's rotated-brace geometry restored (the port also dropped `rot=90`, which is what produced the full-height grey line)
  - Python figures: radar text white, pareto axis/tick/legend text bold, cover logos contain-fitted via PIL

- **Presentation deck feedback fixes** (native §H deck + chart library; deck regenerated to `assets/windows/wsmart_route_results.pptx`):
  - cover-slide institution/conference logos contain-fitted to their natural aspect ratio instead of stretched into fixed boxes
  - slide 4 algorithm taxonomy spells out full names with acronyms (e.g. Hybrid Genetic Search (HGS), Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS))
  - slide 5 simulator diagram matches the reference deck: orange collection-tour arrows (truck → 50 → 60 → 87 → 30 → 80 → truck) instead of miscounted dark connectors, white bin-percentage text, and the stray full-height grey line replaced by the intended horizontal underbrace beneath the constructor→improver chevrons (`rightBrace` rotated 90°)
  - figure font sizes increased across the chart library (axis names 16 bold, axis labels 13, panel titles 17); slide-12 Pareto plots use bold axis/legend text
  - Policy Performance Radar title/legend/axis text now white
  - chart legends with decorative entries (variant colours, region shapes, front dashes) now actually render: placeholder series back the legend names, and the front dash icon uses closed subpaths (a stroke-only `path://` icon has a zero-height bbox that NaNs the legend layout)
- **Studio `useHashSync` deep-link clobbering under StrictMode**: the hash write-effect ran with a stale render closure between the restore effect's double-invocation, rewriting `#m=<mode>` deep-links back to the default page; it now reads fresh store state (caught by the new Cypress deep-link spec)
- **Studio `symexp` inverse**: was not the mathematical inverse of `symlog` for non-default `linthresh` values (behaviour unchanged for the default threshold the app uses)

### Changed

- **Studio source tree reorganised into logical subdirectories** (`app/src/`): `utils/` split into `charts/`, `duckdb/`, `graph/`, `map/`, `benchmark/`, `process/`, `training/`, `sim/`, `runs/`, `app/`; `hooks/` into `brush/`, `process/`, `files/`, `app/`; `components/analysis/` into `benchmark/`, `topology/`, `telemetry/`, `routes/`, `explorer/`, `training/`; `components/monitor/` into `live/`, `eval/`, `process/` — all relative imports rewritten, no behaviour change

- **Archived the `logic/gen` report/deck pipeline to `archive/gen/`** (frozen, bugfix-only per §H): `gen_dataset_analysis.py`, `gen_simulation_analysis.py`, `gen_presentation.py`, `report_utils.py` and their jinja/json/style/js/templates/images/svg/links assets; `gen_dist_matrix.py` and the Studio-support export scripts (`export_for_studio.py`, `export_loss_landscape.py`) remain in `logic/gen/`; ruff excludes `archive/`

#### Studio parity for retired GUI/UI modules

- Training Hub **Meta-RL** mode launching `main.py meta_train` with meta-strategy, meta-LR, history, batch, and step fields (PySide6 "Meta-Learning" tab parity)
- **System Tools** page (Files section) porting the GUI's File System Tools — update / delete / cryptography — and the Program Test Suite runner (`main.py file_system …` / `main.py test_suite …`) with persisted forms, command preview, live log tail, and destructive-delete warning

### Removed

- **PySide6 GUI module** (`gui/`) and the `gui` CLI command (parser, dispatch, tests, PyInstaller spec) — superseded by WSmart-Route Studio (`app/`)
- **Streamlit dashboard** (`logic/src/ui/`, `logic/src/utils/ui/`, `dashboard_entry.py`, `logic/configs/ui/`, `.streamlit/`, `tools/ui/`, dashboard constants, `remove_ui.py` packager) — superseded by the Studio
- PySide6 / Streamlit / folium dependencies; the `gui` extra is now a `viz` extra (matplotlib, seaborn, plotly)

### Changed

- `infra/global/docker/Dockerfile` now builds the Studio into Linux bundles (`.deb` / `.AppImage`) instead of serving the Streamlit dashboard
- Docs, justfiles, CI workflow, agent rules/workflows, and packager configs updated for the Studio-only interface

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirty-eighth pass (§G.1 + §G.6 + §G.11 + §G.15 + §G.17 + §G.18 + §D.7 + §D.8)

Two-hundred-and-thirty-eighth pass closes residual tracker / portfolio brush-label
parity and data-gen completion toast handoffs after pass 237 covered process footers
and eval toasts. HPO trial rows, Experiment Tracker MLflow rows, Training Monitor
run panels, and portfolio ``LoadedRunRow`` shells brush explicit ``run_label`` values
so click-to-brush stays aligned with row rings and ``GlobalFilterBar``. Data-gen
toasts gain **Data** / **Output** actions from sensor CSV (or generated ``.csv``) and
run root; ``LauncherNavMesh`` accepts ``csvPath`` for path-aware Data Explorer post-run.

**React frontend**
- ``genDataPath.ts`` — sensor / TSPLIB / ``Generated`` path extractors + Data Explorer preferer (§G.11 / §D.8)
- ``useProcessMonitor`` — data-gen toast **Data** (CSV) + optional **Output** actions (§D.8 / §G.11 / §G.14)
- ``LauncherNavMesh`` — optional ``csvPath`` for post-run Data Explorer handoff (§G.11 / §G.6 / §D.7)
- Data Generation + Process Monitor data-gen panels pass ``csvPath`` into nav mesh (§G.11 / §G.15)
- HPO Tracker trial log-dir cells — ``brushLabel`` / ``storedLabel`` from ``trialBrushLabel`` (§G.18 / §D.7)
- Experiment Tracker MLflow run-dir cells — explicit ``brushLabel`` with run name (§G.18 / §D.7)
- Training Monitor ``RunPanel`` path header — ``label`` / ``brushLabel`` / ``storedLabel`` from ``run.name`` (§G.17 / §D.7)
- ``LoadedRunRow`` — ``brushLabel`` from resolved portfolio run label (§G.1 / §D.7)

**ROADMAP**
- §G.1 / §G.11 / §G.15 / §G.17 / §G.18 residual table + portfolio brush-label parity checked
- §D.7 tracker / portfolio / run-panel brush-label residual parity checked
- §D.8 data-gen toast + ``LauncherNavMesh`` CSV handoff checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirty-seventh pass (§G.9–§G.12 + §G.15 + §G.17 + §G.18 + §D.7 + §D.8)

Two-hundred-and-thirty-seventh pass closes residual brush-label parity after pass 236
fixed live-header chips. Process-row and live-panel footer ``OpenPathToolbar`` shells
now brush process-derived ``run_label`` so click-to-brush stays aligned with row ring
highlights and ``GlobalFilterBar`` when path stems diverge (train dirs, process-id
fallback). Eval completion toasts gain **Eval** / **Output** handoff actions from the
checkpoint load path and run root.

**React frontend**
- ``ProcessIdFooter`` — optional ``runLabel`` → ``OpenPathToolbar`` ``brushLabel`` (§G.9–§G.18 / §D.7)
- Process Monitor process rows — ``brushLabel`` from ``processRunBrushById`` (§G.15 / §D.7)
- Live-panel footers on all launcher / monitor / tracker surfaces pass process ``runLabel`` (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- ``useProcessMonitor`` — eval toast **Eval** (checkpoint) + optional **Output** actions (§D.8 / §G.12 / §G.14)

**ROADMAP**
- §G.9–§G.12 / §G.15 / §G.17 / §G.18 process-row + footer brush-label residual parity checked
- §D.7 ``ProcessIdFooter`` / process-row brush-label parity checked
- §D.8 eval checkpoint toast handoff checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirty-sixth pass (§G.4 + §G.7 + §G.9–§G.12 + §G.15 + §G.17 + §G.18 + §D.7)

Two-hundred-and-thirty-sixth pass polishes residual ``OpenPathToolbar`` shells after
pass 235 covered live headers and palette migration. Live-panel header chips brush
process-derived ``runLabel`` so click-to-brush stays aligned with ``GlobalFilterBar``;
batch eval footers surface multi-process meta; Command Palette recent rows finish the
kind-as-children shell without nested interactive elements; Graph Topology distance-
matrix paths adopt the shared cluster with CSV handoff.

**React frontend**
- ``RunLabelHeaderSuffix`` — ``brushLabel`` from process ``runLabel`` when set (§G.9–§G.18 / §D.7)
- ``ProcessIdFooter`` — multi-process batch count as ``children`` when ``logPath`` is set (§G.12 / §D.7)
- Command Palette recent rows — kind meta as ``children``, stored ``label`` display, non-nested row (§G.7 / §D.7)
- Graph Topology distance-matrix path → ``OpenPathToolbar`` ``kind="csv"`` (§G.4 / §D.7)

**ROADMAP**
- §G.4 / §G.7 / §G.9–§G.12 / §G.15 / §G.17 / §G.18 residual open-path toolbar shell polish checked
- §D.7 ``OpenPathToolbar`` header brush + palette + topology residual parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirty-fifth pass (§G.7 + §G.9–§G.12 + §G.15 + §G.17 + §G.18 + §D.7)

Two-hundred-and-thirty-fifth pass expands ``OpenPathToolbar`` after pass 234 covered
shared footer / portfolio shells. Residual live-panel header path chips and Command
Palette recent-file rows migrate onto the cluster so every open-path surface shares
one path+handoff shell. ``handoffOnAfterOpen`` closes the palette after icon handoffs;
``RunLabelHeaderSuffix`` composes the toolbar for launcher / train-HPO headers.

**React frontend**
- ``OpenPathToolbar`` — ``handoffOnAfterOpen`` forwarded to labeled + chip handoffs (§G.7 / §D.7)
- ``RunLabelHeaderSuffix`` → ``OpenPathToolbar``; lives beside the toolbar module (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- ``LauncherLivePanelHeader`` / ``TrainHpoLivePanelHeader`` import suffix from ``OpenPathToolbar`` (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- Command Palette recent-file rows → ``OpenPathToolbar``; kind meta as ``children`` + ``handoffOnAfterOpen`` close (§G.7 / §D.7)

**ROADMAP**
- §G.7 / §G.9–§G.12 / §G.15 / §G.17 / §G.18 live-header + palette open-path toolbar parity checked
- §D.7 ``OpenPathToolbar`` header / palette residual surface expansion checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirty-fourth pass (§G.1 + §G.8 + §G.9–§G.15 + §G.17 + §G.18 + §D.7)

Two-hundred-and-thirty-fourth pass expands ``OpenPathToolbar`` after pass 233 covered
residual table-row chips. Shared live-panel footers and portfolio loaded-run rows migrate
onto the cluster so process-id meta and day-count trailing render as ``children``; Output
Browser ``.wsroute`` manifest members adopt the same path shell.

**React frontend**
- ``ProcessIdFooter`` log-path chip → ``OpenPathToolbar``; process id as ``children`` (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- ``LoadedRunRow`` path chip → ``OpenPathToolbar``; meta ``trailing`` as ``children`` (§G.1 / §G.14 / §D.7)
- Output Browser ``.wsroute`` manifest members → ``OpenPathToolbar`` auto-classify (§G.8 / §G.14 / §D.7)

**ROADMAP**
- §G.1 / §G.8 / §G.9–§G.15 / §G.17 / §G.18 shared footer + portfolio open-path toolbar parity checked
- §D.7 ``OpenPathToolbar`` ProcessIdFooter / LoadedRunRow / ``.wsroute`` surface expansion checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirty-third pass (§G.1 + §G.12 + §G.15 + §G.18 + §D.7)

Two-hundred-and-thirty-third pass expands ``OpenPathToolbar`` after pass 232 covered
Training Monitor / Output Browser residual panels. Residual table-row path chips on HPO
trial log dirs, Experiment Tracker MLflow run dirs, Evaluation Runner / Benchmark Analysis
results tables, and Process Monitor process rows migrate onto the shared cluster so dense
list surfaces share one path+handoff shell with meta ``children``.

**React frontend**
- HPO Tracker trial log-dir cells → ``OpenPathToolbar``; trial number as ``children`` (§G.18 / §D.7)
- Experiment Tracker MLflow run-dir cells → ``OpenPathToolbar``; run-id meta as ``children`` (§G.18 / §D.7)
- Evaluation Runner results-table checkpoint cells → ``OpenPathToolbar`` (§G.12 / §D.7)
- Benchmark Analysis results-table checkpoint cells → ``OpenPathToolbar`` (§G.1 / §G.12 / §D.7)
- Process Monitor process-row path chips → ``OpenPathToolbar``; process id as ``children`` (§G.15 / §D.7)

**ROADMAP**
- §G.1 / §G.12 / §G.15 / §G.18 residual table-row open-path toolbar parity checked
- §D.7 ``OpenPathToolbar`` tracker / eval / process-row residual surface expansion checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirty-second pass (§G.8 + §G.14 + §G.17 + §G.12 + §D.7)

Two-hundred-and-thirty-second pass expands ``OpenPathToolbar`` after pass 231 covered
launcher selected-path previews. Residual Training Monitor run-panel / checkpoint-browser
rows and Output Browser checkpoint sidebars migrate onto the shared cluster so training
runs and checkpoint artefacts share one path+handoff surface. Checkpoint rows gain labeled
Eval Runner controls; ``.wsroute`` manifest member paths gain auto-classified icon handoffs.

**React frontend**
- Training Monitor ``RunPanel`` path header → ``OpenPathToolbar``; epochs meta as ``children`` (§G.17 / §D.7)
- Training Monitor checkpoint browser rows → ``OpenPathToolbar``; ``kind="checkpoint"`` labeled + size ``children`` (§G.17 / §G.12 / §D.7)
- Output Browser checkpoint sidebar rows → ``OpenPathToolbar``; labeled Eval Runner + size ``children`` (§G.14 / §G.12 / §D.7)
- Output Browser ``.wsroute`` manifest file table → ``PathRunLabelChip`` ``handoff`` auto-classify (§G.8 / §G.14 / §D.7)

**ROADMAP**
- §G.14 / §G.17 residual panel + checkpoint open-path toolbar parity checked
- §G.8 ``.wsroute`` member path-handoff surface parity checked
- §D.7 ``OpenPathToolbar`` monitor / browser residual surface expansion checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirty-first pass (§G.10 + §G.11 + §G.12 + §G.15 + §D.7)

Two-hundred-and-thirty-first pass expands ``OpenPathToolbar`` after pass 230 covered
tracker / settings / ML residual path previews. Launcher selected-path previews under
eval / train / data-gen form inputs migrate onto the shared cluster so checkpoint, dataset,
sensor CSV, TSPLIB, and instance-preview rows share one path+handoff surface. Training Hub
checkpoints and Data Generation sensor CSVs gain labeled Eval Runner / Data Explorer
controls; eval live/result card headers adopt the same cluster.

**React frontend**
- Evaluation Runner checkpoint-row + dataset path previews → ``OpenPathToolbar`` (§G.12 / §D.7)
- Training Hub eval checkpoint + dataset path previews → ``OpenPathToolbar``; checkpoint ``kind="checkpoint"`` labeled (§G.10 / §G.12 / §D.7)
- Data Generation sensor CSV / TSPLIB / preview paths → ``OpenPathToolbar``; sensor ``kind="csv"`` labeled (§G.11 / §G.6 / §D.7)
- ``EvalResultCard`` / ``EvalCheckpointLiveCard`` checkpoint headers → ``OpenPathToolbar`` (§G.12 / §G.15 / §D.7)

**ROADMAP**
- §G.10 / §G.11 / §G.12 launcher selected-path open-path toolbar parity checked
- §D.7 ``OpenPathToolbar`` launcher + eval-card surface expansion checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirtieth pass (§G.5 + §G.17 + §G.18 + §G.19 + §A.3 + §D.7)

Two-hundred-and-thirtieth pass expands ``OpenPathToolbar`` after pass 229 covered the
remaining analytics/config/output open-path toolbars. Residual path previews on HPO /
Experiment trackers, Training Monitor empty-state, Policy Telemetry SQLite header, ML
Introspection archives, and Settings migrate onto the shared cluster. HPO report
directories gain a labeled Output Browser handoff beside the existing file-manager Open
action.

**React frontend**
- HPO Tracker storage DB path + report directory → ``OpenPathToolbar``; reports ``kind="run"`` labeled + ExternalLink ``children`` (§G.18 / §D.7)
- Experiment Tracker MLflow tracking URI → ``OpenPathToolbar`` (§G.18 / §D.7)
- Training Monitor empty-state logs-root chip → ``OpenPathToolbar`` (§G.17 / §D.7)
- ``PolicyTelemetryTrendsPanel`` SQLite ``db_path`` → ``OpenPathToolbar`` (§A.3 / §D.7)
- ML Introspection tensor archive path → ``OpenPathToolbar`` (§G.5 / §D.7)
- Settings project root / Python / import JSON / Arrow benchmark paths → ``OpenPathToolbar``; benchmark auto-classifies CSV/log handoffs (§G.19 / §D.7)

**ROADMAP**
- §G.5 / §G.17 / §G.18 / §G.19 residual open-path toolbar surface expansion checked
- §D.7 ``OpenPathToolbar`` tracker / settings / ML residual parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-ninth pass (§G.1 + §G.6 + §G.7 + §G.13 + §G.14 + §G.17 + §D.7)

Two-hundred-and-twenty-ninth pass expands ``OpenPathToolbar`` after pass 228 introduced
the shared open-path cluster. Remaining open-path toolbars migrate onto the component so
Data Explorer exports, Config Editor YAML paths, Output Browser run headers, Training
Monitor logs discovery, and Simulation Summary config banners share one path+handoff
surface. Output Browser run headers and Summary config banners gain labeled reverse
Summary / Monitor dual controls when a simulation log is known.

**React frontend**
- Data Explorer open-CSV toolbar → ``OpenPathToolbar`` with export ``children`` (§G.6 / §G.7 / §D.7)
- Config Editor primary + diff path toolbars → ``OpenPathToolbar`` (§G.13 / §D.7)
- Output Browser run header → ``OpenPathToolbar`` with labeled log dual when ``runJsonlPath`` is known (§G.14 / §G.1 / §D.7)
- Training Monitor logs-root discovery chip → ``OpenPathToolbar`` (§G.17 / §D.7)
- Simulation Summary ``ConfigMetaBanner`` → ``OpenPathToolbar`` with reverse Monitor ``labeledTargets`` (§G.1 / §D.7)

**ROADMAP**
- §G.1 / §G.6 / §G.13 / §G.14 / §G.17 remaining open-path toolbar surface expansion checked
- §D.7 ``OpenPathToolbar`` studio-wide open-path parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-eighth pass (§G.1 + §G.6 + §G.7 + §G.14 + §G.16 + §D.7)

Two-hundred-and-twenty-eighth pass unifies open-path toolbars after pass 227 put
icon handoffs on ``PathRunLabelChip``. ``OpenPathToolbar`` composes optional labeled
reverse-handoff buttons with the path chip so analytics open-log rows, OLAP ingest,
and Output Browser file viewers share one cluster. Host pages pass reverse-destination
``labeledTargets`` (Summary → Monitor, Monitor → Summary) so chip icons stay in parity
with the labeled control.

**React frontend**
- ``OpenPathToolbar`` — labeled + chip cluster (``labeledTargets`` / host-aware chip targets / ``order`` / children) (§G.7 / §D.7)
- Simulation Summary / Simulation Monitor / Algorithm Comparison open-log toolbars (§G.1 / §G.16 / §D.7)
- OLAP Explorer ingest path toolbar (§G.6 / §D.7)
- Output Browser file viewer + checkpoint panel (§G.14 / §D.7)

**ROADMAP**
- §G.1 / §G.6 / §G.14 / §G.16 open-path labeled+chip toolbar unification checked
- §D.7 ``OpenPathToolbar`` surface parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-seventh pass (§G.7 + §G.9–§G.18 + §D.7)

Two-hundred-and-twenty-seventh pass unifies path-chip handoffs after pass 226 put
kind-aware icons on analytics and config surfaces. ``PathRunLabelChip`` gains a
composable ``handoff`` prop so live headers, footers, portfolio rows, and open-path
chips no longer wire ``trailing={<PathHandoffButtons/>}`` ad hoc. ``RunLabelHeaderSuffix``
defaults handoffs on every launcher / Process Monitor / train-HPO live header when a
log path is known.

**React frontend**
- ``PathRunLabelChip`` — ``handoff`` / ``handoffStoredLabel`` / ``handoffTargets`` / ``handoffOnAfterOpen`` compose with ``trailing`` (§G.7 / §D.7)
- ``RunLabelHeaderSuffix`` — default header-chip handoffs for live panels (§G.9–§G.18 / §D.7)
- ``ProcessIdFooter`` + Process Monitor process rows + ``LoadedRunRow`` migrate to chip ``handoff`` (§G.15 / §G.1 / §D.7)
- Simulation Summary ConfigMetaBanner + open-path; Output Browser run / checkpoint / viewer chips (§G.1 / §G.14 / §G.16 / §D.7)
- Eval / launcher / tracker / palette / analytics / Config Editor chips drop ad-hoc ``PathHandoffButtons`` trailing wrappers (§G.7 / §G.10–§G.18 / §D.7)

**ROADMAP**
- §G.7 / §G.9–§G.18 path-chip handoff unification via ``PathRunLabelChip`` checked
- §D.7 live-header + studio-wide chip handoff surface parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-sixth pass (§G.1 + §G.6 + §G.7 + §G.9 + §G.13 + §G.16 + §D.7)

Two-hundred-and-twenty-sixth pass completes path-handoff surface parity after pass 225
covered eval, portfolio, and tracker chips. ``PathHandoffButtons`` now accepts log
``targets`` so Summary-only / Monitor-only toolbars and ``LauncherNavMesh`` can drop
direct ``LogHandoffButtons`` imports. Analytics open-path chips (Summary, Monitor,
Algorithm Comparison, OLAP, Data Explorer) and Config Editor YAML chips share the
same kind-aware control.

**React frontend**
- ``PathHandoffButtons`` — optional ``targets`` forwarded to ``LogHandoffButtons`` (§G.1 / §G.16 / §D.7)
- ``LauncherNavMesh`` — sim log shortcuts use ``PathHandoffButtons`` ``kind="log"`` + ``targets`` (§G.9 / §D.7)
- Simulation Summary / Simulation Monitor / Algorithm Comparison — toolbar + open-path chip handoffs (§G.1 / §G.16 / §D.7)
- OLAP Explorer — labeled + chip handoffs for ``.jsonl`` dual and CSV Data Explorer destinations (§G.6 / §D.7)
- Data Explorer open-CSV chip; Config Editor primary + diff path chips (§G.6 / §G.13 / §D.7)

**ROADMAP**
- §G.1 / §G.6 / §G.9 / §G.13 / §G.16 analytics + config path-handoff surface parity checked
- §D.7 ``PathHandoffButtons`` log ``targets`` + remaining LogHandoff call-site migration checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-fifth pass (§G.1 + §G.7 + §G.10 + §G.11 + §G.12 + §G.14 + §G.15 + §G.17 + §G.18 + §D.7)

Two-hundred-and-twenty-fifth pass extends path-handoff surface parity after pass 224
put labeled ``PathHandoffButtons`` on nav meshes, Output Browser, and Training Monitor
run panels. Eval workflow surfaces (checkpoint inputs, results tables, live/result cards)
and portfolio / tracker run lists now share the same kind-aware control. ``LoadedRunRow``
unifies log dual-handoffs and non-log single-icon handoffs behind ``pathHandoffs``.

**React frontend**
- ``LoadedRunRow`` — ``pathHandoffs`` + optional ``handoffKind`` via ``PathHandoffButtons``; ``logHandoffs`` alias retained (§G.1 / §G.7 / §D.7)
- Portfolio lists (Simulation Summary / Benchmark / City Comparison) migrate to ``pathHandoffs`` (§G.1 / §G.1.6 / §D.7)
- Output Browser run list + Training Monitor run discovery + Experiment Tracker output dirs — explicit ``handoffKind`` path handoffs (§G.14 / §G.17 / §G.18 / §D.7)
- Evaluation Runner — checkpoint input + results table checkpoint icon handoffs (§G.12 / §D.7)
- Training Hub eval checkpoint; ``EvalResultCard`` / ``EvalCheckpointLiveCard`` checkpoint icons (§G.10 / §G.12 / §G.15 / §D.7)
- Benchmark Analysis eval-results checkpoint column handoffs (§G.1 / §G.12 / §D.7)
- Data Generation sensor CSV; HPO Tracker trial log dirs + report dir; Experiment Tracker MLflow run dirs (§G.11 / §G.18 / §D.7)

**ROADMAP**
- §G.12 / §G.10 / §G.1 / §G.14 / §G.17 / §G.18 eval + portfolio + tracker path-handoff surface parity checked
- §D.7 ``LoadedRunRow`` path-handoff unification + eval/launcher/tracker coverage checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-fourth pass (§G.7 + §G.9 + §G.10 + §G.12 + §G.14 + §G.15 + §G.17 + §D.7)

Two-hundred-and-twenty-fourth pass continues path-handoff surface parity after
pass 223 introduced ``PathHandoffButtons`` on process rows, footers, and the
Command Palette. The control now supports empty-path mode-only navigation
(kind required) so launcher and train/HPO nav meshes can share the same labeled
buttons used for post-run deep-links. Output Browser file viewer / checkpoint
sidebar and Training Monitor run panels migrate off ad-hoc Open-in buttons onto
the shared control.

**React frontend**
- ``PathHandoffButtons`` optional empty ``path`` + explicit ``kind`` — mode-only navigation (§G.7 / §D.7)
- ``LauncherNavMesh`` — Output / checkpoint / Training / Data Explorer via labeled ``PathHandoffButtons`` (§G.9 / §G.11 / §G.12 / §D.7)
- ``TrainHpoNavMesh`` — Output Browser + Training Monitor via labeled ``PathHandoffButtons`` (§G.10 / §G.15 / §G.17 / §D.7)
- Output Browser — run-header path-kind handoffs; file-viewer CSV / config / checkpoint / log labeled controls; checkpoint sidebar icons (§G.14 / §G.12 / §D.7)
- Training Monitor — run-panel training path + checkpoint browser icon handoffs (§G.17 / §G.12 / §D.7)

**ROADMAP**
- §G.14 / §G.9 / §G.10 / §G.17 path-handoff surface parity checked
- §D.7 empty-path path-handoff + Output Browser / nav-mesh / Training Monitor coverage checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-third pass (§G.7 + §G.9 + §G.10 + §G.14 + §G.15 + §G.17 + §D.7 + §D.8)

Two-hundred-and-twenty-third pass extends path-handoff surface parity beyond
``.jsonl`` dual Summary / Monitor icons (pass 222). Process rows, live-panel
footers, and Command Palette recents now hand off training dirs, run roots,
CSVs, checkpoints, and configs via a shared ``PathHandoffButtons`` control.
Train/HPO completion toasts gain dual **Training** + **Output** actions when
stdout yields both artefacts.

**React frontend**
- ``PathHandoffButtons`` — kind-aware handoffs (log dual + single-icon non-log) (§G.7 / §G.14 / §G.15 / §G.17 / §D.7)
- Process Monitor process rows — non-log path chips use ``PathHandoffButtons`` (§G.15 / §G.10 / §G.17 / §D.7)
- ``ProcessIdFooter`` — path-kind handoffs for training / run / log chips (§G.9 / §G.15 / §D.7)
- Command Palette — recent entries of all known kinds show icon handoffs (§G.7 / §D.7)
- Process train/HPO toasts — dual **Training** + **Output** when both paths present (§D.8 / §G.17 / §G.14 / §D.7)

**ROADMAP**
- §G.15 / §G.7 non-log path-handoff surface parity checked
- §D.8 train/HPO dual toast actions documented
- §D.7 path-kind handoff coverage checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-second pass (§G.1 + §G.9 + §G.14 + §G.15 + §G.16 + §G.17 + §D.7 + §D.8)

Two-hundred-and-twenty-second pass continues log-handoff surface parity after pass 221
put dual handoffs on Process Monitor rows and Command Palette recents. Live-panel
``ProcessIdFooter`` path chips now expose the same Summary / Monitor icons when the
stdout-derived path is a ``.jsonl``, covering Simulation Launcher and Process Monitor
sim panels without relying on the nav mesh alone. Process terminal toasts gain action
buttons that open the matching artefact (Summary + Monitor for logs, Training for
train/HPO runs, Output Browser for run roots) via a new store-based handoff helper.

**React frontend**
- ``applyStoreRecentHandoff`` / ``recentPendingSettersFromStore`` for non-React handoffs (§G.7 / §D.7 / §D.8)
- ``ProcessIdFooter`` — ``LogHandoffButtons`` when ``logPath`` is ``.jsonl`` (§G.9 / §G.15 / §G.1 / §G.16 / §D.7)
- Process completion / failure / cancel toasts — Summary / Monitor / Training / Output actions (§D.8 / §G.1 / §G.14 / §G.16 / §G.17 / §D.7)

**ROADMAP**
- §G.9 / §G.15 live-panel footer log-handoff surface parity checked
- §D.8 toast handoff actions + Option A+B status documented
- §D.7 process-footer + toast log-handoff coverage checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twenty-first pass (§G.1 + §G.7 + §G.15 + §G.16 + §D.7)

Two-hundred-and-twenty-first pass continues log-handoff surface parity after pass 220
put dual handoffs on the Output Browser run-select tree header. Process Monitor process
rows expose icon Summary / Monitor controls when the stdout-derived path is a ``.jsonl``
so live/recent sims can open analytics without expanding the process first. Command
Palette recent log entries gain the same dual icons for explicit Summary vs Digital Twin
open (``onAfterOpen`` closes the palette after handoff).

**React frontend**
- ``LogHandoffButtons`` optional ``onAfterOpen`` callback (§G.1 / §G.7 / §D.7)
- Process Monitor — process-row ``LogHandoffButtons`` when ``logPath`` is ``.jsonl`` (§G.15 / §G.1 / §G.16 / §D.7)
- Command Palette — recent log dual handoffs + close after open (§G.7 / §G.1 / §G.16 / §D.7)

**ROADMAP**
- §G.15 / §G.7 process-row + palette recent-log handoff surface parity checked
- §D.7 process list and command-palette log-handoff coverage checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twentieth pass (§G.1 + §G.14 + §G.16 + §D.7)

Two-hundred-and-twentieth pass closes the Output Browser run-select log-handoff gap
left after pass 218 put dual handoffs on the JSONL file viewer. Selecting a run that
has a discovered ``runJsonlPath`` now shows icon Summary / Monitor controls on the
file-tree header so analytics can open without browsing into the log file first.

**React frontend**
- Output Browser — selected-run tree header ``LogHandoffButtons`` when ``runJsonlPath`` known (§G.14 / §G.1 / §G.16 / §D.7)

**ROADMAP**
- §G.14 run-select log-handoff surface parity with JSONL file viewer checked
- §D.7 Output Browser run-header handoff coverage checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-nineteenth pass (§G.1 + §G.9 + §G.15 + §G.16 + §D.7)

Two-hundred-and-nineteenth pass continues the shared log-handoff control migration
after pass 218 covered analytics toolbars. ``LogHandoffButtons`` accepts an optional
``path`` so launcher nav can still open Summary / Monitor before a ``.jsonl`` is
emitted. ``LauncherNavMesh`` replaces inline sim handoff markup with the shared
labeled control (Monitor-only while running; Summary + Monitor when post-run).

**React frontend**
- ``LogHandoffButtons`` optional ``path`` — mode-only navigation without handoff (§G.1 / §G.16 / §D.7)
- ``LauncherNavMesh`` — sim Summary / Monitor via shared labeled control (§G.9 / §G.15 / §D.7)
- Simulation Launcher + Process Monitor — inherit nav-mesh log handoffs via ``simLogPath`` (§G.9 / §G.15 / §G.16 / §D.7)

**ROADMAP**
- §G.9 / §G.15 launcher nav-mesh ``LogHandoffButtons`` migration checked
- §D.7 empty-path log-handoff navigation coverage checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-eighteenth pass (§G.1 + §G.14 + §G.16 + §D.7)

Two-hundred-and-eighteenth pass completes the shared log-handoff control migration
left after pass 217 introduced ``LogHandoffButtons``. A ``targets`` prop supports
Summary-only / Monitor-only toolbars so pages that already host one view avoid a
redundant sibling button. Simulation Monitor, Algorithm Comparison, Simulation
Summary, and Output Browser JSONL viewer now use the shared control.

**React frontend**
- ``LogHandoffButtons`` ``targets`` — single-direction toolbars (§G.1 / §G.16 / §D.7)
- Simulation Monitor / Algorithm Comparison — Summary via shared control (§G.16 / §G.1 / §D.7)
- Simulation Summary — Monitor via shared control (§G.1 / §G.16 / §D.7)
- Output Browser — JSONL dual handoffs via shared labeled control (§G.14 / §D.7)

**ROADMAP**
- §G.1 / §G.14 / §G.16 studio-wide ``LogHandoffButtons`` toolbar migration checked
- §D.7 single-direction handoff target coverage checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-seventeenth pass (§G.1 + §G.6 + §G.16 + §D.7)

Two-hundred-and-seventeenth pass extracts shared log-handoff controls after passes
214–216 repeated Summary / Monitor button markup across portfolio rows and toolbars.
``LogHandoffButtons`` supports dense icon rows and labeled toolbar buttons via the
shared mode override. ``LoadedRunRow`` delegates to it. OLAP Explorer surfaces
labeled Summary + Monitor handoffs when the selected ingest path is a ``.jsonl``
simulation log.

**React frontend**
- ``LogHandoffButtons`` / ``isSimulationLogPath`` — shared Summary + Monitor handoffs (§G.1 / §G.16 / §D.7)
- ``LoadedRunRow`` — ``logHandoffs`` via shared control (§G.1 / §D.7)
- OLAP Explorer — labeled log handoffs for JSONL ingest paths (§G.6 / §G.1 / §G.16 / §D.7)

**ROADMAP**
- §G.6 / §G.1 / §G.16 shared log-handoff control + OLAP JSONL path parity checked
- §D.7 handoff control deduplication checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-sixteenth pass (§G.1 + §G.16 + §D.7)

Two-hundred-and-sixteenth pass closes the open-log Simulation Summary handoff gap left
after pass 215 wired portfolio row handoffs and pass 214 exposed Monitor handoffs from
Summary, Output Browser, and Algorithm Comparison. Simulation Monitor gains **Simulation
Summary →** for the open ``watchPath``. Algorithm Comparison exposes the same Summary
handoff next to **Compare on Map** so analytics can jump either direction without
re-picking the log.

**React frontend**
- Simulation Monitor — **Simulation Summary →** open-log handoff (§G.16 / §G.1 / §D.7)
- Algorithm Comparison — **Simulation Summary →** ``watchPath`` handoff (§G.1 / §D.7)

**ROADMAP**
- §G.16 / §G.1 open-log Simulation Summary handoff parity checked
- §D.7 reverse open-log Summary surface coverage beyond portfolio rows checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-fifteenth pass (§G.1 + §G.1.6 + §G.16 + §D.7)

Two-hundred-and-fifteenth pass closes the portfolio loaded-run log-handoff gap left after
pass 214 exposed Simulation Monitor mode override on Output Browser, Summary, and
Algorithm Comparison. ``LoadedRunRow`` gains optional ``logHandoffs`` controls that open
a run in Simulation Summary or Digital Twin via the shared handoff helper. Benchmark
Analysis, City Comparison, and Simulation Summary comparison lists enable the controls
so multi-run analytics can jump to per-run map/summary views without re-picking files.

**React frontend**
- ``LoadedRunRow`` — optional Summary + Simulation Monitor log handoffs (§G.1 / §G.16 / §D.7)
- Benchmark Analysis — portfolio loaded-run log handoffs (§G.1 / §D.7)
- City Comparison — portfolio loaded-run log handoffs (§G.1.6 / §D.7)
- Simulation Summary — comparison-run log handoffs (§G.1 / §D.7)

**ROADMAP**
- §G.1 / §G.1.6 portfolio loaded-run Simulation Monitor / Summary handoff parity checked
- §D.7 mode-override coverage on portfolio run lists checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-fourteenth pass (§G.1 + §G.14 + §G.16 + §D.7)

Two-hundred-and-fourteenth pass closes the analytics/file-browser Simulation Monitor
log-handoff gap left after pass 213 introduced the ``mode`` override on nav meshes and
the Command Palette. Output Browser gains **Open in Simulation Monitor →** next to
Summary for ``.jsonl`` files. Simulation Summary exposes **Simulation Monitor →** for
the open log. Algorithm Comparison **Compare on Map** hands off ``watchPath`` with
``mode: "simulation"`` so the Digital Twin reloads the log and recent-file labels stay
aligned with brush/SQL.

**React frontend**
- Output Browser — **Open in Simulation Monitor →** log handoff mode override (§G.14 / §G.16 / §D.7)
- Simulation Summary — **Simulation Monitor →** open-log Digital Twin handoff (§G.1 / §G.16 / §D.7)
- Algorithm Comparison — **Compare on Map** ``watchPath`` handoff + map-compare intent (§G.16 / §D.7)

**ROADMAP**
- §G.14 / §G.1 / §G.16 analytics + Output Browser Simulation Monitor log-handoff parity checked
- §D.7 mode-override surface coverage beyond launcher nav mesh checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-thirteenth pass (§G.1 + §G.6 + §G.7 + §G.9–§G.17 + §D.7)

Two-hundred-and-thirteenth pass closes the post-run Simulation Monitor (Digital Twin)
log-handoff gap left after pass 212 wired Summary navigation only. ``applyRecentHandoff``
/ ``handoff`` accept an optional ``mode`` override so the same ``pendingLogPath`` can open
in Summary or Monitor. ``LauncherNavMesh`` **Simulation Monitor →** pushes recents and
sets ``pendingLogPath`` when ``simLogPath`` is present. Command Palette log open uses the
shared handoff (Summary on successful load, Monitor fallback). Remaining local-open and
portfolio multi-select call sites use ``handoff(…, { navigate: false })`` instead of
inline ``pushRecent(makeRecentEntry(…))``.

**React frontend**
- ``applyRecentHandoff`` / ``useRecentHandoff.handoff`` — optional ``mode`` override (§G.7 / §G.16 / §D.7)
- ``LauncherNavMesh`` — post-run **Simulation Monitor →** log handoff (§G.9 / §G.16 / §D.7)
- Command Palette — log open via shared handoff + mode fallback (§G.7 / §G.1 / §G.16 / §D.7)
- Local-open migration to ``handoff(navigate: false)`` across launchers, monitors, analytics, Output Browser, global drop (§G.1 / §G.6 / §G.10–§G.17 / §D.7)

**ROADMAP**
- §G.16 / §G.9 post-run Simulation Monitor log handoff checked
- §G.7 / §D.7 handoff mode override + studio-wide ``navigate: false`` recent-file parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-twelfth pass (§G.1 + §G.7 + §G.8 + §G.9 + §G.14 + §G.15 + §G.17 + §D.7)

Two-hundred-and-twelfth pass closes the post-run Simulation Summary log-handoff gap
left after pass 211 unified ``applyRecentHandoff`` without wiring completed-sim
``.jsonl`` paths into Summary navigation. ``useRecentHandoff`` consolidates
pending-path setters and the shared ``handoff(path, kind)`` helper across nav meshes,
Command Palette, global drop, wsroute import, Output Browser, and Training Monitor.
``LauncherNavMesh`` accepts ``simLogPath`` so **Simulation Summary →** and the
Simulation Launcher auto-Summary countdown push recents and set ``pendingLogPath``.

**React frontend**
- ``useRecentHandoff`` / ``useRecentPendingSetters`` — shared handoff hook (§G.7 / §G.8 / §D.7)
- ``LauncherNavMesh`` ``simLogPath`` — post-run Summary log handoff (§G.9 / §G.1 / §D.7)
- Simulation Launcher — auto-nav + nav mesh ``simJsonlPath`` via ``extractJsonlPathFromLogLines`` (§G.9 / §D.7)
- Process Monitor — sim panel Summary handoff from stdout ``.jsonl`` (§G.15 / §G.9 / §D.7)
- Call-site migration to ``handoff()`` (nav meshes, palette, drop, import, Output Browser, Training Monitor) (§G.7–§G.15 / §G.17 / §D.7)

**ROADMAP**
- §G.9 / §G.1 post-run Simulation Summary log handoff checked
- §G.7 / §G.8 shared ``useRecentHandoff`` hook parity checked
- §D.7 Shared handoff hook relative-path brush/SQL recent-file parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-eleventh pass (§G.1 + §G.6 + §G.7 + §G.8 + §G.9–§G.14 + §G.16 + §G.17 + §D.7)

Two-hundred-and-eleventh pass closes the studio-wide recent-file helper gap left after
pass 210 introduced ``makeRecentEntry`` / ``recentHandoffSpec`` without migrating all
call sites. ``applyRecentHandoff`` centralises push + pending-path + mode navigation.
``LauncherNavMesh``, ``TrainHpoNavMesh``, Output Browser handoff buttons, Command Palette,
global drop, wsroute import, and Training Monitor checkpoint load use the shared helper.
All remaining page-level ``pushRecent({… portfolioRunLabel …})`` call sites now use
``makeRecentEntry`` for Command Palette brush/SQL label parity.

**React frontend**
- ``applyRecentHandoff`` — shared push + pending-path + mode handoff (§G.7 / §G.8 / §D.7)
- ``LauncherNavMesh`` + ``TrainHpoNavMesh`` — post-run handoffs via ``applyRecentHandoff`` (§G.9–§G.12 / §G.15 / §G.17 / §D.7)
- Output Browser — sim/eval/config/csv/bundle handoffs via ``applyRecentHandoff`` (§G.14 / §D.7)
- Command Palette / global drop / wsroute import / Training Monitor checkpoint via shared handoff (§G.7 / §G.8 / §G.17 / §D.7)
- Analytics + launcher + monitor + Config Editor ``pushRecent`` → ``makeRecentEntry`` (§G.1 / §G.6 / §G.10 / §G.12 / §G.13 / §G.16 / §D.7)

**ROADMAP**
- §G.7 / §G.8 / §G.14 studio-wide ``makeRecentEntry`` / ``applyRecentHandoff`` parity checked
- §G.9–§G.12 / §G.15 / §G.17 nav-mesh recent-file handoff shared helper checked
- §D.7 Shared handoff helper relative-path brush/SQL recent-file parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-tenth pass (§G.6 + §G.7 + §G.8 + §G.10 + §G.12 + §G.14 + §G.17 + §D.7)

Two-hundred-and-tenth pass closes the shared recent-file handoff and directory-drop
gaps left after pass 209 introduced ``recentKindFromPath`` without wiring it through
call sites. ``recentHandoff.ts`` centralises kind → mode / pending-path / toast specs
and ``makeRecentEntry`` for ``portfolioRunLabel`` labels. Command Palette opens all
recent kinds via the shared handoff and keyboard-navigates recents with commands.
``useGlobalFileDrop`` classifies every dropped path (including training ``logs/`` and
``assets/output`` run directories), pushes all matches to recents, and navigates to the
highest-priority artefact. Output Browser directory picker and eval/train CSV dataset
pickers push recents for Command Palette reopen parity.

**React frontend**
- ``recentHandoff.ts`` — shared handoff specs + ``makeRecentEntry`` + drop priority (§G.7 / §G.8 / §D.7)
- ``recentKindFromPath`` — training ``logs/`` + ``assets/output`` run directory heuristics (§G.8 / §G.14 / §G.17 / §D.7)
- Command Palette — unified recent open + keyboard nav over recents and commands (§G.7 / §D.7)
- ``useGlobalFileDrop`` — multi-path classify/push + directory drop + shared handoff (§G.8 / §G.6 / §G.12 / §G.14 / §G.17 / §D.7)
- Output Browser — ``pickOutputDir`` + inline open via ``recentKindFromPath`` / ``makeRecentEntry`` (§G.14 / §D.7)
- Evaluation Runner + Training Hub — CSV dataset pick ``pushRecent`` (§G.12 / §G.10 / §G.6 / §D.7)

**ROADMAP**
- §G.7 Command Palette shared recent handoff + keyboard nav checked
- §G.8 Global multi-path + directory drop recent-file parity checked
- §G.14 Output Browser directory-open recent-file parity checked
- §G.6 / §G.10 / §G.12 CSV dataset pick recent-file handoff checked
- §D.7 Shared handoff relative-path brush/SQL recent-file parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-ninth pass (§G.6 + §G.7 + §G.8 + §G.12 + §G.13 + §G.14 + §D.7)

Two-hundred-and-ninth pass closes the global file-drop and Output Browser multi-kind
gaps left after pass 208 unified Config Editor recent-file handoffs. ``recentKindFromPath``
classifies filesystem paths into recent-file kinds. ``useGlobalFileDrop`` routes dropped
``.csv``, checkpoint (``.pt`` / ``.ckpt`` / ``.pth``), and config files through the same
``portfolioRunLabel`` + pending-path handoffs used by Command Palette recents. Output
Browser pushes checkpoints on inline open and adds **Open in Data Explorer →** for CSV
via ``pendingCsvPath``.

**React frontend**
- ``recentFiles`` store — ``recentKindFromPath`` path → kind classifier (§G.7 / §G.8 / §D.7)
- ``useGlobalFileDrop`` — multi-kind drop for log / csv / checkpoint / config with ``portfolioRunLabel`` + pending handoffs (§G.8 / §G.6 / §G.12 / §G.13 / §D.7)
- Output Browser — inline checkpoint open ``pushRecent`` via ``portfolioRunLabel`` (§G.14 / §G.12 / §D.7)
- Output Browser — **Open in Data Explorer →** CSV handoff via ``pendingCsvPath`` (§G.14 / §G.6 / §D.7)

**ROADMAP**
- §G.8 Global file drop multi-kind recent-file ``portfolioRunLabel`` parity checked
- §G.14 Output Browser checkpoint open + Data Explorer CSV handoff recent-file parity checked
- §G.6 / §G.12 / §G.13 Data Explorer / Eval Runner / Config Editor drop handoff checked
- §D.7 Global drop + Output Browser multi-kind relative-path brush/SQL recent-file parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-eighth pass (§G.7 + §G.13 + §G.14 + §D.7)

Two-hundred-and-eighth pass closes the Config Editor recent-file gaps left after
pass 207 unified eval-checkpoint handoff labels. ``RecentFileKind`` gains
``config`` for YAML / TOML / cfg / ini files. Configuration Editor file picker and
``pendingConfigPath`` consume push recent files via ``portfolioRunLabel``. Output
Browser pushes on inline config open and adds **Open in Config Editor →** via
``pendingConfigPath``. Command Palette opens config recents into Config Editor.

**React frontend**
- ``recentFiles`` store — ``config`` kind for Hydra / YAML config files (§G.13 / §G.7 / §D.7)
- App store — ``pendingConfigPath`` deep-link for Config Editor handoff (§G.13 / §G.14 / §D.7)
- Configuration Editor — file picker + ``pendingConfigPath`` consume ``pushRecent`` via ``portfolioRunLabel`` (§G.13 / §D.7)
- Output Browser — inline config open ``pushRecent`` + **Open in Config Editor →** handoff (§G.14 / §G.13 / §D.7)
- Command Palette — recent config handoff via ``pendingConfigPath`` (§G.7 / §G.13 / §D.7)

**ROADMAP**
- §G.13 Configuration Editor recent-file ``portfolioRunLabel`` parity checked
- §G.7 Command Palette config recent-file handoff checked
- §G.14 Output Browser config open + Config Editor handoff recent-file parity checked
- §D.7 Config relative-path brush/SQL recent-file parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-seventh pass (§G.7 + §G.10 + §G.12 + §G.14 + §G.17 + §D.7)

Two-hundred-and-seventh pass closes the eval-checkpoint recent-file gaps left after
pass 206 unified Training Monitor and nav-mesh handoff labels. ``RecentFileKind``
gains ``checkpoint`` for ``.pt`` / ``.ckpt`` / ``.pth`` model files. Training Monitor
checkpoint browser, Output Browser ``loadInEvalRunner``, and ``LauncherNavMesh``
post-run ``Load in Eval Runner`` push recent files via ``portfolioRunLabel`` before
``pendingCheckpoint`` handoff. Evaluation Runner and Training Hub file pickers and
``pendingCheckpoint`` consume use the same labels. Command Palette opens checkpoint
recents into Evaluation Runner.

**React frontend**
- ``recentFiles`` store — ``checkpoint`` kind for model checkpoint files (§G.12 / §G.7 / §D.7)
- Training Monitor — checkpoint browser ``pushRecent`` via ``portfolioRunLabel`` on eval handoff (§G.17 / §G.12 / §D.7)
- Output Browser — ``loadInEvalRunner`` ``pushRecent`` via ``portfolioRunLabel`` (§G.14 / §G.12 / §D.7)
- ``LauncherNavMesh`` — post-run eval checkpoint handoff ``pushRecent`` via ``portfolioRunLabel`` (§G.9–§G.12 / §G.15 / §D.7)
- Evaluation Runner — ``pendingCheckpoint`` consume + checkpoint file picker ``pushRecent`` (§G.12 / §D.7)
- Training Hub — eval-mode checkpoint file picker ``pushRecent`` (§G.10 / §G.12 / §D.7)
- Command Palette — recent checkpoint handoff via ``pendingCheckpoint`` (§G.7 / §G.12 / §D.7)

**ROADMAP**
- §G.12 Evaluation Runner checkpoint recent-file ``portfolioRunLabel`` parity checked
- §G.7 Command Palette checkpoint recent-file handoff checked
- §G.10 / §G.14 / §G.17 Training Hub + Output Browser + Training Monitor checkpoint handoff recent-file parity checked
- §D.7 Eval checkpoint relative-path brush/SQL recent-file parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-sixth pass (§G.7 + §G.9–§G.12 + §G.15 + §G.17 + §D.7)

Two-hundred-and-sixth pass closes the Training Monitor and launcher/train-HPO
nav-mesh recent-file gaps left after pass 205 unified Command Palette label
refresh and Output Browser inline open. ``RecentFileKind`` gains ``training``
for Lightning log directories. Training Monitor pushes recent files via
``portfolioRunLabel`` on run select, ``pendingTrainingRunPath`` consume, and
post-run auto-select. Command Palette opens training recents into Training
Monitor; ``LauncherNavMesh`` and ``TrainHpoNavMesh`` push Output Browser /
Training Monitor handoff paths with the same relative-path labels.

**React frontend**
- ``recentFiles`` store — ``training`` kind for Lightning log directories (§G.17 / §G.7 / §D.7)
- Training Monitor — ``pushRecent`` via ``portfolioRunLabel`` on select / pending / post-run auto-select; filter-bar ``portfolioRunLabel`` parity (§G.17 / §D.7)
- Command Palette — recent training handoff via ``pendingTrainingRunPath`` (§G.7 / §G.17 / §D.7)
- ``LauncherNavMesh`` + ``TrainHpoNavMesh`` — Output Browser / Training Monitor handoff ``pushRecent`` via ``portfolioRunLabel`` (§G.9–§G.12 / §G.15 / §G.17 / §D.7)

**ROADMAP**
- §G.17 Training Monitor recent-file ``portfolioRunLabel`` parity checked
- §G.7 Command Palette training recent-file handoff checked
- §G.9–§G.12 / §G.15 nav-mesh Output Browser + Training Monitor handoff recent-file parity checked
- §D.7 Training Monitor + nav-mesh relative-path brush/SQL recent-file parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-fifth pass (§G.6 + §G.7 + §G.8 + §G.14 + §D.7)

Two-hundred-and-fifth pass closes the Command Palette on-open label refresh and
Output Browser inline file-open recent-file gaps left after pass 204 unified
portfolio load and bundle/drop handoff labels. ``refreshRecentLabels`` re-derives
persisted recent log/run/csv labels via ``portfolioRunLabel`` when the palette
opens or ``projectRoot`` changes. Command Palette CSV entries hand off through
``pendingCsvPath``; Output Browser inline ``.jsonl`` / ``.csv`` tree opens push
recent files with ``portfolioRunLabel``.

**React frontend**
- ``recentFiles`` store — ``refreshRecentLabels`` via ``portfolioRunLabel`` on palette open / ``projectRoot`` change (§G.7 / §D.7)
- Command Palette — ``refreshRecentLabels`` on open; ``pendingCsvPath`` handoff for recent CSV (§G.7 / §G.6 / §D.7)
- Data Explorer — consume ``pendingCsvPath`` on mount for Command Palette CSV parity (§G.6 / §D.7)
- Output Browser — ``pushRecent`` on inline ``.jsonl`` / ``.csv`` file open via ``portfolioRunLabel`` (§G.14 / §G.8 / §D.7)

**ROADMAP**
- §G.7 Command Palette on-open recent-file label refresh checked
- §G.6 Data Explorer ``pendingCsvPath`` Command Palette CSV handoff checked
- §G.14 Output Browser inline file-open recent-file ``portfolioRunLabel`` parity checked
- §D.7 Recent-file label refresh + inline file-open relative-path brush/SQL parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-fourth pass (§G.1 + §G.1.6 + §G.7 + §G.8 + §G.14 + §D.7)

Two-hundred-and-fourth pass closes the portfolio load, bundle/drop import, and
Simulation Summary handoff recent-file ``run_label`` gaps left after pass 203
unified compare handoff and add-run Command Palette labels. ``portfolioRunLabel``
now drives ``pushRecent`` on Output Browser ``openInSimSummary`` /
``extractBundleAndOpen``, global ``.jsonl`` / ``.wsroute`` drop and import hooks,
and ``loadOutputPortfolio`` on Benchmark Analysis, City Comparison, and Simulation
Summary. Command Palette re-derives stored recent log/run/csv labels on open so
entries stay aligned when ``projectRoot`` changes.

**React frontend**
- Output Browser — ``openInSimSummary`` + ``extractBundleAndOpen`` ``pushRecent`` via ``portfolioRunLabel`` (§G.14 / §G.1 / §D.7)
- ``useGlobalFileDrop`` — ``pushRecent`` on dropped ``.jsonl`` / extracted bundle log (§G.8 / §G.14 / §D.7)
- ``useWsrouteImport`` — ``pushRecent`` on imported bundle log handoff (§G.8 / §G.14 / §D.7)
- Benchmark Analysis — ``pushRecent`` on ``loadOutputPortfolio`` via ``portfolioRunLabel`` (§G.1 / §D.7)
- City Comparison — ``pushRecent`` on ``loadOutputPortfolio`` via ``portfolioRunLabel`` (§G.1.6 / §D.7)
- Simulation Summary — ``pushRecent`` on ``loadOutputPortfolio`` for each scanned log (§G.1 / §D.7)
- Command Palette — refresh recent log/run/csv labels via ``portfolioRunLabel`` on open (§G.7 / §D.7)

**ROADMAP**
- §G.14 Output Browser Simulation Summary handoff recent-file ``portfolioRunLabel`` parity checked
- §G.8 Global file drop + wsroute import recent-file ``portfolioRunLabel`` parity checked
- §G.1 / §G.1.6 Benchmark Analysis + City Comparison + Simulation Summary ``loadOutputPortfolio`` recent-file parity checked
- §G.7 Command Palette recent-file label refresh on open checked
- §D.7 Portfolio load + bundle/drop handoff recent-file relative-path brush/SQL parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-third pass (§G.14 + §G.1 + §G.1.6 + §D.7)

Two-hundred-and-third pass closes the Output Browser multi-run compare and portfolio
add-comparison-run recent-file ``run_label`` gaps left after pass 202 unified single-log
and add-run Command Palette labels. ``compareSelectedRuns`` now derives Benchmark
Analysis handoff refs via ``portfolioRunLabel`` and pushes each compared log to recent
files. Simulation Summary add-comparison-run and Benchmark Analysis / City Comparison
``pendingBenchmarkLogs`` consume paths use the same helper for Command Palette parity
when ``projectRoot`` changes.

**React frontend**
- Output Browser — ``compareSelectedRuns`` Benchmark handoff + ``pushRecent`` via ``portfolioRunLabel`` (§G.14 / §G.1 / §D.7)
- Simulation Summary — ``pushRecent`` on add-comparison-run via ``portfolioRunLabel`` (§G.1 / §D.7)
- Benchmark Analysis — ``pushRecent`` on ``pendingBenchmarkLogs`` consume via ``portfolioRunLabel`` (§G.1 / §D.7)
- City Comparison — ``pushRecent`` on ``pendingBenchmarkLogs`` consume via ``portfolioRunLabel`` (§G.1.6 / §D.7)

**ROADMAP**
- §G.14 Output Browser compare handoff relative-path ``run_label`` + recent-file parity checked
- §G.1 Simulation Summary add-comparison-run ``portfolioRunLabel`` recent-file parity checked
- §G.1 / §G.1.6 Benchmark Analysis + City Comparison ``pendingBenchmarkLogs`` recent-file parity checked
- §D.7 Output Browser compare + portfolio handoff recent-file relative-path brush/SQL parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-second pass (§G.14 + §G.6 + §G.1 + §G.1.6 + §D.7)

Two-hundred-and-second pass closes the Output Browser, OLAP Explorer, and portfolio
add-run recent-file ``run_label`` gaps left after pass 201 unified single-log open-file
UI labels. ``portfolioRunLabel`` now drives Output Browser run-directory select,
Policy Telemetry Trends scoping, and ``.wsroute`` manifest path-chip brush.
OLAP Explorer custom ingest uses the same helper for filter-bar selectors,
DuckDB ``SqlQueryPanel`` brush/SQL sync, and Policy Telemetry Trends when tables
lack a portfolio ``run_label`` column. Benchmark Analysis and City Comparison
push recent files on add-run for Command Palette parity when ``projectRoot`` changes.

**React frontend**
- Output Browser — ``sourceRunLabel`` via ``portfolioRunLabel`` on run select, Policy Telemetry Trends, ``.wsroute`` manifest brush, and ``pushRecent`` (§G.14 / §D.7)
- OLAP Explorer — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, Policy Telemetry Trends, and ``pushRecent`` on ingest (§G.6 / §D.7)
- Benchmark Analysis — ``pushRecent`` on add-run via ``portfolioRunLabel`` (§G.1 / §D.7)
- City Comparison — ``pushRecent`` on add-run via ``portfolioRunLabel`` (§G.1.6 / §D.7)

**ROADMAP**
- §G.14 Output Browser run-directory relative-path ``run_label`` brush + recent-file parity checked
- §G.6 OLAP Explorer custom-ingest single-log DuckDB ``run_label`` brush/SQL parity checked
- §G.1 / §G.1.6 Benchmark Analysis + City Comparison add-run recent-file ``portfolioRunLabel`` parity checked
- §D.7 Output Browser + OLAP Explorer + portfolio add-run recent-file relative-path brush/SQL parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundred-and-first pass (§G.6 + §G.16 + §G.1 + §D.7)

Two-hundred-and-first pass closes the single-log open-file UI ``run_label`` gaps left
after pass 200 unified portfolio loaded-run lists. ``portfolioRunLabel`` now drives
Data Explorer, Algorithm Comparison, and Simulation Monitor filter-bar selectors,
``SqlQueryPanel`` brush/SQL sync, and Policy Telemetry Trends scoping when CSV/logs
lack a ``run_label`` column. Recent-file labels on open use the same helper so
Command Palette entries stay aligned when ``projectRoot`` changes.

**React frontend**
- Data Explorer — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, Policy Telemetry Trends, and recent-file push (§G.6 / §D.7)
- Algorithm Comparison — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, and Policy Telemetry Trends (§G.16 / §D.7)
- Simulation Monitor — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, Policy Telemetry Trends, and recent-file push (§G.16 / §D.7)
- Simulation Summary — recent-file push label via ``portfolioRunLabel`` (§G.1 / §D.7)

**ROADMAP**
- §G.6 Data Explorer single-log open-file relative-path ``run_label`` brush/SQL parity checked
- §G.16 Algorithm Comparison + Simulation Monitor single-log DuckDB ``run_label`` brush/SQL parity checked
- §D.7 single-log open-file UI/DuckDB ``run_label`` relative-path brush/SQL parity across Data Explorer, Algorithm Comparison, Simulation Monitor, and Simulation Summary recent files checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — two-hundredth pass (§G.1 + §G.1.6 + §G.6 + §D.7)

Two-hundredth pass closes the Benchmark Analysis and City Comparison portfolio UI
``run_label`` gaps left after pass 199 unified Simulation Summary portfolio labels.
``normalizedRuns`` + ``portfolioDuckDbLogs`` now derive labels via ``portfolioRunLabel``
so loaded-run rows, ``GlobalFilterBar`` run selectors, and DuckDB ``run_label`` columns
stay aligned for relative paths. OLAP Explorer custom JSONL ingest and Simulation
Summary comparison-run rows use the same helper for display/brush parity.

**React frontend**
- Benchmark Analysis — ``normalizedRuns`` + ``portfolioDuckDbLogs`` ``portfolioRunLabel`` on loaded-run list, ``filteredRuns`` portfolio brush, and DuckDB union ingest (§G.1 / §G.6 / §D.7)
- City Comparison — ``normalizedRuns`` + ``portfolioDuckDbLogs`` ``portfolioRunLabel`` on loaded-run list, ``filteredRuns`` portfolio brush, and DuckDB union ingest (§G.1.6 / §G.6 / §D.7)
- OLAP Explorer — custom JSONL ingest path label via ``portfolioRunLabel`` (§G.6 / §D.7)
- Simulation Summary — comparison-run ``LoadedRunRow`` labels re-derived via ``portfolioRunLabel`` (§G.1 / §D.7)

**ROADMAP**
- §G.1 Benchmark Analysis portfolio loaded-run list + filter-bar relative-path ``run_label`` brush/SQL parity checked
- §G.1.6 City Comparison portfolio loaded-run list + filter-bar relative-path ``run_label`` brush/SQL parity checked
- §D.7 Benchmark Analysis + City Comparison portfolio UI/DuckDB ``run_label`` relative-path brush/SQL parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-ninth pass (§G.1 + §G.6 + §D.7)

Hundred-ninety-ninth pass closes the Simulation Summary portfolio UI ``run_label``
gaps left after pass 198 unified DuckDB union ingest ``projectRoot`` resolution.
Add-comparison-run, output-portfolio load, ``allRuns`` portfolio brush, and
``allDuckDbLogs`` now derive labels via ``portfolioRunLabel`` so loaded-run rows,
``GlobalFilterBar`` run selectors, and DuckDB ``run_label`` columns stay aligned
for relative paths.

**React frontend**
- Simulation Summary — ``portfolioRunLabel`` on add-comparison-run, output-portfolio load, ``allRuns`` portfolio brush, and ``allDuckDbLogs`` union ingest (§G.1 / §G.6 / §D.7)

**ROADMAP**
- §G.1 Simulation Summary portfolio loaded-run list + filter-bar relative-path ``run_label`` brush/SQL parity checked
- §D.7 Simulation Summary portfolio UI/DuckDB ``run_label`` relative-path brush/SQL parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-eighth pass (§G.0 + §G.1 + §G.1.6 + §G.6 + §D.7)

Hundred-ninety-eighth pass closes the portfolio DuckDB union ingest ``run_label``
gaps left after pass 197 unified single-log Arrow pipelines and Simulation Summary
portfolio labels. ``portfolioRunLabel`` + ``runPortfolioSimulationArrowPipeline`` now
accept optional ``projectRoot`` and resolve multi-log ``run_label`` columns via
``runLabelFromSourcePath``. Benchmark Analysis and City Comparison derive portfolio
run labels on load/add-run and pass ``projectRoot`` into DuckDB ingest.

**React frontend**
- ``portfolioRunLabel`` — resolve portfolio ``run_label`` against ``projectRoot`` before DuckDB union (§G.0 / §G.1 / §G.6 / §D.7)
- ``runPortfolioSimulationArrowPipeline`` — optional ``projectRoot`` resolves each log's ``run_label`` via ``runLabelFromSourcePath`` (§G.0 / §G.1 / §G.6 / §D.7)
- Benchmark Analysis + City Comparison — ``portfolioRunLabel`` on add-run, output-portfolio load, and pending compare logs; DuckDB ingest passes ``projectRoot`` (§G.1 / §G.1.6 / §D.7)
- Simulation Summary + OLAP Explorer — portfolio DuckDB pipeline callers pass ``projectRoot`` for multi-log union ingest (§G.1 / §G.6 / §D.7)

**ROADMAP**
- §G.0 / §G.1 portfolio DuckDB union ingest ``run_label`` relative-path resolution checked
- §G.1.6 City Comparison portfolio DuckDB log-label brush/SQL parity checked
- §G.1 Benchmark Analysis portfolio DuckDB log-label brush/SQL parity checked
- §G.6 OLAP Explorer multi-log DuckDB ingest ``projectRoot`` parity checked
- §D.7 portfolio DuckDB ``run_label`` relative-path brush/SQL parity across Benchmark Analysis, City Comparison, Simulation Summary, and OLAP Explorer checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-seventh pass (§G.0 + §G.6 + §G.7 + §G.9–§G.18 + §A.3 + §D.7)

Hundred-ninety-seventh pass closes the DuckDB Arrow ingest and explicit live-panel
``projectRoot`` gaps left after pass 196 unified derived run-label utilities and
live-panel store-fallback path-chip parity. ``runLabelFromSourcePath`` resolves paths
against ``projectRoot`` before ``run_label`` derivation. Single-log Arrow pipelines
annotate DuckDB tables with project-root-resolved labels. Launcher and train/HPO live
panels pass explicit ``projectRoot`` on headers and footers.

**React frontend**
- ``runLabelFromSourcePath`` — resolve paths against ``projectRoot`` before ``run_label`` derivation (§G.0 / §G.6 / §G.7 / §D.7)
- ``annotateTableWithRunLabelIfMissing`` + ``runCsvArrowPipeline`` / ``runSimulationArrowPipeline`` — optional ``projectRoot`` resolves ingest source paths before DuckDB ``run_label`` annotation (§G.0 / §G.6 / §D.7)
- OLAP Explorer + Data Explorer + Algorithm Comparison + Simulation Monitor + Settings — Arrow pipeline callers pass ``projectRoot`` for single-log DuckDB ingest (§G.0 / §G.6 / §G.16 / §G.19 / §D.7)
- Simulation Summary — portfolio DuckDB log labels derived via ``runLabelFromSourcePath`` for brush/SQL parity (§G.1 / §G.6 / §D.7)
- ``PolicyTelemetryTrendsPanel`` — telemetry ``db_path`` ``PathRunLabelChip`` passes ``projectRoot`` (§A.3 / §G.7 / §D.7)
- Simulation Launcher + Data Generation + Evaluation Runner + Training Hub + Process Monitor + Training Monitor + HPO Tracker + Experiment Tracker — explicit ``projectRoot`` on ``LauncherLivePanelHeader`` / ``TrainHpoLivePanelHeader`` + ``ProcessIdFooter`` (§G.9–§G.18 / §D.7)

**ROADMAP**
- §G.0 / §G.6 DuckDB Arrow ingest ``run_label`` relative-path resolution checked
- §G.1 Simulation Summary portfolio DuckDB log-label brush parity checked
- §A.3 Policy Telemetry Trends db_path path-chip ``projectRoot`` parity checked
- §G.9–§G.18 launcher + train/HPO live-panel explicit ``projectRoot`` header/footer parity checked
- §D.7 DuckDB ingest + live-panel explicit ``projectRoot`` path-chip run-label brush parity across Arrow pipelines, Simulation Summary, Policy Telemetry Trends, and all launcher/train/HPO workflow pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-sixth pass (§G.7 + §G.9–§G.18 + §G.15 + §G.6 + §D.7)

Hundred-ninety-sixth pass closes the derived run-label and live-panel relative-path
resolution gaps left after the hundred-ninety-fifth pass (which unified portfolio
``LoadedRunRow`` / open-file ``PathRunLabelChip`` ``projectRoot`` parity).
``runLabelFromLogLines``, ``runLabelMapFrom*``, and ``pathRunLabelBrushActive`` now
resolve paths against ``projectRoot`` before ``run_label`` derivation. Process stdout
brush sync, Process Monitor row ring-highlight, OLAP ingest-table picker highlight,
HPO post-run filter labels, Command Palette recent files, and live-panel footer/header
log-path chips share ``projectRoot``-resolved brush parity.

**React frontend**
- ``runLabelFromLogLines`` / ``runLabelMapFromProcesses`` / ``runLabelMapFromPaths`` /
  ``runLabelMapFromTablePaths`` / ``pathRunLabelBrushActive`` — optional ``projectRoot``
  resolves relative paths before ``run_label`` derivation (§G.7 / §G.15 / §G.16 / §D.7)
- ``useProcessRunLabelBrush`` — resolves stdout log paths against ``useAppStore``
  ``projectRoot`` before global ``run_label`` brush sync (§G.9–§G.18 / §D.7)
- Process Monitor — ``runLabelMapFromProcesses`` passes ``projectRoot`` for process-row
  ring-highlight parity (§G.15 / §D.7)
- OLAP Explorer — ``runLabelMapFromTablePaths`` passes ``projectRoot`` for ingest-table
  picker ring-highlight parity (§G.6 / §D.7)
- HPO Tracker — ``GlobalFilterBar`` ``runLabels`` from ``projectRoot``-resolved post-run
  ``trainingRunPath`` / ``outputRunPath`` (§G.18 / §D.7)
- Command Palette — recent log/run/csv ``PathRunLabelChip`` entries pass ``projectRoot``
  (§G.7 / §D.7)
- ``ProcessIdFooter`` + ``LauncherLivePanelHeader`` + ``TrainHpoLivePanelHeader`` —
  optional ``projectRoot`` prop + store fallback on live-panel log-path suffix chips
  (§G.9–§G.18 / §D.7)

**ROADMAP**
- §G.7 derived run-label utility relative-path resolution checked
- §G.9–§G.18 ``useProcessRunLabelBrush`` + live-panel header/footer path-chip brush parity checked
- §G.15 Process Monitor process-row relative-path ring-highlight parity checked
- §G.6 OLAP Explorer ingest-table picker relative-path ring-highlight parity checked
- §G.18 HPO Tracker post-run ``GlobalFilterBar`` relative-path run-label parity checked
- §D.7 derived run-label + live-panel relative-path path-chip run-label brush parity across Command Palette, Process Monitor, OLAP Explorer, HPO Tracker, and launcher/train/HPO live panels checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-fifth pass (§G.1 + §G.6 + §G.14–§G.17 + §G.15 + §G.16 + §D.7)

Hundred-ninety-fifth pass closes the portfolio and open-file relative-path resolution
gaps left after the hundred-ninety-fourth pass (which added ``PathRunLabelChip``
store fallback and explicit ``projectRoot`` on analysis/monitor/file browsers).
``useLogPathRunLabelBrush`` and ``LoadedRunRow`` now resolve paths against
``projectRoot`` before global ``run_label`` brush sync and portfolio ring-highlight
comparison. Portfolio run lists, analytics open-file headers, and Process Monitor
process rows pass explicit ``projectRoot`` for path-chip brush parity.

**React frontend**
- ``useLogPathRunLabelBrush`` — resolves log/run paths against ``useAppStore``
  ``projectRoot`` before ``run_label`` brush sync (§G.1 / §G.14–§G.16 / §D.7)
- ``LoadedRunRow`` — optional ``projectRoot`` prop + store fallback; portfolio
  ring-highlight compares resolved run labels (§G.1 / §G.14 / §G.17 / §D.7)
- Simulation Summary + Benchmark Analysis + City Comparison + Experiment Tracker +
  Training Monitor + Output Browser — ``LoadedRunRow`` ``projectRoot`` on portfolio
  and run-directory lists (§G.1 / §G.14 / §G.17 / §G.18 / §D.7)
- Simulation Summary + Data Explorer + OLAP Explorer + Algorithm Comparison +
  Simulation Monitor — open-file ``PathRunLabelChip`` headers pass ``projectRoot``
  (§G.1 / §G.6 / §G.16 / §G.15 / §D.7)
- Process Monitor — process-row ``PathRunLabelChip`` passes ``projectRoot`` for
  stdout-resolved log paths (§G.15 / §D.7)
- Training Monitor — ``GlobalFilterBar`` ``runLabels`` from ``projectRoot``-resolved
  Lightning log paths (§G.17 / §D.7)
- Output Browser — ``parentRunBrushLabel`` resolves selected-run path against
  ``projectRoot`` before ``.wsroute`` manifest brush (§G.14 / §G.8 / §D.7)

**ROADMAP**
- §G.1 / §G.6 portfolio loaded-run list relative-path path-chip brush parity checked
- §G.14–§G.17 Training Monitor + Output Browser run-list ``LoadedRunRow`` projectRoot parity checked
- §G.1 / §G.6 / §G.16 analytics open-file header relative-path path-chip brush parity checked
- §G.15 Process Monitor process-row relative-path path-chip brush parity checked
- §D.7 portfolio + open-file relative-path path-chip run-label brush parity across Simulation Summary, Benchmark Analysis, City Comparison, Data Explorer, OLAP Explorer, Algorithm Comparison, Simulation Monitor, Process Monitor, Training Monitor, and Output Browser checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-fourth pass (§G.14 + §G.17 + §G.18 + §G.8 + §G.12 + §G.9–§G.18 + §D.7)

Hundred-ninety-fourth pass closes the analysis/monitor/file-browser relative-path
resolution gaps left after the hundred-ninety-third pass (which unified launcher
workflow ``PathRunLabelChip`` ``projectRoot`` parity). ``PathRunLabelChip`` now
falls back to the persisted ``useAppStore`` project root when the prop is omitted,
so analysis views, monitor pages, and file browsers auto-resolve relative paths.
HPO trial log directories, MLflow artifact run directories, Training Monitor
checkpoint browsers, and Output Browser artefact viewers now pass explicit
``projectRoot`` for parent-run ``brushLabel`` derivation.

**React frontend**
- ``PathRunLabelChip`` — falls back to ``useAppStore`` ``projectRoot`` when prop
  omitted; relative paths resolve before brush + tooltip (§G.1 / §G.14–§G.18 / §D.7)
- ``RunLabelHeaderSuffix`` — optional ``projectRoot`` prop; inherits store fallback
  via ``PathRunLabelChip`` (§G.9–§G.18 / §D.7)
- HPO Tracker — trial ``log_dir`` user-attribute paths resolved against
  ``projectRoot`` before path-chip brush (§G.18 / §D.7)
- Experiment Tracker — MLflow ``artifact_uri`` run directories resolved against
  ``projectRoot`` before path-chip brush (§G.18 / §D.7)
- Training Monitor — logs root, run-discovery list, per-run headers, and checkpoint
  browser use ``projectRoot``-resolved path chips + parent-run ``brushLabel`` (§G.17 / §G.12 / §D.7)
- Output Browser — selected-run, checkpoint sidebar, file viewer, checkpoint preview,
  and ``.wsroute`` manifest rows use ``projectRoot``-resolved path chips (§G.14 / §G.8 / §G.12 / §D.7)

**ROADMAP**
- §G.18 HPO Tracker trial log_dir relative-path path-chip brush parity checked
- §G.18 Experiment Tracker MLflow artifact_uri relative-path path-chip brush parity checked
- §G.17 Training Monitor logs + checkpoint browser relative-path path-chip brush parity checked
- §G.14 / §G.8 Output Browser run/checkpoint/manifest relative-path path-chip brush parity checked
- §D.7 analysis + monitor + file browser relative-path path-chip run-label brush parity across HPO Tracker, Experiment Tracker, Training Monitor, and Output Browser checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-third pass (§G.12 + §G.10 + §G.11 + §G.13 + §G.5 + §G.1 + §G.15 + §G.19 + §D.7)

Hundred-ninety-third pass closes the launcher-workflow relative-path resolution gaps
left after the hundred-ninety-second pass (which unified tracker storage, telemetry
SQLite, data-gen preview, and Settings benchmark/import ``PathRunLabelChip`` parity
against ``projectRoot``). Eval checkpoint and dataset inputs, data-gen source paths,
config editor open/diff files, ML introspection tensor archives, and Settings
secondary paths now resolve against ``projectRoot`` and share ``PathRunLabelChip``
ring-highlight + click-to-brush behaviour with analysis views.

**React frontend**
- ``PathRunLabelChip`` — optional ``projectRoot`` prop resolves relative paths via
  ``resolveLocalProjectPath`` before brush + tooltip (§G.10–§G.13 / §D.7)
- ``parentRunBrushLabelFromCheckpointPath`` — optional ``projectRoot`` for
  checkpoint parent-run brush label derivation (§G.12 / §G.14 / §G.17 / §D.7)
- Evaluation Runner — checkpoint list, dataset input, results table, and live eval
  cards use ``projectRoot``-resolved path chips (§G.12 / §D.7)
- Training Hub — eval checkpoint + dataset path chips use ``projectRoot`` resolution
  (§G.10 / §G.12 / §D.7)
- Data Generation Wizard — TSPLIB/sensor source + instance preview path chips use
  ``projectRoot`` resolution (§G.11 / §D.7)
- Configuration Editor — open YAML + diff comparison path chips use ``projectRoot``
  resolution (§G.13 / §D.7)
- ML Introspection — tensor archive path chip uses ``projectRoot`` resolution
  (§G.5 / §D.7)
- Benchmark Analysis + Process Monitor — eval results / live eval cards use
  ``projectRoot``-resolved checkpoint path chips (§G.1 / §G.12 / §G.15 / §D.7)
- Settings — Python executable + import JSON + Arrow benchmark path chips resolve
  against draft project root (§G.19 / §D.7)

**ROADMAP**
- §G.12 Evaluation Runner launcher path-chip relative-path brush parity checked
- §G.10 / §G.12 Training Hub eval checkpoint + dataset path-chip brush parity checked
- §G.11 Data Generation Wizard source + preview path-chip brush parity checked
- §G.13 Configuration Editor open + diff path-chip brush parity checked
- §G.5 ML Introspection tensor archive path-chip brush parity checked
- §G.1 / §G.15 Benchmark Analysis + Process Monitor eval checkpoint path-chip brush parity checked
- §D.7 launcher + workflow relative-path path-chip run-label brush parity across Evaluation Runner, Training Hub, Data Generation, Config Editor, ML Introspection, and Settings checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-second pass (§G.18 + §G.11 + §G.19 + §G.7 + §A.3 + §D.7)

Hundred-ninety-second pass closes the relative-path resolution and secondary
file-picker path-chip gaps left after the hundred-ninety-first pass (which unified
Settings, MLflow tracking URI, and Optuna storage URL ``PathRunLabelChip`` parity).
Optuna storage URLs, HPO report directories, telemetry SQLite stores, data-gen
preview datasets, and Settings benchmark/import paths now resolve against
``projectRoot`` and share ``PathRunLabelChip`` ring-highlight + click-to-brush
behaviour with analysis views.

**React frontend**
- ``sqliteStoragePathFromUrl`` — resolve Optuna ``sqlite:///`` storage URL against
  ``projectRoot`` for path-chip brush (§G.18 / §G.19 / §D.7)
- HPO Tracker — storage DB + exported report directory ``PathRunLabelChip`` use
  ``projectRoot``-resolved absolute paths (§G.18 / §D.7)
- Data Generation Wizard — instance preview ``.pkl`` / ``.pt`` path ``PathRunLabelChip``
  below preview panel (§G.11 / §D.7)
- Settings — Arrow pipeline benchmark + import-settings JSON ``PathRunLabelChip``
  below filled paths (§G.19 / §D.7)
- ``PolicyTelemetryTrendsPanel`` — SQLite ``db_path`` resolved against ``projectRoot``
  before path-chip brush (§G.7 / §A.3 / §D.7)

**ROADMAP**
- §G.18 HPO Tracker storage/report relative-path path-chip brush parity checked
- §G.11 Data Generation Wizard instance preview path-chip brush parity checked
- §G.19 Settings Arrow benchmark + import JSON path-chip brush parity checked
- §G.7 / §A.3 PolicyTelemetryTrendsPanel ``db_path`` relative-path resolution checked
- §D.7 relative-path storage/preview/import path-chip run-label brush parity across HPO Tracker, Data Generation, Settings, and Policy Telemetry checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninety-first pass (§G.19 + §G.18 + §D.7)

Hundred-ninety-first pass closes the Settings, MLflow tracking URI, and Optuna
storage URL path-chip gaps left after the hundred-ninetieth pass (which unified
live-eval, dataset-input, data-gen source, and telemetry-store ``PathRunLabelChip``
parity). Project root, Python executable, MLflow tracking store, and Optuna
storage inputs now share ``PathRunLabelChip`` ring-highlight + click-to-brush
behaviour with analysis views.

**React frontend**
- ``resolveLocalProjectPath`` — resolve MLflow tracking URI / relative paths against
  ``projectRoot`` for path-chip brush (§G.18 / §G.19 / §D.7)
- Settings — project root + Python executable ``PathRunLabelChip`` below filled path
  inputs (§G.19 / §D.7)
- Experiment Tracker — MLflow tracking URI ``PathRunLabelChip`` below filled tracking
  URI when local path resolves (§G.18 / §D.7)
- HPO Tracker — Optuna storage URL ``PathRunLabelChip`` below filled input; inline
  chip parity with eval dataset inputs (§G.18 / §D.7)

**ROADMAP**
- §G.19 Settings project-root + Python path path-chip brush parity checked
- §G.18 Experiment Tracker MLflow tracking URI path-chip brush parity checked
- §G.18 HPO Tracker Optuna storage URL path-chip brush parity checked
- §D.7 Settings + tracker storage/tracking URI path-chip run-label brush parity across Settings, Experiment Tracker, and HPO Tracker checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninetieth pass (§G.12 + §G.10 + §G.11 + §G.15 + §G.7 + §D.7)

Hundred-ninetieth pass closes the live-eval, dataset-input, data-gen source, and
telemetry-store path-chip gaps left after the hundred-eighty-ninth pass (which unified
post-eval results and bundle-manifest ``PathRunLabelChip`` parity). Live eval
checkpoint cards, eval dataset inputs, TSPLIB/sensor source paths, and the policy
telemetry SQLite header now share ``PathRunLabelChip`` ring-highlight + click-to-brush
behaviour with analysis views.

**React frontend**
- ``EvalCheckpointLiveCard`` — optional ``checkpointPath`` prop renders ``PathRunLabelChip``
  with parent-run ``brushLabel`` on live eval rows (§G.12 / §G.15 / §D.7)
- Evaluation Runner + Training Hub + Process Monitor — live eval cards pass Hydra
  checkpoint path to ``EvalCheckpointLiveCard`` (§G.12 / §G.10 / §G.15 / §D.7)
- Evaluation Runner + Training Hub — eval dataset path ``PathRunLabelChip`` below filled
  dataset inputs (§G.12 / §G.10 / §D.7)
- Evaluation Runner + Training Hub — checkpoint input chips use
  ``parentRunBrushLabelFromCheckpointPath`` ``brushLabel`` parity with results table
  (§G.12 / §G.10 / §D.7)
- Data Generation Wizard — TSPLIB instance + sensor CSV source path ``PathRunLabelChip``
  ring-highlight + click-to-brush parity (§G.11 / §D.7)
- ``PolicyTelemetryTrendsPanel`` — SQLite ``db_path`` header uses ``PathRunLabelChip``
  instead of plain font-mono text (§G.7 / §A.3 / §D.7)

**ROADMAP**
- §G.12 EvalCheckpointLiveCard live-eval path-chip brush parity checked
- §G.10 / §G.12 Evaluation Runner + Training Hub eval dataset path-chip brush parity checked
- §G.11 Data Generation Wizard TSPLIB/sensor source path-chip brush parity checked
- §G.7 PolicyTelemetryTrendsPanel SQLite store path-chip brush parity checked
- §D.7 live-eval + dataset + data-gen source path-chip run-label brush parity across Eval Runner, Training Hub, Process Monitor, and Data Generation checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-ninth pass (§G.12 + §G.1 + §G.8 + §G.10 + §G.15 + §D.7)

Hundred-eighty-ninth pass closes the post-eval results and bundle-manifest path-chip
gaps left after the hundred-eighty-eighth pass (which unified eval-checkpoint input
``PathRunLabelChip`` parity across Output Browser, Evaluation Runner, Training Hub,
and Configuration Editor). Eval results tables, ``EvalResultCard`` headers, and
``.wsroute`` manifest file rows now share ``PathRunLabelChip`` ring-highlight +
click-to-brush behaviour with analysis views.

**React frontend**
- ``parentRunBrushLabelFromCheckpointPath`` — shared helper deriving parent-run brush
  label from ``checkpoints/`` path segments (§G.12 / §G.14 / §G.17 / §D.7)
- ``EvalResult`` / ``EvalAnalyticsRow`` — optional ``checkpointPath`` propagated from
  Hydra eval command via ``checkpointPathFromEvalCommand`` (§G.12 / §G.1 / §D.7)
- Evaluation Runner — post-eval results table renders ``PathRunLabelChip`` when
  checkpoint path is known; parent-run ``brushLabel`` parity with input rows (§G.12 / §D.7)
- Benchmark Analysis — eval results panel checkpoint column ``PathRunLabelChip``
  ring-highlight + click-to-brush parity (§G.1 / §G.12 / §D.7)
- ``EvalResultCard`` — checkpoint header chip on Process Monitor + Training Hub eval
  panels when path known (§G.10 / §G.12 / §G.15 / §D.7)
- Output Browser — ``.wsroute`` manifest file table rows use ``PathRunLabelChip``
  with selected-run brush label (§G.8 / §G.14 / §D.7)
- Output Browser + Training Monitor — checkpoint ``brushLabel`` uses shared
  ``parentRunBrushLabelFromCheckpointPath`` helper (§G.14 / §G.17 / §D.7)

**ROADMAP**
- §G.12 Evaluation Runner eval-results table path-chip brush parity checked
- §G.1 Benchmark Analysis eval-results panel path-chip brush parity checked
- §G.8 Output Browser wsroute manifest file-table path-chip brush parity checked
- §G.10 / §G.15 EvalResultCard checkpoint header path-chip brush parity checked
- §D.7 eval-results + bundle-manifest path-chip run-label brush parity across Eval Runner, Benchmark Analysis, and Output Browser checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-eighth pass (§G.14 + §G.12 + §G.10 + §G.13 + §D.7)

Hundred-eighty-eighth pass closes the eval-checkpoint and Output Browser artefact
path-chip gaps left after the hundred-eighty-seventh pass (which unified Training
Monitor logs-root, checkpoint browser, Configuration Editor primary YAML, and ML
Introspection tensor-archive ``PathRunLabelChip`` parity). Output Browser checkpoint
rows, file viewer headers, Evaluation Runner checkpoint inputs, Training Hub eval
checkpoint input, and Configuration Editor diff comparison paths now share
``PathRunLabelChip`` ring-highlight + click-to-brush behaviour with analysis views.

**React frontend**
- Output Browser — checkpoint sidebar rows render ``PathRunLabelChip`` with parent-run
  ``brushLabel`` parity with Training Monitor (§G.14 / §G.12 / §D.7)
- Output Browser — file viewer header + checkpoint preview use ``PathRunLabelChip`` for
  all artefact paths; checkpoint files brush parent run label (§G.14 / §D.7)
- Output Browser — ``useLogPathRunLabelBrush`` derives label from ``runJsonlPath ??
  selectedRun.path`` (§G.14 / §D.7)
- Evaluation Runner — checkpoint list rows show ``PathRunLabelChip`` when path is set
  (§G.12 / §D.7)
- Training Hub — eval-mode checkpoint path ``PathRunLabelChip`` below input when path
  is set (§G.10 / §G.12 / §D.7)
- Configuration Editor — diff comparison file ``PathRunLabelChip`` +
  ``useLogPathRunLabelBrush`` on ``diffPath``; diff summary uses chips for both files
  (§G.13 / §D.7)

**ROADMAP**
- §G.14 Output Browser checkpoint-browser path-chip run-label brush + ring-highlight parity checked
- §G.12 Evaluation Runner checkpoint-input path-chip brush parity checked
- §G.10 Training Hub eval checkpoint path-chip brush parity checked
- §G.13 Configuration Editor diff comparison path-chip brush parity checked
- §D.7 eval-checkpoint path-chip run-label brush parity across Output Browser, Eval Runner, and Training Hub checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-seventh pass (§G.17 + §G.13 + §G.5 + §D.7)

Hundred-eighty-seventh pass closes the Training Monitor checkpoint/logs path-chip gap left after the
hundred-eighty-sixth pass (which unified HPO Tracker trial-table and storage/report
``PathRunLabelChip`` parity). Lightning logs root, per-checkpoint rows, Configuration Editor
open-file headers, and ML Introspection tensor archive paths now share ``PathRunLabelChip``
ring-highlight + click-to-brush behaviour with analysis views.

**React frontend**
- ``PathRunLabelChip`` — optional ``brushLabel`` prop decouples display text from brush
  run_label when checkpoint filename differs from parent run (§G.17 / §D.7)
- Training Monitor — Lightning ``logs/`` root ``PathRunLabelChip`` in controls + empty-state
  banner (§G.17 / §D.7)
- Training Monitor — checkpoint browser rows render ``PathRunLabelChip`` with parent-run brush
  label + checkpoint filename display (§G.17 / §G.12 / §D.7)
- Configuration Editor — open-file ``PathRunLabelChip`` + ``useLogPathRunLabelBrush`` on primary
  YAML path (§G.13 / §D.7)
- ML Introspection — tensor archive ``PathRunLabelChip`` + ``useLogPathRunLabelBrush`` on open
  ``.npz`` / ``.npy`` / ``.td`` path (§G.5 / §D.7)

**ROADMAP**
- §G.17 Training Monitor logs-root + checkpoint-browser path-chip run-label brush + ring-highlight parity checked
- §G.13 Configuration Editor open-file path-chip brush parity checked
- §G.5 ML Introspection tensor-archive path-chip brush parity checked
- §D.7 file-based workflow path-chip run-label brush parity across Training Monitor, Config Editor, and ML Introspection checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-sixth pass (§G.18 + §D.7)

Hundred-eighty-sixth pass closes the HPO Tracker path-chip gap left after the
hundred-eighty-fifth pass (which unified Experiment Tracker MLflow/output-dir
``PathRunLabelChip`` parity). Optuna trial health rows, storage DB path, exported
report directory, and ``GlobalFilterBar`` ``runLabels`` now share
``PathRunLabelChip`` ring-highlight + click-to-brush behaviour with analysis views.

**React frontend**
- ``trialLogDirFromUserAttrs`` / ``sqlitePathFromStorageUrl`` — resolve Optuna trial
  ``log_dir`` user attribute and local SQLite storage path for path-chip brush
  (§G.18 / §D.7)
- HPO Tracker — trial health table rows render ``PathRunLabelChip`` when trial
  ``log_dir`` is known; muted trial-number suffix parity with Process Monitor
  (§G.18 / §D.7)
- HPO Tracker — Optuna storage DB + exported Plotly report directory
  ``PathRunLabelChip`` ring-highlight + click-to-brush parity (§G.18 / §D.7)
- HPO Tracker — ``GlobalFilterBar`` ``runLabels`` from selected trials, post-run
  ``trainingRunPath`` / ``outputRunPath``, or live process brush (§G.18 / §D.7)

**Python backend**
- ``HpoHealthMetricsCallback`` — persist ``log_dir`` on Optuna trial user attributes
  from Lightning ``trainer.log_dir`` (§G.18 / §A.4 / §D.7)

**ROADMAP**
- §G.18 HPO Tracker trial-table path-chip run-label brush + ring-highlight parity checked
- §G.18 HPO Tracker storage/report path-chip brush parity checked
- §D.7 tracker-page path-chip run-label brush parity across all HPO workflow views checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-fifth pass (§G.18 + §D.7)

Hundred-eighty-fifth pass closes the Experiment Tracker path-chip gap left after the
hundred-eighty-fourth pass (which unified Training Monitor run-discovery ``PathRunLabelChip``
parity). MLflow run table rows, output directory list, and ``GlobalFilterBar`` ``runLabels``
now share ``PathRunLabelChip`` ring-highlight + click-to-brush behaviour with analysis views.

**React frontend**
- ``localPathFromUri`` / ``mlflowRunDirFromArtifactUri`` — resolve MLflow ``artifact_uri`` to
  local run directory for path-chip brush (§G.18 / §D.7)
- ``PathRunLabelChip`` — optional ``label`` prop overrides brush + display text; ``LoadedRunRow``
  passes ``label`` through to chip (§G.1 / §G.14 / §D.7)
- Experiment Tracker — MLflow run table rows render ``PathRunLabelChip`` when ``artifact_uri``
  resolves a local path; muted run-id suffix parity with Process Monitor (§G.18 / §D.7)
- Experiment Tracker — output directory list ``LoadedRunRow`` + ``PathRunLabelChip``
  ring-highlight + click-to-brush parity (§G.18 / §G.14 / §D.7)
- Experiment Tracker — ``GlobalFilterBar`` ``runLabels`` from selected MLflow runs when no
  live process brush is active (§G.18 / §D.7)

**ROADMAP**
- §G.18 Experiment Tracker MLflow run table path-chip run-label brush + ring-highlight parity checked
- §G.18 Experiment Tracker output directory list path-chip brush parity checked
- §D.7 tracker-page path-chip run-label brush parity across all file-based analysis views checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-fourth pass (§G.17 + §G.15 + §D.7)

Hundred-eighty-fourth pass closes the Training Monitor run-discovery path-chip gap left after the
hundred-eighty-third pass (which unified process-row and live-panel footer ``PathRunLabelChip``
parity). Lightning log run list rows, per-run panel headers, and Process Monitor process-id
suffixes now share ``PathRunLabelChip`` ring-highlight + click-to-brush behaviour with analysis
views.

**React frontend**
- Training Monitor — run discovery list uses ``LoadedRunRow`` + ``PathRunLabelChip`` instead of
  inline font-mono run names; ring-highlight parity via global ``run_label`` brush (§G.17 / §D.7)
- Training Monitor — ``RunPanel`` per-run header uses ``PathRunLabelChip`` instead of plain
  font-mono text (§G.17 / §D.7)
- Training Monitor — ``GlobalFilterBar`` ``runLabels`` from selected Lightning log paths when no
  live process brush is active (§G.17 / §D.7)
- Process Monitor — process list rows show muted process-id suffix alongside ``PathRunLabelChip``
  when stdout resolves a log path; footer parity (§G.15 / §D.7)

**ROADMAP**
- §G.17 Training Monitor run-discovery list path-chip run-label brush + ring-highlight parity checked
- §G.15 Process Monitor process-row muted process-id suffix parity with ``ProcessIdFooter`` checked
- §D.7 file-based Lightning log path-chip run-label brush parity across Training Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-third pass (§G.7 + §G.15 + §G.9–§G.18 + §D.7)

Hundred-eighty-third pass closes the process-row and live-panel footer path-chip gap left after the
hundred-eighty-second pass (which unified launcher/monitor card header ``PathRunLabelChip`` parity).
Process Monitor list rows, live panel footers on all launcher/monitor workflow pages, and Command
Palette recent files now share ``PathRunLabelChip`` ring-highlight + click-to-brush behaviour.

**React frontend**
- ``processLogPathKind`` / ``brushLogPathMapFromProcesses`` — derive per-process log/run paths from
  stdout for row + footer path-chip brush (§G.15 / §D.7)
- Process Monitor — process list rows render ``PathRunLabelChip`` when stdout resolves a log path;
  ring-highlight parity preserved (§G.15 / §D.7)
- ``ProcessIdFooter`` — optional ``logPath`` prop renders ``PathRunLabelChip`` with muted
  process-id suffix (§G.9–§G.18 / §D.7)
- Simulation Launcher + Data Generation + Evaluation Runner + Training Hub + Training Monitor +
  HPO Tracker + Experiment Tracker — live panel footers pass ``logPath`` from process stdout
  (§G.9–§G.18 / §D.7)
- Command Palette — recent log/run/csv entries use ``PathRunLabelChip`` ring-highlight +
  click-to-brush parity (§G.7 / §D.7)

**ROADMAP**
- §G.15 Process Monitor process-row path-chip run-label brush + ring-highlight parity checked
- §G.9–§G.18 launcher/monitor live panel footer path-chip brush parity checked
- §G.7 Command Palette recent-file path-chip brush parity checked
- §D.7 process-based workflow path-chip run-label brush parity across all launcher/monitor pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-second pass (§G.9–§G.18 + §G.1 + §D.7)

Hundred-eighty-second pass closes the launcher/monitor card header path-chip gap left after the
hundred-eighty-first pass (which unified portfolio loaded-run list ``PathRunLabelChip`` parity).
Simulation Launcher, Data Generation, Evaluation Runner, Training Hub, Process Monitor,
Training Monitor, HPO Tracker, and Experiment Tracker live/post-run panel headers now share
``PathRunLabelChip`` ring-highlight + click-to-brush behaviour with analysis views.

**React frontend**
- ``brushLogPathFromProcessLines`` — resolve ``.jsonl`` / Lightning ``logs/`` / ``assets/output``
  paths from process stdout for header chip brush (§G.9–§G.18 / §D.7)
- ``RunLabelHeaderSuffix`` — shared inline header suffix rendering ``PathRunLabelChip`` when
  ``logPath`` known, else plain ``· runLabel`` text (§G.9–§G.18 / §D.7)
- ``LauncherLivePanelHeader`` + ``TrainHpoLivePanelHeader`` — optional ``logPath`` prop replaces
  plain run-label suffix with ``PathRunLabelChip`` (§G.9–§G.18 / §G.15 / §D.7)
- Simulation Launcher + Data Generation + Evaluation Runner + Training Hub — live panel headers
  pass ``logPath`` from process stdout (§G.9–§G.12 / §D.7)
- Process Monitor + Training Monitor + HPO Tracker + Experiment Tracker — live/post-run panel
  headers pass ``logPath`` from selected/recent process stdout (§G.15 / §G.17 / §G.18 / §D.7)
- Simulation Summary — ``ConfigMetaBanner`` run path uses ``PathRunLabelChip`` instead of plain
  font-mono text (§G.1 / §G.14 / §D.7)

**ROADMAP**
- §G.9–§G.18 launcher/monitor card header path-chip run-label brush + ring-highlight parity checked
- §G.1 Simulation Summary config banner path-chip brush parity checked
- §D.7 live workflow page header path-chip brush parity across all launchers and monitors checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighty-first pass (§G.1 + §G.6 + §G.14 + §D.7)

Hundred-eighty-first pass closes the portfolio loaded-run list path-chip gap left after the
hundred-eightieth pass (which unified file-based header ``PathRunLabelChip`` parity).
Benchmark Analysis, City Comparison, Simulation Summary comparison runs, and Output Browser
run-directory list now share ``LoadedRunRow`` wrapping ``PathRunLabelChip`` for consistent
ring-highlight + click-to-brush behaviour.

**React frontend**
- ``LoadedRunRow`` — shared portfolio loaded-run row with optional remove, leading slots,
  trailing metadata, and embedded ``PathRunLabelChip`` (§G.1 / §G.14 / §D.7)
- Benchmark Analysis — loaded-run list uses ``LoadedRunRow`` instead of inline font-mono
  brush buttons (§G.1 / §G.6 / §D.7)
- City Comparison — loaded-run list ``LoadedRunRow`` parity with Benchmark Analysis
  (§G.1.6 / §G.6 / §D.7)
- Simulation Summary — comparison-run list ``LoadedRunRow`` parity with portfolio analytics
  pages (§G.1 / §G.6 / §D.7)
- Output Browser — run-directory list ``LoadedRunRow`` with compare checkbox + folder-select
  leading slots; chip click-to-brush parity with selected-run + jsonl viewer header chips
  (§G.14 / §D.7)

**ROADMAP**
- §G.1 Benchmark Analysis + Simulation Summary comparison-run list path-chip brush parity checked
- §G.6 City Comparison loaded-run list path-chip brush parity checked
- §G.14 Output Browser run-directory list path-chip brush parity checked
- §D.7 portfolio loaded-run list path-chip run-label brush + ring-highlight parity across all analysis views checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eightieth pass (§G.1 + §G.6 + §G.14 + §D.7)

Hundred-eightieth pass closes the remaining file-path run-label brush chip gap left after the
hundred-seventy-ninth pass (which unified Monitor / Algorithm Comparison / Data Explorer
``PathRunLabelChip`` parity). Simulation Summary, Output Browser, and OLAP Explorer now share
the same path-chip ring-highlight + click-to-brush behaviour as Simulation Monitor.

**React frontend**
- Simulation Summary — open-log ``PathRunLabelChip`` with DuckDB ingest timing badge (§G.1 / §G.14 / §D.7)
- Output Browser — selected-run ``PathRunLabelChip`` on file-tree header; open-jsonl viewer header
  chip replaces plain path text (§G.14 / §D.7)
- OLAP Explorer — custom-ingest ``PathRunLabelChip`` when selected table tracks a source path;
  ingest timing badge moves to chip trailing slot (§G.6 / §G.14 / §D.7)

**ROADMAP**
- §G.1 Simulation Summary open-log path-chip run-label brush parity checked
- §G.6 OLAP Explorer custom-ingest path-chip run-label brush parity checked
- §G.14 Output Browser selected-run + jsonl viewer path-chip brush parity checked
- §D.7 file-path run-label brush + ring-highlight parity across all file-based analysis views checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-ninth pass (§G.6 + §G.14 + §G.16 + §D.7)

Hundred-seventy-ninth pass closes the single-log DuckDB run-label brush gap left after the
hundred-seventy-eighth pass (which unified OLAP built-in portfolio table brush parity).
Simulation Monitor, Algorithm Comparison, and Data Explorer now share Output Browser
path-chip ring-highlight + click-to-brush behaviour, and single-log Arrow pipelines
annotate DuckDB tables with ``run_label`` + ``city_scale`` when absent.

**React frontend**
- ``annotateTableWithRunLabelIfMissing`` — ``runSimulationArrowPipeline`` / ``runCsvArrowPipeline``
  add path-derived ``run_label`` + ``city_scale`` columns when missing (§G.6 / §D.7)
- ``pathRunLabelBrushActive`` / ``useRunLabelBrushToggle`` / ``PathRunLabelChip`` — shared
  path-chip ring highlight + click-to-brush helpers (§G.14–§G.16 / §D.7)
- Simulation Monitor — watch-path ``PathRunLabelChip``; ``monitor_sim`` ``SqlQueryPanel``
  ``brushSqlSync`` + run-label filter parity (§G.16 / §D.7)
- Algorithm Comparison — watch-path ``PathRunLabelChip``; ``algorithm_sim`` ``SqlQueryPanel``
  run-label brush sync via ``portfolioMode`` (§G.1 / §G.16 / §D.7)
- Data Explorer — open-file ``PathRunLabelChip``; ``useTableRunLabelBrush`` on ``explorer_csv``
  when CSV lacks ``run_label`` column (§G.6 / §G.16 / §D.7)

**ROADMAP**
- §G.6 single-log DuckDB ``run_label`` annotation + ``SqlQueryPanel`` brush sync checked
- §G.14–§G.16 file-path chip run-label brush + ring-highlight parity checked
- §D.7 Monitor / Algorithm Comparison / Data Explorer path-chip brush parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-eighth pass (§G.6 + §G.14 + §G.16 + §D.7)

Hundred-seventy-eighth pass closes the OLAP Explorer built-in DuckDB portfolio table run-label
brush gap left after the hundred-seventy-seventh pass (which unified custom ingested-table
brush hooks). Built-in tables such as ``summary_sim``, ``benchmark_sim``, ``city_sim``, and
``algorithm_sim`` now share table-picker ring-highlight + click-to-brush parity even when no
custom ingest path is tracked.

**React frontend**
- ``runLabelMapFromSingleTableLabels`` / ``tableRunLabelBrushActive`` — DuckDB table
  ``run_label`` helpers for portfolio table-picker brush parity (§G.6 / §D.7)
- ``useTableRunLabelBrush`` — shared hook syncing global brush when a table has exactly one
  distinct ``run_label`` (§G.6 / §D.7)
- OLAP Explorer — ``refreshTables`` indexes per-table ``run_label`` values; built-in portfolio
  tables share ring highlight when global brush matches any contained label (§G.6 / §G.14 / §D.7)
- OLAP Explorer — single-run built-in table brush sync + ``GlobalFilterBar`` / trends fallback
  via ``useTableRunLabelBrush`` when no ingest path is tracked (§G.6 / §G.16 / §D.7)

**ROADMAP**
- §G.6 OLAP Explorer built-in DuckDB portfolio table run-label brush + table-picker ring-highlight parity checked
- §G.14 OLAP Explorer built-in table click-to-brush parity with Output Browser run list checked
- §G.16 OLAP Explorer DuckDB-derived run-label brush hook parity with Data Explorer checked
- §D.7 file-based + built-in DuckDB table run-label brush parity across all analysis views including OLAP Explorer checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-seventh pass (§G.6 + §G.14 + §G.16 + §D.7)

Hundred-seventy-seventh pass closes the OLAP Explorer run-label brush hook gap left
after the hundred-seventy-sixth pass (which unified portfolio analytics run-label brush
parity on Benchmark, City, Algorithm Comparison, and Data Explorer). OLAP Explorer now
syncs the global brush from ingest source paths and shares Output Browser table/run-list
ring-highlight + click-to-brush parity on the ingested-table picker.

**React frontend**
- ``runLabelMapFromTablePaths`` — shared helper deriving per-table ``run_label`` from
  ingest source paths (§G.6 / §D.7)
- OLAP Explorer — ``useLogPathRunLabelBrush`` on selected custom-table ingest path;
  global brush sync on table select (§G.6 / §G.16 / §D.7)
- OLAP Explorer — ingested-table picker ring highlight + click-to-brush via
  ``runLabelMapFromTablePaths`` (§G.6 / §G.14 / §D.7)
- OLAP Explorer — path-derived ``GlobalFilterBar`` ``runLabels`` + trends
  ``initialRunLabel`` fallback when table lacks ``run_label`` column (§G.6 / §G.16 / §D.7)

**ROADMAP**
- §G.6 OLAP Explorer ingest-path run-label brush + table-picker ring-highlight parity checked
- §G.14 OLAP Explorer table-picker click-to-brush parity with Output Browser run list checked
- §G.16 OLAP Explorer path-derived run-label brush hook parity with Data Explorer checked
- §D.7 file-based run-label brush parity across all analysis views including OLAP Explorer checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-sixth pass (§G.1 + §G.6 + §G.16 + §D.7)

Hundred-seventy-sixth pass closes the portfolio analytics run-label brush hook gap left
after the hundred-seventy-fifth pass (which unified file-based brush hooks on Simulation
Summary, Simulation Monitor, and Output Browser). Benchmark Analysis and City Comparison
loaded-run lists now share Simulation Summary ring-highlight + click-to-brush parity, and
Algorithm Comparison / Data Explorer gain path-derived run-label brush hooks.

**React frontend**
- Benchmark Analysis — ``runLabelMapFromPaths`` + ``handleRunLabelClick`` on loaded-run
  list; ring highlight when global brush matches; ``GlobalFilterBar`` ``runLabels`` in
  single-run portfolio mode (§G.1 / §G.6 / §D.7)
- City Comparison — loaded-run list ring highlight + click-to-brush parity with
  Simulation Summary comparison runs (§G.1.6 / §G.6 / §D.7)
- Algorithm Comparison — ``useLogPathRunLabelBrush`` on Simulation Monitor watch path;
  ``GlobalFilterBar`` ``runLabels`` when a log is active (§G.1 / §G.16 / §D.7)
- Data Explorer — ``useLogPathRunLabelBrush`` on open CSV path; path-derived
  ``runLabels`` + ``PolicyTelemetryTrendsPanel`` ``initialRunLabel`` fallback when CSV
  lacks ``run_label`` column (§G.6 / §G.16 / §D.7)

**ROADMAP**
- §G.1 Benchmark Analysis + City Comparison portfolio run-label ring-highlight parity checked
- §G.6 portfolio loaded-run brush + single-run filter bar parity checked
- §G.16 Algorithm Comparison + Data Explorer path-derived run-label brush hook parity checked
- §D.7 portfolio + analytics page run-label brush parity across all analysis views checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-fifth pass (§G.1 + §G.14 + §G.16 + §D.7)

Hundred-seventy-fifth pass closes the Simulation Summary file-based run-label brush hook gap left
after the hundred-seventy-fourth pass (which unified Simulation Monitor and Output Browser
file-based brush hooks). Simulation Summary now auto-syncs the global brush on log open, and
file-based pages share one path-derived run-label map helper for list ring highlights.

**React frontend**
- ``runLabelMapFromPaths`` — shared helper deriving per-run ``run_label`` from paths for
  file-based row ring highlights (§G.1 / §G.14 / §D.7)
- Simulation Summary — ``useLogPathRunLabelBrush`` on primary log open; ``GlobalFilterBar``
  ``runLabels`` in single-log mode; ``PolicyTelemetryTrendsPanel`` uses hook-derived label
  (§G.1 / §G.16 / §D.7)
- Output Browser — ``runLabelMapFromPaths`` replaces inline ``runLabelFromPath`` in run list
  ring highlights (§G.14 / §D.7)
- Simulation Summary — comparison-run list ring highlight via ``runLabelMapFromPaths``
  (§G.1 / §G.6 / §D.7)

**ROADMAP**
- §G.1 Simulation Summary ``useLogPathRunLabelBrush`` + ``GlobalFilterBar`` run-label parity checked
- §G.14 Output Browser ``runLabelMapFromPaths`` ring-highlight parity checked
- §G.16 Simulation Summary file-based run-label brush hook parity with Simulation Monitor checked
- §D.7 file-based run-label brush + ring-highlight parity across Summary / Monitor / Output Browser checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-fourth pass (§G.14 + §G.16 + §D.7)

Hundred-seventy-fourth pass closes the file-based run-label brush hook gap left
after the hundred-seventy-third pass (which unified Process Monitor process-based
run-label brush hooks). Simulation Monitor and Output Browser now use the same
shared path-based brush hook as process launcher/monitor pages.

**React frontend**
- ``useLogPathRunLabelBrush`` — shared hook deriving ``run_label`` from log/run
  paths and syncing the global brush (§G.14 / §G.16 / §D.7)
- Simulation Monitor — ``GlobalFilterBar`` ``runLabels`` when a log is open;
  global brush sync on log open via shared hook (§G.16 / §D.7)
- Output Browser — ``useLogPathRunLabelBrush`` replaces inline ``setRunLabel``
  in ``selectRun``; ``PolicyTelemetryTrendsPanel`` uses hook-derived label
  (§G.14 / §D.7)

**ROADMAP**
- §G.14 Output Browser shared log-path run-label brush hook parity checked
- §G.16 Simulation Monitor ``GlobalFilterBar`` run-label + brush sync parity checked
- §D.7 file-based workflow run-label brush hook parity with process-based pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-third pass (§G.15 + §D.7)

Hundred-seventy-third pass closes the Process Monitor run-label brush hook gap left
after the hundred-seventy-second pass (which unified launcher and monitor card-header
run-label suffixes). Process Monitor now uses the same shared brush hook and per-process
run-label map helper as the launcher and monitor card pages.

**React frontend**
- ``runLabelMapFromProcesses`` — shared helper deriving per-process ``run_label`` from
  stdout for process row ring highlights (§G.15 / §D.7)
- Process Monitor — ``useProcessRunLabelBrush`` replaces inline ``runLabelFromLogLines``
  + manual ``setRunLabel`` effect; global brush sync parity with launcher/monitor pages
  (§G.15 / §D.7)

**ROADMAP**
- §G.15 Process Monitor shared run-label brush hook parity checked
- §D.7 Process Monitor ``useProcessRunLabelBrush`` + ``runLabelMapFromProcesses`` deduplication checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-second pass (§G.9 + §G.10 + §G.11 + §G.12 + §G.15 + §G.17 + §G.18 + §D.7)

Hundred-seventy-second pass closes the launcher and monitor card-header run-label gap left
after the hundred-seventy-first pass (which unified Process Monitor embedded run-label
suffixes). All workflow live panels now show accent-secondary run labels and · live suffixes
in card headers, with shared global ``run_label`` brush sync via ``useProcessRunLabelBrush``.

**React frontend**
- ``useProcessRunLabelBrush`` — shared hook deriving ``run_label`` from process stdout and
  syncing the global brush (§G.9–§G.18 / §D.7)
- ``LauncherLivePanelHeader`` — ``runLabel`` + ``showLiveSuffix`` on card variant headers
  (§G.9 / §G.11 / §G.12 / §G.10 / §D.7)
- ``TrainHpoLivePanelHeader`` — ``runLabel`` + ``showLiveSuffix`` on split and inline card
  layouts (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- Simulation Launcher + Data Generation + Evaluation Runner + Training Hub — pass ``runLabel``
  to live panel headers; ``GlobalFilterBar`` ``runLabels`` when a process is active
  (§G.9 / §G.10 / §G.11 / §G.12 / §D.7)
- Training Monitor + HPO Tracker + Experiment Tracker — pass ``runLabel`` + ``showLiveSuffix``
  to ``TrainHpoLivePanel`` card headers; ``GlobalFilterBar`` ``runLabels`` sync
  (§G.15 / §G.17 / §G.18 / §D.7)

**ROADMAP**
- §G.9 Simulation Launcher card header run-label parity checked
- §G.10 Training Hub card header run-label parity checked
- §G.11 Data Generation card header run-label parity checked
- §G.12 Evaluation Runner card header run-label parity checked
- §G.15 / §G.17 / §G.18 train/HPO monitor card header run-label parity checked
- §D.7 launcher + monitor workflow card header run-label + live suffix parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventy-first pass (§G.9 + §G.11 + §G.12 + §G.15 + §D.7)

Hundred-seventy-first pass closes the Process Monitor embedded run-label and live-suffix
gap left after the hundred-seventieth pass (which unified train/HPO live panel titles).
Eval, data-gen, and train/HPO embedded sections now match the sim panel's muted subtitle
header with run label and · live suffix; global run brush sync applies to all workflow kinds.

**React frontend**
- ``TrainHpoLivePanelHeader`` — ``runLabel`` prop for embedded run-label suffix parity with
  ``LauncherLivePanelHeader`` (§G.15 / §D.7)
- ``TrainHpoLivePanel`` — ``embedded`` variant defaults ``titleTone: muted`` +
  ``showLiveSuffix: true`` for train/HPO analytics subtitles (§G.15 / §D.7)
- Process Monitor — eval + data-gen + train/HPO embedded sections pass ``runLabel``;
  process row ring highlight + global ``run_label`` brush sync for all workflow kinds
  (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)

**ROADMAP**
- §G.9 Simulation Launcher Process Monitor sim run-label parity checked (embedded section)
- §G.11 Data Generation Process Monitor run-label parity checked
- §G.12 Evaluation Runner Process Monitor run-label parity checked
- §G.15 Process Monitor embedded run-label + live suffix parity across all workflow kinds checked
- §D.7 Process Monitor embedded header run-label + live suffix parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventieth pass (§G.10 + §G.15 + §G.17 + §G.18 + §D.7)

Hundred-seventieth pass closes the train/HPO live panel title gap left after the
hundred-sixty-ninth pass (which unified sim and data-gen live panel titles). Training Hub,
Training Monitor, HPO Tracker, Experiment Tracker, and Process Monitor now share one title
helper per workflow kind for running, completed, and failed states.

**React frontend**
- ``trainHpoLivePanelTitle`` — shared live/post-run train/HPO panel title helper in
  ``trainingProcess.ts`` (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- Training Hub + Training Monitor + HPO Tracker + Experiment Tracker — deduplicated inline
  train/HPO live title strings; import shared ``trainHpoLivePanelTitle`` (§G.10 / §G.17 / §G.18 / §D.7)
- Process Monitor — selected ``train_`` / ``hpo_`` embedded sections use dynamic
  ``trainHpoLivePanelTitle`` instead of static ``Training analytics`` subtitle
  (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)

**ROADMAP**
- §G.10 Training Hub ``trainHpoLivePanelTitle`` checked
- §G.15 Process Monitor train/HPO embedded live panel title checked
- §G.17 Training Monitor ``trainHpoLivePanelTitle`` checked
- §G.18 HPO Tracker + Experiment Tracker ``trainHpoLivePanelTitle`` checked
- §D.7 train/HPO workflow live panel title parity across all five pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-ninth pass (§G.9 + §G.11 + §G.15 + §D.7)

Hundred-sixty-ninth pass closes the sim and data-gen live panel title gap left after the
hundred-sixty-eighth pass (which unified eval live panel titles). Simulation Launcher,
Data Generation, and Process Monitor now share one title helper per workflow kind for
running, completed, and failed states.

**React frontend**
- ``simLivePanelTitle`` — shared live/post-run sim panel title helper in
  ``launcherProcess.ts`` (§G.9 / §G.15 / §D.7)
- ``dataGenLivePanelTitle`` — shared live/post-run data-gen panel title helper in
  ``launcherProcess.ts`` (§G.11 / §G.15 / §D.7)
- Simulation Launcher + Data Generation — deduplicated inline sim/data-gen live title
  strings; import shared title helpers (§G.9 / §G.11 / §D.7)
- Process Monitor — selected ``test_sim`` / ``gen_data`` embedded sections use dynamic
  title helpers instead of static subtitles (§G.9 / §G.11 / §G.15 / §D.7)

**ROADMAP**
- §G.9 Simulation Launcher ``simLivePanelTitle`` checked
- §G.11 Data Generation ``dataGenLivePanelTitle`` checked
- §G.15 Process Monitor sim + data-gen embedded live panel titles checked
- §D.7 sim + data-gen launcher live panel title parity across Simulation Launcher, Data Generation, and Process Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-eighth pass (§G.10 + §G.12 + §G.15 + §D.7)

Hundred-sixty-eighth pass closes the eval live panel title gap left after the
hundred-sixty-seventh pass (which fixed Training Hub eval progress duplication and
added ``Training Hub →`` to eval ``LauncherNavMesh``). All three eval surfaces now
share one title helper for running, completed, and failed states.

**React frontend**
- ``evalLivePanelTitle`` — shared live/post-run eval panel title helper in
  ``evalResults.ts``; supports single- and multi-checkpoint batch wording
  (§G.10 / §G.12 / §G.15 / §D.7)
- Training Hub + Evaluation Runner — deduplicated inline eval live title strings;
  import shared ``evalLivePanelTitle`` (§G.10 / §G.12 / §D.7)
- Process Monitor — selected ``eval`` embedded section uses dynamic
  ``evalLivePanelTitle`` instead of static ``Eval results`` subtitle
  (§G.12 / §G.15 / §D.7)

**ROADMAP**
- §G.10 Training Hub ``evalLivePanelTitle`` checked
- §G.12 Evaluation Runner ``evalLivePanelTitle`` checked
- §G.15 Process Monitor eval embedded live panel title checked
- §D.7 eval launcher live panel title parity across Training Hub, Evaluation Runner, and Process Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-seventh pass (§G.10 + §G.12 + §G.15 + §D.7)

Hundred-sixty-seventh pass closes the eval progress and navigation gaps left after the
hundred-sixty-sixth pass (which routed Training Hub eval mode through
``LauncherLivePanel``). Single-checkpoint eval workflows no longer render a duplicate
``LiveTrainProgressBar`` at the panel shell level, and eval ``LauncherNavMesh`` now
links back to Training Hub from Evaluation Runner and Process Monitor.

**React frontend**
- Training Hub — eval live panel omits ``LauncherLivePanel`` ``progress`` prop;
  ``EvalCheckpointLiveCard`` owns the progress bar during runs (§G.10 / §G.12 / §D.7)
- ``LauncherNavMesh`` — ``Training Hub →`` shortcut on eval workflows; optional
  ``hideHub`` prop suppresses self-link on Training Hub eval panel (§G.10 / §G.12 / §G.15 / §D.7)

**ROADMAP**
- §G.10 Training Hub eval single-checkpoint progress bar checked
- §G.12 Evaluation Runner / Process Monitor ``LauncherNavMesh`` Training Hub link checked
- §G.15 Process Monitor eval embedded ``LauncherNavMesh`` Training Hub link checked
- §D.7 eval launcher progress + navigation parity across Training Hub, Evaluation Runner, and Process Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-sixth pass (§G.10 + §G.12 + §D.7)

Hundred-sixty-sixth pass closes the eval live panel shell gap left after the
hundred-sixty-fifth pass (which added ``LauncherLivePanel`` ``logLines`` shell
parity on Evaluation Runner and Process Monitor). Training Hub eval mode now uses
the shared launcher eval panel instead of ``TrainHpoLivePanel``, matching the
Evaluation Runner single-checkpoint pattern.

**React frontend**
- Training Hub — eval mode renders ``LauncherLivePanel`` + ``EvalCheckpointLiveCard``
  / ``EvalResultCard`` instead of ``TrainHpoLivePanel`` (§G.10 / §G.12 / §D.7)
- Training Hub — eval live panel passes ``logLines`` to shared ``LauncherLivePanel``
  shell; ``EvalCheckpointLiveCard`` omits inline tail via ``showLogTail={false}``
  (§G.10 / §G.12 / §D.7)
- Training Hub — eval ``LauncherNavMesh`` post-run shortcuts (Output Browser,
  Evaluation Runner reload, Benchmark Analysis) parity with Evaluation Runner
  (§G.10 / §G.12 / §D.7)

**ROADMAP**
- §G.10 Training Hub eval ``LauncherLivePanel`` shell checked
- §G.12 Training Hub eval ``EvalCheckpointLiveCard`` / ``EvalResultCard`` checked
- §D.7 eval launcher live panel shell parity across Training Hub and Evaluation Runner checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-fifth pass (§G.12 + §G.15 + §D.7)

Hundred-sixty-fifth pass closes the eval log tail shell gap left after the
hundred-sixty-fourth pass (which added ``logLines`` props on ``LauncherLivePanel``
for sim/data-gen workflows). Single-checkpoint eval workflows now pass raw stdout
lines to ``LauncherLivePanel`` instead of rendering inline ``ProcessLogTail`` on
``EvalCheckpointLiveCard``, matching the sim/data-gen panel pattern. Multi-checkpoint
batch eval retains per-card compact tails.

**React frontend**
- ``EvalCheckpointLiveCard`` — optional ``showLogTail`` prop; omit inline tail when
  parent panel renders it (§G.12 / §D.7)
- Evaluation Runner — single-checkpoint live panel passes ``logLines`` to shared
  ``LauncherLivePanel`` shell; multi-checkpoint batch keeps per-card compact tails
  (§G.12 / §D.7)
- Process Monitor — selected ``eval`` embedded section passes ``logLines`` to
  ``LauncherLivePanel`` instead of inline ``ProcessLogTail`` on live card
  (§G.12 / §G.15 / §D.7)

**ROADMAP**
- §G.12 Evaluation Runner ``LauncherLivePanel`` ``logLines`` prop checked
- §G.12 ``EvalCheckpointLiveCard`` ``showLogTail`` prop checked
- §G.15 Process Monitor eval embedded ``LauncherLivePanel`` log tail checked
- §D.7 eval launcher log tail shell parity across Evaluation Runner and Process Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-fourth pass (§G.9 + §G.11 + §G.15 + §D.7)

Hundred-sixty-fourth pass closes the launcher log tail shell gap left after the
hundred-sixty-third pass (which added ``logLines`` props on ``TrainHpoLivePanel``).
Launcher workflows now pass raw stdout lines to ``LauncherLivePanel`` instead of
rendering inline ``ProcessLogTail`` children, matching the train/HPO panel pattern.

**React frontend**
- ``LauncherLivePanel`` — optional ``logLines`` + ``logTailWaiting`` props render
  shared ``ProcessLogTail`` below children (§G.9 / §G.11 / §G.15 / §D.7)
- Simulation Launcher — deduplicated inline ``ProcessLogTail`` child; pass
  ``logLines`` to shared panel shell (§G.9 / §D.7)
- Data Generation Wizard — deduplicated inline ``ProcessLogTail`` child; pass
  ``logLines`` to shared panel shell (§G.11 / §D.7)
- Process Monitor — selected ``test_sim`` / ``gen_data`` embedded sections pass
  ``logLines`` to ``LauncherLivePanel`` instead of inline ``ProcessLogTail``
  (§G.9 / §G.11 / §G.15 / §D.7)

**ROADMAP**
- §G.9 Simulation Launcher ``LauncherLivePanel`` ``logLines`` prop checked
- §G.11 Data Generation ``LauncherLivePanel`` ``logLines`` prop checked
- §G.15 Process Monitor sim/data-gen embedded ``LauncherLivePanel`` log tail checked
- §D.7 launcher workflow log tail shell parity across all launcher pages + Process Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-third pass (§G.10 + §G.15 + §G.17 + §G.18 + §D.7)

Hundred-sixty-third pass closes the train/HPO log tail display gap left after the
hundred-sixty-second pass (which added ``ProcessLogTail`` on Simulation Launcher,
Data Generation, Evaluation Runner, and Process Monitor sim/data-gen/eval
embedded sections). Train/HPO workflows now show the shared stdout tail via
``TrainHpoLivePanel`` across Training Hub, Training Monitor, HPO Tracker,
Experiment Tracker, and Process Monitor train/HPO embedded sections.

**React frontend**
- ``TrainHpoLivePanel`` — optional ``logLines`` + ``logTailWaiting`` props render
  shared ``ProcessLogTail`` below analytics strip (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- Training Hub — ``ProcessLogTail`` in live progress panel during train/hpo/eval runs
  (§G.10 / §D.7)
- Process Monitor — selected ``train_`` / ``hpo_`` processes show ``ProcessLogTail``
  in embedded analytics section (§G.15 / §D.7)
- Training Monitor — ``ProcessLogTail`` on live/recent train panel (§G.17 / §D.7)
- HPO Tracker + Experiment Tracker — ``ProcessLogTail`` on live HPO panels (§G.18 / §D.7)

**ROADMAP**
- §G.10 Training Hub ``ProcessLogTail`` checked
- §G.15 Process Monitor train/HPO embedded log tail checked
- §G.17 Training Monitor ``ProcessLogTail`` checked
- §G.18 HPO Tracker + Experiment Tracker ``ProcessLogTail`` checked
- §D.7 train/HPO workflow log tail display parity across all five pages + Process Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-second pass (§G.9 + §G.12 + §G.15 + §D.7)

Hundred-sixty-second pass closes the remaining launcher log tail display gaps left
after the hundred-sixty-first pass (which added ``ProcessLogTail`` on Data
Generation, ``EvalCheckpointLiveCard``, and Process Monitor ``gen_data``).
Simulation Launcher and Process Monitor ``test_sim`` embedded sections now show
the shared stdout tail, and ``EvalCheckpointLiveCard`` accepts raw ``logLines``
so callers no longer pre-format tails via ``processLogTail``.

**React frontend**
- ``EvalCheckpointLiveCard`` — accepts ``logLines`` + optional ``maxLines``;
  deduplicated ``processLogTail`` calls at Evaluation Runner and Process Monitor
  (§G.12 / §D.7)
- Simulation Launcher — ``ProcessLogTail`` in live status panel during ``test_sim``
  runs (§G.9 / §D.7)
- Process Monitor — selected ``test_sim`` processes show ``ProcessLogTail`` in
  embedded workflow section (§G.9 / §G.15 / §D.7)
- Evaluation Runner — passes raw ``logLines`` to ``EvalCheckpointLiveCard`` instead
  of pre-formatted tail (§G.12 / §D.7)

**ROADMAP**
- §G.9 Simulation Launcher ``ProcessLogTail`` checked
- §G.12 ``EvalCheckpointLiveCard`` ``logLines`` prop checked
- §G.15 Process Monitor ``test_sim`` embedded log tail checked
- §D.7 launcher log tail display parity across all four launcher pages + Process Monitor embedded sections checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixty-first pass (§G.11 + §G.12 + §G.15 + §D.7)

Hundred-sixty-first pass closes the launcher log tail display gap left after the
hundred-sixtieth pass (which added ``processLogTail`` helper + Process Monitor
``EvalCheckpointLiveCard`` eval parity). Log tail rendering is now deduplicated
into a shared ``ProcessLogTail`` component used across Data Generation, eval live
cards, and Process Monitor embedded sections.

**React frontend**
- ``ProcessLogTail`` — shared stdout/stderr tail display with ``compact`` and
  ``default`` variants for launcher live panels (§G.11 / §G.12 / §G.15 / §D.7)
- ``EvalCheckpointLiveCard`` — deduplicated inline log tail markup; import shared
  ``ProcessLogTail`` (§G.12 / §D.7)
- Data Generation Wizard — deduplicated inline log tail formatting; import shared
  ``processLogTail`` + ``ProcessLogTail`` (§G.11 / §D.7)
- Process Monitor — selected ``gen_data`` processes show ``ProcessLogTail`` in
  embedded workflow section (§G.11 / §G.15 / §D.7)

**ROADMAP**
- §G.11 Data Generation ``ProcessLogTail`` checked
- §G.12 ``EvalCheckpointLiveCard`` ``ProcessLogTail`` checked
- §G.15 Process Monitor ``gen_data`` embedded log tail checked
- §D.7 launcher log tail display parity across Data Generation, Evaluation Runner, and Process Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixtieth pass (§G.12 + §G.15 + §D.7)

Hundred-sixtieth pass closes the Process Monitor eval live panel gap left after the
hundred-fifty-ninth pass (which added ``EvalCheckpointLiveCard`` on Evaluation Runner).
Process Monitor now shows the same per-checkpoint live row during running eval
processes, and log tail formatting is deduplicated into a shared helper.

**React frontend**
- ``processLogTail`` — shared stdout/stderr tail helper for live eval panels (§G.12 /
  §G.15 / §D.7)
- Process Monitor — selected ``eval`` processes use ``EvalCheckpointLiveCard`` during
  live runs and while waiting for structured JSON; ``EvalResultCard`` retained on
  completion with metrics + ``Open in Analytics →`` (§G.12 / §G.15 / §D.7)
- Evaluation Runner — deduplicated inline log tail formatting; import shared
  ``processLogTail`` (§G.12 / §D.7)

**ROADMAP**
- §G.12 Evaluation Runner ``processLogTail`` checked
- §G.12 Process Monitor ``EvalCheckpointLiveCard`` live eval parity checked
- §G.15 Process Monitor running eval progress + stdout tail checked
- §D.7 eval live checkpoint card parity across Evaluation Runner and Process Monitor checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-ninth pass (§G.10 + §G.12 + §G.15 + §G.17 + §G.18 + §D.7)

Hundred-fifty-ninth pass closes the monitor-page footer gap left after the
hundred-fifty-eighth pass (which added ``ProcessIdFooter`` on launcher pages and
Training Hub). All train/HPO monitor pages and Process Monitor embedded sections
now show a shared process-id footer row, and Evaluation Runner per-checkpoint
live rows are deduplicated into a shared card component.

**React frontend**
- ``ProcessIdFooter`` — monitor-page footer parity across Training Monitor, HPO
  Tracker, Experiment Tracker, and Process Monitor embedded sections; process id
  removed from inline headers (§G.15 / §G.17 / §G.18 / §D.7)
- Training Monitor + HPO Tracker + Experiment Tracker — ``TrainHpoLivePanel``
  ``footer`` process-id row parity with Training Hub (§G.10 / §G.17 / §G.18 / §D.7)
- Process Monitor — ``LauncherLivePanel`` + ``TrainHpoLivePanel`` embedded
  sections use ``ProcessIdFooter``; simplified analytics subtitles without inline
  process id (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- ``EvalCheckpointLiveCard`` — shared per-checkpoint live eval row with KPI,
  progress bar, and stdout tail (§G.12 / §D.7)
- Evaluation Runner — deduplicated inline per-checkpoint live panel markup;
  import shared ``EvalCheckpointLiveCard`` (§G.12 / §D.7)

**ROADMAP**
- §G.10 Training Hub footer parity extended to Training Monitor checked
- §G.15 Process Monitor embedded ``ProcessIdFooter`` checked
- §G.17 Training Monitor ``ProcessIdFooter`` footer checked
- §G.18 HPO Tracker + Experiment Tracker ``ProcessIdFooter`` footer checked
- §G.12 Evaluation Runner ``EvalCheckpointLiveCard`` checked
- §D.7 train/HPO + launcher workflow footer parity across all pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-eighth pass (§G.9 + §G.10 + §G.11 + §G.12 + §G.15 + §D.7)

Hundred-fifty-eighth pass closes the launcher live panel footer gap left after the
hundred-fifty-seventh pass (which added ``LauncherLivePanel`` shell parity on sim /
data-gen / eval workflow pages). All launcher pages now show a shared process-id
footer row, and eval KPI markup is deduplicated across Evaluation Runner and
Process Monitor.

**React frontend**
- ``ProcessIdFooter`` — shared process-id footer row for launcher and train/HPO
  live panels; supports single id or multi-checkpoint eval batches (§G.9 / §G.10 /
  §G.11 / §G.12 / §D.7)
- ``EvalResultKpiRow`` — shared cost / gap / time / policy KPI row with
  ``compact`` and ``default`` size variants (§G.12 / §G.15 / §D.7)
- ``EvalResultCard`` — shared eval result card with checkpoint title +
  ``Open in Analytics →`` for Process Monitor embedded eval sections (§G.12 /
  §G.15 / §D.7)
- Simulation Launcher + Training Hub — deduplicated inline process-id footer
  markup; import shared ``ProcessIdFooter`` (§G.9 / §G.10 / §D.7)
- Data Generation Wizard + Evaluation Runner — ``LauncherLivePanel`` ``footer``
  process-id row parity with Simulation Launcher (§G.11 / §G.12 / §D.7)
- Evaluation Runner — per-checkpoint live panel uses ``EvalResultKpiRow``
  ``compact`` variant (§G.12 / §D.7)
- Process Monitor — embedded eval section uses ``EvalResultCard`` (§G.12 / §G.15 /
  §D.7)

**ROADMAP**
- §G.9 Simulation Launcher ``ProcessIdFooter`` checked
- §G.10 Training Hub ``ProcessIdFooter`` checked
- §G.11 Data Generation ``footer`` process-id row checked
- §G.12 Evaluation Runner footer + ``EvalResultKpiRow`` checked
- §G.15 Process Monitor ``EvalResultCard`` checked
- §D.7 launcher + eval KPI/footer parity across all workflow pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-seventh pass (§G.9 + §G.11 + §G.12 + §G.15 + §D.7)

Hundred-fifty-seventh pass closes the launcher live panel shell gap left after the
hundred-fifty-sixth pass (which added ``TrainHpoLivePanel`` shell parity on train/HPO
workflow pages). All four sim / data-gen / eval launcher pages now share
``LauncherLivePanel`` so header, progress bar, and body content render inside one
consistent card or embedded shell.

**React frontend**
- ``LauncherLivePanelHeader`` — shared status icon + title + ``LauncherNavMesh`` row
  with ``card`` / ``embedded`` variants; ``runLabel`` + live suffix on Process Monitor
  sim panels (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- ``LauncherLivePanel`` — shared header + ``LiveTrainProgressBar`` + children shell
  with ``card`` / ``embedded`` variants; ``navTrailing`` slot preserves Simulation
  Launcher auto-summary countdown (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- Simulation Launcher — deduplicated inline live status card markup; ``footer``
  process-id row preserved via shared panel (§G.9 / §D.7)
- Data Generation Wizard — deduplicated inline live progress card markup (§G.11 / §D.7)
- Evaluation Runner — deduplicated inline live progress card markup (§G.12 / §D.7)
- Process Monitor — ``embedded`` variant for selected ``test_sim`` / ``gen_data`` /
  ``eval`` analytics sections (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)

**ROADMAP**
- §G.9 Simulation Launcher ``LauncherLivePanel`` shell checked
- §G.11 Data Generation ``LauncherLivePanel`` shell checked
- §G.12 Evaluation Runner ``LauncherLivePanel`` shell checked
- §G.15 Process Monitor ``embedded`` launcher analytics sections checked
- §D.7 launcher workflow panel shell parity across all four pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-sixth pass (§G.10 + §G.15 + §G.17 + §G.18 + §A.2 + §A.4 + §D.7)

Hundred-fifty-sixth pass closes the live panel shell gap left after the
hundred-fifty-fifth pass (which added ``TrainHpoLivePanelHeader`` ``overlaySelect``
parity on Training Monitor). All five train/HPO workflow pages now share
``TrainHpoLivePanel`` so header, progress bar, and analytics strip render inside
one consistent card or embedded shell.

**React frontend**
- ``TrainHpoLivePanel`` — shared header + ``LiveTrainProgressBar`` +
  ``TrainHpoAnalyticsStrip`` shell with ``card`` / ``embedded`` variants (§G.10 /
  §G.15 / §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- Training Hub — ``footer`` process-id row + ``showAnalytics`` /
  ``analyticsWrapperClassName`` slots preserved via shared panel (§G.10 / §D.7)
- Process Monitor — ``embedded`` variant for selected train/HPO analytics section (§G.15 / §D.7)
- Training Monitor — ``overlaySelect`` + ``showHealthAttention={false}`` options
  preserved via shared panel (§G.17 / §A.2 / §A.4 / §D.7)
- HPO Tracker + Experiment Tracker — deduplicated inline live HPO card markup (§G.18 / §G.17 / §D.7)

**ROADMAP**
- §G.10 Training Hub ``TrainHpoLivePanel`` shell checked
- §G.15 Process Monitor ``embedded`` variant checked
- §G.17 Training Monitor overlay + analytics options preserved checked
- §G.18 HPO Tracker + Experiment Tracker shared live panel shell checked
- §A.2 / §A.4 train/HPO workflow live panel shell parity checked
- §D.7 train/HPO workflow panel shell parity across all five pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-fifth pass (§G.17 + §A.2 + §A.4 + §D.7)

Hundred-fifty-fifth pass closes the Training Monitor live panel header gap left after the
hundred-fifty-fourth pass (which deduplicated ``TrainHpoLivePanelHeader`` across Training Hub,
HPO Tracker, Experiment Tracker, and Process Monitor). Training Monitor now shares the same
header component with optional ``overlaySelect`` for ``LIVE_KEY`` multi-run overlay checkbox
parity, completing train/HPO workflow header row consistency across all five pages.

**React frontend**
- ``TrainHpoLivePanelHeader`` — ``overlaySelect`` prop wraps inline title row in a checkbox
  label for Training Monitor ``LIVE_KEY`` overlay selection (§G.17 / §A.2 / §A.4 / §D.7)
- Training Monitor — deduplicated inline live/recent header blocks; shared status icon +
  title + process id + rehydration badges + ``TrainHpoNavMesh`` row (§G.17 / §D.7)
- Live/recent card ``space-y-3`` spacing parity with other train/HPO workflow pages (§G.17 / §D.7)

**ROADMAP**
- §G.17 Training Monitor ``TrainHpoLivePanelHeader`` + ``overlaySelect`` checked
- §A.2 / §A.4 live panel header + nav mesh parity on Training Monitor checked
- §D.7 train/HPO workflow header row parity across all five pages checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-fourth pass (§G.10 + §G.15 + §G.17 + §G.18 + §A.2 + §A.4 + §D.7)

Hundred-fifty-fourth pass closes the live panel header parity gap left after the
hundred-fifty-third pass (which deduplicated ``TrainHpoRehydrationBadges`` across
train/HPO workflow pages). Training Hub, HPO Tracker, Experiment Tracker, and
Process Monitor now share ``TrainHpoLivePanelHeader`` so status icons, titles,
rehydration badges, and ``TrainHpoNavMesh`` shortcuts render consistently.

**React frontend**
- ``TrainHpoLivePanelHeader`` — shared status icon + title + optional process id +
  rehydration badges + nav mesh row for train/HPO live panels (§G.10 / §G.15 /
  §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- Training Hub — ``split`` layout + ``activity`` running icon via shared header (§G.10 / §D.7)
- HPO Tracker + Experiment Tracker — deduplicated inline live HPO header blocks (§G.18 / §G.17 / §D.7)
- Process Monitor — ``muted`` analytics subtitle header + badges-before-nav ordering parity (§G.15 / §D.7)

**ROADMAP**
- §G.10 Training Hub ``TrainHpoLivePanelHeader`` split layout checked
- §G.15 Process Monitor muted analytics header parity checked
- §G.17 HPO Tracker shared live header deduplication checked
- §G.18 Experiment Tracker shared live header deduplication checked
- §A.2 / §A.4 live panel header + nav mesh parity checked
- §D.7 train/HPO workflow header row parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-third pass (§G.10 + §G.15 + §G.17 + §G.18 + §A.2 + §A.4 + §D.7)

Hundred-fifty-third pass closes the header badge parity gap left after the
hundred-fifty-second pass (which aligned post-run banner counts and metric-label
styling). All train/HPO workflow pages now share ``TrainHpoRehydrationBadges``
so live panel headers surface health alerts and attention snapshots alongside
metric updates when rehydrated from ``useProcessStore``.

**React frontend**
- ``TrainHpoRehydrationBadges`` — shared metric / health / attention count badges
  for train/HPO live panel headers (§G.10 / §G.15 / §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- Training Hub + Process Monitor + Training Monitor + HPO Tracker + Experiment
  Tracker — deduplicated inline ``metric updates`` labels; header badges show
  health + attention counts when present (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- Training Monitor — checkbox live/recent header no longer shows ``0 metric updates``
  when only health/attention are rehydrated (§G.17 / §A.2 / §A.4 / §D.7)

**ROADMAP**
- §G.10 Training Hub ``TrainHpoRehydrationBadges`` deduplication checked
- §G.15 Process Monitor shared header badges checked
- §G.17 Training Monitor health/attention header badge parity checked
- §G.18 HPO Tracker + Experiment Tracker ``TrainHpoRehydrationBadges`` checked
- §A.2 / §A.4 health/attention header rehydration counts checked
- §D.7 train/HPO workflow header badge parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-second pass (§G.10 + §G.17 + §A.2 + §A.4 + §D.7)

Hundred-fifty-second pass closes the post-run banner and metric-label parity gap left after the
hundred-fifty-first pass (which deduplicated ``TrainHpoAnalyticsStrip`` across train/HPO workflow
pages). Training Monitor now feeds rehydrated health and attention entries into the shared strip
for accurate ``postRunTrainingRehydrationMessage`` counts while keeping page-level panels
separate, and both Training Hub and Training Monitor align ``metric updates`` label styling and
visibility with Process Monitor / HPO Tracker / Experiment Tracker.

**React frontend**
- Training Monitor — ``TrainHpoAnalyticsStrip`` receives ``effectiveLiveHealth`` +
  ``effectiveLiveAttention`` for post-run banner counts; ``showHealthAttention={false}`` preserves
  page-level ``TrainingHealthPanel`` / ``RuntimeAttentionPanel`` (§G.17 / §A.2 / §A.4 / §D.7)
- Training Monitor — ``metric updates`` label on non-checkbox live/recent header when metrics are
  rehydrated from ``useProcessStore`` (§G.17 / §D.7)
- Training Hub — ``metric updates`` label uses ``text-accent-success`` styling parity (§G.10 / §D.7)

**ROADMAP**
- §G.17 Training Monitor post-run health/attention banner counts checked
- §G.10 Training Hub metric updates styling parity checked
- §A.2 / §A.4 post-run attention/health rehydration banner counts checked
- §D.7 train/HPO workflow metric label parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifty-first pass (§G.10 + §G.15 + §G.17 + §G.18 + §D.7)

Hundred-fifty-first pass closes the shared analytics strip gap left after the
hundred-fiftieth pass (which deduplicated metric snapshots and post-run banners).
Training Hub, Process Monitor, HPO Tracker, and Experiment Tracker now use the
shared ``TrainHpoAnalyticsStrip`` component, and Training Monitor's live/recent
card restores post-run sparklines without requiring ``LIVE_KEY`` selection.

**React frontend**
- ``TrainHpoAnalyticsStrip`` — shared snapshot + sparklines + health/attention +
  post-run banner strip for train/HPO live panels (§G.10 / §G.15 / §G.17 / §G.18)
- Training Hub — ``TrainHpoAnalyticsStrip`` with ``middleContent`` slot for live
  ``LiveChart``; ``metric updates`` label parity (§G.10 / §D.7)
- Process Monitor + HPO Tracker + Experiment Tracker — deduplicated inline
  analytics blocks via ``TrainHpoAnalyticsStrip`` (§G.15 / §G.18 / §D.7)
- Training Monitor — live/recent card uses ``TrainHpoAnalyticsStrip`` for post-run
  sparkline rehydration; removes duplicate ``LIVE_KEY`` sparkline panel (§G.17 / §D.7)

**ROADMAP**
- §G.10 Training Hub ``TrainHpoAnalyticsStrip`` deduplication checked
- §G.15 Process Monitor shared analytics strip checked
- §G.17 Training Monitor live/recent sparkline rehydration checked
- §G.18 HPO Tracker + Experiment Tracker ``TrainHpoAnalyticsStrip`` checked
- §D.7 train/HPO workflow analytics strip parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fiftieth pass (§G.18 + §G.17 + §A.2 + §A.4 + §D.7)

Hundred-fiftieth pass closes the analytics snapshot and health/attention banner gap left after the
hundred-forty-ninth pass (which deduplicated sparklines on Training Hub and Training Monitor).
HPO Tracker and Experiment Tracker now use the shared ``TrainingMetricSnapshot`` component, and
all train/HPO workflow pages share ``postRunTrainingRehydrationMessage`` for post-run banners
that mention metrics, health alerts, and attention snapshots when rehydrated from the process store.

**React frontend**
- ``trainingMetrics.ts`` — ``postRunTrainingRehydrationMessage`` shared post-run banner helper
  (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- HPO Tracker + Experiment Tracker — deduplicated inline metric snapshot rows; import shared
  ``TrainingMetricSnapshot`` (§G.18 / §G.17 / §D.7)
- Training Hub + Training Monitor + Process Monitor — post-run banner uses shared helper for
  health/attention rehydration parity (§G.10 / §G.15 / §G.17 / §D.7)

**ROADMAP**
- §G.18 HPO Tracker + Experiment Tracker ``TrainingMetricSnapshot`` deduplication checked
- §G.17 analytics post-run snapshot parity checked
- §A.2 / §A.4 post-run health/attention banner rehydration checked
- §D.7 train/HPO workflow post-run banner parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-ninth pass (§G.10 + §G.17 + §D.7)

Hundred-forty-ninth pass closes the launcher sparkline gap left after the
hundred-forty-eighth pass (which rehydrated sparklines on Process Monitor,
HPO Tracker, and Experiment Tracker). Training Hub and Training Monitor now
use the shared ``TrainingMetricSparklines`` component for post-run grad-norm
and learning-rate charts.

**React frontend**
- Training Hub — ``GradNormSparkline`` + ``LrSparkline`` + ``TrainingMetricSnapshot``
  replace local ``MiniSparkline``; post-run rehydration banner when train/HPO
  completes (§G.10 / §G.17 / §D.7)
- Training Monitor — deduplicated local sparkline implementations; imports shared
  ``TrainingMetricSparklines``; ``TrainingMetricSnapshot`` on live/recent panel;
  post-run banner mentions sparkline persistence (§G.17 / §D.7)

**ROADMAP**
- §G.10 Training Hub post-run sparkline rehydration checked
- §G.17 Training Monitor shared sparkline deduplication checked
- §D.7 launcher + monitor post-run sparkline parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-eighth pass (§G.15 + §G.17 + §G.18 + §D.7)

Hundred-forty-eighth pass closes the post-run sparkline gap left after the
hundred-forty-seventh pass (which rehydrated metric snapshots on analytics pages).
Process Monitor, HPO Tracker, and Experiment Tracker now restore grad-norm and
learning-rate sparklines from persisted process stdout when navigation clears
live streaming state.

**React frontend**
- ``TrainingMetricSparklines`` — shared ``GradNormSparkline``, ``LrSparkline``, and
  ``TrainingMetricSnapshot`` for train/HPO analytics panels (§G.15 / §G.17 / §G.18)
- Process Monitor — train/HPO metrics rehydrate from ``useProcessStore`` log lines;
  metric snapshot + grad-norm/LR sparklines persist after completion (§G.15 / §D.7)
- HPO Tracker + Experiment Tracker — post-run grad-norm + LR sparklines from persisted
  HPO stdout; rehydration banner when metrics are present (§G.18 / §G.17 / §D.7)

**ROADMAP**
- §G.15 Process Monitor train/HPO post-run sparkline rehydration checked
- §G.18 HPO Tracker + Experiment Tracker post-run sparklines checked
- §G.17 ``TrainingMetricSparklines`` shared component checked
- §D.7 monitor/analytics post-run sparkline parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-seventh pass (§G.17 + §G.18 + §G.10 + §D.7)

Hundred-forty-seventh pass closes the analytics-page post-run metrics gap left after the
hundred-forty-sixth pass (which rehydrated launcher live panels from ``useProcessStore``).
Training Monitor, HPO Tracker, and Experiment Tracker now restore training metrics from
persisted process stdout when live streaming state is cleared by navigation.

**React frontend**
- ``trainingMetrics.ts`` — ``normalizeTrainingMetricRow`` exported for shared CSV + stdout
  metric normalization (§G.17 / §G.10)
- Training Monitor — ``effectiveLiveMetrics`` / ``effectiveLiveHealth`` /
  ``effectiveLiveAttention`` rehydrate from ``useProcessStore`` log lines; ``LIVE_KEY`` overlay
  chart + sparklines persist after train/HPO completion (§G.17 / §D.7)
- HPO Tracker + Experiment Tracker — live metric update count + latest epoch/loss snapshot row
  from ``collectTrainingMetricsFromLogLines`` on persisted HPO stdout (§G.18 / §G.17 / §D.7)

**ROADMAP**
- §G.17 Training Monitor post-run metrics rehydration checked
- §G.18 HPO Tracker + Experiment Tracker live metric snapshot checked
- §G.10 ``normalizeTrainingMetricRow`` shared normalization checked
- §D.7 analytics-page post-run rehydration parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-sixth pass (§G.9 + §G.10 + §G.11 + §G.12 + §D.7)

Hundred-forty-sixth pass closes the launcher post-run panel gap left after the
hundred-forty-fifth pass (which wired analytics-page deep-links on HPO Tracker,
Experiment Tracker, and Training Monitor). Launcher pages now rehydrate live and
completed-run panels from ``useProcessStore`` when navigation clears ephemeral
``liveProcessId`` component state.

**React frontend**
- ``launcherProcess.ts`` — ``findRecentLauncherProcessId`` + ``findRecentEvalProcessIds``
  retain newest sim / data-gen / eval processes after completion (§G.9 / §G.11 / §G.12)
- ``trainingProcess.ts`` — ``findRecentTrainProcessId`` + ``isTrainProcess`` for Training Hub
  train-mode post-run persistence (§G.10)
- ``dayLog.ts`` — ``collectLatestDayLogsByPolicy`` rehydrates Simulation Launcher KPI cards
  from persisted process stdout (§G.9)
- ``trainingMetrics.ts`` — ``collectTrainingMetricsFromLogLines`` rehydrates Training Hub
  live charts from persisted process stdout (§G.10)
- Simulation Launcher — ``displayProcessId`` fallback; suppress auto-navigate countdown on
  rehydrated completed runs (§G.9 / §D.7)
- Data Generation Wizard — ``displayProcessId`` fallback + stdout tail from process store
  (§G.11 / §D.7)
- Training Hub — ``findRecentHubProcessId`` per mode; metrics/health/attention derived from
  process store log lines (§G.10 / §D.7)
- Evaluation Runner — ``findRecentEvalProcessIds`` multi-checkpoint batch restore;
  results grid rehydrated via ``collectEvalResultFromLogLines`` (§G.12 / §D.7)

**ROADMAP**
- §G.9 Simulation Launcher post-run panel persistence checked
- §G.10 Training Hub post-run panel persistence checked
- §G.11 Data Generation post-run panel persistence checked
- §G.12 Evaluation Runner multi-checkpoint batch persistence checked
- §D.7 launcher navigation mesh post-run rehydration parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-fifth pass (§G.17 + §G.18 + §G.14 + §D.7)

Hundred-forty-fifth pass extends post-run ``outputRunPath`` / ``trainingRunPath`` deep-linking to
the HPO Tracker, Experiment Tracker, and Training Monitor — closing the gap left after the
hundred-forty-fourth pass (which wired deep-links on Training Hub and Process Monitor only).

**React frontend**
- ``trainingProcess.ts`` — ``findRecentHpoProcessId`` + ``findRecentTrainOrHpoProcessId`` retain
  the newest train/HPO process after completion for post-run navigation panels (§G.17 / §G.18)
- HPO Tracker — live panel persists after HPO completion; ``TrainHpoNavMesh`` post-run
  ``outputRunPath`` + ``trainingRunPath`` deep-links (§G.18 / §G.14 / §G.17 / §D.7)
- Experiment Tracker — same post-run deep-link parity as HPO Tracker (§G.18 / §D.7)
- Training Monitor — recent train/HPO panel with post-run ``TrainHpoNavMesh`` deep-links;
  auto-refresh run index + select completed run from stdout ``trainingRunPath`` (§G.17 / §G.10)

**ROADMAP**
- §G.18 HPO Tracker + Experiment Tracker post-run deep-links checked
- §G.17 Training Monitor post-run deep-link + auto-select checked
- §G.14 ``outputRunPath`` analytics-page parity checked
- §D.7 train/HPO navigation mesh post-run parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-fourth pass (§G.10 + §G.12 + §G.14 + §G.15 + §G.17 + §D.7)

Hundred-forty-fourth pass extends run deep-linking beyond ``.jsonl`` stdout paths and
closes train/HPO/eval workflow navigation gaps left after the hundred-forty-third pass.

**React frontend**
- ``outputRunPath.ts`` — ``outputRunPathFromHydraArtifact`` + Hydra snapshot / pruned-config /
  ``assets/output`` path parsing as fallback when no ``.jsonl`` in stdout (§G.14 / §G.9 / §G.12)
- ``trainingRunPath.ts`` — ``trainingRunPathFromLogLines`` derives Lightning log directories
  from ``Saved sidecar args.json`` / ``metrics.csv`` stdout paths (§G.10 / §G.17)
- ``store/app.ts`` — ``pendingTrainingRunPath`` for Training Monitor deep-link handoff (§G.17)
- ``TrainHpoNavMesh`` — ``trainingRunPath`` prop sets ``pendingTrainingRunPath`` before
  navigating to Training Monitor (§G.10 / §G.17 / §D.7)
- Training Hub — post-run ``outputRunPath`` + ``trainingRunPath`` on live panel (§G.10 / §D.7)
- Training Monitor — auto-selects run when opened via ``pendingTrainingRunPath``; refreshes
  run index when path not yet listed (§G.17)
- Evaluation Runner — post-run ``outputRunPath`` deep-link on live panel (§G.12 / §G.14)
- Process Monitor — eval ``outputRunPath`` deep-link parity; train/HPO ``outputRunPath`` +
  ``trainingRunPath`` on ``TrainHpoNavMesh`` (§G.12 / §G.14 / §G.15)

**ROADMAP**
- §G.14 Hydra snapshot / pruned-config stdout parsing checked
- §G.10 Training Hub post-run deep-links checked
- §G.12 Evaluation Runner ``outputRunPath`` deep-link checked
- §G.15 Process Monitor eval + train/HPO deep-links checked
- §G.17 ``pendingTrainingRunPath`` Training Monitor auto-select checked
- §D.7 train/HPO/eval navigation mesh deep-link parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-third pass (§G.14 + §G.9 + §G.11 + §G.15 + §D.7)

Hundred-forty-third pass completes the launcher → Output Browser workflow by deep-linking
to the completed run directory parsed from process stdout, and closes Process Monitor
Output Browser parity for simulation and data-generation processes.

**React frontend**
- ``outputRunPath.ts`` — ``outputRunPathFromJsonl`` / ``outputRunPathFromLogLines`` derive
  assets/output run roots from ``.jsonl`` paths in stdout (§G.14 / §G.9 / §G.15)
- ``LauncherNavMesh`` / ``TrainHpoNavMesh`` — ``outputRunPath`` prop sets ``pendingRunPath``
  before navigating to Output Browser (§G.14 / §D.7)
- Simulation Launcher + Data Generation — post-run Output Browser auto-selects the completed
  run when a log path is present in stdout (§G.9 / §G.11 / §G.14)
- Process Monitor — ``Output Browser →`` on completed ``test_sim`` / ``gen_data`` processes
  with the same run deep-link (§G.15 / §G.14)
- Output Browser — refreshes the run index when ``pendingRunPath`` is set but not yet listed
  (§G.14)

**ROADMAP**
- §G.14 ``outputRunPath`` + ``pendingRunPath`` launcher deep-link checked
- §G.9 Simulation Launcher Output Browser run auto-select checked
- §G.11 Data Generation Output Browser run auto-select checked
- §G.15 Process Monitor sim / data-gen Output Browser shortcuts checked
- §D.7 ``outputRunPath`` navigation mesh checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-second pass (§G.12 + §G.14 + §G.9 + §G.11 + §D.7)

Hundred-forty-second pass closes the reverse eval workflow bridge from Output Browser to
Evaluation Runner, adds shared checkpoint helpers, and extends launcher post-run Output
Browser shortcuts to simulation and data-generation workflows.

**React frontend**
- ``checkpoints.ts`` — shared ``isCheckpointEntry`` / ``filterCheckpointEntries`` helpers
  for Training Monitor and Output Browser (§G.14 / §G.12 / §G.17)
- Output Browser — auto-expand ``checkpoints/`` on run select; sidebar checkpoint card with
  **Eval →** shortcuts; file-tree highlight for ``.pt/.ckpt/.pth``; **Load in Eval Runner →**
  on selected checkpoint files via ``pendingCheckpoint`` (§G.14 / §G.12)
- Simulation Launcher + Data Generation — ``LauncherNavMesh`` ``Output Browser →`` on
  completed runs (§G.9 / §G.11 / §G.14 / §D.7)
- Evaluation Runner — single-checkpoint live panel passes ``checkpointPath`` to
  ``LauncherNavMesh`` for post-run reload (§G.12 / §D.7)

**ROADMAP**
- §G.14 Output Browser checkpoint browser + Load in Eval Runner checked
- §G.12 Evaluation Runner single-checkpoint reload shortcut checked
- §G.9 Simulation Launcher Output Browser post-run shortcut checked
- §G.11 Data Generation Output Browser post-run shortcut checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-forty-first pass (§G.12 + §G.14 + §G.15 + §D.7)

Hundred-forty-first pass completes the eval workflow navigation mesh by wiring Output Browser
and Load in Eval Runner shortcuts from completed eval processes, surfacing per-checkpoint KPIs
in the Evaluation Runner live panel, and extending keyboard shortcuts for analytics/file views.

**React frontend**
- ``evalResults.ts`` — ``checkpointPathFromEvalCommand`` extracts Hydra ``load_path`` from eval
  process commands (§G.12 / §G.15)
- ``LauncherNavMesh`` — ``Output Browser →`` + ``Load in Eval Runner →`` on completed eval
  workflows; shared ``showOutputBrowser`` prop mirrors ``TrainHpoNavMesh`` (§G.12 / §G.14 / §D.7)
- Process Monitor — eval processes surface Output Browser + Load in Eval Runner shortcuts when
  complete (§G.15 / §D.7)
- Evaluation Runner — per-checkpoint cost/gap/time KPI row in live progress panel; Output Browser
  post-run shortcut (§G.12 / §A.4)
- Keyboard shortcuts ``B`` → Benchmark Analysis, ``O`` → Output Browser; help overlay updated
  (§D.7)

**ROADMAP**
- §G.12 eval Output Browser + Load in Eval Runner + live KPI row checked
- §G.14 eval Output Browser shortcut checked
- §G.15 Process Monitor eval workflow shortcuts checked
- §D.7 ``B`` / ``O`` keyboard shortcuts checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fortieth pass (§G.12 + §G.15 + §G.10 + §D.7)

Hundred-fortieth pass closes the Process Monitor eval analytics gap by sharing eval
stdout parsing with the Evaluation Runner and wiring Benchmark Analysis navigation
from completed eval and train processes.

**React frontend**
- ``evalResults.ts`` — shared eval JSON line parsing, checkpoint label extraction, and
  ``toEvalAnalyticsRows`` for Benchmark Analysis handoff (§G.12 / §G.15)
- Process Monitor — eval results KPI panel for selected ``eval`` processes; live +
  completed cost / gap / time / policy metrics parsed from stdout (§G.12 / §G.15)
- Process Monitor — ``Open in Analytics →`` + ``LauncherNavMesh`` ``Benchmark Analysis →``
  when eval metrics are present (§D.7 / §G.12)
- Process Monitor — ``TrainHpoNavMesh`` ``Output Browser →`` on completed train/HPO
  processes (§G.10 / §D.7)
- Evaluation Runner — imports shared ``evalResults.ts`` helpers (§G.12)

**ROADMAP**
- §G.12 ``evalResults.ts`` shared parsing checked
- §G.15 Process Monitor eval results panel + analytics shortcut checked
- §G.10 Process Monitor train Output Browser shortcut checked
- §D.7 eval analytics navigation mesh checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-ninth pass (§D.7 + §G.9 + §G.11 + §G.12 + §G.15)

Hundred-thirty-ninth pass consolidates the sim / data-gen / eval launcher navigation mesh
into a shared component, adds Process Monitor return shortcuts for launcher processes, and
extends keyboard shortcuts for the launcher workflow.

**React frontend**
- ``LauncherNavMesh`` — shared cross-page shortcuts on Simulation Launcher, Data Generation
  Wizard, Evaluation Runner, and Process Monitor (§D.7 / §G.9 / §G.11 / §G.12 / §G.15)
- ``launcherProcess.ts`` — shared sim / ``gen_data`` / ``eval`` process detection helpers
- Post-run shortcuts: ``Simulation Summary →``, ``Data Explorer →``, ``Benchmark Analysis →``
  when launcher runs complete successfully
- Keyboard shortcuts ``L`` → Simulation Launcher, ``D`` → Data Generation, ``V`` → Evaluation
  Runner; help overlay updated (§D.7)

**ROADMAP**
- §D.7 ``LauncherNavMesh`` + launcher keyboard shortcuts checked
- §G.9 Simulation Launcher navigation mesh checked
- §G.11 Data Generation navigation mesh checked
- §G.12 Evaluation Runner navigation mesh checked
- §G.15 Process Monitor launcher return shortcuts checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-eighth pass (§D.2 + §G.12 + §A.4)

Hundred-thirty-eighth pass closes the last launcher progress/ETA gap by wiring
``LiveTrainProgressBar`` into the Evaluation Runner live panel for single- and
multi-checkpoint ``eval`` runs.

**React frontend**
- Evaluation Runner — live progress panel with per-checkpoint ``LiveTrainProgressBar``
  during running ``eval`` processes; aggregate status header, stdout tail, and
  Process Monitor shortcut (§D.2 / §G.12 / §A.4)
- Multi-checkpoint launches use unique process IDs and show per-checkpoint progress
  rows with completion/failure badges

**ROADMAP**
- §A.4 Evaluation Runner ``LiveTrainProgressBar`` + live panel checked
- §D.2 eval launcher progress/ETA parity checked
- §G.12 Evaluation Runner live progress + ETA checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-seventh pass (§D.2 + §G.9 + §G.11 + §A.4)

Hundred-thirty-seventh pass extends ``LiveTrainProgressBar`` to the remaining launcher
live panels so simulation and data-generation runs show the same progress bar, elapsed
time, and ETA as train/HPO workflows.

**React frontend**
- Simulation Launcher — ``LiveTrainProgressBar`` in live status panel during running
  ``test_sim`` processes (§D.2 / §G.9 / §A.4)
- Data Generation Wizard — ``LiveTrainProgressBar`` in live progress panel during
  ``gen_data`` runs (§D.2 / §G.11 / §A.4)

**ROADMAP**
- §A.4 Simulation Launcher + Data Generation ``LiveTrainProgressBar`` checked
- §D.2 launcher live progress/ETA parity checked
- §G.9 Simulation Launcher progress + ETA checked
- §G.11 Data Generation progress + ETA checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-sixth pass (§D.2 + §G.15 + §A.4)

Hundred-thirty-sixth pass closes the Process Monitor progress/ETA gap left after the
hundred-thirty-fifth pass consolidated train/HPO workflow pages.

**React frontend**
- Process Monitor — ``LiveTrainProgressBar`` replaces inline ``PROGRESS:`` row renderer;
  elapsed + ETA on every running process (train/hpo/sim/data gen) (§D.2 / §G.15 / §A.4)

**ROADMAP**
- §A.4 Process Monitor ``LiveTrainProgressBar`` + ETA checked
- §D.2 Process Monitor progress/ETA parity checked
- §G.15 process row elapsed + ETA checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-fifth pass (§D.2 + §D.7 + §A.2 / §A.4)

Hundred-thirty-fifth pass consolidates the train/HPO navigation mesh into a shared
component, adds live epoch progress bars with ETA on all train/HPO workflow pages, and
extends keyboard shortcuts for the training workflow.

**React frontend**
- ``TrainHpoNavMesh`` — shared cross-page shortcuts on Training Hub, Training Monitor,
  Process Monitor, HPO Tracker, and Experiment Tracker (§G.7 / §A.2 / §A.4)
- ``LiveTrainProgressBar`` — ``PROGRESS:`` marker progress bar + elapsed + ETA during
  live train/HPO on Training Hub, Training Monitor, HPO Tracker, and Experiment Tracker
  (§D.2 / §G.10 / §G.17 / §G.18)
- ``processProgress.ts`` — shared ``getLatestProgress`` / ``progressPercent`` /
  ``computeEtaMs`` helpers; Process Monitor imports the shared module (§D.2 / §G.15)
- Keyboard shortcuts ``T`` → Training Monitor, ``H`` → Training Hub, ``E`` → Experiment
  Tracker; help overlay updated (§D.7)

**ROADMAP**
- §A.2 ``TrainHpoNavMesh`` shared navigation component checked
- §A.4 ``LiveTrainProgressBar`` epoch progress + ETA checked
- §D.2 live training progress bar + ETA on train/HPO pages checked
- §D.7 train/HPO workflow keyboard shortcuts checked
- §G.10 / §G.17 / §G.18 live progress + navigation consolidation checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-fourth pass (§A.4 + §A.2)

Hundred-thirty-fourth pass completes the bidirectional train/HPO navigation mesh by
adding ``Training Hub →`` return shortcuts on every monitor and tracker page.

**React frontend**
- Training Monitor — ``Training Hub →`` shortcut during live train/HPO runs
  (§G.17 / §A.2 / §A.4)
- Process Monitor — ``Training Hub →`` shortcut for selected ``train_`` / ``hpo_``
  processes (§G.15 / §A.2 / §A.4)
- HPO Tracker — ``Training Hub →`` shortcut during live HPO (§G.18 / §A.2 / §A.4)
- Experiment Tracker — ``Training Hub →`` shortcut during live HPO (§G.18 / §A.2 / §A.4)

**ROADMAP**
- §A.2 Training Hub return navigation mesh checked
- §A.4 Training Hub return navigation mesh checked
- §G.18 Training Hub navigation mesh checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-third pass (§A.4 + §A.2)

Hundred-thirty-third pass completes the live HPO navigation mesh by wiring
``Experiment Tracker →`` shortcuts across all train/HPO workflow pages and
adding the missing ``Training Monitor →`` shortcut on Experiment Tracker.

**React frontend**
- Experiment Tracker — ``Training Monitor →`` navigation shortcut during live
  ``hpo_*`` runs (§G.18 / §A.2)
- HPO Tracker — ``Experiment Tracker →`` navigation shortcut during live HPO
  (§G.18 / §A.2)
- Training Monitor — ``Experiment Tracker →`` when live HPO active (§G.17 / §A.2)
- Process Monitor — ``Experiment Tracker →`` for selected ``hpo_*`` processes
  (§G.15 / §A.2)
- Training Hub — ``Experiment Tracker →`` during live HPO; ``Process Monitor →``
  label parity (§G.10 / §A.4)

**ROADMAP**
- §A.2 Experiment Tracker + live HPO ``Experiment Tracker →`` mesh checked
- §A.4 Training Hub Experiment Tracker shortcut + Process Monitor label checked
- §G.18 Experiment Tracker navigation mesh checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-second pass (§A.4 + §A.2)

Hundred-thirty-second pass completes §G.18 Experiment Tracker live HPO analytics and adds
cross-page navigation shortcuts across the train/HPO workflow pages.

**React frontend**
- Experiment Tracker — ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` during live
  ``hpo_*`` runs; ``HPO Tracker →`` + ``Process Monitor →`` navigation shortcuts
  (§G.18 / §A.4 / §A.2)
- Training Monitor — ``Process Monitor →`` shortcut; ``HPO Tracker →`` when live HPO active
  (§G.17 / §A.2)
- HPO Tracker — ``Training Monitor →`` navigation shortcut (§G.18 / §A.2)
- Process Monitor — ``Training Monitor →`` + ``HPO Tracker →`` for selected train/hpo
  processes (§G.15 / §A.2 / §A.4)
- Training Hub — ``liveTrainProcessLabel`` Live HPO header; ``HPO Tracker →`` during live
  HPO runs (§G.10 / §A.4)

**ROADMAP**
- §A.2 Experiment Tracker live attention + cross-page navigation checked
- §A.4 Experiment Tracker live health + Training Hub HPO shortcuts checked
- §G.18 Experiment Tracker live HPO analytics checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirty-first pass (§A.4 + §A.2)

Hundred-thirty-first pass closes the HPO live-analytics gap: Training Monitor and HPO
Tracker now ingest health alerts and runtime attention for ``hpo_*`` processes, matching
Training Hub and Process Monitor from the prior pass.

**React frontend**
- ``trainingProcess.ts`` — ``isTrainOrHpoProcess``, ``findActiveLiveTrainProcessId``,
  ``findActiveHpoProcessId``, ``liveTrainProcessLabel`` shared helpers
- Training Monitor — live stdout ingest for ``hpo_*`` processes; ``Live HPO`` overlay
  label when HPO active (§G.17 / §A.4 / §A.2)
- HPO Tracker — ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` during live HPO runs;
  ``Process Monitor →`` navigation shortcut (§G.18 / §A.4 / §A.2)
- Process Monitor — ``isTrainOrHpoProcess`` shared matcher (no behaviour change)

**ROADMAP**
- §A.4 Training Monitor + HPO Tracker live health panels checked
- §A.2 Training Monitor + HPO Tracker runtime attention panels checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirtieth pass (§A.4 + §A.2)

Hundred-thirtieth pass extends training health alerts and runtime attention heatmaps
to Training Hub and Process Monitor during live train/hpo runs.

**React frontend**
- ``collectTrainingHealthFromLogLines`` — shared ``TRAINING_HEALTH_START:`` parser for
  process stdout (§A.4)
- ``collectAttentionVizFromLogLines`` — shared ``ATTENTION_VIZ_START:`` parser for process
  stdout (§A.2 Option A)
- Training Hub — ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` in live progress panel
  during train/hpo; ``Training Monitor →`` navigation shortcut (§G.10 / §A.4 / §A.2)
- Process Monitor — training analytics section for selected ``train_`` / ``hpo_`` processes
  with live status badge (§G.15 / §A.4 / §A.2)

**ROADMAP**
- §A.4 Training Hub + Process Monitor health panels checked
- §A.2 Training Hub + Process Monitor runtime attention panels checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-ninth pass (§A.3 Option C)

Hundred-twenty-ninth pass extends live policy telemetry to Simulation Launcher and
tightens Process Monitor ``run_label`` brush parity with Output Browser.

**React frontend**
- ``runLabelFromLogLines`` — shared ``run_label`` derivation from process stdout with
  process-id fallback
- Process Monitor — always ``setRunLabel`` on ``test_sim`` select; process row ring
  highlight when global brush matches (§G.15 / §A.3)
- Simulation Launcher — ``PolicyTelemetryPanel`` + ``PolicyTelemetryTrendsPanel`` during
  live runs; policy chip + KPI card click-to-brush + ``run_label`` auto-sync (§G.9 / §A.3)

**ROADMAP**
- §A.3 Option C Simulation Launcher live telemetry + Process Monitor brush parity checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-eighth pass (§A.3 Option C)

Hundred-twenty-eighth pass extends policy telemetry to Process Monitor for live
``test_sim`` processes and tightens Output Browser ``run_label`` brush sync on run
selection.

**React frontend**
- ``collectPolicyVizFromLogLines`` / ``uniquePolicyVizPolicies`` — per-process
  ``POLICY_VIZ_START:`` parsing from stdout
- ``extractJsonlPathFromLogLines`` — derive SQLite ``run_label`` from ``.jsonl`` paths
  in process logs
- Process Monitor — ``PolicyTelemetryPanel`` + ``PolicyTelemetryTrendsPanel`` when a
  ``test_sim`` process is selected; policy chip brush + 2 Hz live refresh (§G.15 / §A.3)
- Output Browser — auto ``setRunLabel`` on run select; run list ring highlight when
  global brush matches (§G.14 / §A.3)

**ROADMAP**
- §A.3 Option C Process Monitor telemetry + Output Browser run_label auto-brush checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-seventh pass (§A.3 Option C)

Hundred-twenty-seventh pass adds policy telemetry trends to Output Browser when browsing
simulation runs, with run-scoped ``run_label`` brush sync and KPI row click-to-brush.

**React frontend**
- Output Browser — ``PolicyTelemetryTrendsPanel`` when a run is selected; ``initialRunLabel``
  from discovered ``.jsonl`` path stem via ``runLabelFromPath`` (§G.14 / §A.3)
- Output Browser — KPI summary policy rows toggle global policy brush (parity with trends
  history table)

**ROADMAP**
- §A.3 Option C Output Browser trends panel + KPI brush checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-sixth pass (§A.3 Option C)

Hundred-twenty-sixth pass completes policy telemetry trends coverage on Simulation Monitor
and Data Explorer, fixes trajectory chart brush-click indexing, and adds a shared
``run_label`` path helper.

**React frontend**
- ``runLabelFromPath`` — derives SQLite ``run_label`` from log path stem (Python ``Path.stem`` parity)
- ``PolicyTelemetryTrendsPanel`` — trajectory chart click indexes ``allSeries`` (fixes brush when
  chart shows dimmed full dataset, parity with steps chart ``displayStepRows`` fix)
- Simulation Monitor — ``initialRunLabel`` from active log path; cross-run trends scoped to open run
- Data Explorer — ``PolicyTelemetryTrendsPanel`` with policy + ``run_label`` brush sync

**ROADMAP**
- §A.3 Option C trajectory click fix + Simulation Monitor / Data Explorer panels checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-fifth pass (§A.3 Option C)

Hundred-twenty-fifth pass adds portfolio ``run_label`` brush sync across telemetry
trend pages, server-side run scoping in SQLite queries, and OLAP Explorer panel parity.

**Python logic**
- ``query_policy_telemetry_trends`` / ``query_policy_trajectory_series`` — optional
  ``run_label`` SQL filter for portfolio-scoped cross-run queries
- Unit test for ``run_label`` filter roundtrip in
  ``logic/test/unit/tracking/test_policy_telemetry_db.py``

**Rust backend**
- ``load_policy_telemetry_trends`` / ``load_policy_trajectory_trends`` — ``run_label``
  bridge arg forwarded to Python subprocess queries

**React frontend**
- ``PolicyTelemetryTrendsPanel`` — ``initialRunLabel`` prop syncs global run brush;
  SQLite reload passes active ``runLabel`` to Rust commands; steps chart click indexes
  ``displayStepRows`` (fixes brush click when chart shows top-12 rows)
- Simulation Summary / Benchmark Analysis / City Comparison / Algorithm Comparison —
  ``initialRunLabel`` from portfolio single-run brush
- OLAP Explorer — ``PolicyTelemetryTrendsPanel`` with policy + run_label brush sync

**ROADMAP**
- §A.3 Option C run_label brush sync + OLAP Explorer panel checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-fourth pass (§A.3 Option C)

Hundred-twenty-fourth pass completes global brush parity across all analysis pages:
chart dimming, filtered history table, and ``PolicyTelemetryTrendsPanel`` on Algorithm
Comparison + City Comparison.

**React frontend**
- ``TrendBrushFilter`` + chart builders — comparison / steps / trajectory charts dim
  non-brushed policies and runs at 25% opacity (full dataset retained for context)
- ``PolicyTelemetryTrendsPanel`` — history table uses ``filteredRows``; empty-state when
  brush excludes all rows
- Algorithm Comparison — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from brush
- City Comparison — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from brush
- Benchmark Analysis — ``initialPolicy`` brush sync on trends panel (parity with Summary)

**ROADMAP**
- §A.3 Option C chart brush dimming + Algorithm/City Comparison panels checked

### Fixed

- ``RuntimeAttentionPanel`` — ``ChartExportButtons`` prop renamed from ``basename`` to ``filenameStem`` (build fix)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-third pass (§A.3 Option C)

Hundred-twenty-third pass extends global brush sync from the trends history table to all
telemetry charts, adds trajectory CSV export, and surfaces the panel on Simulation Summary.

**Python logic**
- ``query_policy_trajectory_series`` — each series includes ``run_label`` for run-key brush parity
- Unit test asserts ``run_label`` on trajectory roundtrip

**Rust backend**
- ``PolicyTrajectorySeries`` — ``run_label`` field on deserialized trajectory payloads

**React frontend**
- ``filterTrendRows`` / ``filterTrajectorySeries`` — global policy / ``run_label`` brush filters
  comparison, steps, and trajectory chart data
- ``exportPolicyTrajectoryCsv`` — long-format trajectory step export
- ``PolicyTelemetryTrendsPanel`` — chart click brushes global policy / run; active-brush badge +
  clear control; trajectory CSV button
- Simulation Summary — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from active chart brush

**ROADMAP**
- §A.3 Option C chart brush filter + Simulation Summary panel checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-second pass (§A.3 Option C)

Hundred-twenty-second pass polishes cross-run policy telemetry trending: solver-step
trajectory axes, global brush sync, CSV export, and Benchmark Analysis integration.

**React frontend**
- ``buildTrendTrajectoryOption`` — x-axis uses unioned iteration/generation indices from
  persisted ring-buffers (not array index)
- ``exportPolicyTelemetryTrendsCsv`` — history table CSV download
- ``PolicyTelemetryTrendsPanel`` — row click brushes global policy / ``run_label``;
  dimming when global filter active; ``initialPolicy`` prop from Simulation Monitor
- Benchmark Analysis — ``PolicyTelemetryTrendsPanel`` for portfolio solver telemetry

**ROADMAP**
- §A.3 Option C trajectory brush + Benchmark Analysis panel checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twenty-first pass (§A.3 Option C)

Hundred-twenty-first pass adds cross-run improvement trajectory overlays from persisted
SQLite ring-buffer JSON, extending ROADMAP §A.3 Option C.

**Python logic**
- ``query_policy_trajectory_series`` — extracts ``best_cost`` / ``global_best_cost`` improvement
  curves from ``policy_viz_snapshots.data_json`` with iteration/generation x-axis
- Unit tests for trajectory roundtrip and policy-type filtering in
  ``logic/test/unit/tracking/test_policy_telemetry_db.py``

**Rust backend**
- ``load_policy_trajectory_trends`` command — Python subprocess bridge for trajectory series

**React frontend**
- ``buildTrendTrajectoryOption`` — multi-run improvement line chart with optional EMA smoothing
- ``PolicyTelemetryTrendsPanel`` — trajectory chart, policy filter, and EMA toggle above
  cross-run bar charts; PNG export via ``ChartExportButtons`` (§G.7)

**ROADMAP**
- §A.3 Option C cross-run improvement trajectory chart checked (§A complete)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twentieth pass (§A.3 Option C)

Hundred-twentieth pass adds SQLite persistence for policy telemetry cross-run trending,
completing ROADMAP §A.3 Option C.

**Python logic**
- ``policy_telemetry_db.py`` — ``assets/telemetry.db`` schema with ``simulation_runs`` and
  ``policy_viz_snapshots`` tables; ``extract_final_metric`` for comparable terminal KPIs
- ``persist_policy_viz_snapshot`` — upserts on each ``POLICY_VIZ_START:`` emit from
  ``policy_viz_emit.py``
- ``query_policy_telemetry_trends`` — cross-run snapshot query with algorithm-family filter
- Unit tests in ``logic/test/unit/tracking/test_policy_telemetry_db.py``

**Rust backend**
- ``load_policy_telemetry_trends`` command — Python subprocess bridge to query SQLite store

**React frontend**
- ``PolicyTelemetryTrendsPanel`` — cross-run final-metric bar chart, solver-steps chart,
  and history table on Simulation Monitor
- ``policyTelemetryTrends.ts`` — ECharts builders + row formatters

**ROADMAP**
- §A.3 Option C SQLite cross-run trending checked (§A.3 complete)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-nineteenth pass (§A.3 Option B)

Hundred-nineteenth pass adds 2 Hz live policy telemetry streaming during simulation
solver runs, completing ROADMAP §A.3 Option B.

**Python logic**
- ``PolicyVizStreamSession`` — daemon thread emits growing ``PolicyVizMixin`` ring-buffer
  snapshots every 0.5 s during route construction / improvement
- ``route_construction`` / ``route_improvement`` actions wrap solver execution in stream sessions
- Unit tests in ``logic/test/unit/tracking/test_policy_viz_emit.py``

**React frontend**
- Sim store — ``addPolicyVizEntry`` upserts by policy/sample/day/type (replaces stale snapshots)
- ``policyVizDataLen`` helper — picks newest streaming snapshot by metric series length
- ``PolicyTelemetryPanel`` — 2 Hz throttled ECharts refresh + **Live · 2 Hz** badge
- Simulation Monitor — live mode when file-watcher active or ``test_sim`` process running

**ROADMAP**
- §A.3 Option B 2 Hz live telemetry stream checked (Option C SQLite deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighteenth pass (§A.4 Option D)

Hundred-eighteenth pass adds HPO health metrics for early trial pruning, completing
ROADMAP §A.4 Option D.

**Python logic**
- ``HpoHealthMetricsCallback`` — per-epoch ``train/grad_norm`` + ``train/entropy``
  reporting to Optuna user attrs and WSTracker ``hpo/*`` sweep metrics
- ``apply_dehb_health_penalty`` — DEHB fitness penalty on unhealthy trials
- Optuna / Ray Tune / DEHB objectives in ``logic/src/pipeline/features/train/hpo.py``
  wired with health callback alongside existing pruners
- Unit tests in ``logic/test/unit/pipeline/callbacks/test_hpo_health.py``

**Rust backend**
- ``load_optuna_study`` — trial ``user_attrs`` (``last_grad_norm``, ``last_entropy``,
  ``health_pruned``) included in study payload

**React frontend**
- HPO Tracker — **Trial Health** table with grad norm, entropy, and health-pruned badge

**ROADMAP**
- §A.4 Option D HPO prune metrics checked (Options B/C deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventeenth pass (§A.2 Option A)

Hundred-seventeenth pass wires runtime encoder attention ring-buffer capture into Studio
ML introspection, completing ROADMAP §A.2 Option A.

**Python logic**
- ``AttentionRingBuffer`` — fixed-capacity ring-buffer for layer/head/decode-step attention snapshots
- ``install_attention_ring_buffer`` / ``ensure_attention_buffer`` — persistent encoder forward hooks
- ``attention_emit.py`` — ``ATTENTION_VIZ_START:`` marker to stdout + ``attention_viz.jsonl`` append
- ``maybe_log_eval_attention_heatmaps`` — integrates ring-buffer capture + Studio emission
- Unit tests in ``logic/test/unit/tracking/test_attention_buffer.py``

**Rust backend**
- ``parse_attention_viz_line`` + ``load_attention_viz_log`` command

**React frontend**
- ``RuntimeAttentionPanel`` — ECharts heatmap with snapshot/layer/head selectors
- ``attentionViz.ts`` — marker parse + heatmap builders
- Training Monitor — live stdout ingest + historical ``attention_viz.jsonl`` load
- ML Introspection — Attention tab runtime panel + ``attention_viz.jsonl`` file picker

**ROADMAP**
- §A.2 Option A Studio attention ring-buffer checked (Option B BertViz deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixteenth pass (§A.6 Option C)

Hundred-sixteenth pass completes ECharts route-diff overlay parity across Simulation
Monitor and Simulation Summary, extending ROADMAP §A.6 Option C.

**React frontend**
- ``routeViz.ts`` — ``showFailureOverlay`` toggle; dual-policy overlay paths;
  tour-diff ring borders on scatter nodes via ``TOUR_DIFF_RGB``
- ``RouteViz`` — ``compareData`` / ``showTourDiff`` props; combined
  ``FailureOverlayLegend`` for failure + diff modes
- Simulation Monitor — ECharts overlay compare when two map policies visible;
  failure + route-diff toggles propagate to ``RouteViz`` (parity with deck.gl)
- Simulation Summary — **Show/Hide failure overlay** + **Show/Hide route diff**
  toggles; overlay-compare ``RouteViz`` when exactly two brushed policies share a day

**ROADMAP**
- §A.6 Option C ECharts route-diff parity checked on Monitor + Summary

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifteenth pass (§A.6 Option C)

Hundred-fifteenth pass adds failure route-diff overlays on the Studio geospatial view,
completing ROADMAP §A.6 Option C.

**React frontend**
- ``routeFailureOverlay.ts`` — shared overflow/skipped bin sets + tour-diff computation
- ``FailureOverlayLegend`` — reusable legend for failure and tour-diff highlights
- ``DeckRouteMap`` — red overflow + orange skipped ``ScatterplotLayer`` highlights on
  Mercator and OrbitView; cyan/purple tour-diff rings when two policies are overlaid
- Simulation Monitor — **Show/Hide failure overlay** and **Show/Hide route diff** toggles
- ``RouteViz`` — failure legend; ``routeViz.ts`` refactored to shared overlay helper
- Simulation Summary — route panel subtitle notes embedded ``failure_analysis`` highlights

**ROADMAP**
- §A.6 Option C route-diff overlay checked (Options B/D deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fourteenth pass (§A.1)

Hundred-fourteenth pass adds a shared ``RouteViz`` ECharts spatial panel for interactive
route solution visualization, completing ROADMAP §A.1 Option A.

**React frontend**
- ``RouteViz`` — reusable analysis component: star depot, demand-sized tour nodes,
  per-vehicle coloured edges, optional failure overlay (overflow / skipped high-fill)
- ``routeViz.ts`` — ``buildRouteVizOption`` + ``nodeSizeFromDemand`` utilities
- Simulation Monitor — refactored inline ``RouteMapChart`` to ``RouteViz``
- Simulation Summary — day scrubber + multi-policy route comparison grid (§A.1 analysis view)
- PNG/SVG export via ``ChartExportButtons`` (§G.7)

**ROADMAP**
- §A.1 Option A ECharts route viz checked (Option E already via ``DeckRouteMap``; B/C/D deferred)
- §D.1 updated — ``RouteViz`` shared across Monitor + Summary

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-thirteenth pass (§A.2)

Hundred-thirteenth pass adds WandB / TensorBoard attention heatmap logging during
evaluation and validation, completing ROADMAP §A.2 Option C.

**Python logic**
- ``logic/src/tracking/logging/visualization/heatmaps.py`` — runtime attention
  capture via ``add_attention_hooks``, PNG rendering, WandB ``wandb.Image`` and
  TensorBoard image logging
- ``AttentionHeatmapCallback`` — Lightning validation hook; respects
  ``tracking.log_attention``, ``tracking.log_attention_heatmaps``, and
  ``viz_every_n_epochs``
- ``WSTrainer`` — auto-registers ``AttentionHeatmapCallback`` when tracking flags enabled
- Eval engine — ``maybe_log_eval_attention_heatmaps()`` after ``evaluate_policy``
- Unit tests in ``logic/test/unit/tracking/test_attention_heatmaps.py``

**ROADMAP**
- §A.2 Option C WandB attention heatmaps checked (Options A/B deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-twelfth pass (§A.6)

Hundred-twelfth pass adds ``FailureAnalyzer`` post-day root-cause diagnostics and wires
them into the Studio Simulation Monitor, completing ROADMAP §A.6 Option A.

**Python logic**
- ``FailureAnalyzer`` — compares predicted vs. actual fill, flags overflow bins,
  fill-rate spikes, and skipped high-fill bins; severity-coded summary
- ``failure_emit.py`` — ``SIM_FAILURE_START:`` marker to stdout + JSONL append
- ``LogAction`` — runs analyzer after each day; embeds ``failure_analysis`` in day log
- Unit tests in ``logic/test/unit/pipeline/simulations/test_failure_analyzer.py``

**Rust backend**
- ``parse_sim_failure_line`` + ``load_sim_failure_log`` command
- ``sim:failure_update`` watcher events alongside day and policy-viz streams

**React frontend**
- ``FailureAnalysisPanel`` — root-cause badges, overflow bin table, skipped high-fill chips
- ``simFailure.ts`` — marker parse + display helpers
- Simulation Monitor — live stdout ingest + historical ``SIM_FAILURE_START`` load;
  falls back to embedded ``failure_analysis`` in day log payloads

**ROADMAP**
- §A.6 Option A FailureAnalyzer checked (Options B/C/D deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eleventh pass (§A.4)

Hundred-eleventh pass adds ``TrainingHealthCallback`` instability guardrails and wires
them into the Studio Training Monitor, completing ROADMAP §A.4 Option A.

**Python logic**
- ``TrainingHealthCallback`` — detects gradient norm explosion, reward stagnation,
  and entropy collapse; loguru warnings with per-code cooldown
- ``training_health_emit.py`` — ``TRAINING_HEALTH_START:`` marker to stdout +
  ``training_health.jsonl`` under Lightning ``log_dir``
- ``WSTrainer`` — auto-registers health callback in default callback stack
- Unit tests in ``logic/test/unit/pipeline/callbacks/test_training_health.py``

**Rust backend**
- ``parse_training_health_line`` + ``load_training_health_log`` command

**React frontend**
- ``TrainingHealthPanel`` — severity-coded alert list with code counts
- ``trainingHealth.ts`` — marker parse + display helpers
- Training Monitor — live stdout ingest + historical ``training_health.jsonl`` load

**ROADMAP**
- §A.4 Option A TrainingHealthCallback checked (Options B/C/D deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-tenth pass (§A.5)

Hundred-tenth pass adds Optuna ``optuna.visualization`` Plotly report export to
``assets/hpo_reports/``, completing ROADMAP §A.5 Option A.

**Python logic**
- ``hpo_reports.py`` — parallel-coordinates, param-importances, optimisation-history
  HTML export (+ optional PNG when kaleido installed); ``manifest.json`` metadata
- ``run_hpo_sim`` — auto-exports reports after HPO completes (≥2 completed trials)
- Unit tests in ``logic/test/unit/pipeline/simulations/test_hpo_reports.py``

**Rust backend**
- ``export_optuna_reports`` command — invokes Python export; returns report paths

**React frontend**
- HPO Tracker — **Export Plotly** button + **Reports** folder open via shell plugin

**ROADMAP**
- §A.5 Option A Optuna plots checked (Options B/C deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-ninth pass (§A.3)

Hundred-ninth pass wires ``PolicyVizMixin`` iteration telemetry into the Studio
Simulation Monitor, completing ROADMAP §A.3 Option A.

**Python logic**
- ``policy_viz_emit.py`` — ``POLICY_VIZ_START:`` marker emission to stdout + JSONL log
- ``route_construction`` / ``route_improvement`` actions call ``maybe_emit_policy_viz()`` after solver runs
- Unit tests in ``logic/test/unit/tracking/test_policy_viz_emit.py``

**Rust backend**
- ``parse_policy_viz_line`` + ``PolicyVizEntry`` struct in ``sim_watcher.rs``
- ``load_policy_viz_log`` command; ``sim:policy_viz_update`` watcher events

**React frontend**
- ``PolicyTelemetryPanel`` — algorithm-dispatched ECharts (ALNS/HGS/ACO/ILS/selector/generic)
- ``policyTelemetry.ts`` — marker parse + chart builders with EMA smoothing
- Simulation Monitor — panel below route/tour detail; historical + live ingest
- ``useProcessMonitor`` — stdout ``POLICY_VIZ_START:`` lines → sim store

**ROADMAP**
- §A.3 Option A PolicyVizMixin → Studio checked (Options B/C deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-eighth pass (§G.8)

Hundred-eighth pass wires the Tauri updater plugin for signed auto-updates, adds
Settings install UX, and polishes system-theme affordances from the previous pass.

**Rust backend**
- `tauri-plugin-updater` — desktop plugin init; runtime pubkey from `WSMART_UPDATER_PUBKEY`
- `check_for_updates` — signed updater path when pubkey + URL configured; JSON manifest fallback
- `install_app_update` — download/install pending signed update + app restart
- `PendingUpdate` state — holds discovered update between check and install
- `tauri.conf.json` — `createUpdaterArtifacts: true`; `updater:default` capability

**React frontend**
- Settings — "Download & Install" when signed update available; release notes in toast
- Settings — effective theme hint when System appearance selected; draft sync on external theme change
- Command palette — "Cycle Theme (Dark / Light / System)" label

**Assets**
- `app/updater.example.json` — example static Tauri updater manifest

**ROADMAP**
- §G.8 Tauri updater plugin + signed install flow checked (signing keys + CDN deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-seventh pass (§D.3 / §G.19)

Hundred-seventh pass adds system theme following (``prefers-color-scheme``), marks all
§G.0–§G.19 Studio phases complete in the ROADMAP, and updates the §D GUI/UX matrix to
reflect requirements delivered via the Studio.

**React frontend**
- `theme.ts` — ``ThemePreference`` (dark/light/system), ``resolveEffectiveTheme()``,
  ``nextThemePreference()`` cycle helper
- `useThemeSync` — ``matchMedia`` listener keeps DOM + ``effectiveTheme`` in sync when
  preference is ``system``
- `store/app.ts` — ``effectiveTheme`` field; chart/editor consumers use resolved theme
- Settings — System appearance radio; import accepts ``system`` theme
- TopBar + command palette — cycle dark → light → system → dark (Monitor icon)

**ROADMAP**
- §G — Studio Complete banner (all twenty phases delivered)
- §G.19 system theme following checked (§D.3 Option C)
- §D effort matrix updated — theme, session, cancel, training charts, route viz, overrides ✅

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-sixth pass (§G.7)

Hundred-sixth pass introduces a shared ``CanvasExportButton`` component for WebGL/canvas
PNG exports and propagates it to all remaining deck.gl, Sigma.js, Cosmograph, and R3F
panels that still used inline PNG export buttons.

**React frontend**
- `CanvasExportButton` — reusable canvas/container PNG export button with Sonner toasts
- `DeckRouteMap` — refactored deck.gl Mercator/OrbitView map PNG export
- `GraphTopologyPanel` — refactored Sigma.js + Cosmograph WebGL PNG export
- `MLIntrospectionPanel` — refactored Attention Sigma.js + LossLandscape3D R3F PNG export

**ROADMAP**
- §G.7 ``CanvasExportButton`` propagated to deck.gl route map, graph topology WebGL,
  and ML introspection WebGL/R3F panels

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fifth pass (§G.7)

Hundred-fifth pass propagates the shared ``ChartExportButtons`` component to all
remaining portfolio facet, OLAP, monitor, topology, and ML introspection ECharts
panels that still used inline PNG/SVG export buttons.

**React frontend**
- Portfolio facets — ``ChartExportButtons`` on ``BenchmarkParetoPanel``,
  ``BenchmarkPortfolioParallel``, ``BenchmarkDistributionHeatmap``,
  ``BenchmarkGraphHeatmap``, ``BenchmarkPortfolioHeatmap``
- OLAP — ``ChartExportButtons`` on ``PivotTablePanel`` pivot heatmap and
  ``SqlQueryPanel`` auto-chart
- Simulation Monitor — ``ChartExportButtons`` on ECharts route-map preview
- Graph Topology — ``ChartExportButtons`` on ECharts view; WebGL Sigma/Cosmograph
  keeps canvas PNG export
- ML Introspection — ``ChartExportButtons`` on attention graph/heatmap (primary +
  compare), loss contour map; WebGL terrain/Sigma keeps canvas PNG export

**ROADMAP**
- §G.7 ``ChartExportButtons`` propagated to portfolio facets, OLAP, route-map,
  topology ECharts, and ML introspection ECharts panels

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-fourth pass (§G.7)

Hundred-fourth pass adds SVG export on all remaining PNG-only ECharts analytics
panels and introduces a shared ``ChartExportButtons`` component for paired
PNG/SVG export with toast feedback.

**React frontend**
- `ChartExportButtons` — reusable PNG + SVG export button pair with Sonner toasts
- Simulation Summary — SVG export on trajectory, radar, heatmap, parallel,
  hierarchy, Pareto, efficiency ranking, metric bars, and city comparison
- Benchmark Analysis — SVG export on eval/sim metric bars and efficiency ranking
- Algorithm Comparison — SVG export on radar + per-metric bar charts
- City Comparison, PortfolioEfficiencyRanking — SVG export
- Simulation Monitor — SVG export on daily KPI timeseries
- Training Monitor / Training Hub — SVG export on overlay + sparklines
- HPO Tracker — SVG export on history, importance, cross-study, parallel charts
- Experiment Tracker, ZenML pipeline, Data Generation, Evaluation Runner —
  SVG export on remaining ECharts panels

**ROADMAP**
- §G.7 ``ChartExportButtons`` shared export component checked
- §G.7 Global export lists include SVG on all remaining analytics ECharts panels

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-third pass (§G.7)

Hundred-third pass centralises export toast feedback in ``chartExport.ts``, adds SVG
export on portfolio analytics facets and pivot heatmaps, and propagates toast
feedback to all remaining analytics export buttons.

**React frontend**
- `chartExport.ts` — ``exportChartPngWithToast()``, ``exportChartSvgWithToast()``,
  ``exportContainerCanvasPngWithToast()``, ``exportCanvasPngWithToast()`` helpers
- Portfolio facets — SVG export on ``BenchmarkParetoPanel``,
  ``BenchmarkPortfolioParallel``, ``BenchmarkDistributionHeatmap``,
  ``BenchmarkGraphHeatmap``, ``BenchmarkPortfolioHeatmap``, ``PivotTablePanel``
- Analytics pages — toast feedback on Simulation Summary, Benchmark Analysis,
  Algorithm Comparison, City Comparison, HPO Tracker, Experiment Tracker, ZenML
  pipeline, Training Monitor/Hub, Data Generation, Evaluation Runner, Simulation
  Monitor timeseries
- ``MLIntrospectionPanel``, ``GraphTopologyPanel``, ``SqlQueryPanel``,
  ``DeckRouteMap`` — refactored to shared toast helpers

**ROADMAP**
- §G.7 Export helpers with toast feedback checked
- §G.7 Global export lists include portfolio facet SVG + toast on all analytics
  export buttons

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-second pass (§G.1 / §G.4 / §G.6 / §G.7 / §G.16)

Hundred-second pass extends the §G.7 export surface to portfolio analytics
facets (Pareto panels, parallel coordinates, distribution/graph heatmaps, pivot
heatmaps), adds topology ECharts SVG export, and unifies deck.gl PNG export
toast feedback. Marks §G.16 complete in the ROADMAP.

**React frontend**
- `BenchmarkParetoPanel` — per-facet PNG export with toast feedback
- `BenchmarkPortfolioParallel` — portfolio parallel-coordinates PNG export
- `BenchmarkDistributionHeatmap` / `BenchmarkGraphHeatmap` — facet heatmap PNG
  export
- `PivotTablePanel` — pivot heatmap PNG export
- `GraphTopologyPanel` — ECharts SVG export alongside PNG/WebGL export
- `DeckRouteMap` — Mercator vs OrbitView PNG filenames + toast feedback
- `SimulationMonitor` — stale deck.gl deferred comment removed

**ROADMAP**
- §G.1.2 BenchmarkParetoPanel per-facet PNG export checked
- §G.1.3 distribution/graph facet heatmap PNG export checked
- §G.1.4 BenchmarkPortfolioParallel PNG export checked
- §G.4 ECharts topology SVG export checked
- §G.6 PivotTablePanel heatmap PNG export checked
- §G.7 Global export lists include portfolio facets + pivot heatmap + topology
  SVG + deck.gl toast feedback
- §G.16 deck.gl PNG toast + Phase 16 status marked complete
- §G.8 Phase 8 status marked complete (signed releases deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundred-first pass (§G.4 / §G.5 / §G.7)

Hundred-first pass adds WebGL/3D canvas PNG export for loss terrain, attention
Sigma.js, and graph topology Sigma/Cosmograph views, extending the §G.7 export
surface beyond ECharts panels.

**React frontend**
- `chartExport.ts` — ``exportContainerCanvasPng()`` finds the first canvas inside
  a container (Sigma.js, R3F, deck.gl)
- `LossLandscape3D` — ``forwardRef`` on terrain wrapper for canvas capture
- `MLIntrospectionPanel` — 3D terrain PNG on Loss tab; Sigma.js PNG on
  Attention tab; toast feedback
- `GraphTopologyPanel` — unified PNG export for ECharts, Sigma.js, and
  Cosmograph views with toast feedback

**ROADMAP**
- §G.4 Sigma.js / Cosmograph WebGL PNG export checked
- §G.5 Loss landscape 3D terrain + Attention Sigma.js PNG export checked
- §G.7 Global export lists include WebGL/canvas PNG via
  ``exportContainerCanvasPng()``

---

#### WSmart-Route Studio — Tauri App (`app/`) — hundredth pass (§G.5 / §G.7)

Hundredth pass adds PNG/SVG export on ML introspection ECharts panels (including
compare heatmaps), cleans stale §G.5 partial markers, and marks Phase 5 complete
in the ROADMAP.

**React frontend**
- `MLIntrospectionPanel` — PNG + SVG export on attention heatmap (primary +
  side-by-side / distribution compare panels), attention bipartite graph, and
  loss contour map; toast feedback on export success/failure

**ROADMAP**
- §G.5 Stale partial markers removed on tensor pipeline, loss grid export,
  attention decode-step compare, and side-by-side/overlay toggle
- §G.5 ML introspection ECharts PNG/SVG export checked; Phase 5 status marked complete
- §G.7 Global export lists include MLIntrospectionPanel attention/loss charts

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-ninth pass (§G.4 / §G.7)

Ninety-ninth pass propagates global log-scale to graph topology ACO pheromone
edge styling, adds ECharts PNG export on the topology panel, and marks §G.4
complete in the ROADMAP.

**React frontend**
- `chartLogScale.ts` — ``pheromoneWeightDisplay()`` log-transform helper for τ
  edge opacity/width
- `graphTopology.ts` — ``normalizePheromone()`` / ``pheromoneIntensity()`` apply
  log-scale before edge styling; ``buildTopologyFromMatrix`` accepts ``logScale``
- `GraphTopologyPanel` — ``logScale`` prop; ECharts PNG export; subtitle notes
  log-scale τ when active
- `TopologySigmaView` / `TopologyCosmographView` — shared ``pheromoneIntensity()``
  for WebGL edge warmth
- `SimulationMonitor` — passes global ``logScale`` to ``GraphTopologyPanel``

**ROADMAP**
- §G.4 Topology pheromone log-scale + ECharts PNG export checked; stale partial
  markers removed; Phase 4 status marked complete
- §G.7 Global log-scale + export lists include graph topology pheromone styling
  and topology PNG export

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-eighth pass (§G.2 / §G.3)

Ninety-eighth pass extends the shared strategy legend to the policy hierarchy
panel, colours drill-down profit bars by mandatory-selection strategy, and
marks §G.3 geospatial routing complete in the ROADMAP.

**React frontend**
- `policyHierarchy.ts` — ``resolveDrillBarColor()`` colours strategy-depth
  drill bars via ``selectionStrategyColor()``; constructor depth reuses kg/km or
  overflow gradient
- `PolicyHierarchyPanel` — ``StrategyLegend`` chips; drill-down bars use
  ``resolveDrillBarColor()`` instead of flat indigo fill

**ROADMAP**
- §G.2 Shared strategy legend on ``PolicyHierarchyPanel`` + drill-down strategy
  bar colouring checked
- §G.3 Stale partial/deferred markers removed; Phase 3 status marked complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-seventh pass (§G.1.4 / §G.2)

Ninety-seventh pass centralises mandatory-selection strategy legends and adds
strategy-ring border strokes on the policy hierarchy sunburst.

**React frontend**
- `simMetadata.ts` — ``SELECTION_STRATEGY_LEGEND`` constant shared across parallel
  coordinate charts
- `StrategyLegend` — reusable LA · LM · LM-CF70 · LM-CF90 · SL-SL1 · SL-SL2
  colour chips
- `PolicyParallelChart` — strategy legend + subtitle; polylines already coloured
  via ``strategyColor()``
- `BenchmarkPortfolioParallel` — uses shared ``StrategyLegend`` component
- `policyHierarchy.ts` — middle strategy ring segments add
  ``selectionStrategyColor()`` border stroke on sunburst/treemap

**ROADMAP**
- §G.1.4 Shared strategy colour legend on policy + portfolio parallel coords checked
- §G.2 Angular span / kg/km gradient partial marker removed; strategy ring borders checked
- §G.2 Phase 2 status marked complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-sixth pass (§G.1.4)

Ninety-sixth pass colours portfolio parallel-coordinate polylines by
mandatory-selection strategy instead of run index.

**React frontend**
- `simMetadata.ts` — ``selectionStrategyColor()`` + ``resolveRunSelectionStrategy()``
  resolve LA · LM · LM-CF70 · LM-CF90 · SL-SL1 · SL-SL2 from log path segments or
  dominant policy label; ``strategyColor()`` delegates to ``selectionStrategyColor()``
- `BenchmarkPortfolioParallel` — run polylines use strategy colour palette; strategy
  legend chips; tooltips show resolved strategy label

**ROADMAP**
- §G.1.4 Portfolio parallel coordinates strategy colouring checked (partial marker removed)
- §G.1 Phase 1 status marked complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-fifth pass (§G.1 / §G.1.6 / §G.7)

Ninety-fifth pass extends log-scale error-bar whiskers to city comparison
grouped bars and Benchmark Analysis multi-run metric bar charts.

**React frontend**
- `chartLogScale.ts` — ``groupedBarWhiskerX()`` helper for grouped-category
  bar whisker horizontal offsets
- `cityComparison.ts` — ``buildCityComparisonSeries`` computes per-city std;
  ``cityComparisonChartOption`` accepts ``showErrorBars``; profit ·
  symlog-overflows · kg/km whiskers via ``errorBarBounds`` when global
  ``logScale`` on; tooltips show mean ± std
- `BenchmarkAnalysis` — shared ``showErrorBars`` toggle now drives multi-run
  run×policy metric bar whiskers + city comparison chart whiskers
- `CityComparison` — ``showErrorBars`` toggle on dedicated city comparison page
- `SimulationSummary` — portfolio city comparison chart inherits global
  ``showErrorBars`` toggle

**ROADMAP**
- §G.1 BenchmarkAnalysis multi-run metric-bar error-bar whiskers log-scale checked
- §G.1.6 City Comparison error-bar whiskers log-scale checked
- §G.7 Global log-scale propagation includes city-comparison + benchmark metric-bar whiskers

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-fourth pass (§G.1 / §G.7)

Ninety-fourth pass extends log-scale error-bar whiskers and the shared
``showErrorBars`` toggle to Benchmark Analysis and Algorithm Comparison.

**React frontend**
- `BenchmarkAnalysis` — ``showErrorBars`` toggle; single-run efficiency ranking
  and ``PortfolioEfficiencyRanking`` horizontal kg/km whiskers via
  ``errorBarBounds`` when global ``logScale`` on
- `AlgorithmComparison` — ``showErrorBars`` toggle on per-metric bar charts;
  mean ± std whiskers with log/symlog axis via ``errorBarBounds``

**ROADMAP**
- §G.1 AlgorithmComparison metric-bar error-bar whiskers log-scale checked
- §G.1.5 BenchmarkAnalysis efficiency-ranking error-bar whiskers log-scale checked
- §G.7 Global log-scale propagation includes Benchmark + Algorithm error-bar whiskers

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-third pass (§G.1 / §G.2 / §G.7)

Ninety-third pass fixes grouped kg/km error-bar metric keys and extends log-scale
whiskers to hierarchy drill-down profit bars.

**React frontend**
- `GroupedMetricBarChart` — ``metricKey`` prop; kg/km groups pass ``"kg/km"`` so
  ``errorBarBounds`` uses log axis (not profit) when global ``logScale`` on
- `PolicyHierarchyPanel` — drill-down profit bars clamp to log floor; Empirical↔Gamma
  spread whiskers via ``errorBarBounds`` on log-scale profit x-axis

**ROADMAP**
- §G.1.1 grouped metric bar whiskers on log axis checked (stale "hidden" text removed)
- §G.2 hierarchy drill-down error-bar whiskers log-scale checked (partial marker removed)
- §G.7 Global log-scale propagation includes hierarchy drill-down + grouped metric whiskers

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-second pass (§G.1 / §G.5 / §G.7)

Ninety-second pass enables log-scale error-bar whiskers on bar and efficiency
ranking charts, and surfaces per-basin generalization notes on the loss landscape.

**React frontend**
- `chartLogScale.ts` — ``errorBarBounds()`` helper for symlog/log whisker endpoints
- `SimulationSummary` — ``MetricBarChart``, ``GroupedMetricBarChart``, and
  ``EfficiencyRankingChart`` show mean ± std whiskers when global ``logScale`` on
- `PortfolioEfficiencyRanking` — horizontal kg/km whiskers on log x-axis
- `lossLandscape.ts` — ``generalizationNote`` per flat/moderate/sharp basin label
- `MLIntrospectionPanel` / `LossLandscape3D` — display Empirical vs Gamma-3 notes

**ROADMAP**
- §G.1 error-bar whiskers log-scale checked (partial markers removed)
- §G.1.5 efficiency ranking whiskers on log axis checked
- §G.5.2 loss minima generalization notes checked (partial marker removed)
- §G.7 Global log-scale propagation includes error-bar whiskers

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninety-first pass (§G.5 / §G.7)

Ninety-first pass extends global log-scale to attention bipartite graph overlays
(ECharts + Sigma.js) and the React Three Fiber loss landscape 3D terrain.

**React frontend**
- `chartLogScale.ts` — ``attentionWeightDisplay()`` helper for edge opacity/width mapping
- `attentionGraph.ts` — ``buildAttentionGraphOption`` log-transforms edge styling when
  global ``logScale`` on; tooltips retain raw attention weights
- `AttentionSigmaView` — log-scale edge size/opacity via ``attentionWeightDisplay``;
  edge ``weight`` attribute stores raw values
- `LossLandscape3D` — log-transformed height/colour via ``transformMatrixLogScale`` when
  on; minima sharpness analysis stays on raw loss grid
- `MLIntrospectionPanel` — passes ``logScale`` to graph/sigma/3D views; subtitles reflect mode

**ROADMAP**
- §G.5.2 Loss landscape 3D terrain log-scale checked
- §G.5.3 Attention bipartite graph overlays log-scale checked
- §G.7 Global log-scale propagation includes attention graphs + 3D loss terrain

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninetieth pass (§G.5 / §G.7)

Ninetieth pass extends global log-scale to §G.5.3 attention weight heatmaps so
low-magnitude Q/K/V cells are visible without distorting overlay Δ diff panels.

**React frontend**
- `chartLogScale.ts` — ``transformMatrixLogScale()`` helper; ``attention``/``weight``
  metrics recognised by ``isLogScaleMetric``
- `MLIntrospectionPanel` — ``buildLogAwareMatrixHeatmap`` log-transforms raw attention
  cells when global ``logScale`` on; overlay/distribution Δ diff stays linear;
  tooltips retain raw weight values; subtitle reflects mode

**ROADMAP**
- §G.5.3 Attention weight heatmaps log-scale checked
- §G.7 Global log-scale propagation includes ML attention heatmaps

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-ninth pass (§G.1 / §G.7)

Eighty-ninth pass extends global log-scale to §G.1.3 policy configuration
heatmaps so KPI cells are symlog/log-transformed before min–max normalisation.

**React frontend**
- `heatmapMetrics.ts` — ``buildNormalizedHeatmapCells`` accepts ``logScale``;
  applies ``displayBarValue`` symlog/log transform before row normalisation;
  tooltips retain raw KPI values
- `PolicyHeatmapChart` / `DistributionFacetHeatmaps` — log-normalised cells
  when global ``logScale`` on; subtitle reflects mode
- `BenchmarkPortfolioHeatmap` / `BenchmarkDistributionHeatmap` /
  `BenchmarkGraphHeatmap` — accept ``logScale`` prop from Simulation Summary
  and Benchmark Analysis

**ROADMAP**
- §G.1.3 Policy configuration heatmaps log-scale checked
- §G.7 Global log-scale propagation includes policy configuration heatmaps

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-eighth pass (§G.1 / §G.6 / §G.7)

Eighty-eighth pass aligns Pareto scatter charts with symlog-overflows convention and
extends global log-scale to OLAP auto-chart and pivot heatmaps.

**React frontend**
- `BenchmarkParetoPanel` — symlog overflows y-axis + log profit x-axis when global
  ``logScale`` on; raw KPI tooltips preserved
- `SimulationSummary` — ``PolicyParetoChart`` symlog overflows + log profit x-axis;
  subtitle reflects linear vs symlog/log mode
- `queryAutoChart.ts` — profit vs overflows scatter uses ``chartMetricDisplay`` symlog;
  heatmap ``visualMap`` transforms KPI cells when ``logScale`` on
- `pivotTable.ts` — ``pivotHeatmapOption`` accepts ``logScale`` + ``valueKey`` for
  log-normalised pivot heatmap cells
- `PivotTablePanel` / `SqlQueryPanel` — pivot heatmap follows global ``logScale``

**ROADMAP**
- §G.1.2 Pareto scatter symlog overflows + log profit x-axis checked
- §G.6 Auto-chart heatmap + pivot heatmap log-scale checked
- §G.7 Global log-scale propagation includes Pareto symlog + OLAP/pivot heatmaps

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-seventh pass (§G.1 / §G.2 / §G.6 / §G.7)

Eighty-seventh pass extends global log-scale to policy/portfolio parallel
coordinates, hierarchy drill-down profit bars, and OLAP profit vs overflows
scatter charts via shared ``parallelAxisValue`` helpers.

**React frontend**
- `chartLogScale.ts` — ``parallelAxisValue()`` + ``invertParallelAxisValue()`` for
  parallel-coordinates KPI transforms and symlog corridor brush inversion
- `SimulationSummary` — ``PolicyParallelChart`` log-normalised axes; ``PolicyHierarchyPanel``
  drill-down profit log x-axis; portfolio parallel passes ``logScale``
- `BenchmarkPortfolioParallel` — log-normalised profit · kg/km · km; symlog overflows
- `BenchmarkAnalysis` — portfolio parallel passes global ``logScale``
- `queryAutoChart.ts` — profit vs overflows scatter log x + log y when ``logScale`` on

**ROADMAP**
- §G.1.4 Policy + portfolio parallel coordinates log-normalised axes checked
- §G.2 Hierarchy drill-down profit bars log-scale checked
- §G.6 Auto-chart profit vs overflows scatter log-scale checked (partial marker removed)
- §G.7 Global log-scale propagation includes parallel coords + hierarchy drill-down

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-sixth pass (§G.1 / §G.5 / §G.7 / §G.18)

Eighty-sixth pass extends global log-scale to policy radar charts, HPO parallel
coordinates, and ML loss contour heatmaps via shared ``radarAxisValue`` helper.

**React frontend**
- `chartLogScale.ts` — ``radarAxisValue()`` for radar / parallel-axis metric transforms
- `SimulationSummary` — ``PolicyRadarChart`` log-normalised axes when global ``logScale`` on
- `AlgorithmComparison` — radar chart log-normalised metric axes + subtitle
- `HPOTracker` — parallel-coordinates objective axis log transform + subtitle
- `MLIntrospectionPanel` — loss contour log colour map; raw-loss tooltips;
  ``ExperimentTracker`` passes global toggle

**ROADMAP**
- §G.1 Policy radar log-normalised axes checked (partial marker removed)
- §G.5.2 ML loss contour log-scale colour map checked
- §G.7 Global log-scale propagation includes radar, HPO parallel, loss contour
- §G.18 HPO parallel-coordinates objective log-scale checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-fifth pass (§G.1 / §G.7 / §G.11 / §G.16 / §G.18)

Eighty-fifth pass extends global log-scale to Simulation Summary per-day
trajectory, Data Generation demand histogram, and ZenML step-duration charts.

**React frontend**
- `chartLogScale.ts` — duration/count/histogram metric heuristics for launcher charts
- `SimulationSummary` — ``TrajectoryChart`` symlog overflows + log profit/km/kg when
  global ``logScale`` on; linear vs log subtitle per selected metric
- `DataGeneration` — demand preview histogram log y-axis; ``GlobalFilterBar`` toggle
- `ZenMLPipelineView` — step-duration Gantt bars use log x-axis when ``logScale`` on;
  tooltips show raw seconds; ``ExperimentTracker`` passes global toggle

**ROADMAP**
- §G.1 Simulation Summary per-day trajectory log-scale checked
- §G.7 Global log-scale propagation includes trajectory, Data Generation, ZenML
- §G.11 Data Generation demand histogram log-scale checked
- §G.16 Simulation Summary trajectory log-scale noted
- §G.18 ZenML step-duration chart log-scale checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-fourth pass (§G.6 / §G.7 / §G.16 / §G.18)

Eighty-fourth pass extends global log-scale to Experiment Tracker, Simulation
Monitor daily KPI charts, and OLAP auto-chart bar/line types.

**React frontend**
- `chartLogScale.ts` — shared metric heuristics for symlog overflows and log KPI axes
- `ExperimentTracker` — MLflow metric comparison log y-axis when global ``logScale`` on;
  ``GlobalFilterBar`` + linear vs log subtitle (disabled when Normalize Y is on)
- `SimulationMonitor` — ``MetricTimeseries`` symlog overflows + log profit/km/kg;
  ``GlobalFilterBar`` when a log is loaded; daily KPI subtitle
- `queryAutoChart.ts` — bar / grouped-bar / line auto-charts follow ``logScale`` on
  overflow, loss, and KPI y-axis metrics

**ROADMAP**
- §G.7 Global log-scale propagation includes Experiment Tracker + Simulation Monitor
- §G.6 Auto-chart log-scale on bar / grouped-bar / line checked
- §G.16 Simulation Monitor daily KPI timeseries log-scale checked
- §G.18 Experiment Tracker MLflow metric comparison log-scale checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-third pass (§G.7 / §G.10 / §G.17 / §G.18 / §D.7)

Eighty-third pass extends global log-scale to training and HPO charts and adds
``Ctrl+S`` save in the Config Editor.

**React frontend**
- `TrainingMonitor` — ``MultiRunChart`` log loss axis; grad-norm/LR sparklines log
  y-axis when global ``logScale`` on; ``GlobalFilterBar`` above run list
- `TrainingHub` — ``LiveChart`` + ``MiniSparkline`` follow global ``logScale``;
  ``GlobalFilterBar`` in live progress panel; linear vs log subtitle
- `HPOTracker` — optimisation history scatter + best-so-far + cross-study overlay
  use log objective axis when ``logScale`` on; ``GlobalFilterBar`` + subtitle
- `BenchmarkAnalysis` — eval checkpoint panel subtitle reflects linear vs log mode
- `ConfigEditor` — ``Ctrl+S`` / ``Cmd+S`` saves when dirty; documented in shortcuts help

**ROADMAP**
- §G.7 Global log-scale propagation includes Training Monitor, Training Hub, HPO Tracker
- §G.10 / §G.17 / §G.18 training + HPO log-scale items checked
- §G.13 ``Ctrl+S`` config save checked; §D.7 keyboard shortcuts marked complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-second pass (§G.7 / §G.12)

Eighty-second pass extends global log-scale to Evaluation Runner inline checkpoint
comparison charts, closing the remaining §G.12 partial marker.

**React frontend**
- `EvaluationRunner` — inline cost/gap/time bar charts follow global ``logScale``;
  ``GlobalFilterBar`` toggle above results grid; linear vs log subtitle

**ROADMAP**
- §G.12 EvaluationRunner inline charts global log-scale checked; partial marker removed
- §G.7 Global log-scale propagation includes Evaluation Runner

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighty-first pass (§G.1 / §G.7 / §G.12)

Eighty-first pass extends symlog-overflows log-scale polish and responsive chart grids
to Algorithm Comparison and Evaluation Runner inline charts.

**React frontend**
- `AlgorithmComparison` — symlog overflows y-axis when global ``logScale`` on; profit/km/kg/km
  use log axis; linear vs log subtitle; metric bar grid ``sm:grid-cols-2 lg:grid-cols-4``
- `EvaluationRunner` — inline checkpoint bar charts use ``grid-cols-1 sm:grid-cols-2 lg:grid-cols-3``

**ROADMAP**
- §G.1.1 AlgorithmComparison symlog overflows on log-scale metric bars checked
- §G.7 AlgorithmComparison responsive chart grids checked
- §G.12 EvaluationRunner responsive inline chart grid checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — eightieth pass (§G.1 / §G.7)

Eightieth pass extends global log-scale and responsive layout polish to Benchmark
Analysis, including portfolio efficiency ranking and eval checkpoint charts.

**React frontend**
- `BenchmarkAnalysis` — multi-run ``PortfolioEfficiencyRanking`` with global
  ``logScale``; single-run efficiency chart log x-axis; symlog overflows on
  multi-run metric bars when log on; eval checkpoint charts follow ``logScale``
- `BenchmarkAnalysis` — responsive grids: Pareto ``md:grid-cols-2``, metric bars
  ``sm:grid-cols-2``, eval results ``sm:grid-cols-2 lg:grid-cols-3``; city
  comparison subtitle reflects linear vs log mode

**ROADMAP**
- §G.1.5 BenchmarkAnalysis efficiency ranking global log-scale checked
- §G.1.1 BenchmarkAnalysis symlog overflows on log-scale metric bars checked
- §G.7 BenchmarkAnalysis responsive chart grids checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-ninth pass (§G.1 / §G.7)

Seventy-ninth pass extends global log-scale to grouped metric bar charts on Simulation
Summary and polishes responsive layout for narrow viewports.

**React frontend**
- `GroupedMetricBarChart` — ``logScale`` + ``useSymlog`` props; symlog overflows axis;
  log kg/km axis; whiskers suppressed on log scale
- `SimulationSummary` — grouped overflow/kg/km charts follow global ``logScale``; Pareto
  panel grid `md:grid-cols-2`; metric bar grid `sm:grid-cols-2`
- `Layout` — sidebar auto-collapses below `lg` via `matchMedia` listener

**ROADMAP**
- §G.1.1 Grouped metric bar charts global log-scale checked
- §G.7 Responsive layout partial marker removed; mobile sidebar + chart grids checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-eighth pass (§G.7)

Seventy-eighth pass completes full startup prefetch for every lazy route and heavy
vendor chunk, and extends global log-scale to efficiency ranking charts.

**React frontend**
- `App.tsx` — startup prefetch warms all 18 lazy routes plus duckdb-wasm, sigma,
  and @react-three/fiber vendor chunks
- `EfficiencyRankingChart` / `PortfolioEfficiencyRanking` — log x-axis when global
  ``logScale`` on; error-bar whiskers suppressed on log scale
- `SimulationSummary` — passes global ``logScale`` to efficiency ranking panels

**ROADMAP**
- §G.7 Startup route prefetch (all routes) checked
- §G.7 Startup vendor prefetch (duckdb + sigma + r3f) checked
- §G.7 performance partial markers removed from lazy-load / manualChunks / timing items
- §G.1.5 Efficiency ranking global log-scale checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-seventh pass (§G.7)

Seventy-seventh pass extends global log-scale propagation to City Comparison charts and
completes analytics startup prefetch for city + algorithm routes.

**React frontend**
- `cityComparison.ts` — `cityComparisonChartOption()` accepts ``logScale``; symlog-overflows
  when on, linear raw values when off
- `CityComparison` / `SimulationSummary` / `BenchmarkAnalysis` — city comparison bars follow
  global ``logScale``; City Comparison page shows ``showLogScale`` in filter bar
- `App.tsx` — startup prefetch warms city comparison + algorithm comparison route chunks

**ROADMAP**
- §G.1.6 City Comparison global log-scale toggle checked
- §G.1 log-scale partial markers removed (Pareto, Benchmark, Algorithm Comparison)
- §G.7 Startup route prefetch (city + algorithms) checked
- §G.7 marked complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-sixth pass (§G.7)

Seventy-sixth pass unifies log-scale chart toggles under global filter state and extends
startup prefetch for analytics routes.

**React frontend**
- `useGlobalFiltersStore` — global ``logScale`` boolean shared across analytics views
- `GlobalFilterBar` — ``showLogScale`` prop adds app-wide log-scale toggle; Clear resets it
- `useHashSync` — bookmarkable ``l=1`` query param for log-scale deep-links
- `SimulationSummary` / `BenchmarkAnalysis` / `AlgorithmComparison` — consume global
  ``logScale``; per-page toggles removed
- `SqlQueryPanel` — auto-chart scatter log overflows axis follows global ``logScale``
- `App.tsx` — startup prefetch warms benchmark + OLAP explorer routes and Monaco editor chunk

**ROADMAP**
- §G.7 Global log-scale filter + bookmarkable ``l=1`` hash sync checked
- §G.7 Startup route prefetch (benchmark + OLAP) checked
- §G.7 Startup vendor prefetch (@monaco-editor/react) checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-fifth pass (§G.6)

Seventy-fifth pass closes §G.6 auto-chart Pareto polish: frontier step-line overlay on
labeled scatter, log-scale overflows axis toggle, and line-chart cross-filter / type
override for time-series queries.

**React frontend**
- `queryAutoChart` — Pareto frontier step-line + frontier point highlight on labeled
  scatter; ``logScale`` option for overflows axis; line chart in ``suggestChartAlternatives()``
- `SqlQueryPanel` — log overflows toggle; line point click → ``onDaySelect``; ignore
  Pareto front line clicks on scatter

**ROADMAP**
- §G.6 Pareto frontier step-line overlay checked
- §G.6 Auto-chart log-scale overflows toggle checked
- §G.6 Auto-chart line cross-filter + line type override checked
- §G.6 marked complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-fourth pass (§G.6)

Seventy-fourth pass completes §G.6 pre-built query templates and auto-chart scatter
polish: Pareto efficiency frontier SQL, labeled scatter cross-filter, and SVG export.

**React frontend**
- `duckdbTemplates` — ``Pareto efficiency frontier`` template (single-log + portfolio)
- `queryAutoChart` — labeled profit vs overflows scatter with ``labelKey`` for brush
  resolution; point labels when ≤24 rows
- `SqlQueryPanel` — scatter click cross-filter; SVG export alongside PNG

**ROADMAP**
- §G.6 Pareto efficiency frontier SQL template checked
- §G.6 Auto-chart scatter cross-filter checked
- §G.6 Auto-chart SVG export checked
- §G.6 Pre-built query templates marked complete (partial removed)
- §G.6 Data Explorer sort/filter/export partial markers removed

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-third pass (§G.6)

Seventy-third pass closes remaining §G.6 auto-chart polish: click-to-cross-filter on
suggested charts, PNG export, chart-type override chips, and a run×policy matrix SQL
template.

**React frontend**
- `queryAutoChart` — ``suggestChartAlternatives()`` + ``heatmapCellLabels()`` for
  multi-type suggestions and heatmap brush resolution
- `SqlQueryPanel` — auto-chart click cross-filter (bar / grouped-bar / heatmap);
  type override chips; PNG export via ``exportChartPng()``
- `duckdbTemplates` — ``Run×policy matrix (kg/km)`` portfolio template

**ROADMAP**
- §G.6 Auto-chart click cross-filter checked
- §G.6 Auto-chart PNG export checked
- §G.6 Auto-chart type override selector checked
- §G.6 Run×policy matrix SQL template checked
- §G.6 Auto-chart suggestions marked complete (heatmap included)

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-second pass (§G.6)

Seventy-second pass closes partial §G.6 OLAP/Data Explorer polish: DuckDB-derived
filter options in OLAP Explorer, heatmap auto-charts for matrix query results,
cell-level cross-filtering, and brush-aware CSV export.

**React frontend**
- `OlapExplorer` — DuckDB-derived ``policy`` + ``city_scale`` options for
  ``GlobalFilterBar`` on any ingested table
- `queryAutoChart` — ``heatmap`` chart type for ``city_scale`` × ``policy`` and
  ``run_label`` × ``policy`` matrix results
- `DataExplorer` — cell-level brush cross-filter; export respects global brush +
  text filter + sort order
- `SqlQueryPanel` — cell-level brush cross-filter on result grid (policy /
  ``run_label`` / ``city_scale`` columns)

**ROADMAP**
- §G.6 Auto-chart heatmap for city×policy matrix checked
- §G.6 OLAP DuckDB-derived policy / city_scale filter bar checked
- §G.6 Data Explorer cell-level cross-filter + brush-aware export checked
- §G.6 SQL result grid cell-level cross-filter checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventy-first pass (§G.6)

Seventy-first pass extends Data Explorer portfolio brushing to CSV-derived filter
options, detects portfolio tables dynamically in OLAP Explorer, and adds grouped
auto-charts for multi-dimension SQL results.

**React frontend**
- `GlobalFilterBar` — optional ``policies`` prop for CSV/DuckDB-derived policy options
- `DataExplorer` — CSV-derived policy / ``run_label`` / city selectors; SQL + HTML
  table row cross-filter dimming when brush columns present
- `OlapExplorer` — portfolio mode via ``duckDbHasColumn(run_label)`` instead of
  hardcoded table set (custom ``olap_*`` ingests included)
- `queryAutoChart` — ``grouped-bar`` chart type for ``city_scale`` × ``policy`` results
- `duckdbClient` — ``duckDbTableColumns()`` + ``duckDbHasColumn()`` helpers

**ROADMAP**
- §G.6 Data Explorer CSV-derived filter bar + row cross-filter checked
- §G.6 OLAP dynamic portfolio column detection checked
- §G.6 Auto-chart grouped bar for multi-dimension GROUP BY checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventieth pass (§G.6)

Seventieth pass ensures single-log portfolio ingests always carry ``run_label`` and
``city_scale``, extends pivot/auto-chart/SQL tooling for city groups, and wires Data
Explorer policy brush sync.

**React frontend**
- `arrowPipeline` — `runPortfolioSimulationArrowPipeline()` always adds ``run_label`` +
  ``city_scale`` (removes single-log shortcut)
- `SimulationSummary` — always uses portfolio pipeline for DuckDB ingest
- `OlapExplorer` — JSONL ingest via portfolio pipeline with filename ``run_label``
- `PivotTablePanel` — ``city_scale`` row highlight + cross-filter click
- `SqlQueryPanel` — passes ``highlightCityScaleLabels`` to pivot panel
- `queryAutoChart` — prefers ``city_scale`` / ``run_label`` / ``policy`` dimensions
- `duckdbTemplates` — City×policy matrix (kg/km) template
- `DataExplorer` — ``GlobalFilterBar`` + SQL ``brushSqlSync`` when CSV has policy column

**ROADMAP**
- §G.6 Portfolio single-log ``run_label`` + ``city_scale`` columns checked
- §G.6 Pivot table ``city_scale`` cross-filter checked
- §G.6 City×policy matrix SQL template checked
- §G.6 Auto-chart portfolio GROUP BY detection checked
- §G.6 Data Explorer global filter bar + SQL brush sync checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-ninth pass (§G.6)

Sixty-ninth pass wires GlobalFilterBar selections into DuckDB SQL brush sync,
adds a ``city_scale`` column on portfolio ingest, and extends SQL cross-filtering.

**React frontend**
- `SqlQueryPanel` — ``brushFilter`` merges global policy / ``run_label`` / city
  brush when chart props are absent; ``city_scale`` row cross-filter + dimming
- `arrowPipeline` — portfolio union adds ``city_scale`` via `cityScaleFromRunLabel()`
- `cityComparison` — `cityScaleFromRunLabel()` helper for ingest + SQL
- `duckdbTemplates` — city leaderboard template; ``city_scale`` WHERE clause in
  ``brushedPortfolioSql()``
- `SimulationSummary` / `BenchmarkAnalysis` / `CityComparison` — pass
  ``portfolioRunLabels`` to SQL panel for filter-bar city expansion

**ROADMAP**
- §G.6 Global filter bar → SQL brush sync checked
- §G.6 Portfolio ``city_scale`` column + city leaderboard template checked
- §G.6 SQL ``city_scale`` row cross-filter checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-eighth pass (§G.6)

Sixty-eighth pass wires OLAP Explorer city brushing into DuckDB SQL sync and
centralizes portfolio ``run_label`` expansion for city groups.

**React frontend**
- `cityComparison.ts` — `groupRunLabelsByCity()` + `resolveBrushedRunLabels()` for
  DuckDB ``run_label`` city grouping
- `usePortfolioRunBrush` — delegates run-label expansion to `resolveBrushedRunLabels()`
- `SqlQueryPanel` — `portfolioRunLabels` prop; city brush expands to ``run_label`` IN
  clause; SQL row ``run_label`` cross-filter clears ``brushedCity``
- `OlapExplorer` — city/scale dropdown on portfolio tables; SQL panel receives
  ``portfolioRunLabels`` for city brush sync

**ROADMAP**
- §G.6 OLAP Explorer global city/scale brush SQL sync checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-seventh pass (§G.6 / §G.7)

Sixty-seventh pass promotes portfolio city brushing to global filter state with
bookmarkable deep links, and adds SQL result row search with filtered export.

**React frontend**
- `useGlobalFiltersStore` — `brushedCity` + `setBrushedCity`; cleared on filter reset
  and mutually exclusive with ``runLabel`` selections
- `usePortfolioRunBrush` — city brush reads/writes global store (fixes filter bar /
  chart desync when run selector changes)
- `GlobalFilterBar` — city/scale dropdown when ≥2 city groups loaded on portfolio views
- `useHashSync` — serializes ``brushedCity`` as ``c`` URL hash param
- `SqlQueryPanel` — row filter search box; CSV export respects active filter + sort

**ROADMAP**
- §G.6 Portfolio global city/scale filter bar checked
- §G.6 SQL result grid row filter + filtered CSV export checked
- §G.7 Bookmarkable city brush URL hash sync checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-sixth pass (§G.6 / §G.7)

Sixty-sixth pass unifies portfolio ``run_label`` brushing with the global filter store
across Summary, Benchmark, and City views, and adds bookmarkable ``run_label`` deep links.

**React frontend**
- `usePortfolioRunBrush` — shared city/run brush hook; chart clicks set global ``runLabel``;
  city chart expands to all runs in the group
- `SimulationSummary` / `BenchmarkAnalysis` / `CityComparison` — `GlobalFilterBar` run
  selector when ≥2 runs loaded; SQL panels mirror global brush
- `useHashSync` — serializes ``runLabel`` as ``r`` URL hash param; restores on load and
  browser back/forward

**ROADMAP**
- §G.6 Portfolio global run_label filter bar on Summary/Benchmark/City checked
- §G.7 Bookmarkable run_label URL hash sync checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-fifth pass (§G.6)

Sixty-fifth pass adds global ``run_label`` brush SQL sync to the OLAP Explorer and
bidirectional ``run_label`` cross-filtering from SQL result rows and pivot tables.

**React frontend**
- `useGlobalFiltersStore` — `runLabel` + `setRunLabel`; cleared on filter reset
- `GlobalFilterBar` — optional ``run_label`` dropdown when portfolio run options are supplied
- `SqlQueryPanel` — result row + pivot click sets global ``run_label``; row dimming +
  `brushSqlSync` mirrors policy + run brushes
- `PivotTablePanel` — separate policy / ``run_label`` pivot highlight props
- `OlapExplorer` — loads distinct ``run_label`` values per portfolio table; passes run
  filter to `GlobalFilterBar` + `SqlQueryPanel`

**Utilities**
- `duckdbClient.ts` — `listDuckDbDistinctValues()` for OLAP run selector options

**ROADMAP**
- §G.6 OLAP Explorer global run_label brush SQL sync checked
- §G.6 SQL result row + pivot run_label cross-filter checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-fourth pass (§G.6)

Sixty-fourth pass extends portfolio ``run_label`` brush SQL sync to Simulation Summary
and Benchmark Analysis, and connects the standalone OLAP Explorer to global policy filters.

**React frontend**
- `SimulationSummary` — comparison-run click, city chart click, and portfolio efficiency
  ranking click set ``run_label`` brush; `SqlQueryPanel` `highlightRunLabels` on `summary_sim`
- `BenchmarkAnalysis` — city comparison chart click filters by ``run_label`` on `benchmark_sim`
- `PortfolioEfficiencyRanking` — `onConfigClick(policy, runLabel)` for run×policy bar clicks
- `OlapExplorer` — `brushSqlSync` + `autoRunOnBrushSync` from `GlobalFilterBar` policy;
  portfolio/algorithm template modes for known ingested tables

**ROADMAP**
- §G.6 Simulation Summary + Benchmark Analysis run_label brush SQL sync checked
- §G.6 OLAP Explorer global policy brush SQL sync checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-third pass (§G.6)

Sixty-third pass extends bidirectional chart ↔ DuckDB brush sync to Benchmark Analysis
and City Comparison, and unifies portfolio SQL brush filters.

**React frontend**
- `duckdbTemplates.ts` — `brushedPortfolioSql()` combines policy + `run_label` WHERE clauses
- `SqlQueryPanel` — `highlightRunLabels` prop; brush sync uses portfolio filter helper
- `BenchmarkAnalysis` — efficiency ranking + metric bar click sets global policy filter with
  dimming; `brushSqlSync` + `autoRunOnBrushSync` on `benchmark_sim`
- `CityComparison` — city chart + summary table click filters by `run_label`; brush SQL sync
  on `city_sim`

**ROADMAP**
- §G.6 Benchmark Analysis + City Comparison brush SQL sync checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-second pass (§G.6)

Sixty-second pass adds Algorithm Comparison policy-analysis SQL templates and
bidirectional chart ↔ DuckDB brush sync on the algorithms view.

**React frontend**
- `duckdbTemplates.ts` — `algorithmSqlTemplates()` for policy ranking, worst overflow
  days, zero-overflow rate, and day-over-day profit Δ
- `SqlQueryPanel` — `algorithmMode` prop merges algorithm templates
- `AlgorithmComparison` — radar/bar click sets global policy filter with dimming;
  `brushSqlSync` + `autoRunOnBrushSync` on `algorithm_sim`

**ROADMAP**
- §G.6 Algorithm Comparison SQL templates + brush SQL sync checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixty-first pass (§G.6 / §G.7)

Sixty-first pass adds portfolio-aware OLAP query templates and closes the DuckDB ingest
gap on Algorithm Comparison.

**React frontend**
- `duckdbTemplates.ts` — `portfolioSqlTemplates()` for cross-run robustness, run
  leaderboard, run×policy variance, and Pareto-by-run when `run_label` is present
- `SqlQueryPanel` — `portfolioMode` prop merges portfolio templates on multi-log views
- `AlgorithmComparison` — DuckDB ingest into `algorithm_sim`, timing badge, `SqlQueryPanel`
- `Settings` — last-ingest summary uses shared `formatPipelineTimingBadge()`

**ROADMAP**
- §G.6 portfolio query templates + Algorithm Comparison DuckDB checked
- §G.7 Settings timing badge helper checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixtieth pass (§G.1.4 / §G.6 / §G.7)

Sixtieth pass unions multi-run simulation portfolios into DuckDB-Wasm and surfaces
consistent ingest timing badges across all portfolio analytics views.

**React frontend**
- `arrowPipeline.ts` — `runPortfolioSimulationArrowPipeline()` unions JSONL logs with
  `run_label`; `formatPipelineTimingBadge()` shared timing text (sidecar count, budget)
- `SimulationSummary` — portfolio mode re-ingests primary + comparison runs into
  `summary_sim`
- `BenchmarkAnalysis` / `CityComparison` — DuckDB ingest + `SqlQueryPanel` on loaded
  portfolios (`benchmark_sim` / `city_sim`)
- `DataExplorer` / `OlapExplorer` / `SimulationMonitor` — unified timing badge format

**ROADMAP**
- §G.0 portfolio DuckDB union + timing badge helper checked
- §G.1.4 portfolio DuckDB ingest across summary/benchmark/city views checked
- §G.6 portfolio SQL panels on Benchmark Analysis + City Comparison checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-ninth pass (§G.6 / §G.7)

Fifty-ninth pass extends JSONL Arrow sidecar coverage to the standalone OLAP Explorer and
surfaces DuckDB ingest timing badges across all simulation log views.

**React frontend**
- `OlapExplorer` — "Ingest CSV / JSONL" uses `runSimulationArrowPipeline()` with sidecar
  fast-path; last-ingest timing badge notes Arrow sidecar hits
- `SimulationSummary` / `SimulationMonitor` — DuckDB row count + latency badge on loaded logs;
  notes sidecar fast-path when a sibling ``.arrow`` is present
- `arrowPipeline.ts` — `runSimulationArrowPipeline()` slow path sets `usedSidecar: false`

**ROADMAP**
- §G.6 OLAP JSONL ingest + sidecar fast-path checked
- Effort × Impact matrix updated: §G.1–§G.18 phases marked ✅ complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-eighth pass (§G.0 / §G.8)

Fifty-eighth pass extends the Arrow IPC loop to simulation JSONL logs: DuckDB ingest
prefers pre-built ``.arrow`` sidecars for logs, bundles can emit log sidecars, and
integration tests verify row parity.

**React frontend**
- `arrowPipeline.ts` — `jsonlArrowSidecarPath()`, `runSimulationArrowPipeline()`
  sidecar fast-path via `path_exists` + `runArrowSidecarPipeline()`
- `Settings` — Arrow benchmark accepts CSV or JSONL; timing badge notes sidecar path
- `OutputBrowser` — export toggle label covers CSV + JSONL sidecars

**Rust**
- `arrow.rs` — `write_simulation_log_arrow_sidecar()` for on-disk JSONL → Arrow IPC
- `data.rs` — `create_wsroute_bundle(..., include_arrow)` emits sidecars for CSV and
  JSONL; `simulation_arrow_sidecar_row_parity` + updated round-trip tests

**Python**
- `export_for_studio.py` — `--arrow` emits Arrow IPC sidecars for JSONL logs via
  `parse_day_log_line()` + `jsonl_to_arrow_ipc()`

**ROADMAP**
- §G.0 JSONL Arrow sidecar fast-path ingest checked
- §G.8 JSONL bundle export + simulation row parity tests checked
- §G.17 Training Monitor + §G.18 Experiment & HPO Tracker marked ✅ complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-seventh pass (§G.0 / §G.8)

Fifty-seventh pass closes the Arrow IPC loop: Studio DuckDB ingest prefers pre-built
``.arrow`` sidecars from extracted bundles, and the Output Browser can emit sidecars
when packaging runs.

**React frontend**
- `arrowPipeline.ts` — `csvArrowSidecarPath()`, `runArrowSidecarPipeline()` sidecar
  fast-path; `runCsvArrowPipeline()` auto-detects sibling ``.arrow`` via `path_exists`
- `DataExplorer` / `Settings` — pipeline timing badge notes sidecar fast-path
- `OutputBrowser` — “Include Arrow IPC sidecars” export toggle; manifest
  `arrow_sidecars` count in bundle inspector

**Rust**
- `arrow.rs` — `write_csv_arrow_sidecar()`, `path_exists` command
- `data.rs` — `create_wsroute_bundle(..., include_arrow)` emits ``.arrow`` sidecars;
  `inspect_wsroute_bundle` surfaces `arrow_sidecars` from manifest; `.arrow` in bundle extensions

**ROADMAP**
- §G.0 Arrow sidecar fast-path ingest checked
- §G.8 Studio sidecar ingest + Rust bundle Arrow export checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-sixth pass (§G.1.3 / §G.2 / §G.8)

Fifty-sixth pass closes deferred **§G.1.3** portfolio policy×metric heatmap and
unified heatmap mode across facets, **§G.2** hierarchy breadcrumb root reset,
and **§G.8** Arrow IPC sidecar export for `.wsroute` bundles.

**React frontend**
- `heatmapMetrics.ts` — shared `HeatmapMode`, metric schema, normalised cell builder
- `BenchmarkPortfolioHeatmap` — portfolio-wide policy×metric heatmap with brush dimming
- `BenchmarkDistributionHeatmap` / `BenchmarkGraphHeatmap` — support `all` / `overflows` / `kg/km` modes
- `SimulationSummary` — portfolio heatmap panel; unified `heatmapMode` drives distribution/graph facets
- `BenchmarkAnalysis` — graph facet heatmaps use same three-mode toggle
- `HierarchyBreadcrumb` — root **All** button resets sunburst drill-down (§G.2)

**Python**
- `export_for_studio.py` — `--arrow` flag writes Arrow IPC (`.arrow`) sidecars for each CSV;
  manifest records `arrow_sidecars` count

**ROADMAP**
- §G.1.3 unified heatmap mode + portfolio policy×metric heatmap checked
- §G.2 breadcrumb root **All** navigation checked
- §G.8 Arrow IPC bundle export checked
- §G.9–§G.15, §G.19 phase headers marked ✅ complete

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-fifth pass (§G.1.2 / §G.1.3 / §G.1.4 / §G.1.5)

Fifty-fifth pass closes deferred **§G.1** portfolio Pareto markers/tooltips,
distribution facet heatmaps, Simulation Summary portfolio parallel coordinates,
and multi-config efficiency ranking.

**React frontend**
- `BenchmarkParetoPanel` — `citySymbol()` per run×policy point; tooltips with
  `formatLogMeta` + `formatPolicyMeta` + Pareto-optimal badge
- `paretoPortfolio.ts` — `ParetoPoint` carries `path` + `logMeta` for multi-run scatter
- `BenchmarkDistributionHeatmap` — per-distribution policy heatmap facets in portfolio mode
- `portfolioDistribution.ts` — `groupRunsByDistribution()` buckets loaded runs
- `BenchmarkPortfolioParallel` — shared component extracted from Benchmark Analysis;
  wired on Simulation Summary when ≥2 runs loaded
- `PortfolioEfficiencyRanking` — top run×policy configs ranked by mean kg/km with whiskers
- `SimulationSummary` — portfolio overflows-by-city bars; distribution + graph heatmap
  facets; portfolio parallel + efficiency ranking panels

**ROADMAP**
- §G.1.1 multi-city overflows grouped bars on Simulation Summary checked
- §G.1.2 Pareto marker shapes + config tooltips on portfolio panels checked
- §G.1.3 distribution facet heatmaps in portfolio mode checked
- §G.1.4 portfolio parallel on Simulation Summary checked
- §G.1.5 portfolio efficiency ranking checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-fourth pass (§G.1 / §G.2 / §G.7)

Fifty-fourth pass closes deferred **§G.1** Simulation Summary portfolio analytics,
**§G.2** multi-log hierarchy sunburst, and **§G.7** chart-render benchmark.

**React frontend**
- `SimulationSummary` — add comparison log + output portfolio load; 4-panel Pareto
  grid on single- or multi-log; graph heatmap facets + city comparison when ≥2 runs;
  kg/km grouped by city/scale in portfolio mode
- `BenchmarkParetoPanel` / `BenchmarkGraphHeatmap` — shared components extracted
  from Benchmark Analysis
- `paretoPortfolio.ts` — `buildParetoByPanel()` shared Pareto point builder
- `policyHierarchy.ts` — `buildPortfolioHierarchy()` multi-root sunburst per city/scale
- `chartRenderBenchmark.ts` — off-screen ECharts render timing probe
- `Settings` — "Run Chart Render Benchmark" button + 500 ms budget badge in About

**ROADMAP**
- §G.1.1 multi-city kg/km grouped bars on Simulation Summary checked
- §G.1.2 single-log Simulation Summary 4-panel Pareto checked
- §G.1.3 Simulation Summary graph heatmap facets checked
- §G.2 multi-log portfolio hierarchy sunburst checked
- §G.7 chart-render benchmark checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-third pass (§G.1.6 / §G.2 / §G.6)

Fifty-third pass closes deferred **§G.1.6** dedicated City Comparison page,
**§G.2** DuckDB auto-run on hierarchy segment brush, and **§G.6** standalone
OLAP Explorer + pivot drag wells.

**React frontend**
- `CityComparison` — dedicated city/graph comparison page; portfolio load;
  log-scale profit · symlog-overflows · kg/km bars + summary table
- `cityComparison.ts` — shared `groupRunsByCity` + chart builders (reused by BenchmarkAnalysis)
- `OlapExplorer` — standalone DuckDB-Wasm OLAP page; `listDuckDbTables` table
  picker; CSV ingest into `olap_*` tables
- `SqlQueryPanel` — `autoRunOnBrushSync` executes brush SQL; auto-expands on brush
- `PivotTablePanel` — draggable column chips + HTML5 drop wells (row/column/value)
- `duckdbClient.ts` — `listDuckDbTables()` for OLAP table discovery

**Navigation**
- Sidebar + command palette: City Comparison, OLAP Explorer
- `AppMode`: `city_comparison`, `olap_explorer`

**ROADMAP**
- §G.1.6 dedicated City Comparison page checked
- §G.2 DuckDB auto-run on segment brush checked
- §G.6 standalone OLAP page checked
- §G.6 pivot drag wells checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-second pass (§G.1 / §G.2 / §G.3)

Fifty-second pass closes deferred **§G.1** DuckDB brush sync and 480-log portfolio
scan, **§G.2** animated sunburst morph, and **§G.3** Cartesian TripsLayer +
collected-kg node radius.

**React frontend**
- `SimulationSummary` — ingests log → DuckDB `summary_sim`; `SqlQueryPanel` with
  `brushSqlSync` + multi-policy `highlightPolicies`
- `duckdbTemplates.ts` — `brushedPoliciesSql()` mirrors chart policy brush
- `SqlQueryPanel` — `highlightPolicies` / `brushSqlSync` props for multi-policy dim
- `PolicyHierarchyPanel` — `universalTransition` morphs sunburst/treemap → drill bars
- `DeckRouteMap` — `TripsLayer` in OrbitView Cartesian mode; stop radius ∝ collected kg
- `outputRunLogs.ts` — `PORTFOLIO_SCAN_DEFAULT` (480) + `loadPortfolioLogs()` batches
- `BenchmarkAnalysis` — progressive portfolio load with toast progress

**Types**
- `SimDayData.bin_state_collected` corrected to `number[]` (kg collected per bin)

**ROADMAP**
- §G.1 DuckDB SQL brush sync checked
- §G.1.4 full 480-log portfolio scan checked
- §G.2 animated sunburst→bar morph checked
- §G.3.1 collected-kg node radius checked
- §G.3.2 Cartesian TripsLayer animation checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifty-first pass (§G.1 / §G.3)

Fifty-first pass closes deferred **§G.1** kg symlog, ten-axis parallel coordinates,
overflow-corridor axis brush, and **§G.3.2** per-vehicle tour-stop scatter.

**React frontend**
- `parallelPolicyAxes.ts` — ten-axis schema: city · N · dist · improver · strategy ·
  constructor · overflows · kg/km · km · profit
- `PolicyParallelChart` — uses full schema; overflows-axis `brushEnd` syncs corridor slider
- `SimulationSummary` — kg symlog on primary + secondary log-scale rows
- `DeckRouteMap` — per-vehicle `ScatterplotLayer` tour stops when multi-vehicle tour

**ROADMAP**
- §G.1 kg symlog + ten-axis parallel + overflow corridor axis brush checked
- §G.3.2 per-vehicle stop scatter checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fiftieth pass (§G.1 / §G.6)

Fiftieth pass closes deferred **§G.1** symlog/axis-brush items and adds **§G.6**
bidirectional OLAP brush highlighting.

**React frontend**
- `SimulationSummary` — profit/km `MetricBarChart` uses `useSymlog` when log scale on;
  secondary log-scale row adds km symlog duplicate
- `PolicyParallelChart` — ECharts parallel-axis brush toolbox; `brushselected` →
  `handleBrushPolicies` cross-filter
- `SqlQueryPanel` — reads `useGlobalFiltersStore.policy`; highlights matching SQL
  rows; dims non-matching rows when filter active
- `PivotTablePanel` / `pivotTable.ts` — `highlightRowLabels` dims non-matching pivot
  heatmap rows (bidirectional brush with `GlobalFilterBar`)

**ROADMAP**
- §G.1 profit/km symlog checked
- §G.1 parallel-axis brush checked
- §G.6 bidirectional pivot/SQL brush checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-ninth pass (§G.2 / §G.4)

Forty-ninth pass closes remaining **§G.4** deferred timeline/brush items and adds
**§G.2** treemap overflows colour mode.

**React frontend**
- `graphTopology.ts` — `accumulateTourPheromoneByStep`, `countTourEdgeSteps` for
  per-tour-edge ACO τ stepping
- `GraphTopologyPanel` — pheromone mode toggle (by day / by tour step); click node
  → fill-% bidirectional brush across ECharts, Sigma.js, and Cosmograph views
- `TopologySigmaView` / `TopologyCosmographView` — `clickNode` handler for fill brush
- `policyHierarchy.ts` — `HierarchyColorMode` (`kgkm` | `overflows`); green→red
  overflows gradient on treemap/sunburst segments
- `PolicyHierarchyPanel` — kg/km vs overflows colour mode selector

**ROADMAP**
- §G.4 per-ACO-iteration stepping checked
- §G.4 bidirectional chart brush checked
- §G.2 overflows treemap colour mode checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-eighth pass (§G.4 / §G.5)

Forty-eighth pass closes the remaining **§G.4** deferred WebGL/layout items and adds
**§G.5.1** compressed NPZ plane slicing.

**Rust backend (`app/src-tauri/`)**
- `commands/tensor.rs` — `load_npz_plane_decompress` inflates deflated `.npz` entries and
  slices the trailing 2-D plane via `load_plane_from_npy_bytes`; `TensorSlicePreview.used_decompress_slice`
  flag; `probe_npy_mmap` reports large compressed entries; unit test
  `npz_decompress_plane_reads_trailing_2d_slice`

**React frontend**
- `TopologyCosmographView` — Cosmograph-style dense Sigma.js point renderer (no labels,
  `hideEdgesOnMove`, ForceAtlas2 strong-gravity settings)
- `TopologySigmaView` — Graphology ForceAtlas2 layout on force mode
- `GraphTopologyPanel` — ECharts / Sigma.js / Cosmograph view toggle
- `MLIntrospectionPanel` — decompress-slice timing badge

**ROADMAP**
- §G.4 Cosmograph WebGL + Graphology/ForceAtlas2 checked
- §G.5.1 compressed NPZ decompress slice checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-seventh pass (§G.4 / §G.5)

Forty-seventh pass closes remaining deferred **§G.5** infrastructure items and adds
**§G.4** Sigma.js WebGL topology rendering on Simulation Monitor.

**Rust backend (`app/src-tauri/`)**
- `commands/tensor.rs` — `load_npz_plane_mmap` reads trailing 2-D planes from stored
  `.npz` entries via zip `data_start` + `memmap2`; `probe_npy_mmap` covers `.npz`;
  unit test `npz_mmap_plane_reads_trailing_2d_slice`

**Python**
- `logic/gen/export_loss_landscape.py` — `--batch-size` (default 4) averages training
  forward-loss across multiple synthetic instances per grid point; `batch_size` bundled in NPZ

**React frontend**
- `TopologySigmaView` — Sigma.js WebGL k-NN topology graph with fill/pheromone styling
- `GraphTopologyPanel` — ECharts / Sigma.js view toggle
- `graphTopology.ts` — exported `topologyNodeStyle()` shared by both renderers
- `MLIntrospectionPanel` — mmap badge text covers `.npz` archives

**ROADMAP**
- §G.4 Sigma.js WebGL topology overlay checked (partial — Cosmograph deferred)
- §G.5.1 NPZ-in-zip mmap slice checked (partial — compressed entries deferred)
- §G.5.2 multi-batch training-loss probe checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-sixth pass (§G.5)

Forty-sixth pass closes the remaining **§G.5** deferred infrastructure items:
memory-mapped `.npy` slice loading, Sigma.js WebGL attention overlay, and a full
training-loss probe for loss landscape export.

**Rust backend (`app/src-tauri/`)**
- `commands/tensor.rs` — `load_npy_plane_mmap` via `memmap2` for standalone `.npy` > 8 MB;
  `TensorSlicePreview.used_memmap` flag; fixed NPY `descr` header parsing; unit test
  `mmap_plane_reads_trailing_2d_slice`

**Python**
- `logic/gen/export_loss_landscape.py` — `--probe-mode auto|training|proxy`; greedy
  forward-loss grid via `load_model` when hyperparameters are discoverable; `probe_mode`
  metadata bundled in NPZ

**React frontend**
- `AttentionSigmaView` — Sigma.js + Graphology ForceAtlas2 bipartite attention graph
- `MLIntrospectionPanel` — View toggle adds Sigma.js WebGL; mmap slice badge on timing row
- `vite.config.ts` — lazy `sigma` vendor chunk (`sigma`, `graphology`)

**Dependencies**
- `sigma`, `graphology`, `graphology-layout-forceatlas2` (frontend)
- `memmap2` (Rust)

**ROADMAP**
- §G.5.1 full mmap slice for large `.npy` checked (partial — NPZ-in-zip mmap deferred)
- §G.5.2 training-loss forward probe checked (partial — multi-batch deferred)
- §G.5.3 Sigma.js WebGL attention overlay checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-fifth pass (§G.5)

Forty-fifth pass closes the remaining **§G.5** infrastructure deferred items:
TensorDict (`.td`) inspect/slice, InstancedMesh loss voxels, and spherical k-means
attention clustering.

**Rust backend (`app/src-tauri/`)**
- `commands/tensor.rs` — `.td` inspect + 2-D slice via Python subprocess (`torch.load`);
  `project_root` + `python_executable` params on `inspect_npz_archive`, `load_tensor_slice`,
  `tensor_slice_to_arrow_ipc`

**Python**
- `logic/gen/export_for_studio.py` — includes `.td` TensorDict datasets in `.wsroute` bundles

**React frontend**
- `utils/sphericalKMeans.ts` — spherical k-means row clustering + cluster-band reorder
- `LossLandscape3D` — `InstancedMesh` voxel view; surface/voxels toggle
- `tensorHeatmap.ts` — cluster `markArea` bands on attention heatmaps
- `MLIntrospectionPanel` — `.td` file picker; K-means selector; loss 3D view toggle;
  project-root threaded into tensor commands
- `arrowPipeline.ts` — `runTensorArrowPipeline` passes `projectRoot` for `.td` ingest

**ROADMAP**
- §G.5.1 `.td` TensorDict inspect/slice + DuckDB ingest checked
- §G.5.2 InstancedMesh voxels checked
- §G.5.3 spherical k-means clustering checked (partial — Sigma.js WebGL deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-fourth pass (§G.5)

Forty-fourth pass closes remaining **§G.5** deferred items: DuckDB tensor ingest,
Q/K/V attention colour coding, and bipartite graph overlay on bin coordinates.

**React frontend**
- `utils/arrowPipeline.ts` — `runTensorArrowPipeline` (NPZ slice → Arrow IPC → DuckDB-Wasm `studio_tensor`)
- `utils/tensorHeatmap.ts` — `classifyAttentionRole`, `groupAttentionKeys`, per-role colour palettes (Query blue · Key green · Value amber)
- `utils/attentionGraph.ts` — `buildAttentionGraphOption` ECharts graph overlay; edge opacity ∝ weight; query node at decode step
- `MLIntrospectionPanel` — Archive tab "Ingest slice → DuckDB"; Attention tab Q/K/V filter, Heatmap/Graph view toggle, graph preset selector

**ROADMAP**
- §G.5.1 DuckDB tensor ingest checked (partial — `.td` TensorDict deferred)
- §G.5.3 Q/K/V colour coding + graph-on-coords overlay checked (partial — Sigma.js WebGL deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-third pass (§G.5)

Forty-third pass completes the remaining **§G.5 Machine Learning Introspection**
checklist items: BPC exact-solver landscape marker and Empirical vs Gamma-3
attention distribution compare.

**Rust backend (`app/src-tauri/`)**
- `commands/tensor.rs` — `load_npz_vectors` reads 0-D/1-D NPZ arrays (θ axes, BPC marker coords)

**Python**
- `logic/gen/export_loss_landscape.py` — bundles `bpc_theta1`, `bpc_theta2`, `bpc_loss`, and `distribution` metadata; `--bpc-theta1`/`--bpc-theta2`/`--distribution` CLI flags

**React frontend**
- `utils/lossLandscape.ts` — `resolveBpcMarker`, `thetaToGridCell`, `gridCellToTerrainPosition`
- `utils/distributionCompare.ts` — `inferDistributionLabel`, Empirical/Gamma-3 path heuristics
- `LossLandscape3D` — amber BPC octahedron marker on 3D topography
- `MLIntrospectionPanel` — BPC `markPoint` on 2D contour; "Empirical vs Gamma-3" dual-archive attention compare (side-by-side + overlay Δ)

**ROADMAP**
- §G.5.2 BPC optimum landscape marker checked
- §G.5.3 Empirical vs Gamma-3 attention compare checked (partial — Sigma.js, Q/K/V deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-second pass (§G.5)

Forty-second pass completes remaining **§G.5 Machine Learning Introspection** items:
React Three Fiber 3D loss topography, minima sharpness annotations, attention head
selector, sparse top-k, and decode-step compare modes.

**Dependencies**
- `three`, `@react-three/fiber`, `@react-three/drei` — lazy `r3f` vendor chunk in `vite.config.ts`

**React frontend**
- `LossLandscape3D` — vertex-coloured `PlaneGeometry` topography, cyan global-min marker, `OrbitControls`
- `utils/lossLandscape.ts` — `analyzeLossMinima`, `lossToColor`, `normalizeGrid`
- `MLIntrospectionPanel` — Loss tab 3D + 2D side-by-side grid; attention head selector; sparse top-k; side-by-side / overlay Δ compare
- `utils/tensorHeatmap.ts` — `detectHeadAxis`, `applySparseTopK`, `diffMatrices`

**ROADMAP**
- §G.5.2 R3F topography + colour gradient + orbit camera + minima annotation checked (partial — BPC marker deferred)
- §G.5.3 head selector + sparse top-k + side-by-side/overlay compare checked (partial — Sigma.js, Q/K/V, Empirical vs Gamma-3 deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — forty-first pass (§G.5)

Forty-first pass starts **§G.5 Machine Learning Introspection**: NPZ/NPY tensor
pipeline, attention heatmaps, and loss landscape contour.

**Rust backend (`app/src-tauri/`)**
- `commands/tensor.rs` — `inspect_npz_archive`, `load_tensor_slice`, `tensor_slice_to_arrow_ipc`, `probe_npy_mmap` via `ndarray-npy` + `zip`
- Downsampled 2D slice preview with leading-dimension index selection

**Python**
- `logic/gen/export_loss_landscape.py` — export `loss_grid` NPZ (demo Rosenbrock or checkpoint filter-normalized probe)

**React frontend**
- `MLIntrospectionPanel` — Archive / Attention / Loss tabs on Experiment Tracker
- `utils/tensorHeatmap.ts` — ECharts heatmap builder + attention key heuristics
- `ExperimentTracker` — embeds ML Introspection section (§G.5)

**ROADMAP**
- §G.5.1 NPZ inspect + slice + Arrow IPC checked (partial)
- §G.5.2 loss export script + ECharts contour checked (partial — R3F deferred)
- §G.5.3 attention heatmap + decode-step timeline checked (partial — Sigma.js deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — fortieth pass (§G.4)

Fortieth pass completes remaining **§G.4 topological graph analytics** items:
ACO pheromone trails, radial dense layout for large graphs, and day-synced timeline.

**React frontend**
- `utils/graphTopology.ts` — `accumulateTourPheromone()`, `radialDenseLayout()`, `resolveLayoutMode()`; pheromone-aware edge styling; tour edges injected when τ overlay active
- `GraphTopologyPanel` — ACO pheromone toggle + day timeline slider; layout mode (auto/force/radial dense); sync with day scrubber
- `SimulationMonitor` — passes `filteredEntries`, `displayDay`, `dayRange`, `onDaySelect` into topology panel

**ROADMAP**
- §G.4 ACO pheromone trails, Cosmograph-style radial dense layout (N≥200), timeline slider checked (partial — Sigma.js/Cosmograph WebGL, live solver τ deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-ninth pass (§G.4)

Thirty-ninth pass starts **§G.4 topological graph analytics** on Simulation Monitor.

**React frontend**
- `utils/graphTopology.ts` — distance-matrix CSV parser, k-NN edge list, Fruchterman-Reingold layout, ECharts graph option builder
- `GraphTopologyPanel` — collapsible topology view with k-NN selector, fill-% cross-filter, re-layout toggle
- `SimulationMonitor` — topology panel below route map; SQL panel day click → day scrubber; profit brush → topology hint
- `SqlQueryPanel` — optional `onDaySelect` / `onProfitRange` callbacks for §G.4 cross-filter

**ROADMAP**
- §G.4 distance matrix load, ECharts topology graph, force layout, fill/profit cross-filter, dynamic re-layout checked (partial — Sigma.js, ACO pheromone, Cosmograph deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-eighth pass (§G.1.4 / §G.6)

Thirty-eighth pass adds **pivot table OLAP**, **SQL cross-filtering**, and **output
portfolio batch loader** for multi-log parallel coordinates.

**React frontend**
- `utils/pivotTable.ts` — client-side pivot aggregation + heatmap option builder
- `utils/outputRunLogs.ts` — scan `assets/output` run folders for JSONL logs (cap 48)
- `PivotTablePanel` — row/column/value/agg selectors with ECharts heatmap
- `SqlQueryPanel` — pivot below auto-chart; row click sets global `policy` cross-filter
- `BenchmarkAnalysis` — "Load output portfolio" scans output dirs into portfolio parallel chart
- `OutputBrowser` — shared `findRunJsonl()` helper

**ROADMAP**
- §G.6 pivot table UI + cross-filter to Phase 1–2 charts checked (partial — drag wells deferred)
- §G.1.4 output portfolio batch loader checked (partial — full 480-log scan deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-seventh pass (§G.1.4 / §G.3.4 / §G.6)

Thirty-seventh pass adds **OrbitView Cartesian deck.gl**, **portfolio parallel coords**,
**SQL auto-chart**, and **Simulation Monitor DuckDB SQL** panel.

**React frontend**
- `utils/mapPositions.ts` — shared geographic vs circular abstract bin layout
- `utils/queryAutoChart.ts` — infer bar/line/scatter from query columns; build ECharts option
- `DeckRouteMap` — OrbitView 3D point cloud when no lat/lng (fill-scaled Z); Mercator tile map when geo present
- `SimulationMonitor` — deck.gl available without geo coords; Mercator/OrbitView mode labels; SQL panel on `monitor_sim`
- `SqlQueryPanel` — auto-chart suggestion below query results (§G.6)
- `BenchmarkAnalysis` — `BenchmarkPortfolioParallel` one polyline per loaded simulation log

**ROADMAP**
- §G.1.4 multi-log parallel coords checked (partial — 480-log batch deferred)
- §G.3.4 Mercator vs Cartesian/OrbitView toggle checked (partial)
- §G.6 auto-chart from SQL results checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-sixth pass (§G.3 / §G.6)

Thirty-sixth pass adds **multi-vehicle route rendering**, **DuckDB SQL explorer**,
**graph-split benchmark heatmaps**, and **§G.2 drill-down error bars**.

**React frontend**
- `utils/vehicleTours.ts` — split depot-delimited tours into per-vehicle segments (ColorBrewer palette)
- `DeckRouteMap` — distinct PathLayer/TripsLayer per vehicle; legend chips per vehicle
- `SimulationMonitor` `RouteMapChart` — multi-vehicle colored route lines on Cartesian map
- `components/analysis/SqlQueryPanel.tsx` — Monaco SQL editor + templates + sortable result grid + CSV export
- `utils/duckdbTemplates.ts` — robustness, variance, Pareto candidate query templates
- `DataExplorer` — DuckDB SQL panel when CSV ingested into Wasm worker
- `BenchmarkAnalysis` — graph-facet heatmaps (RM-100 / RM-170 / FFZ-350) with overflows/kg/km toggle
- `policyHierarchy.ts` — drill-down profit std + Empirical↔Gamma spread for error-bar whiskers
- `SimulationSummary` — drill-down bars show distribution variance when error bars enabled

**ROADMAP**
- §G.3.2 multi-vehicle rendering checked (partial — per-vehicle stop colors deferred)
- §G.1.3 graph-split heatmaps, §G.2 drill-down error bars checked (partial)
- §G.6 DuckDB query editor, templates, result grid + CSV export checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-fifth pass (§G.1 / §G.2)

Thirty-fifth pass adds **§G.2 hierarchical drill-down** on Simulation Summary and
closes more **§G.1** multi-run / distribution-split items on Benchmark Analysis.

**React frontend**
- `utils/policyHierarchy.ts` — build sunburst/treemap tree (city → strategy → constructor); breadcrumb path helpers
- `utils/paretoPanels.ts` — classify runs into Gamma-3/FTSP · Empirical/FTSP · Gamma-3/CLS · Empirical/CLS panels
- `utils/simMetadata.ts` — shared `strategyColor`, `citySymbol`, `cityScaleLabel` helpers
- `SimulationSummary` — `PolicyHierarchyPanel` sunburst/treemap toggle; drill-down bar chart + breadcrumb trail
- `SimulationSummary` — `DistributionFacetHeatmaps` splits heatmaps when multiple distributions present
- `SimulationSummary` — zero-overflow corridor slider cross-filters parallel coords + all brushed panels
- `SimulationSummary` — parallel polylines colored by selection strategy
- `BenchmarkAnalysis` — 4-panel Pareto grid from loaded runs; City Comparison log-scale bar chart (§G.1.6)

**ROADMAP**
- §G.2 sunburst/treemap, drill-down bars, breadcrumb checked (partial — DuckDB filter deferred)
- §G.1.2 four-panel Pareto, §G.1.3 distribution heatmap split, §G.1.4 strategy colors + overflow corridor,
  §G.1.6 city comparison log scale checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-fourth pass (§G.1)

Thirty-fourth pass implements **§G.1 Statistical Overview Dashboard** cross-filter
brushing, grouped KPI charts, parallel coordinates, and richer policy metadata tooltips.

**React frontend**
- `utils/simMetadata.ts` — parse log paths and policy labels into city/scale/distribution/strategy metadata
- `utils/chartHighlight.ts` — `isHighlighted`, `barOpacity`, `toggleBrush` for dashboard cross-filtering
- `SimulationSummary` — `ConfigMetaBanner` run-config strip; `PolicyBrushBar` chip cross-filter
- `GroupedMetricBarChart` — overflows by selection strategy; kg/km by constructor (mean ± std)
- `PolicyParallelChart` — ECharts parallel coordinates (profit · kg/km · overflows · km)
- `PolicyHeatmapChart` — metric mode toggle (all / overflows / kg/km); brush dimming
- `PolicyParetoChart` — strategy color + city/scale marker shape encoding; brush dimming
- `EfficiencyRankingChart` / `MetricBarChart` — bar opacity by brush; click-to-filter; rich tooltips
- §G.1.6 — auto log-scale duplicate row for profit and overflows when global log toggle is off

**ROADMAP**
- §G.1 grouped KPI bars, interactive brushing, parallel coords (partial), heatmap metric toggle,
  Pareto color/shape encoding, rich tooltips, secondary log-scale views checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-third pass (Phase 0)

Thirty-third pass completes **§G.0 Phase 0** foundation items deferred by later-phase
work: Arrow IPC serialization, DuckDB-Wasm worker, and end-to-end latency probe.

**Rust backend (`app/src-tauri/`)**
- `commands/arrow.rs` — CSV + simulation JSONL → Arrow IPC file; `read_binary_file` for zero-copy handoff
- Simulation Arrow schema: policy, sample_id, day, profit, km, overflows, kg, kg_per_km, cost, ncol, kg_lost
- `benchmark_arrow_pipeline` command for Rust-side timing

**React frontend**
- `@duckdb/duckdb-wasm` + `apache-arrow` dependencies; `duckdb` vendor chunk in Vite
- `duckdbClient.ts` — DuckDB-Wasm worker singleton; `insertArrowFromIPCStream` table registration
- `arrowPipeline.ts` — CSV/log → Rust → Arrow → DuckDB orchestration with 500 ms budget
- `useDuckDbInit` — spawns worker on app mount; startup timing milestone `duckdbReady`
- `Settings` — Phase 0 pipeline panel + "Run Arrow Pipeline Benchmark" button
- `DataExplorer` — auto-ingests opened CSV into DuckDB; shows row count + latency
- `SimulationMonitor` — auto-ingests opened simulation log into DuckDB

**ROADMAP**
- §G.0 Arrow IPC + DuckDB-Wasm worker + latency benchmark checked (Phase 0 complete)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-second pass

Thirty-second implementation pass: policy heatmap (§G.1); deck.gl 3D pitch;
AlgorithmComparison log scale; filtered CSV export.

**React frontend**
- `SimulationSummary` — policy × metric heatmap with normalised scores and PNG export
- `DeckRouteMap` — 3D pitch toggle (0°/45°); controlled pan/zoom view state
- `AlgorithmComparison` — log-scale toggle on per-metric bar charts
- `DataExplorer` — CSV export respects active filter/sort (exports visible subset)

**ROADMAP**
- §G.1 policy configuration heatmap checked (partial — multi-config/multi-city deferred)
- §G.3.1 deck.gl 3D pitch toggle checked (partial — OrbitView deferred)
- §G.1 AlgorithmComparison log-scale toggle noted
- §G.6 Data Explorer filtered CSV export noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-first pass

Thirty-first implementation pass: graph preset auto-detect (§G.3.1); symlog
overflows; Data Explorer filter; deck.gl fill-scaled nodes; benchmark log scale.

**React frontend**
- `utils/graphCoords.ts` — `guessGraphPreset()` infers RM-100/RM-170/FFZ-350 from log path or bin count
- `utils/symlog.ts` — symmetric log transform for near-zero overflow values
- `SimulationMonitor` — auto-selects graph preset on log load; shows "auto-detected" badge
- `SimulationSummary` — symlog overflows bar when log scale on; efficiency ranking error bars
- `DataExplorer` — row filter search across all columns with match count
- `DeckRouteMap` — tour-stop scatter radius scales with fill level
- `BenchmarkAnalysis` — log-scale toggle on multi-run comparison bar charts

**ROADMAP**
- §G.3.1 graph preset auto-detect from log path/bin count checked (partial)
- §G.1 symlog overflows bar + efficiency ranking error bars checked (partial)
- §G.6 Data Explorer row filter checked (partial)
- §G.16 deck.gl node radius ∝ fill level checked (partial — profit deferred)
- §G.1 BenchmarkAnalysis log-scale toggle noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirtieth pass

Thirtieth implementation pass: graph JSON coordinate loader (§G.3.1); Pareto
log-scale; BenchmarkAnalysis efficiency rank; Evaluation Runner charts (§G.12).

**React frontend**
- `utils/graphCoords.ts` — load RM-100/RM-170/FFZ-350 coordinates from graph JSON + area CSV via project root
- `SimulationMonitor` — graph preset selector + "Load graph coords" enriches logs for deck.gl tile map
- `SimulationSummary` — log-scale toggle applies to Pareto scatter y-axis (overflows)
- `BenchmarkAnalysis` — horizontal efficiency ranking chart (kg/km) with PNG export
- `EvaluationRunner` — inline cost/gap/time bar charts with PNG export on results grid
- `App.tsx` — maplibre-gl + @deck.gl/react included in startup prefetch batch

**ROADMAP**
- §G.3.1 graph JSON coordinate loader checked (partial — auto-detect from log metadata deferred)
- §G.1 Pareto log-scale y-axis checked (partial — true symlog deferred)
- §G.12 Evaluation Runner inline charts + PNG export checked
- §G.7 maplibre/deck.gl vendor prefetch noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-ninth pass

Twenty-ninth implementation pass: Pareto + efficiency charts (§G.1); Data
Explorer column sort; BenchmarkAnalysis kg/km metric.

**React frontend**
- `SimulationSummary` — horizontal efficiency ranking chart (kg/km); profit vs overflows Pareto scatter with dashed frontier
- `utils/pareto.ts` — Pareto front + step-line helpers for policy comparison
- `BenchmarkAnalysis` — `kg/km` added to simulation comparison metrics
- `DataExplorer` — sortable column headers (asc/desc toggle on click)

**ROADMAP**
- §G.1 Pareto front + horizontal kg/km ranking checked (partial — multi-config deferred)
- §G.6 Data Explorer column sort checked (partial)
- §G.7 BenchmarkAnalysis kg/km export line updated

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-eighth pass

Twenty-eighth implementation pass: launcher/monitor PNG exports (§G.7);
Simulation Summary radar + error bars (§G.1); Data Explorer CSV export.

**React frontend**
- `TrainingMonitor` — PNG export on multi-run overlay chart and grad-norm / LR sparklines
- `TrainingHub` — PNG export on live training chart and grad-norm / entropy sparklines
- `DataGeneration` — PNG export on dataset demand histogram preview
- `SimulationSummary` — policy radar chart; error-bar whiskers toggle on bar charts (linear scale)
- `DataExplorer` — Export CSV button for loaded table data
- `App.tsx` — `DeckRouteMap` chunk included in startup prefetch batch

**ROADMAP**
- §G.1 policy radar + error bars checked (partial — symlog/Pareto deferred)
- §G.7 TrainingMonitor / TrainingHub / DataGeneration PNG export checked
- §G.7 Data Explorer CSV export checked
- §G.10 / §G.11 / §G.17 launcher & monitor PNG export noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-seventh pass

Twenty-seventh implementation pass: SimulationSummary chart PNG export (§G.7);
deck.gl tile map PNG capture (§G.16); startup 2s budget indicator (§G.7).

**React frontend**
- `SimulationSummary` — PNG export on per-day trajectory chart and all four policy bar charts
- `DeckRouteMap` — PNG export button captures WebGL canvas via `exportCanvasPng`
- `chartExport.ts` — `exportCanvasPng` helper for deck.gl / canvas screenshots
- `Settings` — prefetch timing shows pass/fail against 2s load budget
- `useStartupTiming` — `withinBudget` flag derived from prefetch milestone
- `App.tsx` — echarts vendor chunk included in startup prefetch batch

**ROADMAP**
- §G.7 SimulationSummary + deck.gl PNG export + 2s budget probe checked (partial — hardware benchmark deferred)
- §G.3.1 ScatterplotLayer fill-coded nodes checked (partial)
- §G.16 deck.gl PNG export checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-sixth pass

Twenty-sixth implementation pass: global file drop (§G.8); extended startup
timing (§G.7); chart PNG exports; guided tour spotlights (§G.19).

**React frontend**
- `hooks/useGlobalFileDrop.ts` — app-wide `.wsroute` extract + `.jsonl` log open from OS file drop
- `utils/startupTiming.ts` — shared startup milestone marks; prefetch-complete timing in Settings
- `GuidedTour` — `data-tour` spotlight rings highlight sidebar, palette, and nav targets per step
- `OnboardingDialog` — auto-offers guided tour after first project-root configuration
- `BenchmarkAnalysis` — PNG export on simulation and eval comparison bar charts
- `AlgorithmComparison` — PNG export on per-metric bar charts (radar already supported)
- `App.tsx` — `Ctrl+Shift+/` opens guided tour; Escape dismisses tour overlay

**ROADMAP**
- §G.3.1 deck.gl + MapLibre integration checked (was implemented, now documented)
- §G.7 prefetch timing probe + BenchmarkAnalysis PNG export checked (partial — <2s target deferred)
- §G.8 global file drop checked
- §G.19 guided tour spotlight + auto-offer checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-fifth pass

Twenty-fifth implementation pass: `.wsroute` drag-drop (§G.14); map compare
deep link (§G.16); ECharts side-by-side split; guided tour (§G.19).

**React frontend**
- `hooks/useFileDrop.ts` — Tauri window `onDragDropEvent` listener for OS file drops
- `OutputBrowser` — drag-drop `.wsroute` onto file viewer; dashed overlay + manifest inspect
- `store/app.ts` — `pendingMapCompare` ephemeral state for Algorithm Comparison → map navigation
- `AlgorithmComparison` — "Compare on Map" sets policy filters + split layout when 2 policies
- `SimulationMonitor` — consumes `pendingMapCompare`; ECharts Cartesian side-by-side when split + 2 policies
- `GuidedTour` — 5-step studio walkthrough; TopBar compass, command palette, Settings entry
- `store/layout.ts` — `guidedTourOpen` / `guidedTourStep` / `guidedTourDismissed` persistence

**ROADMAP**
- §G.14 `.wsroute` drag-drop checked
- §G.16 ECharts side-by-side + map deep link checked
- §G.19 guided tour checked (partial — spotlight deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-fourth pass

Twenty-fourth implementation pass: side-by-side route map compare (§G.16);
Algorithm Comparison map link + PNG export; update check command; startup timing probe.

**Rust backend**
- `system::check_for_updates` — fetches optional `WSMART_UPDATE_URL` JSON manifest; compares `version` field to `CARGO_PKG_VERSION`
- `reqwest` dependency (rustls) for async update manifest fetch

**React frontend**
- `SimulationMonitor` — overlay/split layout toggle on deck.gl tile map when exactly 2 policies visible; split renders dual labelled `DeckRouteMap` panels
- `AlgorithmComparison` — "Compare on Map" navigates to Simulation Monitor; radar chart PNG export via `exportChartPng`
- `hooks/useStartupTiming.ts` — module-load → first-mount timing probe surfaced in Settings About
- `Settings` — startup timing display; "Check for Updates" button wired to `check_for_updates`

**ROADMAP**
- §G.16 side-by-side route compare checked (partial — ECharts Cartesian deferred)
- §G.3.3 algorithm comparison side-by-side map checked (partial)
- §G.7 startup timing probe + AlgorithmComparison PNG export checked (partial — <2s load target deferred)
- §G.8 `check_for_updates` checked (partial — Tauri updater plugin deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-third pass

Twenty-third implementation pass: multi-policy route map overlay (§G.16);
log-scale bar charts (§G.1 partial); app version command; recent-run navigation.

**Rust backend**
- `system::get_app_version` — returns `CARGO_PKG_VERSION` for Settings About panel

**React frontend**
- `DeckRouteMap` — refactored for multi-policy `routes[]` overlay with per-policy colour paths and legend
- `SimulationMonitor` — map policy visibility chips; overlays all policies for the selected day on tile map
- `SimulationSummary` — log-scale toggle on policy ranking bar charts (values clamped to 0.001 for log axis)
- `store/app.ts` — `pendingRunPath` for command-palette recent-run deep link
- `OutputBrowser` — auto-selects run when opened via `pendingRunPath`
- `Settings` — version loaded from Rust; notes auto-update requires release endpoint

**ROADMAP**
- §G.16 multi-policy map overlay + toggle visibility checked (partial — side-by-side deferred)
- §G.1 log-scale toggle on bar charts checked (partial — symlog/Pareto deferred)
- §G.8 `get_app_version` checked (partial — updater plugin deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-second pass

Twenty-second implementation pass: TripsLayer route trail animation and depot
marker (§G.16); recent-files quick open (§G.7 / §G.14); startup route prefetch.

**React frontend**
- `components/maps/DeckRouteMap.tsx` — `TripsLayer` trail animation during day playback; gold depot `ScatterplotLayer`; dimmed idle bins
- `store/recentFiles.ts` — persisted recent logs, runs, and CSVs (max 12)
- `CommandPalette` — Recent section for quick reopen; logs navigate to Simulation Summary
- `SimulationMonitor`, `SimulationSummary`, `OutputBrowser`, `DataExplorer` — track opened files/runs
- `App.tsx` — prefetch simulation, summary, process monitor, and output browser on startup

**Dependencies**
- `@deck.gl/geo-layers` — TripsLayer for animated route trails

**ROADMAP**
- §G.16 TripsLayer animation + depot marker checked (partial — multi-vehicle deferred)
- §G.7 recent files + startup prefetch checked (partial — <2s load target deferred)
- §G.14 recent run/file tracking checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-first pass

Twenty-first implementation pass: first-run onboarding wizard (§G.19);
simulation day playback controls (§G.16); Tauri bundler configuration (§G.8);
sidebar page prefetch and `.wsroute` import via command palette (§G.7).

**React frontend**
- `components/layout/OnboardingDialog.tsx` — welcome modal when `projectRoot` is unset; directory picker + `validate_project_root`; dismissible with persistence
- `pages/monitor/SimulationMonitor.tsx` — play/pause day playback with 1×/2×/4× speed multiplier on the day scrubber
- `utils/pagePrefetch.ts` — warms lazy route chunks on sidebar `mouseEnter`
- `hooks/useWsrouteImport.ts` — pick bundle → extract → navigate to Simulation Summary
- `constants/commands.ts` — "Import .wsroute Bundle" command palette action
- `package.json` — `tauri:dev`, `tauri:build`, `tauri:build:linux` scripts

**Tauri bundler**
- `tauri.conf.json` — explicit `deb`/`appimage`/`msi`/`dmg` targets; short/long description; Linux deb section; Windows NSIS install mode

**ROADMAP**
- §G.19 first-run onboarding wizard checked (partial — guided tour deferred)
- §G.16 day playback controls checked (partial — TripsLayer animation deferred)
- §G.8 Tauri bundler config + build scripts checked (partial — signing/auto-update deferred)
- §G.7 sidebar prefetch + palette bundle import checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — twentieth pass

Twentieth implementation pass: command palette (§G.7); Parquet table export;
bundle round-trip integration test; Vite manual chunk splitting for faster
initial load.

**Rust backend**
- `data::export_csv_to_parquet` — converts an on-disk CSV to Parquet via pandas/pyarrow subprocess
- `data::export_table_parquet` — writes in-memory tabular data to Parquet (temp CSV → convert)
- `wsroute_bundle_round_trip_preserves_jsonl` unit test — create → extract → verify `.jsonl` content

**React frontend**
- `components/layout/CommandPalette.tsx` — fuzzy-search overlay for all 17 views + theme/shortcuts actions; `Ctrl+K` or TopBar search button
- `constants/commands.ts` — shared palette command registry
- `utils/tableExport.ts` — `downloadParquetFromCsv()` and `downloadParquetTable()` helpers
- `DataExplorer`, `OutputBrowser`, `SimulationSummary` — Parquet export buttons alongside CSV
- `vite.config.ts` — `manualChunks` for echarts, maplibre, deck.gl, monaco vendor bundles

**ROADMAP**
- §G.7 command palette and Parquet export checked; manual chunk splitting noted (partial — <2s load target deferred)
- §G.8 bundle round-trip integration test checked (partial — Tauri bundler/updater deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — nineteenth pass

Nineteenth implementation pass: `.wsroute` bundle create/extract (§G.8); deck.gl
tile route map (§G.16); keyboard shortcuts help overlay; lazy-loaded pages and
SVG chart export (§G.7).

**Rust backend**
- `data::create_wsroute_bundle(source_dir, output_path)` — walks a run directory, zips eligible artefacts plus `manifest.json`
- `data::extract_wsroute_bundle(path, dest_dir)` — decompresses a `.wsroute` zip; returns first `.jsonl` log path for Simulation Summary

**React frontend**
- `components/maps/DeckRouteMap.tsx` — deck.gl `PathLayer` + `ScatterplotLayer` over MapLibre dark basemap; lazy-loaded from SimulationMonitor
- `components/layout/KeyboardShortcutsHelp.tsx` — modal overlay listing all global shortcuts; opened via `?` or TopBar button
- `App.tsx` — all page components lazy-loaded behind `Suspense`; `?` opens shortcuts help; `Escape` dismisses
- `pages/files/OutputBrowser.tsx` — "Export as .wsroute" on selected run; "Extract & Open" on bundle files
- `pages/monitor/SimulationMonitor.tsx` — ECharts / deck.gl route map toggle; SVG export on Cartesian map
- `utils/chartExport.ts` — `exportChartSvg()` for ECharts SVG download
- `vite.config.ts` — build target bumped to `es2022` for deck.gl BigInt literals

**Dependencies**
- `@deck.gl/core`, `@deck.gl/layers`, `@deck.gl/react`, `maplibre-gl`, `react-map-gl`

**ROADMAP**
- §G.8 bundle create/extract commands + Output Browser UI checked (partial — Tauri bundler/updater deferred)
- §G.16 deck.gl `PathLayer` tile route map checked
- §G.7 lazy-loaded pages, shortcuts help overlay, SVG export checked (partial — Parquet/command palette deferred)

### Fixed

#### WSmart-Route Studio — Tauri build

- Removed unused `protocol-asset` feature from `Cargo.toml` (mismatched Tauri allowlist)
- Corrected capability permissions to `core:*` identifiers for Tauri 2 ACL
- Added placeholder RGBA app icons required by `generate_context!()`

### Added

#### WSmart-Route Studio — Tauri App (`app/`) — eighteenth pass

Eighteenth implementation pass: analytical workflow navigation strip and
collapsible sidebar (§G.7); `P`/`M` keyboard shortcuts; `GlobalFilterBar`
propagated to Benchmark Analysis; MLflow dashboard iframe embed (§G.18);
`.wsroute` bundle export script and inspector (§G.8 partial).

**Python**
- `logic/gen/export_for_studio.py` — packages run output artefacts (CSV, JSON/JSONL, YAML, NPZ, PKL, PT, Parquet) into a `.wsroute` zip with `manifest.json`

**Rust backend**
- `data::inspect_wsroute_bundle(path)` — lists zip contents and parses bundle manifest
- `zip` crate dependency for bundle inspection

**React frontend**
- `components/layout/WorkflowNav.tsx` — Overview → Drill-Down → Geospatial → Registry → ML → HPO → Launch strip (§G.7)
- `components/layout/GlobalFilterBar.tsx` — shared policy/sample filter controls
- `store/layout.ts` — `sidebarOpen` state with persistence; TopBar toggle + mobile overlay backdrop
- `App.tsx` — `P` → Process Monitor, `M` → Simulation Digital Twin
- `pages/analysis/BenchmarkAnalysis.tsx` — global filter propagation + comparison CSV export
- `pages/analysis/ExperimentTracker.tsx` — MLflow Runs/Dashboard tabs; iframe embed + open-in-browser
- `pages/files/OutputBrowser.tsx` — `.wsroute` bundle manifest viewer
- `SimulationSummary`, `AlgorithmComparison` — `GlobalFilterBar` integration

**ROADMAP**
- §G.7 workflow nav, P/M shortcuts, sidebar collapse (partial), global filters to Benchmark checked
- §G.18 MLflow iframe embed fallback checked
- §G.8 export script + bundle inspector checked (partial — full import deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventeenth pass

Seventeenth implementation pass: Monaco YAML editor in Config Editor (§G.13);
ZenML pipeline run browser with step-duration Gantt chart (§G.18); table CSV
export utility; global filters in Simulation Summary; HPO chart PNG export;
responsive layout container (§G.7).

**Rust backend**
- `commands/zenml.rs` — `list_zenml_pipeline_runs`, `load_zenml_run_steps`: Python subprocess queries ZenML via `Client.list_pipeline_runs` and `get_pipeline_run`

**React frontend**
- `components/editors/YamlEditor.tsx` — Monaco YAML editor (lazy-loaded) with dark/light theme sync; replaces raw textarea in ConfigEditor
- `pages/analysis/ZenMLPipelineView.tsx` — pipeline run table, step-duration horizontal bar chart (Gantt-style), CSV/PNG export
- `pages/analysis/ExperimentTracker.tsx` — embeds ZenML section; MLflow runs CSV export
- `pages/analysis/SimulationSummary.tsx` — respects `useGlobalFiltersStore`; ranking table CSV export; active filter badge
- `pages/analysis/HPOTracker.tsx` — PNG export buttons on all four ECharts panels
- `utils/tableExport.ts` — reusable `downloadCsv()` for table data export
- `components/layout/Layout.tsx` — max-width container (`1920px`) and responsive padding
- `types/index.ts` — `ZenmlPipelineRun`, `ZenmlPipelineStep` interfaces
- `package.json` — `@monaco-editor/react` dependency

**ROADMAP**
- §G.13 Monaco Editor integration checked
- §G.18 ZenML pipeline view checked
- §G.7 table CSV export (partial), responsive layout (partial), theme toggle noted done

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixteenth pass

Sixteenth implementation pass: MLflow run browser and metric comparison (§G.18);
ECharts route map preview (§G.16 partial); global filter store, URL hash
bookmarking, chart PNG export, and `Ctrl+R` launch shortcut (§G.7).

**Rust backend**
- `commands/mlflow.rs` — `list_mlflow_runs`, `list_mlflow_metric_keys`, `load_mlflow_metric_history`: Python subprocess queries local/remote MLflow tracking via `mlflow.search_runs` and `MlflowClient`

**React frontend**
- `pages/analysis/ExperimentTracker.tsx` — MLflow run table with multi-select; metric comparison ECharts chart with normalize toggle; params panel; output dirs retained
- `pages/monitor/SimulationMonitor.tsx` — `RouteMapChart` ECharts scatter + path using `all_bin_coords` + `tour_indices`; fill-level colour coding; PNG export on charts
- `store/filters.ts` — `useGlobalFiltersStore` (policy + sampleId) propagates across SimulationMonitor and AlgorithmComparison
- `store/launchTrigger.ts` — nonce-based launch triggers for `Ctrl+R` on launcher pages
- `hooks/useHashSync.ts` — serializes `mode` + filters to URL hash for deep-linking
- `utils/chartExport.ts` — reusable `exportChartPng()` via ECharts `getDataURL()`
- `App.tsx` — `Ctrl+R` launches on active launcher page; `useHashSync()` on mount
- Launcher pages (`SimulationLauncher`, `TrainingHub`, `DataGeneration`, `EvaluationRunner`) — subscribe to launch trigger nonces
- `types/index.ts` — `MlflowRun`, `MlflowMetricPoint` interfaces

**ROADMAP**
- §G.18 MLflow run table and metric comparison chart checked
- §G.16 ECharts route map preview checked (deck.gl tile basemap still open)
- §G.7 global filters, URL hash bookmarking, chart PNG export, `Ctrl+R` checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifteenth pass

Fifteenth implementation pass: session profiles in Output Browser (§G.14);
sensor CSV data source and dataset preview panel (§G.11); Config Editor Form
mode (§G.13); Optuna cross-study comparison (§G.18).

**Rust backend**
- `data::preview_dataset_stats(path, project_root, python_executable)`: Python subprocess inspects `.pkl`/`.pt` datasets; returns `DatasetPreviewStats` (instances, nodes, demand μ±σ, histogram, file size)

**React frontend**
- `store/sessionProfiles.ts` — `useSessionProfilesStore` (persist, max 20 profiles); captures/restores all three launcher Zustand stores via `captureLauncherSnapshot()` / `applyLauncherSnapshot()`
- `pages/files/OutputBrowser.tsx` — Session Profiles sidebar (§G.14): name input + Save button; load/delete profile list
- `pages/launch/DataGeneration.tsx` — sensor source (§G.11): third `dataSource` radio; CSV file picker; Hydra `data.source=sensor` + `data.sensor_file=<path>`
- `pages/launch/DataGeneration.tsx` — Instance Preview panel (§G.11): "Preview .pkl/.pt" button; KPI cards + ECharts demand histogram via `preview_dataset_stats`
- `pages/files/ConfigEditor.tsx` — Form mode (§G.13): fourth view toggle; typed widgets (checkbox/number/text) inferred from value; edits sync back to Raw YAML via `rowsToYaml()`
- `pages/analysis/HPOTracker.tsx` — cross-study comparison (§G.18): "Compare with" study dropdown; overlaid best-so-far line chart; side-by-side best-value KPI cards
- `store/launchers.ts` — `sensorCsvPath` field in `useDataGenStore`
- `types/index.ts` — `DatasetPreviewStats` interface

**ROADMAP**
- §G.14 session profiles checked
- §G.11 sensor source and preview panel checked
- §G.13 Form mode checked (partial — flat YAML, no OmegaConf schema introspection)
- §G.18 cross-study comparison checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fourteenth pass

Fourteenth implementation pass: Output Browser compare-runs multi-select (§G.14);
structured directory tree with hydra/ auto-expand (§G.14); Config Editor Apply to
Launcher (§G.13); Data Generation TSPLIB source option (§G.11).

**React frontend**
- `pages/files/OutputBrowser.tsx` — compare runs (§G.14) + structured tree (§G.14):
  - Per-run checkbox multi-select; "Compare N Runs →" button when ≥2 selected
  - `findRunJsonl()` scans top-level and `hydra/` for `.jsonl` logs
  - `setPendingBenchmarkLogs` + navigate to `benchmark` mode
  - Auto-expand `hydra/` on run selection; `sortEntries()` prioritises config and log files
  - Highlight `pruned_config.yaml` / `.jsonl` entries in the file tree
- `pages/analysis/BenchmarkAnalysis.tsx` — consumes `pendingBenchmarkLogs` on mount; loads multiple simulation logs for side-by-side comparison
- `pages/files/ConfigEditor.tsx` — Apply to Launcher (§G.13):
  - Target selector (Simulation Launcher / Training Hub / Data Generation)
  - `applyConfigToLauncher()` maps flat YAML keys to Zustand store patches; navigates to target page
- `utils/configToLauncher.ts` — key-mapping utility for sim/train/data-gen Hydra fields + unmapped keys → `extraOverrides`
- `pages/launch/DataGeneration.tsx` — TSPLIB source option (§G.11):
  - `dataSource` radio: synthetic vs TSPLIB; `.vrp`/`.tsp` file picker via Tauri dialog
  - Hydra overrides `data.source=tsplib` + `data.tsplib_instance=<path>`; graph form hidden for TSPLIB mode
- `store/launchers.ts` — `dataSource` + `tsplibPath` persisted in `useDataGenStore`
- `store/app.ts` — `pendingBenchmarkLogs: BenchmarkLogRef[] | null` ephemeral handoff field

**ROADMAP**
- §G.14 compare runs and structured directory tree checked
- §G.13 Apply to Launcher checked
- §G.11 TSPLIB source option checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirteenth pass

Thirteenth implementation pass: dynamic policy registry in SimulationLauncher (§G.9);
Eval Runner → Benchmark Analysis handoff (§G.12); resolved Hydra config dump in ConfigEditor
(§G.13); Optuna study browser in HPOTracker (§G.18); Tauri OS notifications and Ctrl+. cancel
(§D.8/§G.7).

**Rust backend**
- `commands/policies.rs` — `list_sim_policies(project_root)`: parses `logic/configs/tasks/test_sim.yaml` for `/policies@p.{id}:` entries; returns sorted `SimPolicyEntry` list; falls back to 8 default policies when file is missing
- `commands/hpo.rs` — Optuna integration via Python subprocess:
  - `list_optuna_studies(storage_url, project_root, python_executable)`: enumerates studies with trial counts and best values
  - `load_optuna_study(storage_url, study_name, project_root, python_executable)`: returns trials, FANOVA importances, best value, and best params as JSON
- `commands/system.rs` — `dump_hydra_config(task, project_root, python_executable)`: runs `python main.py <task> --cfg job` and returns resolved YAML
- `commands/process.rs` — `resolve_python()` extracted as public helper shared by spawn, HPO, and Hydra commands

**React frontend**
- `pages/launch/SimulationLauncher.tsx` — dynamic policy registry (§G.9):
  - `availablePolicies` state loaded via `list_sim_policies` on `projectRoot` change
  - Scrollable checkbox grid (89 policies from `test_sim.yaml`); reload button with `RefreshCw` spinner
  - Stale selections pruned when registry reloads; count badge in header
- `pages/launch/EvaluationRunner.tsx` — "Open in Analytics →" button in `ResultsGrid` (§G.12):
  - Serialises result rows to `pendingEvalResults` in app store; navigates to `benchmark` mode
- `pages/analysis/BenchmarkAnalysis.tsx` — eval results panel (§G.12):
  - `EvalResultsPanel` component: 3-column bar charts (cost / gap / time) + summary table
  - Consumes `pendingEvalResults` on mount via `useEffect`; dismissible independently of simulation runs
- `pages/files/ConfigEditor.tsx` — resolved Hydra config loader (§G.13):
  - Task selector (test_sim / train / hpo / eval / gen_data) + "Load via --cfg job" button
  - Calls `dump_hydra_config`; populates Raw view without requiring a file on disk
- `pages/analysis/HPOTracker.tsx` — Optuna study browser rewrite (§G.18):
  - Storage URL input with SQLite file picker; study dropdown with trial counts
  - ECharts: optimisation history scatter + best-so-far line; FANOVA parameter importance bars; parallel coordinates
  - KPI cards (trials / completed / best value / param count); "Copy best params" as Hydra overrides
- `hooks/useProcessMonitor.ts` — OS notifications (§D.8) + cancel shortcut (§D.7):
  - `maybeSendOsNotification()`: requests permission and fires native notification when `document.hidden` on completed/failed
  - Global `Ctrl+.` listener cancels first running process via `cancel_process`
- `store/app.ts` — `pendingEvalResults: EvalAnalyticsRow[] | null` + `setPendingEvalResults` (ephemeral)
- `types/index.ts` — `SimPolicyEntry`, `EvalAnalyticsRow`, `OptunaStudySummary`, `OptunaTrial`, `OptunaStudyData`
- `App.tsx` — additional keyboard shortcuts: `G` → simulation monitor, `Q` → HPO tracker

**ROADMAP**
- §G.9 policy registry loading checked
- §G.12 Open in Analytics checked
- §G.13 Load resolved Hydra config checked
- §G.18 Optuna study browser (partial — history, importance, parallel coords, copy best params) checked
- §D.8 OS notifications checked; §D.7 Ctrl+. cancel checked

---

#### Analysis script & report — Pareto-front policy catalogue

- `logic/gen/gen_simulation_analysis.py` — new `build_pareto_front_table(df)` function:
  - Computes the Pareto front (min overflows, max kg/km) independently for each `(dist, improver)` panel
  - Groups front members by unique `(selection variant, constructor, improver)` key; merges `cf`/`sl_var` into a human-readable label (`LM (CF70)`, `SL (SL1)`, …)
  - Outputs a markdown table with columns: Selection | Constructor | Improver | Overflows | kg/km | Pareto-Front Scenarios
  - Scenarios column lists every `Region-N / Distribution` combination where that configuration reached the front; sorted descending by scenario count
  - Wired into `generate_markdown` at the end of section 2 (Analytics Comparison — Pareto View)
- `public/simulation_analysis.md` — "Pareto-Front Policy Catalogue" table inserted at the end of §2 (22 rows; BPC + ACO_HH + PG-CLNS dominate the front across all panels)

#### WSmart-Route Studio — Tauri App (`app/`) — twelfth pass

Twelfth implementation pass: live training mode in TrainingMonitor (§G.17); Lightning column
normalization in TrainingMonitor and TrainingHub (§G.17 parity); §G.16 Streamlit parity confirmed.

**React frontend**
- `pages/monitor/TrainingMonitor.tsx` — live training mode (§G.17) + column normalization:
  - `LIVE_KEY = "__live__"` constant: virtual run key for the live process entry in `metricsMap`
  - `normalizeMetricRow(raw)`: maps Lightning CSV column aliases to canonical `TrainingMetricsRow` keys — `train/rl_loss` / `train/il_loss` → `train_loss`; `val/cost` / `val_cost` → `val_loss`; `lr-*` prefix variants → `lr`; applied at both CSV load time and live stdout parse time
  - `parseMetricLine(line)` extended with `/`-containing key patterns (`\w[\w/]*`) to handle Lightning's `/`-separated metric names in key=value format
  - `METRIC_SIGNAL_KEYS` extended with Lightning variants: `train/rl_loss`, `train/il_loss`, `val/cost`, `val_cost`
  - `activeTrainId`: `useMemo` over `useProcessStore` — first `train_*` process with `status === "running"`
  - Live stdout `useEffect`: when `activeTrainId` is set, initializes `metricsMap[LIVE_KEY] = []` and attaches a `process:stdout` listener that calls `parseMetricLine` and appends parsed rows; cleans up on `activeTrainId` change
  - Auto-select `useEffect`: prepends `LIVE_KEY` to `selected` when `activeTrainId` appears; removes it when process exits
  - `runsMetrics` memo: live entry inserted first with `name: "Live Training"`
  - Live entry in run selector: `Radio` icon with `animate-pulse`; update count shown; checkbox to toggle manually
  - Live `RunPanel`-style block: green pulsing dot header + `GradNormSparkline` + `LrSparkline` for the live row set
  - CSV loading now applies `normalizeMetricRow` via `rows.map(normalizeMetricRow)` in `loadMetrics`
- `pages/launch/TrainingHub.tsx` — column normalization sync:
  - `METRIC_SIGNAL_KEYS` extended with Lightning column variants (same set as `TrainingMonitor.tsx`)
  - `normalizeMetricRow()` added (identical implementation); applied inside `parseMetricLine` for both JSON and key=value code paths
  - key=value regex updated to `(\w[\w/]*)` to capture `/`-separated metric names

**ROADMAP**
- §G.16 Streamlit parity check confirmed and checked
- §G.17 live training mode checked
- §G.17 column normalization checked
- §G.17 Streamlit parity check checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — eleventh pass

Eleventh implementation pass: session persistence for all three launcher forms (§G.9/G.10/G.11);
auto-navigate countdown in SimulationLauncher (§G.9); grad_norm + entropy sparklines in
TrainingHub (§G.10).

**React frontend**
- `store/launchers.ts` — new file; three persisted Zustand stores using a single `patch` action:
  - `useSimLauncherStore` (`wsroute-sim-launcher`): `selectedPolicies`, `area`, `numLoc`, `samples`, `nCores`, `seed`, `distribution`, `extraOverrides`
  - `useTrainHubStore` (`wsroute-train-hub`): all train/hpo/eval form fields
  - `useDataGenStore` (`wsroute-data-gen`): `problem`, `distributions`, `datasetType`, `seed`, `overwrite`, `area`, `numLoc`, `nSamples`, `nDays`, `extraOverrides`
- `pages/launch/SimulationLauncher.tsx` — session persistence (§G.9) + auto-navigate (§G.9):
  - Local `useState` for all form fields replaced with `useSimLauncherStore`
  - `navCountdown: number | null` state; first `useEffect` sets it to 5 when `simStatus === "completed"`; second `useEffect` decrements every second via `setTimeout` and calls `setMode("simulation_summary")` on 0
  - Countdown label `"(auto in Xs — cancel)"` shown beside "View Summary →" button; cancel clears countdown
- `pages/launch/TrainingHub.tsx` — session persistence (§G.10) + sparklines (§G.10):
  - Local `useState` for all form fields replaced with `useTrainHubStore`
  - `MiniSparkline` component: compact 70 px ECharts `line` chart; area fill at `color + "22"` opacity; returns `null` when all data values are null (metric not emitted by the run)
  - Grad norm sparkline (red `#f87171`) + entropy sparkline (purple `#a78bfa`) rendered as a 2-column grid below `LiveChart` when ≥2 metric updates have been received
- `pages/launch/DataGeneration.tsx` — session persistence (§G.11):
  - Local `useState` for all form fields replaced with `useDataGenStore`
  - `toggleDist` rewritten to avoid functional updater (incompatible with store `patch` signature)

**ROADMAP**
- §G.9 auto-navigate and session persistence checked
- §G.10 grad_norm + entropy sparklines and session persistence checked
- §G.11 session persistence checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — tenth pass

Tenth implementation pass: process toast notifications (§G.15); per-process progress bar (§G.15);
policy multi-select overlay on KPI timeseries (§G.16).

**React frontend**
- `hooks/useProcessMonitor.ts` — toast notifications (§G.15):
  - `import { toast } from "sonner"` added
  - `StatusUpdate` listener fires `toast.success` (4 s) / `toast.error` (6 s) / `toast.info` (3 s) on terminal status transitions; human-readable label extracted via `id.split("_")[0]`
- `pages/monitor/ProcessMonitor.tsx` — progress bar (§G.15):
  - `PROGRESS_MARKER = "PROGRESS:"` constant + `ProgressInfo` interface added
  - `getLatestProgress(logLines)` scans last 30 log lines for `PROGRESS:{json}` markers; returns `{ value, total?, label? }`; accepts both `value` and `current` keys
  - Progress bar rendered in `ProcessRow` between header row and log viewer when process is running and progress data is present; deterministic `width: pct%` bar when `total` is known, indeterminate pulsing bar otherwise
- `pages/monitor/SimulationMonitor.tsx` — policy multi-select overlay (§G.16):
  - `POLICY_COLORS` 8-colour palette (`#6366f1`, `#34d399`, `#f87171`, …) defined at module level
  - `MetricTimeseries` refactored: replaces `entries` + implicit single series with `policySeries: { policy; entries; color }[]`; builds one ECharts line series per policy; shows legend when >1 series; top grid margin increases to 20 when legend is visible; area fill only when single series
  - `chartPolicies: string[]` state + `activeChartPolicies` memo (defaults to all policies when `chartPolicies` is empty)
  - `toggleChartPolicy(p)` callback: XOR toggle; prevents deselecting all (resets to full set)
  - `policySeries` memo: maps each `activeChartPolicy` to filtered entries + assigned color
  - Chip-toggle row rendered below header controls when ≥2 policies present; chip border/text/background tinted with policy color; inactive chips at 35% opacity

**ROADMAP**
- §G.15 progress bar checked; cancel button confirmed already wired (no code change); toast notifications checked
- §G.16 policy/sample multi-select checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninth pass

Ninth implementation pass: DataGeneration live progress panel (§G.11); OutputBrowser simulation
KPI summary card on run selection (§G.14).

**React frontend**
- `pages/launch/DataGeneration.tsx` — live progress panel (§G.11):
  - `liveProcessId`, `runStatus`, `logTail` state added
  - `useEffect([liveProcessId])`: subscribes to `process:stdout` (appends last 20 non-empty lines to `logTail`) and `process:status` (updates `runStatus`)
  - `launch` now generates a stable process ID and clears state before spawn
  - Live panel renders below the Launch button: `Activity`/`CheckCircle`/`XCircle` status icon; scrollable pre-block with last 20 stdout lines; "Process Monitor" navigation button
- `pages/files/OutputBrowser.tsx` — simulation KPI summary (§G.14):
  - `runKpi` state: `Array<{ policy, overflows, kgkm, profit }> | null`
  - `selectRun` now scans top-level entries for the first `.jsonl` ≤ 20 MB; reads via `read_text_file`; parses each line as `DayLogEntry`; aggregates per-policy means; sorted ascending by overflows
  - KPI card rendered below the config metadata card: 3-column micro-table (Policy | Overflows | kg/km); overflows colour-coded (green = 0, amber = low, red > 20)

**ROADMAP**
- §G.11 live progress checked; §G.14 simulation result summary checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — eighth pass

Eighth implementation pass: LR schedule sparkline in Training Monitor (§G.17); completion
navigation in Training Hub (§G.10); eval results grid with CSV export in Evaluation Runner (§G.12).

**React frontend**
- `pages/monitor/TrainingMonitor.tsx` — LR schedule sparkline (§G.17):
  - Refactored `GradNormSparkline` and new `LrSparkline` to share a `MetricSparkline` base component (avoids duplication; same ECharts config parameterised by `label`, `data`, `color`)
  - `LrSparkline` plots `lr` vs `step` in amber (`#fbbf24`); shown per selected run below the grad-norm sparkline
- `pages/launch/TrainingHub.tsx` — completion navigation (§G.10):
  - "Output Browser →" button appears in live progress header when `runStatus === "completed"`; navigates to `output_browser` mode so users can inspect checkpoints immediately after training
- `pages/launch/EvaluationRunner.tsx` — results grid (§G.12):
  - `EvalResult` interface; `EVAL_RESULT_KEYS` sentinel list (`cost`, `gap`, `tour_cost`, `obj`, `time`, `policy`, `checkpoint`)
  - `processToCheckpoint` ref: maps process ID → checkpoint filename; populated at launch, used by the global `process:stdout` listener to attribute result rows
  - `ResultsGrid` component: dynamic columns from first result; numeric values formatted to 4 dp; updates live as rows arrive; replaces static placeholder card
  - "Export CSV" button in `ResultsGrid`: builds CSV string from all result rows, triggers `<a>` download via `Blob` + `URL.createObjectURL`

**ROADMAP**
- `docs/moon/ROADMAP.md` — §G.17 LR sparkline checked; §G.10 completion navigation checked; §G.12 results grid + CSV export checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventh pass

Seventh implementation pass: TrainingHub live progress chart (§G.10); OutputBrowser run
metadata panel + Sim Summary handoff (§G.14); Settings import/export JSON (§G.19); global
keyboard shortcuts (§G.7); pages directory reorganised into five subdirectories.

**React frontend**
- `pages/launch/TrainingHub.tsx` — live progress panel (§G.10):
  - `parseMetricLine`: tries JSON parse first; falls back to `key=value` scanning; detects rows with `train_loss`, `val_loss`, `reward`, `grad_norm`, `epoch`, or `step` keys
  - `LiveChart` component: ECharts canvas with train_loss (solid indigo), val_loss (dashed green), reward (dotted amber, right y-axis); shown once ≥ 2 metric rows received
  - Live snapshot row: epoch / train_loss / val_loss / reward / ‖∇‖ inline below chart
  - "Process Monitor" navigation button; `CheckCircle`/`XCircle` status header on completion
- `pages/files/OutputBrowser.tsx` — enhancements (§G.14):
  - Run metadata card: on `selectRun` auto-loads `pruned_config.yaml` / `config.yaml`; parses flat key-value pairs filtered by `META_KEYS` (task, seed, envs, area, policies, …); shown below the file tree as a compact two-column card
  - "Open in Sim Summary" button: shown for `.jsonl` files after loading; sets `store.pendingLogPath` + navigates to `simulation_summary`
- `pages/analysis/SimulationSummary.tsx` — consumes `pendingLogPath` via `useEffect` on mount; calls `loadLog` (extracted from button handler) and clears the store field
- `pages/app/Settings.tsx` — Backup & Restore card (§G.19):
  - "Export Settings": opens `save` dialog, serialises `{projectRoot, pythonPath, theme}` to JSON via `write_text_file`
  - "Import Settings": opens file picker, parses JSON, populates draft fields for review before saving
- `store/app.ts` — `pendingLogPath: string | null` + `setPendingLogPath` action (ephemeral, not persisted)
- `App.tsx` — global keyboard shortcuts (§G.7):
  - `Ctrl+,` → `settings`; `Ctrl+Shift+P` → `process_monitor`
  - Digit `1`–`8` (when no input focused): quick-switch to simulation / simulation_summary / training / benchmark / sim_launcher / training_hub / process_monitor / settings

**Project structure**
- `app/src/pages/` reorganised into five subdirectories mirroring sidebar sections:
  - `monitor/` — SimulationMonitor, TrainingMonitor, ProcessMonitor
  - `analysis/` — SimulationSummary, BenchmarkAnalysis, DataExplorer, ExperimentTracker, AlgorithmComparison, HPOTracker
  - `launch/` — SimulationLauncher, TrainingHub, DataGeneration, EvaluationRunner
  - `files/` — ConfigEditor, OutputBrowser
  - `app/` — Settings
- All intra-page imports updated from `../` to `../../`; `App.tsx` import paths updated to `pages/<subdir>/`

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.10 live training progress checked; §G.14 metadata panel and Open in Sim Summary checked; §G.19 import/export checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixth pass

Sixth implementation pass: SimulationLauncher gains a live-status panel (§G.9); ConfigEditor gains a
Save button (§G.13); ProcessMonitor gains structured JSON log rendering and persistent history (§G.15);
SimulationSummary is rewritten with a ranking table, per-day trajectory chart, and four metric charts.

**Rust backend**
- `commands/data.rs` — `write_text_file(path, content)`: writes (or overwrites) any text file; creates parent directories; used by ConfigEditor Save button; registered in `lib.rs`

**React frontend**
- `pages/SimulationLauncher.tsx` — live-status panel (§G.9):
  - After launch, subscribes to `process:stdout` Tauri events filtered by the spawned process ID
  - Parses `GUI_DAY_LOG_START:` markers (same protocol as `sim_watcher.rs`) to extract `DayLogEntry` JSON
  - Displays a per-policy card grid with latest day / profit / km / overflows in real time
  - Status header: animated `Activity` icon while running; `CheckCircle`/`XCircle` on completion
  - "View Summary →" button navigates to `simulation_summary` mode; "Process Monitor" button to `process_monitor`
- `pages/ConfigEditor.tsx` — Save button (§G.13):
  - Calls `write_text_file` Tauri command with the currently open path and textarea content
  - Tracks dirty state via `savedContentRef` (updates on open + save); button label shows `Save*` when unsaved edits exist; disabled when no changes
  - `Save` icon from lucide-react; spinner shown during write
- `pages/SimulationSummary.tsx` — full rewrite:
  - `RankingTable` component: sortable by any of 4 metrics (profit / km / overflows / kg); click column header to sort ascending/descending; shows mean ± std in `font-mono`; coloured policy dot + rank number
  - `TrajectoryChart` component: single ECharts line chart overlaying all policies across simulation days (mean per day, averaged across samples); metric selector tabs (Overflows / Profit / Distance / Waste); 8-colour palette
  - `MetricBarChart` component: per-metric bar chart with std dev exposed in tooltip hover
  - `aggregateByPolicyAndDay` helper for trajectory data: groups entries by `(policy, day)`, averages across samples
  - `std()` helper function
- `pages/ProcessMonitor.tsx` — improvements (§G.15):
  - `LogLine` component: attempts `JSON.parse` on each log line; if the result has `level`/`levelname`/`severity` and `msg`/`message`/`text` fields, renders timestamp prefix + colour-coded level badge (danger/warning/muted/default) + message body; falls back to raw string otherwise
  - Per-row `Trash2` "Remove" button for completed processes
  - "Clear completed (N)" header button calls `clearCompleted` store action
- `store/process.ts` — persistence and bulk-clear (§G.15):
  - Wrapped `create` in `persist` middleware; `partialize` strips `logLines` and retains only the last 50 non-running processes; stored under key `"wsmart-studio-processes"`
  - `clearCompleted()` action: removes all entries with `status !== "running"` from the map

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.9 live-status item checked; §G.13 `write_text_file` and Save button checked; §G.15 structured log parsing, remove/clear buttons, and history persistence checked; §G.16 Simulation Summary rewrite noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifth pass

Fifth implementation pass: SimulationMonitor gains day-scrubber controls, a bin-fill strip chart,
and a tour sequence table (§G.16); TrainingMonitor gains multi-run overlay chart, hyperparameter
panel, gradient norm sparkline, and checkpoint browser with one-click Eval Runner handoff (§G.17).

**React frontend**
- `pages/SimulationMonitor.tsx` — rewritten:
  - Day scrubber: `◀`/`▶` step buttons flanking the range input; "Following" badge (green pulse, shown when `selectedDay` is null and watcher active); "Latest ↓" button releases back to auto-follow
  - `BinFillStrip` component: top-25 bins sorted by fill %, 0-100% horizontal bars (green <80%, amber ≥80%, red ≥100%), mandatory (!) and collected (✓) badges; show/hide toggle
  - `TourTable` component: stop #, bin ID, fill %, collected, mandatory columns; reads `tour_indices` preferentially; capped at 60 rows; show/hide toggle
- `pages/TrainingMonitor.tsx` — rewritten:
  - `MultiRunChart`: single ECharts canvas overlaying all selected runs; 8-colour palette; solid train loss, dashed val loss, dotted reward (right y-axis); scrollable legend
  - `GradNormSparkline`: compact `grad_norm` chart per run
  - `HparamsPanel`: collapsible; reads `hparams.yaml` via `read_text_file`; flat YAML parser; 8-row preview with "Show all" expand
  - `CheckpointBrowser`: `list_dir` on `<run>/checkpoints/`; "Load in Eval Runner →" button sets `pendingCheckpoint` in app store and navigates to Eval Runner
  - `RunPanel`: groups grad norm + hparams + checkpoints per run below the shared overlay chart
- `pages/EvaluationRunner.tsx` — reads `pendingCheckpoint` on mount via `useEffect`; pre-populates first checkpoint entry and clears the store field
- `store/app.ts` — `pendingCheckpoint: string | null` + `setPendingCheckpoint` action (not persisted)

#### Build tooling

- `tools/app/justfile` — added `bundle` (list installer output), `logs-dir` (print platform data dir), `reset-data` (delete Tauri Store files)
- Root `justfile` — added `studio-reset` shorthand (→ `app::reset-data`)

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.16 items checked (bin-fill, tour table, day scrubber, secondary KPI toggle); §G.17 items checked (multi-run overlay, grad norm, hparams panel, checkpoint browser)

---

#### WSmart-Route Studio — Tauri App (`app/`) — fourth pass

Fourth implementation pass: Evaluation Runner page (§G.12), full DataGeneration form (§G.11),
Settings validation with Rust backend probes, `tools/app/justfile` Clippy/outdated recipes,
and `studio-check`/`studio-clippy` root shorthands.

**Rust backend**
- `commands/system.rs` — new module with two commands:
  - `validate_project_root(path)`: checks path exists, is a directory, and contains `main.py`
  - `probe_python(python_path)`: runs `<path> --version`, handles Python 2 (stderr) and 3 (stdout), returns version string
- `lib.rs` — registers `validate_project_root` and `probe_python`; imports `system` module

**React frontend**
- `pages/EvaluationRunner.tsx` — new: dynamic checkpoint list (add/remove/file-picker), dataset path picker, problem/strategy/device/val_size selects, multi-checkpoint launch (one process per checkpoint, tagged by filename), Advanced Overrides, command preview, results placeholder (§G.12)
- `pages/DataGeneration.tsx` — rewritten: problem selector, distribution checkboxes (Gamma-3/Empirical), dataset type selector, overwrite toggle, graph form (area/num_loc/n_samples/n_days), Advanced Overrides, command preview; Hydra args mirror `gen_data.yaml` (§G.11)
- `pages/Settings.tsx` — validation wiring: `onBlur` and pre-save calls to `validate_project_root` and `probe_python`; inline `CheckCircle`/`XCircle` badges; save blocked on validation errors
- `types/index.ts` — `"eval_runner"` added to `AppMode` union
- `components/layout/Sidebar.tsx` — `"Evaluation Runner"` entry added to Launch section; `ClipboardList` icon
- `components/layout/TopBar.tsx` — `"Evaluation Runner"` title added to TITLES map
- `App.tsx` — `EvaluationRunner` import and router case added

#### Build tooling

- `tools/app/justfile` — added `clippy` (`cargo clippy -- -D warnings`) and `outdated` (`npm outdated`) recipes
- Root `justfile` — added `studio-check` (→ `app::check`) and `studio-clippy` (→ `app::clippy`) shorthands

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.11 additional items checked (full form); §G.12 marked 🚧 In Progress with completed items; §G.19 additional items checked (validation commands)

---

#### WSmart-Route Studio — Tauri App (`app/`) — third pass

Third implementation pass: full-featured Simulation Launcher and Training Hub forms, tabular
Process Monitor with live duration, Settings page (§G.19) with project root / Python path
persistence, first-run onboarding banner, and extended `tools/app/justfile`.

**Rust backend**
- `process::spawn_python_process`: new `python_executable: Option<String>` parameter; empty string treated as `None`, falling back to `which_python`
- `process::which_python`: now takes `working_dir` parameter; checks `<workingDir>/.venv/bin/python` (uv-managed venv) and `<workingDir>/.venv/Scripts/python.exe` (Windows) before system PATH

**React frontend**
- `pages/SimulationLauncher.tsx` — rewritten: 8-policy multi-select checkboxes; area / num_loc / n_samples / cpu_cores / seed inputs; distribution radio (Normal/Gamma/Empirical); Advanced Overrides collapsible; `useMemo` command preview; Hydra args exactly mirror `just controller::test-sim`
- `pages/TrainingHub.tsx` — rewritten: mode selector (Train / HPO Sweep / Evaluate); problem/model/encoder selects; mode-specific param groups (epochs/batch_size for train; method/trials/workers for HPO; checkpoint picker / dataset picker / strategy / val_size for eval); WandB toggle; command preview
- `pages/ProcessMonitor.tsx` — rewritten: tabular `ProcessRow` components with `StatusPill`, process ID, command, PID, live duration (`useLiveDuration` 1s tick), exit code; expand/collapse inline log with auto-scroll toggle; stderr lines coloured warning
- `pages/Settings.tsx` — new: Project Root (text input + directory picker), Python Executable (override `which_python`), Appearance (dark/light radio), About section; dirty-state detection; Save / Discard buttons
- `store/app.ts` — `pythonPath` field + `setPythonPath` action added; persisted via `partialize`
- `types/index.ts` — `"settings"` added to `AppMode` union
- `hooks/useSpawnProcess.ts` — reads `pythonPath` from app store; passes `pythonExecutable: pythonPath || null` to `spawn_python_process`
- `components/layout/Sidebar.tsx` — "App" section added with Settings entry; `FolderOpen` icon for output_browser; `Settings` icon for settings entry
- `components/layout/TopBar.tsx` — first-run warning banner: shown when `projectRoot` is empty and mode ≠ `"settings"`; "Open Settings" quick-link
- `App.tsx` — Settings page import and router case added

#### Build tooling

- `tools/app/justfile` — extended with `check-rust` (`cargo check`), `fmt-rust` (`cargo fmt`), `preview` (build + serve), `update` (`npm update`) recipes

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.9 additional items checked (full form); §G.10 additional items checked (full form, all three modes); §G.15 additional items checked (tabular layout, live duration); §G.19 added (Settings & First-Run Onboarding); Effort × Impact matrix updated

---

#### WSmart-Route Studio — Tauri App (`app/`) — second pass

Second implementation pass: completes all page stubs, wires process lifecycle events, adds
Config Editor (§G.13) and Output Browser (§G.14), and introduces `tools/app/justfile`.

**Rust backend additions**
- `data::read_text_file` — reads any text file (YAML, JSON, plain text) as a `String`; used by ConfigEditor and OutputBrowser
- `data::list_dir` — lists files and subdirectories in a path; returns `DirEntry` with `name`, `path`, `is_dir`, `size_bytes`, `extension`
- `process::ProcessSpawned` event — emitted immediately when a process is spawned (before any stdout); frontend registers the process in the store automatically via `useProcessMonitor`
- `process::which_python` — now resolves `<workingDir>/.venv/bin/python` first (uv-managed project venv), then `.venv/Scripts/python.exe` (Windows), then system PATH

**React frontend additions**
- `hooks/useSpawnProcess.ts` — wraps `spawn_python_process` invoke with loading state and `sonner` toasts; used by all three launcher pages
- `hooks/useProcessMonitor.ts` — now subscribes to `process:spawn` (new) in addition to `process:stdout` and `process:status`; process is registered in the store on spawn, not on first stdout line
- `pages/ConfigEditor.tsx` — Raw / Table / Diff view modes for any YAML/TOML config file; flat YAML parser; "Copy Overrides" button via `navigator.clipboard`; Diff view highlights changed keys between two files (e.g. `pruned_config.yaml` from two runs)
- `pages/OutputBrowser.tsx` — three-pane layout: run list (`list_output_dirs`), file tree (`list_dir`, lazy-loads subdirs), file viewer (CSV table up to 200 rows; raw text for YAML/JSON/log); arbitrary directory picker via Tauri dialog
- `components/layout/Sidebar.tsx` — added "Files" section with Output Browser and Config Editor entries
- `types/index.ts` — added `ProcessSpawned`, `DirEntry`, `OutputDir` interfaces
- Updated `SimulationLauncher`, `TrainingHub`, `DataGeneration` to use `useSpawnProcess` (removes direct `invoke` calls and manual state management)

#### Build tooling

- `tools/app/justfile` — new just module with `install`, `dev`, `tauri-dev`, `build`, `check`, `clean-js`, `clean-rust`, `clean` recipes
- Root `justfile` — added `mod app 'tools/app'` and shorthands: `just studio` (→ `app::tauri-dev`), `just studio-build` (→ `app::build`), `just studio-install` (→ `app::install`)
- `tools/helper/justfile` — updated help text to list `app` module and `just studio` shorthand

#### ROADMAP

- `docs/moon/ROADMAP.md` — marked §G.0, §G.9–§G.15 as 🚧 In Progress with completed items checked; remaining items clearly separated

---

#### WSmart-Route Studio — Tauri App (`app/`) — initial scaffold

Initial scaffold and core implementation of the WSmart-Route Studio desktop app,
a Tauri 2.0 + React 19 replacement for the PySide6 GUI and the Streamlit dashboard.
Implements §G.0, §G.9–§G.12, §G.15 from `docs/moon/ROADMAP.md`.

**Rust backend (`app/src-tauri/`)**
- `src-tauri/src/lib.rs` — plugin registration (notification, store, dialog, shell) and all command handlers
- `src-tauri/src/commands/sim_watcher.rs` — real-time `GUI_DAY_LOG_START:` log line watcher; polls every 200 ms, emits `sim:day_update` Tauri events; replaces Streamlit's `time.sleep()` + `st.rerun()` polling loop
- `src-tauri/src/commands/data.rs` — `load_simulation_log`, `load_csv_file` (returns `CsvFile` with headers+rows), `list_output_dirs` (returns `OutputDir` with metadata), `list_training_runs`, `load_training_metrics`
- `src-tauri/src/commands/process.rs` — `spawn_python_process` (stdout/stderr streamed as `process:stdout` events), `cancel_process` (tokio watch channel), `list_processes`; global `PROCESS_REGISTRY`
- `src-tauri/Cargo.toml` — tauri 2.0, tauri-plugin-{notification,store,dialog,shell}, serde, tokio (full), csv, anyhow
- `src-tauri/tauri.conf.json` — window 1600×1000, min 1200×700
- `src-tauri/capabilities/default.json` — Tauri 2.0 capability grants for all plugins

**React frontend (`app/src/`)**
- `types/index.ts` — `DayLogEntry`, `SimDayData`, `TrainingRun`, `TrainingMetricsRow`, `ProcessEntry`, `ProcessStatus`, `StdoutLine`, `StatusUpdate`, `AppMode`, `NavSection`, `NavItem`
- `store/app.ts` — Zustand with persist: `mode`, `theme` (syncs `dark` class), `projectRoot`
- `store/sim.ts` — `entries`, `selectedPolicy/Sample/Day`, `watchPath`, `isWatching`; `addEntry` deduplicates by `(policy, sample_id, day)`; exports `uniquePolicies`, `uniqueSamples`, `filterEntries`
- `store/process.ts` — `processes` map; `appendLog` caps at 2000 lines per process
- `hooks/useSimWatcher.ts` — subscribes to `sim:day_update`, calls `start_sim_watcher`/`stop_sim_watcher`
- `hooks/useProcessMonitor.ts` — subscribes to `process:stdout` and `process:status` events
- `components/layout/Layout.tsx`, `Sidebar.tsx`, `TopBar.tsx` — 3-section nav (Monitor / Analysis / Launch), running-process count badge, theme toggle
- `components/ui/KpiCard.tsx` — label, value, unit, delta with trend icons, `lowerIsBetter` prop
- `components/ui/StatusPill.tsx` — animated pulse badge for process status
- `index.css` — Tailwind base + component layer (`card`, `kpi-card`, `btn-primary`, `btn-ghost`, `input-base`, `select-base`, `log-line`, `kpi-delta-pos/neg`)

**Pages**
- `pages/SimulationMonitor.tsx` — real-time digital twin; file picker for log, Rust watcher, KPI dashboard (primary + secondary), day-slider, ECharts timeseries; ports `logic/src/ui/pages/simulation/`
- `pages/TrainingMonitor.tsx` — training run discovery, metrics.csv loading, ECharts loss/reward curves; ports `logic/src/ui/pages/training.py`
- `pages/SimulationSummary.tsx` — per-policy aggregate KPIs and bar charts from completed logs
- `pages/BenchmarkAnalysis.tsx` — multi-run, multi-policy comparison with overlaid bar charts
- `pages/DataExplorer.tsx` — paginated CSV table viewer (50 rows/page)
- `pages/ExperimentTracker.tsx` — output directory browser with creation time and size
- `pages/AlgorithmComparison.tsx` — radar chart + per-metric bars comparing all policies in loaded log
- `pages/HPOTracker.tsx` — training run final-reward bar chart; Optuna embedding planned §G.18
- `pages/ProcessMonitor.tsx` — live table of all spawned processes, inline log viewer, cancel button
- `pages/SimulationLauncher.tsx` — Hydra override textarea → `spawn_python_process main.py test_sim`
- `pages/TrainingHub.tsx` — mode selector (train/hpo/eval) + Hydra overrides → `spawn_python_process main.py <mode>`
- `pages/DataGeneration.tsx` — script picker + extra args → `spawn_python_process`

**Config**
- `package.json` — React 19, Tauri 2, ECharts, Zustand 5, react-router-dom 7, sonner, lucide-react
- `vite.config.ts` — Tauri build settings, `VITE_` + `TAURI_` env prefix
- `tsconfig.json` — strict TypeScript, `@/*` path alias
- `tailwind.config.ts` — custom `canvas-*` and `accent-*` color palette, `darkMode: "class"`
- `index.html` — `<html class="dark">` shell

#### ROADMAP

- `docs/moon/ROADMAP.md` — rewrote §D (GUI/UX) for Tauri/React architecture; added §G (WSmart-Route Studio, 16 phases); added §G.16 (Simulation Digital Twin), §G.17 (Training Monitor), §G.18 (Experiment & HPO Tracker)

---


---

## [4.3.0] - 2026-01-27

### Simulator Pipeline Overhaul & Selection Strategies

Introduced a modular **Selection Strategy** system for the Simulator, allowing for granular control over bin collection criteria (Last Minute, Lookahead, Revenue, etc.) via XML/YAML configurations.

### Added

- **Selection Strategies** (`logic/src/policies/selection/`):
  - `LastMinuteSelection`: Reactive threshold-based collection.
  - `LookaheadSelection`: Predictive overflow prevention.
  - `RevenueThresholdSelection`: Profit-driven selection.
  - `RegularSelection`: Periodic schedule enforcement.
  - `MeansAndStdDevSelection`: Statistical overflow prediction.
- **Configuration System**:
  - `assets/configs/policies/`: Modular XML/YAML policy configs.
  - `logic/src/utils/configs/yaml_to_env.py`: Utility for environment variable conversion.
- **Decoding Strategies** (`logic/src/utils/functions/decoding.py`):
  - Unified `Greedy`, `Sampling`, `BeamSearch`, and `Evaluate` strategies.

### Changed

- **Simulator Engine**: Refactored `actions.py` to use `mandatorySelectionAction` and `PostProcessAction` for composable behavior.
- **Testing**: Added comprehensive unit tests for selection actions and fixtures in `logic/test/fixtures/sim_fixtures.py`.

## [4.2.0] - 2026-01-25

### Comprehensive Testing Suite (Phase 2)

Completed a major testing overhaul, introducing End-to-End (E2E) and Integration testing layers to ensure system reliability and stability.

### Added

- **E2E Testing**:
  - `logic/test/e2e/test_cli_commands.py`: Smoke tests for core CLI commands (`gen_data`, `train_lightning`, `eval`, `test_sim`).
- **Integration Testing**:
  - `logic/test/integration/test_training_advanced.py`: Validated advanced RL algorithms (DR-GRPO, GDPO) and checkpoint resumption.
  - `logic/test/integration/test_simulation.py`: Component-level tests for `SimulationContext`, `run_day`, and wrappers.
  - `logic/test/integration/test_evaluation_workflow.py`: Verified evaluation pipeline (checkpoint loading, inference loop, and metrics).
- **Test Infrastructure**:
  - Patched `VRPPEnv` and `CriticNetwork` for compatibility during testing.
  - `walkthrough.md`: Documentation of test results and coverage.

### Changed

- **Implementation Plan**: Updated to reflect completion of Phase 2.1 and 2.2.
- **Testing Standards**: Enforcing explicit `cpu` accelerator for CI/Integration tests to prevent GPU contention.

## [4.1.0] - 2026-01-24

### Legacy Pipeline Removal

The legacy `reinforcement_learning/` pipeline has been permanently removed after the successful migration to the PyTorch Lightning architecture.

### Changed

- **Directory Structure**: Removed `logic/src/pipeline/reinforcement_learning/` and corresponding symlinks.
- **Training Entry Point**: Centralized training logic in `logic/src/pipeline/features/train.py`.
- **Documentation**: Updated all architecture guides to reflect the consolidated pipeline structure.

## [4.0.0] - 2026-01-22

### Lightning Pipeline Migration Complete ✅

Successfully completed the full migration from the legacy `reinforcement_learning/` pipeline to the new Lightning-based `rl/` pipeline. All 6 migration phases finished with backward compatibility preserved.

### Migration Summary

- **Phase 1-2**: CLI compatibility and vectorized policy relocation complete
- **Phase 3-4**: External imports updated, 580 tests passing (167 core + 413 extended)
- **Phase 5**: Documentation updated (TUTORIAL.md, AGENTS.md)
- **Phase 6**: Legacy pipeline backed up and deprecated with symlink for backward compatibility

### Added

- **Symlink Approach**: `reinforcement_learning/` → `reinforcement_learning.bak/` symlink maintains backward compatibility for tests and existing code
- **Migration Documentation**: Comprehensive `MIGRATION_PLAN.md` with phase tracking and verification checklists

### Changed

- **Primary Pipeline**: `logic/src/pipeline/rl/` is now the active, maintained codebase
- **Deprecated**: `logic/src/pipeline/reinforcement_learning/` marked as deprecated (symlink to backup)
- **Documentation**: All tutorial examples, architecture docs, and agent instructions updated to reference new pipeline
- **Import Patterns**: Updated test fixtures to use new `wrap_dataset(dataset, policy=None, env=None)` signature

### Fixed

- **TensorDict Compatibility**: Fixed membership checks across 7 files (`key in list(td.keys())`)
- **RolloutBaseline**: Added backward compatibility for old `(model, problem, opts)` calling convention
- **List Datasets**: Added TensorDict conversion in `_rollout` for list-type datasets
- **NonTensorData Handling**: Fixed `get_tour_length` to handle TensorDict's None wrapper
- **Crossover Robustness**: Made `vectorized_ordered_crossover` handle duplicate values gracefully
- **Linting**: Fixed E402 errors in `train.py` by moving imports before deprecation warning

### Test Results

- ✅ 167 core tests passing (test_models, test_train, test_integration, test_hp_optim, test_il_train)
- ⚠️ 12 pre-existing failures in SAPO/GSPO integration (unrelated to migration)
- ✅ No import errors, full backward compatibility maintained

### Migration Notes

The old `reinforcement_learning/` directory is preserved as `reinforcement_learning.bak/` and accessed via symlink. This allows existing tests and code to continue working while new development uses the Lightning pipeline. After 1 week of stable operation, the backup can be removed.

## [3.2.1] - 2026-01-21

### Defaults Optimization & Stabilization

Standardized the training pipeline on `train_lightning`, optimized default performance settings, and resolved critical data loading concurrency issues.

### Changed

- **Entry Point**: Standardized on `python main.py train_lightning` as the primary training command in all documentation.
- **Performance Defaults**: Enabled **Automatic Mixed Precision (AMP)** (`precision="16-mixed"`) and set `num_workers=4` by default for faster training on modern GPUs.
- **Generator Handling**: `VRPPGenerator` and `WCVRPGenerator` now correctly handle `capacity=None` defaults (1.0 and 100.0 respectively).

### Fixed

- **Data Loading Concurrency**: Resolved CUDA forking errors with `num_workers > 0` by ensuring `Generator` instances are moved to CPU before being passed to `DataLoader`, enabling efficient parallel data generation.
- **Documentation**: Updated `CONTRIBUTING.md`, `TROUBLESHOOTING.md`, `MIGRATION_PLAN.md`, and `REFACTORING_PLAN2.md` to align with the new Lightning pipeline standards.

## [3.2.0] - 2026-01-21

### CWCVRP Lightning Migration & GDPO Support

Successfully migrated the Capacitated Waste Collection VRP (CWCVRP) pipeline to the new PyTorch Lightning architecture, achieving verifyied parity, and integrated Gradient-Decomposed Policy Optimization (GDPO).

### Added

- **GDPO Algorithm**: Implemented `GDPO` (Gradient-Decomposed Policy Optimization) in `pipeline/rl/core/gdpo.py` and consolidated it into `train_lightning.py`.
- **Decomposed Rewards**: Updated `VRPPEnv` and `WCVRPEnv` to calculate and store decomposed reward components (`reward_overflows`, `reward_cost`, `reward_collection`) for advanced gradient analysis.
- **Baseline Improvements**: Enabled `bl_warmup_epochs` configuration and refactored `Baseline` classes to inherit from `torch.nn.Module` for automatic device management.
- **Parity Report**: Added `CWCVRP_LIGHTNING_MIGRATION_REPORT.md` documenting the successful parity verification between Legacy and Lightning pipelines.

### Changed

- **`CWCVRP` Pipeline**: Fully migrated to Lightning, verified with `val/reward` (~ -2.78) matching inverted legacy costs.
- **`RLConfig`**: Added GDPO-specific configurations (`gdpo_objective_keys`, `gdpo_objective_weights`, etc.) and `bl_warmup_epochs`.
- **Infrastructure**: Refactored `generators.py` and `data_utils.py` for cleaner data handling.

### Fixed

- **Device Mismatch**: Resolved critical `RuntimeError` in `reinforce.py` where reward and baseline tensors were on different devices.
- **Optimization**: Support for `RMSprop` in Lightning `configure_optimizers` to match legacy defaults.

## [3.1.0] - 2026-01-21

## [3.2.0] - 2026-01-21

### Multi-Objective Optimization (GDPO)

Introduction of Group reward-Decoupled Normalization Policy Optimization (GDPO) for stable multi-objective learning.

### Added

- **GDPO Algorithm**: Implementation of `GDPO` in `logic/src/pipeline/rl/core/gdpo.py`, featuring decoupled Z-score normalization and weighted aggregation.
- **Decomposed Rewards**: `VRPPEnv` and `WCVRPEnv` now expose individual reward components (`reward_waste`, `reward_cost`, `reward_overflow`) in `TensorDict`.
- **Config**: Added fields to `RLConfig` for GDPO objectives, weights, and conditional keys.
- **Integration**: Added `gdpo` support to `train_lightning.py`.

### Vectorization & Expert Policy Suite

A major performance upgrade and the introduction of a new stochastic expert for imitation learning.

### Added

- **Vectorized ALNS Policy**: Modular `VectorizedALNS` implementation for GPU-accelerated batch optimization, replacing instance-by-instance loops.
- **Random Local Search Policy**: New `RandomLocalSearchPolicy` expert for imitation learning, featuring pre-sampled operator sequences for maximum efficiency.
- **Full LS Vectorization**: Complete migration of local search operators (`relocate`, `two_opt_star`, `swap_star`, `three_opt`) in `local_search.py` to native PyTorch vectorized operations.
- **Imitation Learning Integration**: Added `random_ls` expert support in `ImitationLearning` and `AdaptiveImitation` RL modules.
- **Benchmarking & Validation**:
  - `benchmark_ls.py`: Performance auditing tool (achieving ~1,700 instances/sec on RTX 3090 Ti).
  - `test_random_local_search.py`: Comprehensive correctness suite for stochastic operators.

### Changed

- **`local_search.py`**: Architecture refactored to use advanced vectorized techniques (v-map equivalent) like priority-based `argsort` for double-relocations and case-masking for 3-opt.
- **`RLConfig`**: Expanded to support configurable iterations (`random_ls_iterations`) and probabilities (`random_ls_op_probs`) for expertos.
- **`ALNSPolicy` & `HGSPolicy`**: Now fully utilize low-level vectorized solvers for GPU throughput.

### Fixed

- Resolved CUDA `RuntimeError` and `IndexError` in vectorized operators by implementing strict tensor expansion and explicit device placement.
- Cleaned up redundant expressions and variables in the local search implementation.

## [3.0.0] - 2026-01-21

### Major Refactoring Completion (Phases 1-5)

The **Old RL Pipeline** (`logic/src/pipeline/reinforcement_learning/`) features have been fully migrated to the **New Lightning Pipeline** (`logic/src/pipeline/rl/`), achieving 100% parity and enhanced modularity.

### Added

- **Robust Baselines**:
  - `RolloutBaseline`: Now performs correct greedy rollout with T-test significance updates.
  - `POMOBaseline`: Computes mean reward across instance augmentations/starts.
  - `WarmupBaseline` & `CriticBaseline`: Restored and integrated.
- **Meta-Learning Suite**:
  - `CostWeightManager` (TD Learning): Reimplemented using tabular TD methods for adaptive weight tuning.
  - `HyperNetworkTrainer`: Implemented via `HyperNetworkStrategy` wrapper.
  - Registery now supports: `rnn` (RWA), `bandit` (Contextual), `morl` (Pareto), `tdl` (TD), `hypernet`.
- **Training Infrastructure**:
  - `Epoch Utilities`: Rich validation metrics (overflows, efficiency, cost) in `epoch.py`.
  - `route improvement`: Full `calculate_efficiency` logic and `EfficiencyOptimizer` parity.
- **HPO Module** (`logic/src/pipeline/rl/hpo/`):
  - Consolidated HPO logic into dedicated module.
  - `OptunaHPO`: Clean class-based interface for Optuna (TPE, Grid, Random, Hyperband).
  - `DEHB`: Integrated Differential Evolution Hyperband.

### Changed

- **Folder Structure**:
  - Moved `dehb` from `features/` to `hpo/`.
  - Created alias proxies in `pipeline/rl/policies/` for classical solvers (`hgs`, `alns`).
- **Logic**:
  - `MetaRLModule` now automatically passes environment context to strategies like HyperNet and TDL.
  - `HRLModule` Manager now computes Actor-Critic (PPO) loss instead of no-op.

## [2.0.0] - 2026-01-21

### Added

- **New RL Pipeline (`logic/src/pipeline/rl/`)**: Modular, PyTorch Lightning-based architecture.
  - **Algorithms**: REINFORCE, PPO, SAPO, GSPO, DR-GRPO, POMO, SymNCO.
  - **Baselines**: Rollout (greedy), Warmup (exponential->greedy), POMO (mean-augmented), Critic (network).
  - **Meta-Learning**: Unified `MetaRLModule` with RNN, Contextual Bandit, and Pareto (MORL) strategies.
  - **Hierarchical RL**: `HRLModule` with Manager-Worker architecture.
  - **HPO**: Lightning-integrated Hyperparameter Optimization using Optuna and DEHB.
  - **Hybrid Policies**: `NeuralHeuristicHybrid` combining AM/TAM with ALNS/HGS.
- **Environment Layer (`logic/src/envs/`)**: RL4CO-compatible `VRPPEnv` and `WCVRPEnv` with diverse generators.
- **Data Layer**: Optimized `TensorDict`-based datasets and generators in `logic/src/data/`.

### Changed

- Refactored `logic/src/pipeline/reinforcement_learning/` into the new `logic/src/pipeline/rl/` package.
- Standardized all policy inputs/outputs to `TensorDict`.
- Unified training loop via `WSTrainer` (Lightning Wrapper).
- Updated `alns.py` and `hgs.py` to support `pd.DataFrame` coordinate inputs for distance calculation.

### Removed

- Legacy training scripts (`manager_train.py`, `worker_train.py`) in favor of `main.py` entry points.

## [Unreleased]

### Added

- `logic/src/dehb/` directory containing extracted DEHB library (internalized).
- `TESTING.md`: Comprehensive documentation on the project's testing strategy and organization.
- Added property-based tests using `Hypothesis` for robust input validation.
- Implemented **Contract Tests** for Gurobi, Hexaly, PyVRP, and OR-Tools solvers.
- Introduced **Performance Benchmark Suite** for neural decoders and OR solvers.

### Changed

- Moved DEHB implementation from `logic/src/pipeline/reinforcement_learning/hyperparameter_optimization/` to `logic/src/dehb/`.
- Updated all import statements in `hpo.py` and test files to reflect DEHB move.
- Updated `.gitignore` to exclude `.pytest_cache`, `.mypy_cache`, and `.ruff_cache`.
- Increased code coverage threshold to 50% with enforcement in `pyproject.toml`.
- Improved `BaseProblem` and `VRPP` logic for full `TensorDict` compatibility.
- Fixed inconsistent return types in `multi_vehicle.py` solver wrappers.

### Fixed

- Cleaned up 500+ `__pycache__` directories from the repository.
- Resolved DEHB import errors in the test suite by updating `@patch` decorators and import paths.

---

## [0.5.0] — 2026-06-XX

### Added
- Figueira da Foz 350-bin dataset with plastic-bin results using Empirical distribution and Classical Local Search route improver
- City comparison simulation analysis (`global/`)

### Changed
- Analysis CSVs moved to `global/` directory; markdown references updated

---

## [0.4.0] — earlier

*(Earlier history not yet documented in this changelog)*

---

[Unreleased]: https://github.com/ACFHarbinger/WSmart-Route/compare/HEAD...HEAD
