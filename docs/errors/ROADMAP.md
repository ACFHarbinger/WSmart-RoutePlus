# Systematic Bug-Hunt Roadmap

Tracker for issue #61. The sequence follows the component flow in
`docs/moon/ARCHITECTURE.md` §§1–7 and command flows in §8. Each completed
component is checked for cross-boundary shape/type errors, invalid state
mutation, device handling, and silent routing/solver failures.

- [COMPLETED] CLI entry point and Hydra command dispatch (`main.py`, task routing) — normalized documented aliases `evaluation` → `eval` and `sim_hpo` → `hpo_sim`; regression coverage added.
- [COMPLETED] Typed configuration composition and validation (`logic/src/configs/`) — all canonical task groups compose through Hydra; no additional typed-config defect confirmed in this pass.
- [COMPLETED] Feature engines: train, evaluation, data generation, and simulation dispatch — repaired task-scoped curriculum graphs, CPU multiprocessing validation, failed-run tracking, and duplicate simulator validation.
- [COMPLETED] Routing environments, generators, and task objectives — 2026-09-29 Cursor pass (see below).
- [SKIPPED] Model encoders, decoders, embeddings, and critic interfaces — covered by the 2026-09-26 logic review; not re-traced in this increment.
- [SKIPPED] Policy and solver construction, including exact and heuristic routes — covered by the 2026-09-26 logic review.
- [SKIPPED] RL training lifecycle, baselines, callbacks, and tracking — covered by the 2026-09-26 logic review.
- [SKIPPED] Multi-day simulation state machine, day context, and result aggregation — covered by Grok's logic-review lane and #41 this round.
- [COMPLETED] Data repositories, processors, and dataset builders — lazy-load optional dashboard/geo extras so core imports no longer hard-fail.
- [COMPLETED] Controllers (`logic/controllers`) — Hydra dispatch now accepts `evaluation` when `main.py` did not rewrite `tasks=evaluation`.
- [COMPLETED] Cross-cutting utilities, package imports, lint/type checks, and regression sweep — `as_batch_nodes` added in `envs/base/ops.py`; `utils/functions` and `utils/data` inspected, no additional clear defect in this increment.

## Current pass (2026-09-29, Cursor, components the logic review did not cover)

Base `dfb049e7e`. Fixes landed in this increment:

1. **WCVRP / OpsMixin action shape.** Decoder actions of shape `[B, 1]` were
   unsqueezed again before `gather`, which is a silent index-shape error.
   Shared helper `as_batch_nodes` squeezes to `[B]`. WCVRP also prepends a
   depot column on a customer-only `mandatory` mask (same as CVRPP/VRPP).
2. **CVRPP remaining capacity.** An illegal (unmasked) pickup could drive
   `remaining_capacity` negative. It is now clamped at 0.
3. **SCWCVRP overflow comparison.** `_get_reward` blindly unsqueezed
   `max_waste` whenever `dim > 0`, so a per-node `[B, N+1]` tensor became
   `[B, N+1, 1]` against customer waste `[B, N]`. It now slices like WCVRP.
4. **Hydra `evaluation` alias.** `python main.py tasks=evaluation` bypasses
   `main.py`'s rewrite and used to raise `Unknown task`. Dispatch now maps
   `evaluation` → `eval` (and still maps `sim_hpo` → `hpo_sim`).
5. **Optional extras on the data import path.** `logic.src.data.datasets`
   eagerly imported the HTML dashboard crawler (hard `ImportError` without
   beautifulsoup4), and `logic.src.data.network` eagerly imported geopy /
   geopandas / googlemaps / OSM backends. Both are lazy. `haversine_distance`
   and core dataset classes import without those extras.

## Design decisions (not patched)

- **WCVRP vehicle-capacity clamp vs `max_waste` credit.** The docstring says
  collection is `min(waste, max_waste, remaining_capacity)`, but
  `test_update_waste_clamping` (waste=15, `max_waste`=10, default capacity=1)
  asserts collected credit 10. Waste / `max_waste` / `capacity` are not
  guaranteed to share units. Clamping `current_load` to remaining capacity
  would change that legal-path fixture. Left as-is; report rather than mix
  the unit systems.
- **`VRPP.get_costs` always prepends a depot waste column.** Dataset
  generation stores customer-only waste, so the prepend is correct for the
  training evaluator. After `VRPPEnv.reset` waste is already `[B, N+1]`;
  calling `get_costs` on a reset TensorDict would shift every customer.
  Unifying those conventions needs a design call (same family as #82's
  training-vs-simulator unit gap).

## Config-propagation pass (2026-09-29, Cursor, open-issues-2)

Base `606f79690`. Silent-default class outside route constructors:

1. **Mandatory selection list variants.** `{file.yaml: [variant]}` (including
   OmegaConf ListConfig) was treated as a strategy named after the filename
   with empty params. Last-minute `threshold: 70/90` and lookahead
   `current_collection_day: 0` now load from yaml.
2. **Acceptance yaml never became `acceptance_criterion`.** Constructors read
   the singular field and ignored `acceptance_criteria: {ac_bmc.yaml: [bmc]}`.
   PSOMA therefore ran BMC at dataclass T0=3 / α=0.9 instead of yaml 100 /
   0.995. MandatorySelectionAction (runs before construction) now injects a
   typed `AcceptanceConfig` from the yaml.
3. **CLS `ls_operator: 2opt` was ignored.** Classical local search always ran
   the full relocate/swap/2-opt/… suite. `2opt` now means intra-route 2-opt
   only, and `time_limit` is honoured. Yaml params are passed to `process()`
   without overwriting the policy's constructor `time_limit` on the day
   context.

4. **Codex review amendment (DictConfig / RI leak).** `_gather_strategies`
   treated a Hydra DictConfig mapping as a sequence of keys, so
   `{ms_service_level.yaml: service_level2}` loaded every service-level
   variant. Sequence detection now excludes mappings. `_create_processors`
   clears `_ri_yaml_params` per entry so CLS yaml does not leak into the
   next improver.

5. **Consumer-level capture.** `AcceptanceCriterionFactory.create` records
   the kwargs constructors actually pass (BMC instance T/α), last-minute
   records `SelectionContext.threshold`, lookahead records
   `current_collection_day`, and CLS `process()` records the operator /
   iterations / time_limit it consumes. Live `main.py test_sim` must show
   BMC 100/0.995, last-minute 70/90, CLS 2opt/1000/30 — not action-hook
   snapshots alone. ALNS ``from_config`` reads ``getattr`` even on a dict, so
   the live config is wrapped as an attribute-dict; otherwise BMC falls
   back to ``start_temp`` (0) despite the action-level yaml snapshot.

6. **Live `test_sim` context.** Nested Hydra DictConfig children stayed
   struct-locked (`acceptance_criterion` not in struct). `SimulationDayContext`
   is a Mapping without `setdefault`/`pop`; capture now uses setattr-style
   list append and a pop helper.

7. **Tagged live capture.** Consumer JSONL records include `policy` / `constructor` / `day`.
   NA is untested (missing AMGAT weights). The raw files and invocation live under
   `.agent/cache/patches/cursor/issue-61-config-propagation-live*`.

## NA live capture (2026-09-30, Cursor, open-issues-3)

Base `43b422e27`. Round 2 left NA untested. A trained AM checkpoint at
`~/.cache/wsr-review/open-int-smoke/am` is now used (`p.na.na.amgat.0.model_path`).

1. **Decoding yaml dropped at the adapter.** `policy_na.yaml` sets
   `decoding.beam_width: 5`. `NeuralParams.from_config` already unpacked that
   nested map, but `BaseRoutingPolicy._build_config` kept only dataclass field
   names, so the adapter ran at the default `beam_width=1`. NA `_build_config`
   now calls `from_config`. Fail-before:
   `test_na_adapter_honours_yaml_decoding_not_dataclass_beam_width`.
2. **Consumer capture.** `execute` records `consumer_decoding` (strategy,
   beam_width, reward_weight, length_penalty_alpha) before the empty-mandatory
   early return. The decoder records the same with `applied: true`. Empty
   `route_improvement: []` records `consumer_ri` `{entries: []}` even when the
   tour is `[0, 0]`. Lookahead is the existing `consumer_lookahead` path.
3. **Live `test_sim`.** NA-only riomaior-20 run with the AM checkpoint; report
   sits next to the eight-constructor table.

## Next session

Resume is not required for the uncovered-component increment above. A later
#61 pass can take the WCVRP unit-system decision and the `get_costs` depot
column, or walk `utils/{decoding,model,security}` if those are still in
scope after the logic review.
