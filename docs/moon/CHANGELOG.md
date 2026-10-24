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

### Claude: the 1.0.2 package ships only the paper policies' configs (`2026-10-24`)

#### Changed

- **Release builder** (`logic/package/release_closure.py`, `release_staging.py`): `logic.src.configs.policies` re-exports every policy family's config dataclass, and the closure used to follow those imports, so the package carried the configs of about 90 unreleased policies. The closure now indexes that package's eager re-exports like a lazy map (`INDEXED_INITS`): a config ships when shipped code imports it, by module path or by name. The staged `__init__` drops the imports, `__all__` names and docstring `Attributes:` lines of the pruned configs (`drop_unbound_exports`). `package/1.0.2` was rebuilt: 90 config files fewer; the release tests and the 10-algorithm smoke are unchanged.
- `logic/src/configs/policies/__init__.py`: the docstring example uses `HGSConfig` instead of the unreleased `ARCOConfig`.

### Claude: Studio follows the simulator's output layout (`2026-10-24`)

#### Fixed

- **Studio looked for runs in `assets/output`**, where the simulator stopped writing on 2026-10-13. The Output Browser, Experiment Tracker, Simulation Summary, Benchmark Analysis and City Comparison now scan `outputs/simulations` first and still the older `assets/output` (`app/src/utils/runs/outputRoots.ts`); dropped bundles are imported into `outputs/simulations/.imports`, and the report launcher's default parse folder is `outputs/simulations/90days`.
- **Run detection from stdout**: the Hydra-snapshot and pruned-config parsers matched only `→`; they accept the `->` the simulator prints since 8f0ce9b77, and bare run paths under `outputs/simulations`.
- The Studio sets `WSR_STUDIO=1` for the processes it launches (8f0ce9b77), so it still receives the live policy telemetry stream.

### Claude: run folders and console output from the third Windows test (`2026-10-23`)

Third Windows test of the 1.0.2 package: 0 test failures, all 20 smoke runs identical to Linux, Ctrl+C clean. Fixed what it still found:

#### Fixed

- **Run files under `assets/`**: the config snapshot (`hydra/`), the folders `_ensure_directories` makes, and the final realtime log still used `<root>/assets/<output_dir>/...`, left over from before results moved to `<root>/<output_dir>` (2026-10-13). They now go into the run folder with the results, so the final GUI output also lands in the same `.jsonl` as the daily lines.
- **Policy telemetry flooded the console**: `PolicyVizStreamSession` printed the whole ALNS ring buffer to stdout (and appended it to the realtime log) every 0.5 s, so one 3-day run logged ~28 MB. The live stream and the stdout copy are now only for WSmart-Route Studio, which sets `WSR_STUDIO=1` when it launches Python (`spawn_python_process`); other runs write one final snapshot per solve to the log.
- **ASCII arrows** in the simulator's config-snapshot and SP-pool log lines (`→` came out garbled on the Windows console).

### Claude: Windows portability from the second Windows test (`2026-10-23`)

Second Windows test of the 1.0.2 package (Windows 11 VM): all 10 routing algorithms ran from source and from the executable with results identical to Linux; 17 tests failed and Ctrl+C left a process behind. Fixed:

#### Fixed

- **Ctrl+C on Windows**: the simulator's shutdown handler (and `handle_shutdown`) killed the workers only through a POSIX process group, so on Windows the `multiprocessing` Manager outlived the run and kept the executable's unpacked `_MEI*` folder locked. `parallel_runner.kill_child_processes` kills every live child first.
- **`get_path_until_string`** split only on `os.sep`, so on Windows a forward-slash path (a `focus_graph` from a config) was never truncated and data loading got `None`; it now splits on both separators there and keeps the path's own separators.
- **Atomic distance-matrix write**: `os.replace` onto a file another writer is replacing fails on Windows with `PermissionError`; `_replace_with_retry` retries for up to 2 s.
- **Evaluation throughput**: the evaluators divided by a `time.time()` duration that Windows' ~15 ms clock can report as 0; they use `time.perf_counter()` and `samples_per_second` returns 0.0 for an unmeasured duration.
- **`--help` of the executable** printed `python .`; it names the executable when frozen.
- **Setup scripts** install the `test` extra, install the Visual C++ runtime without a UAC prompt from an elevated shell, and say how to install it by hand when the prompt is declined.
- **Tests on Windows**: path assertions use `os.path.join`; the symlink security tests use a `make_symlink` fixture that skips when Windows refuses to create symlinks (WinError 1314); new `test_windows_portability.py`.

### Claude: release installs on a clean Windows (`2026-10-23`)

First Windows test of the 1.0.2 package (Windows 11 VM): setup could not complete. Fixed:

#### Fixed

- **Release Python version**: `logic/release.yaml` `python:` sets the release's `requires-python` to `>=3.10,<3.13` (ortools 9.9 has no wheels past 3.12) and writes `.python-version` (3.12, the tested interpreter). Without it uv picked CPython 3.15, for which tensordict has no wheels.
- **torch-scatter / torch-sparse dropped from the release** (`dropped_requirements:`, applied by `write_release_pyproject`): they have no Windows wheels, and their only wheels (Linux, torch 2.12.1) do not load with the pinned torch 2.13, so `uv sync` failed on Windows and the packages were dead weight on Linux. The pyg index goes with them.
- **`GNNLayer` without torch-scatter**: `nargnn/gnn_layer.py` falls back to a plain-torch `scatter_mean` (`index_add_`), so NARGNN no longer raises ImportError when torch-scatter is missing or broken; its tests now run instead of being skipped.
- **NARGNN log-likelihood**: `NARGNNPolicy` passed the node action mask (`[B, N+1]`) to `get_log_likelihood` as a per-step mask, a shape error hidden while the tests were skipped; it now sums every step.
- **Summary table of a failed policy**: `display_simulation_summary_table` crashed with `TypeError` on the `None` metrics of a policy whose samples all failed, hiding the real error (for example a missing data file); it shows a dash.
- **Windows setup scripts** (`desktop/windows/*/setup_package.*`): install the Microsoft Visual C++ x64 runtime when it is missing (torch's DLLs fail with WinError 126 without it), and create `executables\`; the Linux script creates `executables/` too.

### Claude: autoregressive paper submodule renamed (`2026-10-23`)

#### Changed

- **`assets/papers/Autoregressive-Deep-Reinforcement-Learning-for-Waste-Collection-Routing`**: the autoregressive DRL paper submodule (formerly `...-Waste-Collection-Route-Planning`) now has the name, path and URL of its GitHub repository. Existing checkouts: run `git submodule sync` after pulling.

### Claude: new problem CPTPTWTC (`2026-10-22`)

#### Added

- **Capacitated Profitable Tour Problem with Time Windows and Time Constraints (CPTPTWTC)**, `logic/src/pipeline/learning/envs/cptptwtc.py` (`CPTPTWTCGenerator`, `CPTPTWTCEnv`, `CPTPTWTC`, `CPTPTWTC_SPEC`) and `logic/configs/envs/cptptwtc.yaml`. It is CPTPTC (vehicle capacity and per-trip shift budget) plus a time window `[tw_early, tw_late]` per bin, in hours from the start of the working day: an early vehicle waits, service must start by `tw_late`, and the vehicle must be back before the day (the depot's window, `day_hours`) ends. The day clock runs across trips; a depot return resets the shift budget, not the clock.
- **`TimeWindowResource`** (`envs/base/resources.py`) tracks the clock and masks bins whose window would be missed or after which the depot can no longer be reached in time. `CPTPTWTC.evaluate_tour` adds `tw_violations` (and raises with `check_feasibility`).
- **Model:** `CPTPTWTCInitEmbedding` adds each node's window (as fractions of the day) to CPTP's node features through a new `PTPInitEmbedding._extra_node_features` hook (empty for PTP/CPTP, so their parameters and outputs are unchanged); the context, edge and state embedders are CPTP's.
- **Data:** windows are drawn by `logic/src/data/time/node_time_windows.py` (default day 10 h, windows 2–6 h; `day_hours`, `tw_min_width_h` and `tw_max_width_h` on `EnvConfig`), both by the generator and by `gen_data data.problem=cptptwtc`, whose datasets carry `day_hours`, `tw_early` and `tw_late`.
- Learning side only: the simulator has no bin time windows, so `test_sim` does not take `cptptwtc`, and the problem is not in the release.

### Claude: TCPTP renamed CPTPTC (`2026-10-22`)

#### Changed

- **The Time-constrained Capacitated Profitable Tour Problem (TCPTP) is now the Capacitated Profitable Tour Problem with Time Constraints (CPTPTC)**, in code and file names: `envs/cptptc.py` (`CPTPTCEnv`, `CPTPTCGenerator`, `CPTPTC_SPEC`), `logic/configs/envs/cptptc.yaml`, the task configs, the release manifest and the tests. Hard break, as for the earlier renames: checkpoints and datasets that record `tcptp` are rejected; regenerate them.

### Claude: configuration layout mirrors the code (`2026-10-22`)

#### Changed

- **Bin-analysis submodule links removed:** `logic/submodules/wsmart_bin_analysis` and `logic/submodules/WSmart_Bin_Analysis` were tracked symlinks to `logic/submodules/WSmart-Bin-Analysis` (the only real submodule). Both are gone. `logic/__init__.py` installs an import hook that maps the import name `logic.submodules.wsmart_bin_analysis` onto the submodule (Python cannot import the hyphenated folder), so existing imports keep working; a real `wsmart_bin_analysis` folder, as the release package stages, takes precedence. The release builder resolves that import name through the same alias.
- **IRP and ThOP modules:** their generators moved to `logic/src/pipeline/learning/envs/generators/irp.py` and `generators/thop.py`, and their environments joined their task modules: `envs/irp.py` now holds `IRPEnv`, `IRP` and `IRP_SPEC` (formerly split with `irp_env.py`), and `envs/thop.py` holds `ThOPEnv`, `ThOP` and `ThOP_SPEC`. `get_env`/`get_generator` and the lazy class exports follow; `irp_env`, `irp_generator`, `thop_env` and `thop_generator` no longer exist (hard break).
- **Policy configs are organised like `logic/src/policies`.** In `logic/src/configs/policies/`: `routing/<category>/` (`meta_heuristics`, `hyper_heuristics`, `matheuristics`, `exact_and_decomposition_solvers`, `learning_algorithms`, `learning_heuristic_algorithms`, `learning_matheuristic_algorithms`, `other_algorithms`), `packing_and_routing/`, `vectorized/routing/`, `packing/` (`dispatch`, `learned`, `threshold`), `route_improvement/`, `acceptance_criteria/` and `helpers/learning/`; the `other/` package is gone. In `logic/configs/policies/`: `routing/<category>/policy_<key>.yaml`, `packing_and_routing/policy_nds_brkga.yaml`, and `other/packing/pk_*.yaml`, `other/route_improvement/ri_*.yaml`, `other/acceptance_criteria/ac_*.yaml`. The policy YAMLs reference their stage files by the new paths, and `test_sim.yaml` composes each policy from its category group (`- /policies/routing/meta_heuristics@p.alns: policy_alns`). `from logic.src.configs.policies import ALNSConfig` (and the other re-exports) still works; module paths such as `configs.policies.alns` and `configs.policies.other.route_improvement` are now `configs.policies.routing.meta_heuristics.alns` and `configs.policies.route_improvement` (hard break).
- **One resolver for policy YAML files** (`logic/src/utils/configs/policy_files.py`): the simulator, the test-config variants, the batch executor, the policy link updater and the release builder find policy and stage files through it. File names are unique, so a pre-reorganisation flat path (`other/pk_last_minute.yaml`, `policy_alns.yaml`) still resolves. The Studio app reads the grouped `test_sim.yaml` defaults; the API docs are regenerated.
- **`logic/src/configs/learning/`** now holds `core/` (the training-loop and algorithm dataclasses, formerly directly in `configs/learning/`) and `envs/` (formerly `configs/envs/`). `from logic.src.configs.learning import RLConfig` (and the other algorithm and environment configs) still works; module paths such as `configs.learning.imitation` and `configs.envs.graph` are now `configs.learning.core.imitation` and `configs.learning.envs.graph` (hard break).

### Claude: package evaluation fixes (`2026-10-22`)

#### Fixed

- **Simulations on Windows** (`orchestrator/parallel_runner.py`, `orchestrator/__init__.py`, `__main__.py`): CPU runs (`sim.no_cuda=true`) asked for the `fork` start method, which Windows does not have, so every `test_sim` failed with `cannot find context for 'fork'` (found running the Windows executable under Wine). The pool now uses `spawn` where `fork` does not exist; the Ctrl+C handler no longer calls the POSIX-only `os.killpg` there; and `__main__.py`, the executables' entry point, calls `multiprocessing.freeze_support()` so spawned workers start correctly in a frozen program.
- **Executables bundle mypyc helper modules** (`ci/simulator_release.spec`): `tomli` 2.5 (imported by pyvrp) is compiled with mypyc and loads a hashed `<hash>__mypyc` extension that PyInstaller does not detect, so HGS, PSOMA and CVRP crashed in the executable with `ModuleNotFoundError`. The spec now adds every `*__mypyc` module of the build environment to the hidden imports.
- **Release build imports every shipped module** (`release_staging.import_sweep`): the static closure check missed module-level imports that resolve through a lazy registry (the `load_problem` bug above dropped 16 test files instead of failing the build). The build now imports each module in the release environment and fails on any error; it also removes the `__pycache__` the test run leaves across the tree, and no longer ships the bin-analysis `*_experiment` research script (it needs `statsmodels`; nothing imports it).
- **`load_problem` imports only the requested problem** (`utils/model/problem_factory.py`): it imported every problem class at module level, so in the release (which does not ship `envs/cptptwtc.py`) loading any model failed with `ModuleNotFoundError`, breaking NA and eval there.
- **The release ships only tests that pass in it** (`build_release.py --test-venv/--test-data`, `release_staging.py`, `release_closure.py`): the package carried 262 test files, but no `conftest.py`, pytest settings or test dependencies, and in the pruned tree 143 tests failed, 90 errored and 9 files did not collect. The build now stages a `conftest.py` that registers only the shipped fixture modules, copies the repository's `[tool.pytest.ini_options]`, adds a `test` extra (pytest, pytest-mock, hypothesis), installs the staged release into a venv and runs its tests there with the package data, dropping every file with a failure until a run is clean. The dropped files and their first failure are listed in `logic/test/release_tests.excluded.txt`. Fixture modules are now selected by their imports, not by module names in their strings; `integration_fixtures` patches the HPO handler only when it exists; the CLI end-to-end and no-torch tests run `__main__.py` when there is no `main.py`.
- **Release dependencies pinned to the tested versions** (`logic/package/release_files.py`): the release `pyproject.toml` only had lower bounds, so a fresh install resolved much newer libraries than main is tested on (pyvrp 0.14.1, pandas 3.0, numpy 2.5, OR-Tools 9.10 against 0.11.3, 2.3, 1.26 and 9.9). pyvrp 0.14 changed `Client`'s signature and broke the CVRP pyvrp engine. Each requirement is now pinned to the version installed where the release is built and tested; custom CUDA wheels (`+cu130`) keep their specifier.
- **`python . --help`** (`__main__.py`): `--help` was passed to Hydra as a task (`Could not find 'tasks/--help'`). `-h`, `--help` and `help` now print the usage, listing the tasks whose configs the tree ships (the release lists only its own), and exit 0.
- **Release ships only its tasks** (`logic/release.yaml` `tasks:`, `build_release.py`): the package carried the `hpo`, `hpo_sim`, `meta_train` and `slurm` task configs although their code is excluded, so those tasks crashed with import errors. The manifest now lists the released tasks (`test_sim`, `train`, `eval`, `gen_data`); `--check` validates them and the build drops the other task configs.
- **`gen_data` no longer overwrites by default** (`configs/tasks/gen_data.yaml`): its default graphs regenerate the paper's Figueira da Foz 350 Gamma-3 test datasets under their own names in `data/simulator/datasets/`, and `overwrite: true` replaced the shipped files silently. The default is now `false` (as in `DataConfig`); pass `data.overwrite=true` to replace a file.
- **Summary tables when output is redirected** (`tracking/logging/modules/summary_display.py`): Rich assumes 80 columns when stdout is not a terminal, so next to the long policy names every metric column of the end-of-run tables printed empty (in log files, pipes and the setup-script runs). Those tables now use 200 columns when not on a terminal.

### Release 1.0.2 package (`2026-10-22`)

- Version 1.0.2 in `pyproject.toml`, `logic/pyproject.toml` and `logic/release.yaml`. It is 1.0.1 plus the review-swarm fixes (Kimi, Grok, Cursor, Gemini, Codex, the Claude ultrareview and its follow-ups), the reproducible iteration budgets of the paper heuristics, and the `policies/vectorized/` layout. As in 1.0.1, the cloud runner (`logic.controllers.cloud`) is excluded; it ships with v1.1.0 (#120). The package is published as the branch `package/1.0.2`, without `data.zip` and the executables.

### Claude: reproducible routing budgets (`2026-10-22`)

#### Changed

- **The paper's heuristics now stop on an iteration budget, not the clock.** ALNS, HGS, SANS (`new`) and PG-CLNS all ended at their 60 s `time_limit`, so a run's result depended on machine speed and load (ALNS gave 93.5 to 109.6 km over the same 2 days in four runs of the same code). Each budget is about what the laptop reaches in 60 s: ALNS `max_iterations` 5000 → 4500; HGS a new `max_iterations` 650 (offspring generations); SANS a new `max_iterations` 1,800,000 (neighbour moves); PG-CLNS `max_iterations` 50 → 30. `time_limit` becomes a 600 s guard there and in ACO-HH, PSOMA, PG-CLNS's ACO/LNS stages and the `cls` improver, which already stopped on their iteration counts. **This changes the ALNS, HGS, SANS and PG-CLNS results; their paper numbers need a rerun.**
- **HGS** (`hybrid_genetic_search/hgs.py`, `HGSConfig.max_iterations`, 0 = no cap): the cap is checked next to the time limit; restarts are unchanged.
- **SANS** (`heuristics/sans.py`, `SANSConfig.max_iterations`, 0 = no cap): a total move budget for the `new` engine.

### Claude: HGS repeatability (`2026-10-22`)

#### Fixed

- **HGS diversity caches keyed by a serial number** (`hybrid_genetic_search/individual.py`, `evolution.py`, `hgs_population.py`): the broken-pairs distance cache was keyed by `id(individual)`. Python reuses the `id` of a garbage-collected individual, and individuals that leave the population outside `_trim_pop` were not evicted, so a new individual could read a stale distance. When that happened depended on garbage-collection timing, so the same seed could give different HGS routes (seen when an earlier test had run HGS in the same process). Each `Individual` now has a `serial` from a process-wide counter.

### Studio launch on a Snap-tainted desktop (`2026-10-22`)

#### Fixed

- **WSmart-Route Studio showed "WebKit encountered an internal error"** when the desktop session or a Snap-packaged IDE put `/snap/core20` on `LD_LIBRARY_PATH`. `WebKitNetworkProcess` loaded that `libpthread` and died on `__libc_pthread_init` (`GLIBC_PRIVATE`). On Linux, Studio now drops Snap entries from the library and GTK paths and clears an inherited `SNAP*` identity before GTK starts, unless the Studio binary itself is under `/snap`. After that scrub, a `GTK_MODULES` name with no host `lib*.so` is dropped. Host modules such as `colorreload-gtk-module` stay once Snap's `GTK_PATH` no longer hides them.
- **`ProcessEntry`** in `app/src-tauri/src/commands/process.rs` was never built. Spawned processes are reported with `ProcessSpawned`. The unused struct is removed, which clears the `dead_code` warning.

### Claude: vectorised policies in one package (`2026-10-22`)

#### Changed

- **`logic/src/policies/vectorized/`** now holds both batched policy families: `routing/` (formerly `logic/src/policies/vector/`) and `packing/` (formerly `logic/src/policies/packing/vectorized/`). Imports, lazy registries, the release manifest, tests, the API docs and the notebooks use the new paths; `from logic.src.policies.packing import LastMinuteSelector` (and the other re-exported selectors) still works. Hard break: the old module paths no longer import.

### Claude: ultrareview follow-ups (`2026-10-22`)

#### Fixed

- **`ProblemContext.from_kwargs`** (`interfaces/context/problem_context.py`): node `g`'s coordinates now come from row `g` of `coords`, as in `RoutingExecutionMixin.execute` (`1fbd41e`); the `ID` column holds real bin IDs, so the old lookup left every location at zero. `day_index` now falls back to the day context's `day`, so multi-period policies no longer always see day 0. Off-paper (multi-period policies only).
- **`import_billing_csv`** (`controllers/cloud/cost/billing.py`): a cost with comma thousands separators (`"1,234.56"`) is now read, and any other unparseable cost raises instead of being skipped, which under-reported the job's bill.
- **Explicit `None` values:** `CollectAction` no longer raises a `TypeError` on a TCPTP trip when `vehicle_capacity` is `None` (no capacity check, as for the payload report), and `BaseJointPolicy.execute` reads `n_vehicles=None` as one vehicle. `0` is still passed through unchanged. Neither is reached by a packaged config today.
- **Release builder** (`logic/package/build_release.py`): the bin-analysis runtime modules are staged as a real `logic/submodules/wsmart_bin_analysis/` directory instead of `WSmart-Bin-Analysis/` plus a `wsmart_bin_analysis` symlink. A symlink becomes a plain text file in a zip or a Windows checkout (the import then fails), and PyInstaller's `collect_submodules("logic")` skips the hyphenated directory. The submodule's Sphinx `docs/` are no longer copied.

### Claude ultrareview fixes (`2026-10-22`)

Fixes from the ultrareview of the 137 targeted functions. Apart from the two entries under "Fixed (moves paper results)", which Afonso approved on 2026-10-22, none of these changes moves a paper policy's results or the trained NA weights. The other owner-level findings are in FINDINGS.md §6.

#### Fixed (moves paper results)

- **SANS `new` engine revenue** (`configs/policies/sans.py`, `simulated_annealing_neighborhood_search/dispatcher.py::execute_new`): `SANSConfig` declared `V: float = 0.0`, and `_load_area_params` lets the policy config overwrite the area's bin volume, so `sans.new` loaded `stock · 0 · density / 100 = 0` per bin and searched with zero revenue (distance and the mandatory penalty only). With the area's volume restored the units would still be wrong: kg against a capacity in % and revenue in EUR per %. `execute_new` now uses the percent-unit identity `V · B = 100`, as `execute_og` already does with its identity factors, so a full bin loads 100 % and earns `R` per %. The unused `V` field is removed. **This changes `sans.new` routes and KPIs**; the `og` engines are unchanged. Covered by `test_sans_new_percent_units.py`.
- **Sub-problem coordinates for classical policies** (`policies/routing/base/routing_execution.py::RoutingExecutionMixin.execute`) were looked up in `coords["ID"]` using the global node index. In the simulator, `coords` holds the depot in row 0 and then the bins sorted by their real IDs (192, 211 … 22056 for Rio Maior), so on real data every lookup missed and `x_coords`/`y_coords` were `None`. As a result HGS's SWAP* polar-sector pruning was silently off, and ACO-KS raised. Node `g` is now read from row `g`, the same mapping `get_daily_results` uses for the logged tour IDs. **This changes HGS's search on real data** (routes stay feasible). Covered by `test_routing_coords_by_node_index.py`. `ProblemContext.from_kwargs` (multi-period, off-paper) has the same lookup and is still open in FINDINGS §6.

#### Fixed

- **`AttentionModel._load_from_state_dict`** (`models/core/attention_model/model.py`) walked every BatchNorm buffer key from the model itself with the full key, prefix included. PyTorch passes each module the whole state dict, so loaded nested (a Lightning checkpoint, `policy.` prefix) every `running_mean`/`running_var`/`num_batches_tracked` was dropped, the policy's and its siblings' alike, and strict loading failed. Only keys under `prefix` are now considered, walked relative to it. Dormant for the paper model (`track_stats` is off, so it has no such buffers). Covered by `test_am_nested_bn_buffers.py`.
- **`NeuralAgent.compute_batch_sim`** (`policies/routing/learning_algorithms/neural_agent/batch.py`): `out.get("actions") or out.get("pi")` raised "Boolean value of Tensor … is ambiguous" for any dict output with a multi-element action tensor (the AttentionModel's own output); the attention hooks it registers were never removed, so each call added another set to the encoder; and a per-instance `B×N×N` distance matrix was indexed as if shared (`dist[src, dst]`, and instance 0's depot rows for every instance). The fallback is now an `is None` check, the hooks are removed in a `finally`, and 3-D matrices are indexed per instance. Shared-matrix `km` is unchanged. Only tests call this path; the simulator's NA uses `run_day`. Covered by `test_batch_sim_review.py`.
- **`plateau` LR scheduler** (`pipeline/learning/common/base/optimization.py`) was returned as a bare `lr_scheduler`, which Lightning rejects for `ReduceLROnPlateau` ("must include a monitor"), so selecting it crashed at fit start; its default `mode="min"` would also have cut the LR while the reward was still rising. It now returns `{"scheduler", "monitor"}`, monitoring `lr_scheduler_kwargs["monitor"]` (default `val/reward`) with `mode="max"` for reward metrics unless a mode is given. Other schedulers are unchanged; no paper config selects `plateau`. Covered by `test_plateau_scheduler.py`.
- **`Gamma._sample_tensor`** (`data/distributions/statistical_gamma.py`) built a fresh `torch.Generator()` when none was passed. A new generator always has the same default seed, so every unseeded draw of a per-node gamma option returned the same sample. It now uses `torch.default_generator`, so unseeded draws follow `torch.manual_seed`. Seeded calls are unchanged, and the simulator and training generators take the numpy `sample_array` path, so the paper data is unaffected. Covered by `test_gamma_tensor_rng.py`.
- **`regular` packing selector** (`policies/packing/spec.py`, `policies/packing/vectorized/spec.py`): a frequency strictly between 0 and 1 made `int(period)` zero, so the day test divided by zero. A period below one day now means every day, the same as a non-positive one, in both the scalar and the vectorized rule. Whole-day periods are unchanged, and the paper configs use whole-day periods. Covered by `test_regular_subdaily_period.py`.

#### Optimizations

- **`get_route_cost`** (`utils/routing/tours.py`) copied the full N×N distance matrix on every call (it runs at least twice per policy per day in collection and route improvement). It now reads the matrix with `np.asarray`; fancy indexing already copies the edges it needs, so results are bit-identical. Microbenchmark (one third of the nodes in the tour): N=101 12.5→9.9 µs, N=318 60.7→11.1 µs, N=1001 936→24 µs per call. Covered by `test_route_cost_no_copy.py`.
- **`two_opt_steepest`** (`policies/helpers/operators/improvement_descent/steepest_two_opt.py`, the `steepest_two_opt` route improver) re-ran an O(N²) `np.allclose(d, d.T)` symmetry check inside `_find_best_2opt` on every descent iteration. It is now checked once per call and passed in; `_find_best_2opt` still checks when called alone. Routes are identical (tested against the per-iteration loop on symmetric and asymmetric matrices). Three 30-bin routes to a 2-opt minimum: N=101 140→112 ms, N=1001 2.16 s→140 ms. Covered by `test_steepest_two_opt_symmetry_once.py`.
- **`CostMixin._check_budget`** (`controllers/cloud/cost/budget.py`): with no cost estimate and a monthly cap, it called `month_committed` twice, once for the test and once for the message. Each call lists every job, task and event of the month, and nothing is memoized outside a scheduling pass (`submit` is outside one). It now reads the value once; refusals and messages are unchanged. Covered by `test_budget_month_committed_once.py`.

### Review-swarm follow-up fixes (`2026-10-21`)

#### Fixed

- **MACO empty mandatory set** — `None` mandatory nodes reached the subproblem builder; now falls back to `[]` (Lucas, `95717fa`).
- **HVPL `_random_construction` docstring** (`policies/routing/meta_heuristics/hybrid_volleyball_premier_league/solver.py`) claimed a random construction but the body calls `build_greedy_routes`; the docstring and the ACO-initialization comment now say greedy. The method name is kept for compatibility.
- **ARCO `best_sequence`** (`policies/routing/other_algorithms/adaptive_routing_algorithm_orchestrator/`) temporarily overwrote `params.selection_strategy` and `params.epsilon` and restored them without `try/finally` — not re-entrant, and an exception mid-selection would leak the override. `_select_sequence`/`_apply_strategy` now take optional `strategy`/`epsilon` overrides and `best_sequence` passes `greedy` explicitly, leaving `params` untouched.
- **ARCO `execute` docstring** carried a duplicated trailing `Args: kwargs` block; removed.

#### Changed

- **Collection frequency/level** (`pipeline/simulations/bins/`): new `calculate_frequency_and_level_vectorized` evaluates all bins in one batched gamma-quantile pass with semantics identical to the scalar loop (frequency = first n in 1..49 with quantile > 100, else 49; level = 100 − quantile at n = 1). `set_collection_level_and_freq` calls it once instead of looping per bin. The scalar function is kept.
- Covered by `test_frequency_vectorization.py` (vectorized vs scalar, exact match).

#### Verified

- Vectorized vs scalar outputs identical on 10,000 randomized (ui, vi) cases including zeros and extreme ratios (cf ∈ {0.5, 0.9, 0.99}).
- Merged as PR #142 on 2026-10-22: full suite 2,788 passed, 9 skipped; 2-day Rio Maior 100 parity of the paper policies plus NA against main, 14 of 14 logs identical.

### Codex: atomic telemetry schema initialization (`2026-10-21`)

- Initialize SQLite schema metadata with an atomic insert that ignores only an existing version key, preserving its value. Independent simulator processes can now initialize the same fresh telemetry database without a unique-key error dropping a policy snapshot. A two-process regression forces the original race and verifies both snapshots are persisted.

### Codex: respect per-instance evaluation days (`2026-10-21`)

- Select each row's own temporal waste slice in both environment reset and mandatory packing. Missing days retain day zero; out-of-range days clamp to the available series. Regression tests cover mixed, reordered, uniform and clamped days across both consumers. Existing single-day/uniform-day paper inputs and checkpoints are unchanged; future affected training requires owner review.

### Codex: preserve zero validation objective weights (`2026-10-21`)

- Keep explicitly zero cost/revenue weights when constructing validation environments; only missing or null weights receive the unit default. Tests cover per-eval and training-graph fallback objectives. Existing nonzero paper objectives and checkpoints are unchanged; affected future training needs owner review.

### Codex: preserve Hydra model constructor overrides (`2026-10-21`)

- Forward configured top-level model values rather than DictConfig internals, preserving temporal horizons and MoE overrides. Regression tests cover Hydra and object configurations. Existing checkpoints are unchanged; changes to future affected training require owner review.

### Codex explicit-half decoding likelihoods (2026-10-21)

- Compute explicit fp16/bfloat16 action log probabilities and entropy in float32 log space, preserving zero masked mass and avoiding large additive-epsilon gradient bias. Float32 paths retain their prior behavior; this does not retrain checkpoints.

### Codex beam identity and cache compatibility (2026-10-21)

- Preserve original instance IDs through beam slicing and environment steps, align all decoder cache tensors by those IDs, and support unequal and reordered beam widths with Glimpse/GAT decoders.
- Restore the public beam precomputation and terminal result contracts; simulator beam backtracking now handles fewer feasible candidates than the requested width. Greedy decoding and checkpoint parameter shapes are unchanged.
- Check returned public beam costs independently against route geometry and unique collections; exercise reward-aware simulator scoring in the ragged-beam regression.

### Codex CPTP checkpoint compatibility (2026-10-21)

- Restore coordinate Euclidean distances in the learned CPTP step-context feature. Road objective matrices in kilometers no longer change inputs to existing checkpoints.

### Codex resource-history compatibility (2026-10-21)

- Preserve the latest cloud resource sample when more than 20 progress, day, or cost records follow it; bounded reads fall back to history when necessary. Covered for SQLite and document control planes.

### Codex rollout cache compatibility follow-up (`2026-10-21`)

- Restrict skipped rollout precomputation to the built-in TensorDictDataset. Custom datasets retain iteration and RNG side effects, including stochastic item transforms; regression checks compare their cached rewards and RNG states.

### Codex rollout baseline precomputation (`2026-10-21`)

- Skip the unused epoch-wide rollout cache in Lightning for the built-in Attention Model with PTP/CPTP data. Keep per-batch packed baseline evaluation and the original DataLoader seed draw; custom contracts retain explicit precomputation. On CPU, a 64-instance/100-customer small Attention Model avoids about 0.24 seconds per epoch preparation. Regression checks compare sampling RNG, loss and gradients with and without mandatory packing. Existing checkpoint weights are unchanged.

### Codex adaptive imitation resume (`2026-10-21`)

- Preserve imitation weight, stop flag, patience count and best validation reward in Lightning checkpoints. Legacy checkpoints retain constructor defaults. This corrects future resumed training and requires owner review; existing paper weights are unchanged.

### Gemini swarm review fixes across Lanes A, B, C, D (`2026-10-20`)

#### Fixed

- **Lane A: Attention Models & Decoding.**
  - `embeddings/context`: Fixed context embedders (`cptp`, `generic`, `ptp`, `base`) crashing on multi-instance beam search by deriving `batch_size = current_node.size(0)` and dynamically broadcasting embeddings across beams (A1).
  - `models/core/attention_model/policy.py`: Prevented finished instances from accumulating log-probabilities and entropy indefinitely during decoding steps via active instance masking (`active.float()`) (A2); added bounds check preventing IndexError in teacher-forcing when trajectories exceed action length (A3); safeguarded log-probability computations against float16 numerical epsilon underflow (`eps = 1e-4` under `torch.float16`) (A5).
  - `utils/decoding/sampling.py`: Prevented masked probabilities from clamping to positive values by zeroing invalid actions and computing exact entropy over non-zero actions (A4, A11).
  - `utils/decoding/greedy.py`: Applied float16-safe numerical epsilon (`1e-4`) under `torch.float16` preventing `-inf` log-probabilities (A5).
  - `utils/model/loader.py`: Preserved `decoder_type`, `decoder`, and `decoder_normalization` in both parser branches of `_parse_hydra_config` (A6).
  - `pipeline/learning/envs/base/problem.py`: Unpacked `(parent, action, score)` triplet from `propose_expansions` in `BaseProblem.beam_search` (A7).
  - `models/subnets/modules/normalization.py` & `configs/models/normalization.py`: Changed `BatchNorm1d` default `track_stats` / `track_running_stats` to `True` so singleton inputs in `eval()` do not raise `ValueError` (A8).
  - `decoders/glimpse/decoder_logits.py` & `attention.py`: Forwarded `mask_inner` flag to `one_to_many_logits` for proper internal attention masking (A9).
  - `embeddings/context/cptp.py`: Precomputed distance matrix lookup replacing on-the-fly euclidean distance recalculation (A10: 3.72x speedup).

- **Lane B: Environments & Rewards.**
  - `pipeline/learning/envs/tcptp.py`: Enabled `explicit_stop: bool = True` and updated `_check_done` to terminate when finished or customer actions are exhausted (B12).
  - `pipeline/learning/common/base/data.py`: Read from `_waste_full` to restore full 3D tensor across multi-day evaluation replication (B13).
  - `data/distributions/statistical_gamma.py`: Normalized `_sample_tensor` by `/ 100.0` matching `_sample_array` distribution scale (B14).
  - `pipeline/features/eval/eval_loop.py`: Checked `"dist_matrix"`, `"dm"`, and `"dist"` keys for distance matrix input, and scaled reported `"kg"` by `bin_capacity_kg` (default 47.5 kg) (B15).
  - `pipeline/learning/envs/ptp.py`: Initialized `td["max_waste"] = torch.ones(bs)` in `_generate` (B16).
  - `pipeline/learning/envs/op.py`: Prepended depot 0 to `prize_all` whenever indexing customer nodes in partial tours (B17).
  - `pipeline/learning/envs/pctsp.py`: Prepended depot 0 to `penalty` whenever indexing customer nodes in partial tours (B18).
  - `policies/routing/learning_algorithms/neural_agent/simulation.py`: Scaled incremental reward waste term by `waste_weight` (default 27.74 €/bin) in `compute_simulator_day` (B19).
  - `pipeline/learning/envs/base/problem.py`: Batched `dist_matrix[batch_indices]` and preserved intermediate depot legs in `get_tour_length` (B20).
  - `pipeline/learning/envs/base/routing.py`: Vectorized road distance step indexing via tensor slicing (B21: 73.41x speedup).

- **Lane C: Cloud Money & Scheduling.**
  - `controllers/cloud/service.py`: Coordinator accrues costs for pool jobs on each scheduling tick (C22); decoupled pool worker local slot availability from global parallelism cap in `_tick_job` (C23).
  - `controllers/cloud/service_status.py`: Preserved attempt interval durations and costs across retried tasks (C24).
  - `controllers/cloud/scheduler/mixin.py`: Cancelled backend task on placing lease expiry in `_place_and_start` (C25); enforced monthly budget cap in `_parallelism` auto-scaling against accrued monthly spend (C27).
  - `controllers/cloud/cost/budget.py`: Enforced monthly budget cap check for unestimated remote jobs (C26); exempted zero-cost local jobs from monthly budget cap refusals (C29); memoized `month_committed` calculations within each scheduling pass via `_month_memo` (C33: 74.90x speedup).
  - `controllers/cloud/cost/billing.py`: Validated currency is `"USD"` before storing billing export totals (C28).
  - `controllers/cloud/prediction/features.py`: Supported unbracketed `sim.policies`, `sim.days=N`, and fallback policy parsing (C30).
  - `controllers/cloud/prediction/model.py` & `features.py`: Mapped `na.amgat`, `amgat`, and `na` to `am` for runtime prediction (C31).
  - `controllers/cloud/prediction/history.py`: Cached static fitted predictor `RuntimePredictor(load_table())` via `get_base_predictor()` (C34: 105.64x speedup).

- **Lane D: Cloud State, Retry & Results.**
  - `controllers/cloud/telemetry.py`: Supported `sim.graph.n_days=N`, `sim.days=N`, and `days=N` in `days_from_argv` (D35).
  - `controllers/cloud/service.py`: Requeued retried tasks directly to `TaskState.QUEUED` with event emission, eliminating stuck `RETRYING` transitions (D36); partitioned multi-cell job downloads under `dest / cell_id` when paths do not already contain cell_id (D39); skipped redundant `record_results` on remote tasks in `_poll` (D43: 5.62x speedup).
  - `controllers/cloud/providers/azure/batch.py`: Appended `-a{attempt}` to Azure Batch `task_id` for distinct retry tracking (D38).
  - `controllers/cloud/runtime.py`: Filtered deleted local files from `OutputMirror.index()` (D40).
  - `controllers/cloud/cli.py`: Acquired `worker_lock` during `cloud submit --wait` to prevent scheduler race conditions (D41).
  - `controllers/cloud/service_status.py`: Optimized `_latest_resource` via `get_metrics(limit=20, reverse=True)` (D42: 7.92x speedup).

#### Optimizations (Performance Deltas)

- **A10 (CPTP distance lookup)**: 119.96 ms -> 32.22 ms (3.72x speedup).
- **A11 (Sampling entropy & multinomial)**: 200.58 ms -> 240.69 ms (exact entropy calculation without illegal probability leakage).
- **B21 (Routing vectorized DM indexing)**: 1108.92 ms -> 15.11 ms (73.41x speedup).
- **C33 (Budget month committed memoization)**: 61.75 ms -> 0.82 ms (74.90x speedup).
- **C34 (Fitted predictor static cache)**: 111,798.05 ms -> 1058.33 ms (105.64x speedup).
- **D42 (Latest resource query limit)**: 26.23 ms -> 3.31 ms (7.92x speedup).
- **D43 (Remote record_results bypass)**: 12.05 ms -> 2.14 ms (5.62x speedup).

#### Verified

- Dedicated regression unit tests added in `logic/test/unit/models/test_review_fixes_gemini_lane_a.py`, `logic/test/unit/pipeline/test_review_fixes_gemini_lane_b.py`, `logic/test/unit/controllers/cloud/test_review_fixes_gemini_lane_c.py`, and `logic/test/unit/controllers/cloud/test_review_fixes_gemini_lane_d.py` (33/33 passed).
- Full regression unit test suite passes cleanly with PyTorch under isolated execution (238 passed in models, 632 passed in pipeline, 200 passed in cloud; 0 failures).
- 2-day Rio Maior 100 simulation parity verified against `main` for standard paper policies (`alns`, `hgs`, `aco_hh`, `sans`, `psoma`, `pg_clns`, `bp`): all 13 logs identical to `main` in kg, km, kg lost, overflows, and profit (0.000 unintended deviation).

### Stage-plumbing fixes from the Cursor review of PR #140 (`2026-10-20`)

#### Fixed

- **Packing.** `submodular_greedy` and `supermodular_greedy` no longer crash on a tuple gain. Fractional knapsack's 1/2 fallback compares net profit, the objective it packs by. `combined` passes each child its parameters. `PackingAction` takes `cost_per_km` from `bins.expenses`, reads the day's `coords`, and builds a scenario tree for `mip_knapsack`/`fptas_knapsack`. Whittle's cardinality ignores empty bins. The Thompson dispatcher draws from `PackingContext.seed`, and `CollectAction` records the day's profit on its posterior. Deadline compares the unfloored days to overflow. Rollout's Monte Carlo follows `PackingContext.seed`. None of the paper's packers (`last_minute`, `service_level`, `lookahead`, `regular`) changed.
- **Route improvement and acceptance.** Simulated annealing and ruin-recreate (`lns_acceptance: sa`) pass negated tour length to the maximising Boltzmann rule. Steepest 2-opt prices interior arcs on a directed matrix. The relocate sweeps try the slot after the depot. SWAP* prices every slot on a Top-3 cache miss. Tabu search forces the best rejected neighbour, not the first. The ten packaged acceptance files (`ac_gd`, `ac_rrt`, `ac_nlgd`, `ac_ta`, `ac_lahc`, `ac_schc`, `ac_oba`, `ac_svns`, `ac_gt_sa`, `ac_ema`) build the criterion they name. `RouteImprovementAction` rejects a refined tour that visits the same stops and is longer; tours that add bins or depot returns are kept as before.
- **Routing base.** `ProblemContext` reads the fill rates `Bins` stores (`means`/`std`) and the policy's horizon; multi-period policies pass their configured horizon, and the routing scenario tree reads `horizon`/`num_days`. `jgo` and `jsa` are in the routing manifest, and the joint base unpacks the area-parameter tuple, so `nds_brkga`, `jgo` and `jsa` run.
- **Config and CLI.** `meta_train`, `hpo_sim` and `tasks=slurm` compose. `sim.days`, `sim.n_samples` and the root `seed` reach the simulator (defaults unchanged: 30 days, 1 sample, seed 42); `eval.seed` and `data.seed` follow `seed`. `file_system delete` selects nothing by default and can select simulator outputs. `update_ms`/`update_ri` rewrite block-style policy YAML. `test_suite` points at `logic/test`.

Tests: `test_review_fixes_2026_10_20.py` under `logic/test/unit/policies/{packing,route_improvement,routing}` and `logic/test/unit/configs`.

Verified: full suite with torch, 2,698 passed and 9 skipped. The paper policies over 2 days of Rio Maior 100 give 13 of 13 logs identical to `main`.

### Simulator fixes from the Kimi review of PR #138 (`2026-10-20`)

#### Fixed

- **Days to overflow** (`Bins.predict_days_to_overflow`) passed the fill-rate standard deviation where `predict_days_to_overflow` expects the variance. Its other caller already passed the variance. Nothing in the simulator calls the `Bins` method today.
- **TCPTP trip check** (`CollectAction`) summed the observed fill `bins.c`, while the capacity report used the true fill `bins.real_c`. It now checks the true fill (`test_capacity_violation_does_not_mutate_bins` now has an observed fill that fits and a true fill that doesn't). A zero vehicle capacity no longer reads as unlimited in the payload telemetry.
- **Observed fill from a noisy-fill file** is clipped at 0 as well as at 100, as the noise-variance branch already was.
- **Packing economics** (`PackingAction`). The `PackingContext` capacity, revenue, density and bin volume came from context attributes that don't exist, so the defaults always applied: a 2.5 kg bin at €1/kg and a 100 kg vehicle. They now come from the bins and the day context, with the vehicle capacity converted from percent of a bin to kg, which is what the knapsack rules compare it with. The paper's four packing rules don't read these fields. Covered by `test_packing_context_gets_the_area_economics_in_kg`.
- **Rollout packing** tested the `(selected, context)` tuple returned by `pack()`, which is always truthy, so every simulated day collected. It now tests the selection.
- **Sample aggregation** (`FinishingState`) adds `time_spent` only when every daily record has it. One record without it raised `KeyError`.
- **`values` passed to the solvers** carry the scaled capacity and revenue under `capacity` and `revenue` too, not the raw config values. No solver reads those keys today.
- **Unknown area or waste type** in `get_area_params` raises `ValueError`, not an `assert` that `python -O` removes. The docstring units are corrected to m³ and kg/m³.

#### Changed

- `predict_days_to_overflow` computes its loop-invariant terms once.
- `get_area_params` raises `ValueError` for an unknown area or waste type; its tests expect that now.

#### Not taken from the PR

- The empty-mandatory short-circuit in `BaseMultiPeriodRoutingPolicy.execute`. Multi-period planners (CP-SAT, LBBD, PH, ST-EF) are usually given no mandatory set and choose the visits themselves, so it stopped them from ever routing; their tests caught it. Ruling D3 is about single-day policies.

#### Verified

- Full test suite: 2,638 passed, 0 failed.
- A 2-day Rio Maior 100 run of ALNS, HGS, ACO-HH, SANS, PSOMA, PG-CLNS and BP: all 13 logs identical to `main` in kg, km, kg lost, overflows and profit.

### Hybrid policies ran the wrong engine (`2026-10-20`)

#### Fixed

- **Routing engine resolution** (`pipeline/simulations/actions/routing.py`). When a policy's config named no engine, the routing action took the first config key that matched a registered engine, in set order, before it looked at the policy name. Hybrids nest their components' sections at the top level, so they ran a component instead:
  - HVPL, HMLNS and RL-HVPL ran ALNS, with their own `alns` section;
  - HGS-ALNS ran ALNS;
  - CALM ran TPKS.

  The engine named in the policy name now comes first, and a config key is a deterministic fallback. All 87 other policy YAMLs resolve as before; none of the affected policies is in the paper or the 1.0 release. Covered by `test_routing_engine_resolution_prefers_the_policy_name`.

#### Removed

- `policy_hms.yaml` and `policy_rl_ahvpl.yaml`, configs for policies that no longer exist. They only ever ran ALNS.

#### Known issues

Running for the first time, four of the five hybrids crash on a 2-day Rio Maior 100 run (HVPL completes):
- HMLNS and RL-HVPL: `ALNSConfig` has no `max_removal_pct`.
- HGS-ALNS: `_select_parents()` is called without `pop_infeasible`.
- CALM: `float()` is given a dict.

These are listed for the ultra review.

### Compatibility shims removed (`2026-10-20`)

#### Removed

- **`GlobalRegistry`** (`logic/src/enums/registry.py`), a wrapper that forwarded to `ALL`. The 131 `@GlobalRegistry.register(...)` decorators now call `@ALL.register(...)` (`logic.src.utils.registry`).
- **`LegacyCriticNetwork`** (`models/common/critic_network/model.py`) and its tests.
- **`AttentionModel.context_embedder`**, a proxy for `init_embedding`, along with its adapter class. Checkpoints saved with `context_embedder.*` keys still load, because the loader's key remap stays. `PTPEmbedding.init_node_embeddings`, an alias of `forward`, is removed too.
- **Flattened parameters in HVPL and HMLNS** (`alns_iterations`, `aco_n_ants`, ...). These branches never ran, because the policy values are plain dicts and the nested `aco`/`maco`/`alns` sections are always mappings.
- **Explicit solver arguments**:
  - `BranchAndPriceSolver` takes `params: BPConfig` (default `BPConfig()`) instead of 18 keyword arguments.
  - `BranchAndCutSolver` no longer takes `**kwargs`.
- **Other aliases**:
  - `HybridMemeticLargeNeighborhoodSearchParams.max_iterations`; the solver reads `max_generations`.
  - `apply_lns` parsing of `regret_<k>[_profit]` names; use `repair_op="regret"` with `repair_k`.
  - The no-op `decay_weights` of the HGS-RR operator manager.
  - The `horizon_days` lookup in deadline packing; the context has no such field, so the threshold was always used.

#### Fixed

- **Branch-and-Price overwrote its configuration.** With `params` given, the constructor reset `branching_strategy` and `tree_search_strategy` to its argument defaults (`edge`, `best_first`), so configured values were ignored. It now reads them from `params`, and `use_ryan_foster_branching` selects Ryan-Foster.
- **The multi-day path dropped the fleet size.** It passed `vehicle_limit` next to `params`, where it was ignored. A non-zero `n_vehicles` now sets `params.vehicle_limit`.
- Both fixes leave the default configuration's behaviour unchanged.

#### Kept

- These are live paths, not shims; misleading "legacy" comments on them were reworded:
  - the branch-and-bound tree's explicit arguments, which Branch-and-Price uses;
  - `calculate_reward`, used by RL-ALNS;
  - the base routing policy's untyped-config path;
  - POPMUSIC's `greedy`/`nearest_neighbor` initial solvers, which are selectable options.

#### Verified

- Full suite 2,605 passed (the shim-only tests are gone); the release build, closure verification and smoke pass. A 2-day Rio Maior 100 simulation of ALNS, HGS, ACO-HH, SANS, PSOMA, PG-CLNS, HVPL and BP gives identical kg, km, overflows and profit to main for all 14 policy logs. HMLNS crashes on both (see the next entry).

### Unused functions, classes and methods removed (`2026-10-20`)

#### Added

- **`logic/validation/check_unused_symbols.py`**: runs vulture over the repository (tests count as users) and sorts each unused function, class, method or property in production code by evidence of use:
  - `STRING`: a token in YAML, JSON, TOML, TS, Rust or notebook files, or in a Python string literal;
  - `DECORATED`: has registration decorators;
  - `HOOK`: torch, TorchRL, Lightning, argparse or `ast` callbacks, or a dunder;
  - `OVERRIDE`: a method of a class with an external ancestor;
  - `FSTRING`: matches an f-string prefix or suffix (`getattr(self, f"_try_{move}")`);
  - `CANDIDATE`: no evidence. `--remove` deletes the `CANDIDATE` definitions, with their decorators, and their module-docstring entries.

#### Removed

- **116 unused definitions in 69 files (2,522 lines)**: 24 functions, 2 classes (`KoptTopologyFactory`, `PolicyOutput`), 81 methods and 9 properties with no reference anywhere. They include unused local-search move variants (`_try_cross_exchange`, `three_opt_star`, `four_opt_star`, ...), statistics getters, unused interface methods and config property aliases (`RLHVPLConfig.qlearning_*`).
- The Azure provider's three candidates are left for its lane (CR-07). Two names used only in notebooks (`set_daily_waste`, `to_dataframe`) stay.

#### Verified

- Full suite 2,608 passed; the release build, closure verification and smoke pass. A 2-day Rio Maior 100 simulation of ALNS, HGS, ACO-HH, SANS, PSOMA and PG-CLNS gives identical kg, km and overflows to main for all 12 policy logs.

### Compatibility re-exports removed (`2026-10-20`)

#### Added

- **`logic/validation/check_reexports.py`**: finds re-exports, meaning names a non-`__init__` module imports but never uses itself. For each it follows the import chain to the defining module and lists the importers (module-level or deferred, absolute or relative), the string references (`mock.patch` targets, YAML) and the module-attribute uses.
  - `--fix` rewrites the importers to the defining module and removes the re-export from the module and its `__all__`.
  - Modules whose namespace is read dynamically (`getmembers`, `vars`, `dir`, `__dict__`, `getattr`, `import_module`), targets an f-string may build (`envs.{name}:{Prefix}Generator`), attribute uses and string references are reported for review by hand instead (`--ignore-dynamic`, `--no-strings`).
- `vulture` in the `dev` dependency group (for the unused-symbol pass).

#### Removed

- **260 re-exports** (248 and 6 by `--fix`, 6 by hand), including the facades the file-size refactor left so old import paths kept working; 186 files changed, importers now import from the defining modules.
- **Modules that were only re-exports**: `configs/policies/other/packing.py`, `tracking/logging/log_utils.py`, `policies/vector/local_search.py`, `pheromone_guided_cooperative_large_neighborhood_search/operators/{repair,destroy,exchange,move,route}_operators.py` and `perturbation_shaking/sans/{move,swap}.py`.
- `policies/helpers/learning/agents/bandits.py`: dead, shadowed by the `agents/bandits/` package of the same name, which every import already resolved to.
- Test-only re-exports (`ms_bpc_sp_engine.GRB` and `_column_generation_loop`, `build_release.LAZY_LINE_RE` and `lazy_targets`): the tests use the defining modules.

#### Fixed

- **No-op patches deleted**: `mock.patch` targets on names the module never used (`initializing.ROOT_DIR`, `node_selection.ROOT_DIR`, `route_improvement.ROOT_DIR`, `simulator.update_policy_log_section`) changed nothing.
- `test_sequential_loop_does_not_write_mean` patched `simulator.update_policy_log_section`, which `simulator.py` never calls, so its assertion was vacuous. It now patches the mean writer's module, `simulator_progress`, and still passes.
- The env and generator targets for `irp` and `thop` and their `ProblemSpec.env_target` point at `irp_env`, `irp_generator`, `thop_env` and `thop_generator`; the f-string targets had relied on the re-exports.
- `generate_packing_templates` reads the three modules that define the packing configs (the same 34 configs).

Full suite: 2,608 passed, no failures; release build, closure verification and smoke pass.

### Remaining test failures and Launcher scheduling fields (`2026-10-20`)

#### Fixed

- `test_integration.py::test_sim_sequential_basic` and `::test_sim_policy_neural_mock`: their mock routing adapters returned the old 3-tuple; they now return the routing contract `(tour, cost, profit, search_context, multi_day_context)`.
- `test_hrl_manager.py::test_gatlst_manager_init` checked `MandatoryManager.hidden_dim`, the compatibility alias removed in `e33884f49`; it now checks `embed_dim` (`hidden_dim` remains the legacy constructor argument).

#### Added

- **Launcher**: priority, placement targets (one per line) and warm pool inputs (`PlacementFields`) for the CR-09 scheduler. They are added to the CLI preview and passed by `cloud_submit` (`--priority`, `--target`, `--pool`).

### Cloud runner CR-17: docs, release extra, CI gate and close-out fixes (`2026-10-20`)

#### Added

- **`docs/CLOUD_RUNNER_GUIDE.md`** (#137): profiles, credentials and secrets, launching from the CLI and the Studio, following jobs, placement and the warm pool, costs and limits, prediction, and testing with fakes. It states that nothing is validated against a real API yet.
- **`--profile NAME`** on every cloud command (`profile_service.py`). It builds the service from a profile: control plane, store, and the provider's backends for every compute kind plus local processes.
  - Constructor arguments are filtered by each backend's signature. The new profile field `backend_options` passes non-secret provider settings, and keys that look like secrets are rejected.
  - `submit` defaults to the profile's provider, compute kind, region, image and Gurobi secret. The spawned worker inherits the profile.
- **CI**: `test-cloud` runs the cloud suites with and without the `[cloud]` extra and uploads JUnit reports; `release-smoke` builds the release tree and runs its smoke.
- **Release**: provider backends, stores, price tables and `cloud.testing` are seeded in `logic/release.yaml`. They are reached only through strings, so the release had pruned `providers/` completely. The smoke now checks that the cloud runner imports every provider and runs a local job with no cloud SDK. With `[cloud]`, the release resolves.

#### Fixed

- **Task runtime crash**: `OutputMirror.sync` failed with `FileNotFoundError` when a file was renamed between the scan and the upload (the simulator replacing `checkpoints/day.json.tmp`). The runtime exited 1 mid-run. It now skips the file and uploads its replacement on the next sync. This was the 'flaky' `test_preempted_task_resumes_on_another_machine_with_identical_results`: 1 failure in 20 loaded runs before, 0 in 30 after. Regression test: `test_mirror_skips_a_file_renamed_during_its_upload`.
- **Release build**: closure verification failed on main after the bin-analysis submodule moved to `logic/submodules/WSmart-Bin-Analysis`. The staged module scan does not follow the `wsmart_bin_analysis` symlink, so the imported name was missing. Its modules are now registered under both names (`staged_module_names`).
- **Stuck cancels**: a cancelled attempt that its backend no longer knows (deleted Batch jobs or VMs after a coordinator restart) polled `UNKNOWN` forever. A cancel-requested task with an `UNKNOWN` attempt is now cancelled.

### Cloud runner CR-05: GCP backends (`2026-10-20`)

#### Added

- **`logic/controllers/cloud/providers/gcp/`** (#125). Tested against in-process fakes only; no real API call.
  - **`GcpBatchBackend`**: one Batch job per attempt, with `task_count` 1 and no Batch retries (the scheduler retries with resume). Attempts are not grouped into task arrays: the runner's unit of retry and accounting is the task. Spot runs use `provisioning_model: SPOT`, and preemption (exit code 50001) maps to `PREEMPTED`. `secret_env` becomes `secret_variables` that reference Secret Manager versions, and `max_run_duration` follows `max_wall_seconds`.
  - **`CloudRunJobsBackend`**: one job and one execution per attempt. Launches beyond Cloud Run's 24 h, 8 vCPU or 32 GiB limits raise `CapacityError`, so the scheduler places the task elsewhere. Secrets are Secret Manager env sources, there is no spot, and the job is deleted once its execution ends.
  - **`ComputeEngineBackend`**: self-deleting Container-Optimized OS VMs. The startup script takes a metadata-server token, runs the image, uploads the exit code to the GCS status store and deletes its own instance. Secrets reach the runtime as `WSR_SECRET_REFS` `gcpsm://` references, never as values in metadata. Polling reads the exit object even after the VM is gone or the coordinator has restarted (the ids are in the handle). A spot VM gone without an exit code counts as preempted.
  - **Supporting pieces**: `GcsObjectStore` (`gs://` profiles), `GcpSecretManagerStore`, `ArtifactRegistry` (pushes through Docker, resolves through the API when a client is given), price hints (`pricing.py`: list prices for e2, n2, c2 and Cloud Run sizes, with regional factors) and machine capabilities.
  - **Shared behaviour**: quota and stock-out errors become `CapacityError`. Cancelled Batch jobs and VMs (stopped by deletion) report `FAILED` (cancelled), not `UNKNOWN`. Logs come from the runtime's uploaded attempt log, or the VM serial port.
- **Shared fixtures**: `testing.contract_launch(outcome)` and `testing.outcome_of(env)`. The scripted outcome travels in `WSR_FAKE_OUTCOME`, which any provider's fake can read from its request.
- Tests: `test_gcp_contracts.py` (11) with fakes in `gcp_fakes.py`. All three backends pass the CR-01 backend contract, GCS the object-store contract and Secret Manager the secret-store contract.

#### Changed

- `test_profiles_resolve_local_uris_and_name_missing_providers`: `gs://` now resolves, so the missing-provider message is checked with a provider that does not exist.

### Cloud runner CR-09: scheduler (`2026-10-20`)

#### Added

- **`logic/controllers/cloud/scheduler/`** (#129). Retries with resume, wall-time limits, idle kill and cancellation already existed; this package adds:
  - **Priorities**: `JobSpec.priority`. Jobs are served highest first, then oldest, so they take parallelism and quota first.
  - **Cross-provider placement**: `JobSpec.placement` lists `PlacementTarget`s (provider, compute, region, machine type, spot, `max_tasks` quota across jobs). Each start ranks the eligible targets by expected cost (CR-11 p50 × CR-10 price; spot inflated by 1 + 2 × its recent preemption rate), then by tasks already starting there, then by list order.
  - **Spill-over**: a capacity error or launch failure moves the task to the next target in the same pass. A per-target circuit breaker (3 consecutive failures, 300 s cooldown, then half-open) keeps a failing target out. The chosen target is stored per attempt (`TaskRecord.placement`) and drives polling, cancelling, logs, collection and pricing. A task that every target refuses fails with `launch failed: ...`; otherwise it waits with the reasons in its message. Jobs without `placement` behave exactly as before.
  - **Auto-scaling**: a job's running tasks grow with its queue up to the smallest of `max_parallel`, its targets' free quota and the tasks its remaining budget pays for at the estimated task cost. Running tasks are never stopped to shrink. A job whose remaining budget cannot pay for one more task, with nothing running, is stopped (`budget_stop`, `reason: exhausted`) instead of waiting forever.
  - **Warm pool**: `JobSpec.pool` jobs are served by `cloud worker --pool --slots N --idle S`, a `CloudService` in pool mode on its own machine. It leases pool tasks, runs them through the task runtime as local processes, refreshes their heartbeats each tick and exits after `S` idle seconds (scale to zero).
    - The coordinator, configured by `PoolPolicy` in `<root>/pool.json`, never starts pool tasks. It keeps `min(max_workers, max(min_workers, machines needed))` workers. Machines needed is a first-fit-decreasing packing (`pack`, `slots_per_machine`) of the queued and running pool tasks' CR-11 memory onto the worker machine.
    - It requeues the tasks of a worker whose heartbeats stop (`heartbeat_timeout`).
- **CLI and manifests**: `cloud submit --priority`, `--pool` and repeatable `--target provider/compute[/region[/machine_type[/spot]]][:max_tasks]`. Native manifests accept `priority`, `pool` and `placement`.
- Tests: `test_scheduler.py` (9). The completion check drains a 20-task queue over three fake providers, with spot preemptions, a 2-task capacity limit, a target whose first three launches fail, a 2-task quota and a scripted failure. Every task succeeds within the quota, capacity and `max_parallel` limits.

#### Changed

- The monthly spend is computed once per scheduling pass rather than once per job.
- `_launch` and `_remote_launch` moved unchanged to `service_launch.py` (`ServiceLaunchMixin`), keeping `service.py` at 448 lines.

### Cloud runner CR-10: cost estimates, budget caps and accrual (`2026-10-20`)

#### Added

- **`logic/controllers/cloud/cost/`** (#130). All amounts are list-price estimates, not invoices.
  - **`PriceTable`**: hourly prices per provider, region, machine type and spot. Overrides in `<root>/prices.json`, refreshed per provider and region by `update(provider, region, fetcher)`, take precedence over the providers' static tables. `provider_fetcher` picks up a provider pricing module's `fetch_prices` (none exist yet: fakes only). The status view and accrual both use this table.
  - **Pre-launch estimate** (`estimate_job`, `CloudService.estimate`): CR-11 p10/p50/p90 × hourly price, one task per instance, summed per job. Local jobs cost 0; an unknown price gives None, never 0.
  - **Caps**: `Resources.budget_usd` per job, and `BudgetPolicy(monthly_usd, warn_fractions)` read from `<root>/budget.json`.
    - At submit, a p50 estimate over the job cap, or over the month's remaining budget, raises `BudgetExceeded` before anything is stored. The month counts unfinished jobs at max(p50 estimate, accrued). A p90 over a cap only warns.
    - Each scheduling pass writes `kind: cost` records (`accrued_usd`, `rate_usd_h`, `elapsed_s`) to the CR-12 stream, at most every 60 s and once more at the end. It emits one `budget_warning` per scope and fraction, and cancels the job when its accrued cost reaches its cap. When the month reaches its cap, every running job of that month is cancelled (`budget_stop`).
  - **Billing exports** (`import_billing_csv`): per-job totals from a CSV with a `wsr_job_id` label column (GCP, AWS and Azure column names), recorded as `billing` events.
- **CLI**: `cloud submit --machine-type --region --budget`, plus `cloud cost estimate` (with `cost_usd`, `eta_seconds` from longest-first packing onto `--max-parallel` slots, and `memory_required_gb`), `cloud cost show <job>`, `cloud cost prices <provider> <machine_type>` and `cloud cost import-billing <csv> --provider`.
- **App**: the `cloud_cost_estimate` Tauri command and `getCloudCostEstimate`. The Launcher has machine type, region and budget fields (`CostLimitsFields`) and a debounced estimate (`useCloudEstimate`) in its `EstimatePanel`. Submit passes the new flags.
- Tests: `test_cost.py` (8) and `CostLimitsFields.test.tsx`.

### Cloud runner CR-11: runtime and memory prediction (`2026-10-20`)

#### Added

- **`logic/controllers/cloud/prediction/`** (#131): wall-time quantiles (p10, p50, p90) and memory for every cell of a job.
  - **Model:** three gradient-boosted quantile regressions (scikit-learn) on log wall time. Features: routing algorithm, packing variant, improver, distribution, area, network size and horizon, read from the cell's `test_sim` arguments or its rerun id.
  - **Training data:** the bundled `runtime_training.csv` (810 runs on one laptop), built by `tools/build_runtime_training.py`. 210 cells come from the rerun drivers' timed `done` lines. The other 600 are summary runs whose wall time is their policy time plus the measured overhead of 19–32 s; on the timed cells, wall time is 1.00–1.03 × policy time.
  - **Held-out error** (52 timed cells): median APE 20%, median error 265 s, p10–p90 coverage 77%. Run `cloud predict evaluate` to reproduce.
  - **Machine types:** the median observed/predicted ratio of each type's completed single-attempt tasks (at least 3), else 1.0.
  - **Memory:** the 90th percentile of observed CR-12 peaks per routing algorithm and size × 1.25, else the rerun drivers' reservation.
  - **Fallback:** the manifest's `expected_wall_seconds` (0.5–2×) when scikit-learn is missing or the cell has no size or horizon.
- **Submit:** `CloudService(predictor_factory=...)`, set by the local profile, fills each cell's `expected_wall_seconds` that the submitter left empty (the scheduler and the Analysis page read it). It records the full predictions as a `prediction` event. `CloudService.predict(spec)` serves the CR-10 estimate and the CR-09 packing.
- **CLI:** `cloud predict estimate --manifest/--command [--machine-type]`, `cloud predict evaluate` and `cloud predict accuracy <job>` (predicted against actual wall time, median APE and interval coverage).
- Tests: `test_prediction.py` (7).

#### Changed

- `CloudService`'s `status`, `metrics` and their price and duration helpers moved unchanged to `service_status.py` (`ServiceStatusMixin`). This keeps `service.py` under 500 lines. `extract_policy_from_cell` is re-exported from `service`.

### Azure provisioning crash recovery (`2026-10-20`)

- Persist a scoped allocation intent before creating a NIC or VM, flush journal data and its directory entry, and retain it until the control plane accepts the backend handle. On restart, expire placing leases before reconciling; preserve live owners and accepted handles, reclaim abandoned allocations, and retain failed deletion work durably.
- Renew leases during slow provisioning, use unique resource names per leased submission and reject stale scheduler handoffs. NIC provisioning failures now pass through the same teardown path as VM failures.
- Keep journal files under the configured persistent `WSR_STATE_DIR` (or the explicit orphan-file directory); recovery requires the same directory and control-plane scope on restart. Unknown account/control-plane ownership is preserved rather than deleted.
- Added crash tests before/after NIC creation, after VM creation, around durable handoff, with competing ownership, plus an actual child process exiting during provisioning. The initial cloud-controller suite passed (142 tests, four Docker/Postgres-only skips); after integrating CR-12 instrumentation, the combined suite passes 150 tests with the same four skips.

### Codex Azure orphan/logging review (`2026-10-20`)

- Scope orphan cleanup by subscription/profile, serialize queue mutations, and surface corrupt/unwritable state. Handle disappearance after VM cancellation, distinguish retry VM names, block reuse pending cleanup, and drain cleanup during scheduler maintenance without active jobs. Fix Log Analytics timespan/dependency/error handling; refresh the lock. Added offline account-boundary, queue, SDK and service regressions (67 tests pass). The provisioning crash window still requires a write-ahead intent and ownership reconciliation before merge.

### Azure compute backend blockers resolved and service integration verified (`2026-10-20`)

- **Partial-VM Teardown & Durable Orphan Queue**:
  - Enforced partial VM resource deletion *before* NIC deletion on failed submit to avoid NIC-in-use errors.
  - Implemented idempotent not-found handling (treating 404, `ResourceNotFound`, and `NotFound` as clean deletions).
  - Added durable JSON orphan queue (`orphan_file`, `_record_orphan`, `drain_orphans`) tracking uncleaned resources across scheduler timeouts, loss, or crashes, drained automatically on `submit()` and `poll()`.
- **Boot Diagnostics & Container Apps Log Analytics**:
  - Enabled managed boot diagnostics in `vm_params` (`{"diagnostics_profile": {"boot_diagnostics": {"enabled": True}}}`).
  - Added Azure Monitor Log Analytics query support (`ContainerAppConsoleLogs_CL`) for `ContainerAppsJobsBackend.logs()` with `logs_client` and `workspace_id`, maintaining fallback for offline test fakes.
- **Integrated CloudService Tests**:
  - Added end-to-end service integration test suite driving Azure VM and Container Apps through `CloudService` lifecycle (submit -> run -> success, job cancellation propagating to backend deletion, spot eviction retry handling, and submit capacity partial teardown).
  - All 46 provider contract, regression, and service integration tests pass offline without cloud credentials.

### Azure cloud provider integration blockers resolved (`2026-10-20`)

- **Batch Placement & Job API**: Replaced invalid task-level `pool_id` placement with job-level targeting (`job_id_spot` for spot tasks, `job_id` for on-demand). Inferred spot status from job handle rather than non-existent task field.
- **Container Apps Jobs SDK Adapter**: Implemented real `azure-mgmt-appcontainers==3.0.0` execution lifecycle using `jobs_executions.list(...)`, `jobs.begin_stop_execution(...)`, and awaited LRO poller result on `jobs.begin_start(...)`.
- **Container Apps Secret Patching**: Corrected patch envelope structure to `{"properties": {"configuration": {"secrets": ...}}}` with snake_case `key_vault_url`, awaited update LRO poller, and propagated update failures without proceeding without required secrets.
- **Batch Authentication Adapter**: Added `MsrestTokenCredentialAdapter` wrapping `azure.identity` TokenCredentials with `signed_session(session)` for `azure.batch` 14.x msrest pipelines.
- **Dependency Contracts**: Added `azure-mgmt-network>=25.0.0,<26.0.0` to `cloud` optional dependencies in `logic/pyproject.toml` and pinned compatible version bounds for `azure-batch`, `azure-mgmt-appcontainers`, and `azure-mgmt-compute`.
- **VM Cleanup & Billing Safety**: Cleaned up pre-created NIC if VM creation fails; triggered VM deletion upon asynchronous provisioning failure or spot eviction; surfaced deletion errors on deallocated/stopped VMs.
- **Diagnostics & Failure Handling**: Fetched log content from boot diagnostics URL in VM `logs()`, handled absent Container Apps log stream gracefully, and re-raised authorization/service errors in Key Vault and Blob Storage rather than swallowing into `None` or `False`.
- **Validation**: Expanded test suite to 25 unit/contract/regression tests covering all resolved blockers.

### Codex cloud review corrections (`2026-10-20`)

- Corrected Azure VM ACR token exchange, numeric exit-code Merge tagging, token refresh, SSH-key generation failure handling, and the Compute 30 NIC deletion field. Added local bootstrap regressions. CR-07 remains pending real Batch/Container Apps SDK adapters, installation/version bounds and reliable cleanup; see the Codex cloud review of 2026-10-20 on the main checkout.

### Cloud runner CR-12: instrumentation and tracking (`2026-10-20`)

#### Added

- **Task telemetry** (`logic/controllers/cloud/telemetry.py`, #132). While the simulator runs, the task runtime writes three kinds of control-plane metric records:
  - **`resource`**: CPU, `memory_rss_mb`, threads and processes of the simulator's process tree, plus work-directory size and network counters, every `WSR_CLOUD_SAMPLE_SECONDS` (default 5). Resource sampling needs `psutil` and is skipped without it.
  - **`progress`**: one record per policy and day, read incrementally from the day logger's `log_realtime_*.jsonl`, with the day, the horizon, the solver status and the day's KPIs. The task's `progress` field carries the latest day.
  - **`log_tail`** and **`telemetry_error`**: lifecycle events for a failed simulator and for instrumentation errors. Instrumentation errors never stop a task.
- **Batched, rate-limited writes** (`TelemetryBuffer`): at most one write per 10 s, at most 200 records each. Cost records (`kind: cost`, CR-10) use the same stream.
- **Tracking module**: a `run_events` table, `log_run_event`, `get_run_events` and `clear_run_data` on `TrackingStore`. `import_job` creates one `cloud_task` run per task, tagged with job, cell, provider, compute and attempt. It records metrics as `resource/*`, `progress/*` and `cost/*` and lifecycle events in `run_events`. The run's artifact directory is the job's `cloud download` folder, which links it to the local outputs. Re-importing is idempotent.
- **CLI**:
  - `cloud metrics <task>`: resource points, peaks, latest progress and per-day KPIs.
  - `cloud track <job>`: imports the job into the simulator's tracking database (`assets/tracking/tracking.db`).
  - `cloud status`: each task now has `cpu_percent`, `memory_rss_mb` and `progress`.
- **App**: the `cloud_task_metrics` Tauri command and `getCloudTaskMetrics`. The Monitor's telemetry tab plots the selected task's samples and shows its current day, refreshing every 10 s. Its empty state no longer says CR-12 is pending.
- Tests: `test_telemetry.py` (8).

### Cloud local backend import path (`2026-10-20`)

#### Fixed

- **Local backend tasks failed when the repository was not on the import path.** `LocalProcessBackend` starts `python -m logic.controllers.cloud.local._wrap` in the task's work directory, which need not contain this checkout. It now prepends the repository root to `PYTHONPATH`, as `spawn_worker` already did. This fixes the four cloud tests that failed in the main suite: `test_local_process_backend`, `test_local_backend_resolves_secrets_without_logging_them`, `test_cli_runs_a_manifest_end_to_end_and_resumes_a_killed_task` and `test_download_lists_files_with_hashes`.

### Codex CR-16 telemetry review corrections (`2026-10-20`)

- Align retry-wide timestamp spans with summed attempt durations; preserve unavailable/invalid totals and policy identity; restrict prediction accuracy to completed single-attempt samples. Monetary values are prorated scheduler-time/price-hint estimates with transported provenance, not invoiced costs or universal provider billing minimums. Index lifecycle events by task. Validation: 12 Python service, 95 frontend and 18 Rust cloud tests pass, plus TypeScript/Ruff/diff checks.

### Learning config layout (`2026-10-20`)

#### Changed

- **`logic/src/configs/rl` is now `logic/src/configs/learning`**. Algorithm dataclasses sit next to `RLConfig`. The expert dataclasses from `rl/policies/` are one module, `logic.src.configs.learning.experts`, and the short aliases (`ALNSConfig`, `HGSConfig`, `ILSConfig`, `HGSALNSConfig`, `HVPLConfig` on that package) are gone. Hydra targets use `logic.src.configs.learning.experts.<Class>` with the `Vector*` names. `VectorHVPLConfig` keeps `alns_iterations` so the HVPL expert yaml matches `VectorizedHVPL`, and both imitation trainers dispatch that config.
- **Removed `logic/configs/models/policies/ahvpl.yaml`**. It targeted `AHVPLConfig`, which had no class.
- **Expert dataclasses now carry the fields the vector policies already accept** (`population_size`, `elite_size`, `n_generations`, `max_iterations`, `alns_start_temp`, `alns_cooling_rate`, `max_vehicles` where that policy reads them). `alns.yaml` no longer sets `weight_update_frequency` or `min_removal_pct`; the vector ALNS policy does not read them, so Hydra could not build the config.

### Dead-code sweep (`2026-10-20`)

#### Added

- **`logic/validation/check_dead_code.py`** (recipe `just check-dead-code`): reports modules under `logic/src` with no importer and no Hydra or registry string, and module-level `Alias = Canonical` re-exports that nothing imports. F-string registry targets, `module:name` targets, test imports, and `__main__` entry points count as uses. Assignments to `Any` or `object` stay as optional-dependency fallbacks.

#### Removed

- **Compatibility aliases with no remaining caller**:
  - `MSBPCSPPruningException` from the MS-BPC-SP engine. Callers use `BPCPruningException`.
  - `MultiStartGreedyEval` and `MultiStartGreedyAugmentEval` from the evaluator package. Callers use `MultiStartEval` and `MultiStartAugmentEval`.
  - `ENV_REGISTRY` from the environment package. Callers use `ENV_TARGETS`.
  - `LLH_POOL` from the low-level heuristic pool. Callers use `INTRA_ROUTE_POOL`.
  - Unused MS-BPC-SP re-exports `_reset_master_constraints` and `_detect_cycles` on the engine, and `_reset_master_constraints`, `_detect_cycles`, and `_column_generation_loop` on node selection. The engine still re-exports `_column_generation_loop` for the shared column-generation test.

### Backwards Compatibility Cleanup (`2026-10-20`)

#### Removed

- **Removed legacy & backwards-compatibility stubs across the codebase**:
  - Deleted `logic/src/data/processor/temporal.py` deprecation stub; all callers use `logic.src.data.time.travel_matrix`.
  - Removed backwards-compatibility re-exports of `TimeWindowSchedule` and `build_time_window_pattern` from `logic/src/data/time/time_windows.py` (callers import from `logic.src.data.time.schedule` or `logic.src.data.time`).
  - Removed `validate_gen_data_args` alias from `logic/src/data/generators/validators.py` and exported canonical `validate_data_config` in `logic/src/data/generators/__init__.py`.
  - Removed `validate_eval_args` alias from `logic/src/pipeline/features/eval/validation.py` and exported canonical `validate_eval_config` in `logic/src/pipeline/features/eval/__init__.py`.
  - Removed `context.policy` backward-compatibility property alias from `SimulationContext` (`logic/src/pipeline/simulations/states/base/context.py`); code uses canonical `pol_name`/`policy_name`.
  - Removed `context_node_projected` getter/setter alias from `AttentionDecoderCache` (`logic/src/models/subnets/decoders/common/cache.py`); callers use canonical `graph_context`.
  - Removed `BranchingConstraint = EdgeBranchingConstraint` alias from `logic/src/policies/helpers/solvers_and_matheuristics/branching/constraints.py` and its re-exports.
  - Removed private backwards-compatibility aliases `_vectorized_split_limited`, `_reconstruct_routes`, `_reconstruct_limited` from `logic/src/policies/vector/shared/__init__.py`, exporting canonical public functions directly.
  - Removed `self.num_heads = n_heads` alias from `MultiHeadAttentionMDAM` (`logic/src/models/subnets/modules/mdam_attention.py`).
  - Removed `self.hidden_dim = self.embed_dim` alias from `MandatoryManager` (`logic/src/models/meta/hrl_manager/model.py`).
  - Removed `LegacyCriticNetwork` alias in `A2C` (`logic/src/pipeline/learning/core/a2c.py`) in favor of direct `CriticNetwork` import.
  - Removed `"lac"` backward-compatibility registry entry from `SANSPolicy` (`logic/src/policies/routing/meta_heuristics/simulated_annealing_neighborhood_search/policy_sans.py`) and `logic/src/policies/routing/_manifest.py`.
  - Removed `POLICY_SEARCH_SPACES = ROUTING_SPACES` alias from `logic/src/pipeline/simulations/hpo/search_spaces.py` and updated callers to canonical `ROUTING_SPACES`.

### Simulation Time Windows & Temporal Module Reorganization (`2026-10-20`)

#### Added

- **Time Windows constraints in simulator** (`logic/src/data/time/`):
  - Structured across `schedule.py` (`TimeWindowSchedule` and `build_time_window_pattern`, 216 LoC) and `time_windows.py` (file loader & unified dispatching, 260 LoC) to ensure both stay well within the <500 LoC threshold.
  - Support for optional simulation time window constraints where no routing occurs on designated days.
  - Can be specified via CSV/Excel file with boolean indicators per day, or via a 3-parameter periodic pattern (`start_day`, `days_after`, `interval_between`).
  - Added `TimeWindowSchedule` class and `load_time_windows`, `build_time_window_pattern`, `load_time_windows_file` utilities.
  - Added `SimulationRepository.get_time_window_params` and `load_time_windows` repository wrapper.
  - Integrated with `SimConfig`, `GraphConfig`, `SimulationContext`, `InitializingState`, `RunningState`, and `SimulationDayContext`.
  - On non-routing days, bins accumulate filling as normal while routing is bypassed (`tour=[0]`, `solver_status="no_routing"`).
  - Comprehensive unit test suite in `logic/test/unit/pipeline/simulations/test_time_windows.py`.

#### Changed

- **Moved travel time matrices to `logic/src/data/time/`**:
  - `logic/src/data/processor/temporal.py` moved and renamed to `logic/src/data/time/travel_matrix.py`.
  - Maintained backward-compatibility stub at `logic/src/data/processor/temporal.py` re-exporting `compute_time_matrix`.
  - Updated repository and test imports to use `logic.src.data.time.travel_matrix`.

### Autoregressive DRL paper submodule (`2026-10-20`)

#### Added

- **`assets/papers/Autoregressive-Deep-Reinforcement-Learning-for-Waste-Collection-Route-Planning`**: the ACFHarbinger GitHub repository as a submodule. Its Overleaf project is the local remote `overleaf` (`https://git@git.overleaf.com/671f898ed9913ebc7ae5fefa`, branch `master`) for future merges; remotes live in the clone's `.git/config`, so a fresh clone needs `git remote add overleaf ...` again. The Overleaf and GitHub histories share no commit, so the first merge needs `--allow-unrelated-histories`.

### File-size refactor (`2026-10-20`)

#### Changed

- **Code files over 500 lines split to 150-500 lines** across `logic/` (about 90 files, seven agent lanes: Claude, Codex, Gemini, Kimi, Mistral, Grok, Cursor; brief in `.agent/tasks/file-size-refactor-2026-10-20.md`, lane reports on `.agent/bus/2026-10-20.md`). Code was moved, not changed: whole functions and classes to sibling modules, large classes into mixins, and every original module keeps its import path by re-exporting what moved. `tools/check_pure_move.py 957185ee1` reports 7,436 of 7,439 definitions unchanged; the three intended changes are below. Test changes are mock patch targets that follow the moved code, plus split test files.
- **Two functions too long to move whole** were shortened by extracting private helpers: `column_generation_loop` (its node set-up, into `branching/pruning._prepare_node_relaxation`) and `run_ms_bpc_sp` (its stages, into the new `ms_bpc_sp_phases.py`). `run_ms_bpc_sp` returns identical results on 12 seeded instances, and the names the tests patch stay on the engine module.
- A few modules that only re-export after the split are under 150 lines (for example `irp.py`, `thop.py`, `store.py`).

#### Fixed

- **`generate_packing_templates`** matched configs by `__module__ == packing.__name__` and would have skipped the configs moved to `packing_dispatch`, `packing_learned` and `packing_threshold`; it now matches the `packing*` modules and finds the same 34 configs as before.

#### Notes

- The full suite gives the same result before and after (2,472 passed; the same 7 failures on both: three simulation integration tests and four cloud-runner tests).
- Pre-existing bugs the lanes noticed but did not fix are listed on the bus for the ultra-review.

### Paper results from the 2026-10-19 vintage (`2026-10-20`)

#### Changed

- **Simulation paper** (submodule at `dd76028`): tables, evidence and prose now use `docs/private/global/simulation/rerun-2026-10-19` (BPC with the CPTP pricer, ACO-HH with the mandatory-bin fix). The statements that ACO-HH omits mandatory requests are removed; the remaining executed-service gaps are SWC-TCF's empty bin-days. The observation-contract audit replays the rerun logs with the closed-form lookahead.

### Codex cloud review corrections (`2026-10-20`)

- **Telemetry analysis transport & CR-16 integration:**
  - Implemented real telemetry and status analysis payload in Python `CloudService.status` (`logic/controllers/cloud/service.py`) exposing provider, compute kind, spot flag, machine type, region, hourly rates (with on-demand baseline), per-task timestamps (`started_at`, `finished_at`, `duration`), pre-run predictions (`expected_wall_seconds`), policy identity (extracted from `CellSpec`), and billed costs across all retry attempts adhering to cloud provider billing minimums.
  - Mirrored status models through Rust Tauri commands (`CloudTaskSummary`, `CloudJobStatus` in `app/src-tauri/src/commands/cloud/models.rs`) and TypeScript interfaces (`app/src/types/cloud.ts`).
  - Rebased `feat/cloud-16-analysis` onto `main`, dropping duplicate launcher files and duplicate route/nav declarations from CR-14.
  - Updated `computeJobAnalysis` in `app/src/utils/cloud/analysis.ts` to preserve and aggregate real transported metrics (total compute hours, wall clock time from timestamp spans, accrued expenditure, spot savings percentage, and surrogate duration predictions), keeping missing observations nullable.
  - Corrected concurrency metrics in `CloudAnalysis.tsx` to display true average concurrent tasks rather than claiming a sequential baseline speedup factor.
- Removed invented cloud execution durations, predictions, prices and concurrency from analysis; missing telemetry stays unavailable. Corrected per-task prediction MAPE, plain-text chart tooltips and stale detail responses.

### Cloud runner planned (`2026-10-19`)

#### Added

- **CR-01 (#121): cloud runner core** in `logic/controllers/cloud/`:
  - `model.py`: `JobSpec`, `CellSpec`, `Resources`, `TaskRecord`, `JobRecord`, events and result records; the task state machine (`TRANSITIONS`, `check_transition`) and the derived job state.
  - `manifest.py`: job specs from native manifests, the paper rerun manifests (filtered by `policy`) and single `test_sim` commands. Runner-owned settings (`sim.cpu_cores`, `sim.resume`, `sim.checkpoint_days`, `sim.output_dir`) are dropped from the cells.
  - `interfaces.py`: `ControlPlane`, `ComputeBackend`, `ObjectStore`, `SecretStore` and `ImageRegistry`.
  - `local/`: the SQLite control plane (atomic leases across processes), a filesystem object store, environment and in-memory secret stores, and a subprocess backend whose attempts survive a scheduler restart and wait for free memory.
  - `service.py`: `CloudService` submits jobs, starts tasks up to `max_parallel`, retries failures and preemptions (resuming from the daily checkpoints), enforces the wall-time and idle limits, cancels, requeues and downloads hash-checked results into the `outputs/` layout.
  - `testing/`: fakes and the contract suites every backend, control plane, object store and secret store must pass.
  - CLI: `python main.py cloud submit|status|logs|cancel|retry|download|backends|worker`, with `--json` output for the Studio. A real two-day `test_sim` cell ran end to end through it.
  - `logic.controllers.cloud` is excluded from the release until CR-17 ships it as the `[cloud]` extra.
- **CR-13 (#133): Studio cloud job runner backend** in `app/src-tauri/src/commands/cloud/` and `app/src/utils/cloud/`:
  - `commands/cloud/mod.rs` & `cli.rs`: Tauri commands wrapping `python main.py cloud --root <root> --json <sub-command>`: `cloud_list_jobs`, `cloud_job_status`, `cloud_submit`, `cloud_cancel`, `cloud_retry`, `cloud_task_logs`, `cloud_download`, `cloud_verify_download`, `cloud_backends`, `cloud_worker_tick`. Subprocess outputs are deserialized into typed models mirroring Python payloads with comprehensive error reporting for non-zero exits and malformed JSON.
  - `commands/cloud/poller.rs`: background poller emitting `cloud://job-updated` Tauri events with `CloudJobStatusUpdateEvent` payloads. Abstracted behind the `CloudJobWatcher` trait boundary so Firestore and Supabase realtime (CR-04) can feed the identical event contract. Provides `cloud_watch_job`, `cloud_unwatch_job`, `cloud_stop_all_watchers`, and `cloud_list_watched_jobs`.
  - `commands/cloud/download.rs`: download progress tracking emitting `cloud://download-progress` lifecycle events and SHA-256 verification (`verify_file_sha256`, `cloud_verify_download`).
  - `commands/cloud/keychain.rs`: OS keychain credential management via the `keyring` crate (`cloud_store_credential`, `cloud_delete_credential`, `cloud_has_credential`) under service `wsmart-route.cloud` and account `{provider}/{profile}`. Secrets are never logged, returned to frontend, or written to disk.
  - Rust unit tests verifying fixture deserialization across all CLI subcommands against real CLI-generated fixtures, validation of request parameters, SHA-256 hashing, and poller lifecycle.
  - TypeScript client in `app/src/utils/cloud/` and types in `app/src/types/cloud.ts` wrapping Tauri invokes and event subscriptions.
- **CR-14 (#134): Studio Cloud Launcher** in `app/src/pages/cloud/CloudLauncher.tsx`:
  - `CloudLauncher.tsx`: Studio launcher interface for submitting distributed simulation runs and multi-cell experiment manifests. Supports single `test_sim` commands with preset templates, batch manifests with regex policy filtering, and custom job names and local state roots.
  - `ProviderSelector.tsx`: dynamic target provider (`local`, `gcp`, `aws`, `azure`) and compute topology (`process`, `batch`, `serverless`, `vm`) configuration with active backend capability inspection (`listCloudBackends`).
  - `CredentialModal.tsx`: secure OS keychain credential management for cloud profiles (`cloud_has_credential`, `cloud_store_credential`, `cloud_delete_credential`) without exposing secrets on disk or in logs.
  - `EstimatePanel.tsx`: typed resource and cost estimation panel (`CloudJobEstimate`). Explicitly renders "Estimate unavailable (CR-10 cost and ETA engine pending)" when estimates are absent or pending without mocking arbitrary numbers.
  - Resource and fault tolerance controls: parallel task caps, CPUs, RAM, Spot/Preemptible toggle, retries, wall-clock time, and idle timeouts.
  - Gurobi licensing integration: WLS secret and BYO license configuration options.
  - Live CLI command preview mirroring `logic/controllers/cli/cloud_parser.py` with one-click clipboard copying.
  - Form state persistence in `useCloudLauncherStore` via Zustand (`wsroute-cloud-launcher`).
  - App navigation integration in `Sidebar.tsx` (under Launch section with Cloud icon) and lazy loading in `App.tsx`.
- **CR-15 (#135): Studio Cloud Job Monitor and Manager Actions** in `app/src/pages/cloud/CloudMonitor.tsx`:
  - `CloudMonitor.tsx`: Real-time monitoring and management dashboard for cloud routing jobs. Provides master-detail view with live updates fed by CR-13 background poller (`watchCloudJob`), state breakdown counters, proportional progress bars, and state/search filtering.
  - Manager actions: On-demand cancellation (`cancelCloudJob`), requeuing failed/cancelled tasks (`retryCloudJob`), scheduler placement stepping (`tickCloudWorker`), and hash-checked results downloading (`downloadCloudJobResults`).
  - `CloudStatusBadge.tsx`: Visual status pill for all job and task states (`running`, `succeeded`, `failed`, `cancelled`, `queued`, `placing`, `starting`, `preempted`, `retrying`) with animation pulse for active states.
  - `TaskMetricsPanel.tsx`: Dedicated resource telemetry panel with structured placeholders for CPU load, memory RSS, and throughput, cleanly rendering the "Metrics pending (CR-12)" empty state until the instrumentation engine lands.
  - `TaskLogsModal.tsx`: Real-time log inspector supporting configurable tail counts (50 to all), live poll refresh, auto-scroll to bottom, and clipboard export.
  - `DownloadModal.tsx`: Interactive download manager tracking stages, file counts, and checksum verification status via `onCloudDownloadProgress`.
  - `useCloudMonitorStore`: Zustand persistence store for UI state, root path, search filters, and active tabs.
  - App navigation integration in `Sidebar.tsx` (under Monitor section), `App.tsx` (lazy route), and `pagePrefetch.ts`.
- **Cloud runner, local worker and light entry point** (found while reviewing CR-13):
  - `cloud submit` now starts a detached background worker (`logic/controllers/cloud/worker.py`) unless one already holds `<root>/worker.lock`. The worker schedules every job under the root and exits when all have finished. Before, a job submitted from the Studio stayed queued, because nothing ran the scheduler. `--no-worker` keeps the old behaviour.
  - `python -m logic.controllers.cloud` runs the same commands without `main.py`'s start-up imports: about 0.3 s and 33 MB per call instead of 2 s and 690 MB. The Studio's status polling uses it.
  - `--project-root` sets the repository whose `main.py` local tasks run.
  - `cloud download --json` adds `file_list` (`relpath`, `sha256`, `size` per file) next to the `files` count.
- **CR-02 (#122): task runtime** (`python -m logic.controllers.cloud.runtime`, the entry point of the release image):
  - `staging.py`: at submit, a remote job's shared inputs (`JobSpec.inputs`, default `data/simulator`) and each cell's policy directory are uploaded content-addressed (unchanged files once across jobs) with a manifest (`JobSpec.inputs_manifest`). The runtime fetches them into the project root and its work directory, checking every hash.
  - `runtime.py`: restores the task's outputs mirror (`work/<job>/<task>/outputs/` in the object store), runs `main.py test_sim` with daily checkpoints, uploads changed outputs every `WSR_CLOUD_SYNC_SECONDS`, and records a heartbeat when outputs change. It stops a run after `idle_seconds` without new output (exit 124). On SIGTERM it syncs and exits 143, so the next attempt, on any machine, resumes from the last finished day. On success it records the results index.
  - Gurobi: the backend injects `WSR_GUROBI_LICENSE` from the secret store; the runtime writes it to a mode-0600 license file for `GRB_LICENSE_FILE`, keeps the secret out of the simulator's environment, and fails before the run (exit 3) when a cell needs Gurobi and no license was injected.
  - `profiles.py`: control planes and object stores from URIs (`sqlite://`, `file://`; `firestore://`, `supabase://`, `gs://`, `s3://` and `az://` resolve to the provider packages).
  - `CloudService(remote=RemoteConfig(...))`: jobs for GCP, AWS and Azure need an image, stage their inputs at submit, and launch the runtime with the control-plane and store URIs in the environment and the Gurobi secret by name.
  - A task SIGTERMed after day 2 resumed on a fresh machine (empty work and project directories) and finished with the same results as an uninterrupted run.
- **CR-04 (#124): Firebase and Supabase control planes** in `logic/controllers/cloud/control/`:
  - `documents.py`: `DocumentControlPlane` implements the `ControlPlane` contract (records, state machine, conditional updates) over a `DocumentAdapter` of atomic storage primitives.
  - `firebase.py`: Cloud Firestore adapter. Leases and every read-modify-write run in Firestore transactions, and batches are capped at 500 writes. Result document ids are hashes, because Firestore ids cannot contain `/`. Composite indexes and security rules are in `infra/cloud/firebase/`.
  - `supabase.py`: Postgres adapter through PostgREST. `infra/cloud/supabase/migrations/0001_cloud_runner.sql` creates the tables, row-level security, Realtime publication and four functions (`wsr_insert_job`, `wsr_lease_task` with `FOR UPDATE SKIP LOCKED`, `wsr_expire_leases`, `wsr_replace_results`). Other updates are optimistic on a `rev` column, retried on conflict.
  - `testing/fake_firestore.py` and `testing/fake_supabase.py`: strict in-memory clients that reject what the services reject (ids with `/`, nested arrays, documents over 1 MiB, writes before reads in a transaction, unknown columns, NOT NULL, duplicate keys). Both control planes pass the CR-01 contract suite on them.
  - `test_supabase_sql.py` (opt-in, `WSR_TEST_DOCKER_POSTGRES=1`): the migration on a real Postgres 16, including two sessions leasing concurrently. It passed.
  - URIs: `firestore://<project>[/<database>]` and `supabase://<project-ref>` (key from `SUPABASE_KEY` until CR-08). The SDKs (`google-cloud-firestore`, `supabase`) belong to the `[cloud]` extra (CR-03). A URI whose SDK is missing raises a `ValueError` that names it.
- **CR-08 (#128): credentials and secrets** in `logic/controllers/cloud/`:
  - `credentials.py`: `detect_logins()` reports usable gcloud ADC, AWS profiles and environment keys, and `az login` sessions, from files and environment variables only. `KeychainSecretStore` reads the OS keychain under the Studio's naming (service `wsmart-route.cloud`, account `{provider}/{profile}`; named secrets `secret/{name}`). `ProviderCredentials` builds `boto3.Session` keyword arguments, `google.auth` credentials or `azure.identity` credentials from a CLI login or a keychain key.
  - `secret_refs.py`: `WSR_SECRET_REFS` (`akv://`, `awssm://`, `gcpsm://`, `env://`) for services that cannot inject secrets natively. The task runtime resolves the references with the machine's identity before starting the simulator, and exits 4, naming the variable but never a value, when one cannot be resolved. `redact()` removes values from stored text.
  - `config.py`: cloud profiles in `$WSR_CLOUD_CONFIG` or `~/.config/wsmart-route/cloud.yaml` (provider, compute kind, region, credential source, control-plane and store URIs, image, Gurobi secret name). Unknown keys are rejected, so secrets cannot be put there.
  - CLI: `cloud credentials`, `cloud profiles`, `cloud secrets set|delete|exists NAME` (a value is read only from `--from-file` or `--stdin`, never from an argument, and is never printed). `run_cloud` dispatches through a command table.
  - A Gurobi license resolved through a reference reaches the simulator as a mode-0600 file, and its value appears nowhere in the control-plane database or the object store.
- **`[cloud]` extra completed** (`logic/pyproject.toml`, moved from `[dependency-groups]` to `[project.optional-dependencies]` so that `wsmart-route[cloud]` and the release build see it): `google` (an unrelated search-scraping package) and `google-cloud` (a deprecated, empty meta-package) are replaced with the SDKs the cloud runner uses: `google-cloud-batch`, `-run`, `-compute`, `-storage` and `-secret-manager`, `boto3`, `azure-identity`, `azure-batch`, `azure-mgmt-appcontainers`, `azure-mgmt-compute`, `azure-storage-blob`, `azure-keyvault-secrets` and `keyring`, next to `google-cloud-firestore` and `supabase`. `uv.lock` adds them; no existing package changes version. `logic/release.yaml` now lists the `cloud` group by name.
- **Fixed:** CR-03 had inserted the release's `cloud` group inside `neural`, which moved `tensorboard`, `loss-landscapes` and `plotly` out of the release's neural extra. They are back under `neural`.
- **CR-06 (#126) review fixes, made during integration:**
  - The EC2 user data now has `trap 'shutdown -h now' EXIT`, so the instance stops (and, with `InstanceInitiatedShutdownBehavior=terminate`, is deleted) even when a step fails. It also logs Docker in to ECR before pulling an ECR image; before, a private image could not be pulled.
  - The image uses `CMD`, not `ENTRYPOINT`, for the task runtime: backends pass `LaunchSpec.argv` as the whole container command, which an entry point would have doubled.
  - `firestore://` URIs raise a `ValueError` with the login command when no Google credentials are found, instead of `DefaultCredentialsError`.
- **Moved:** `staging.py` and `storage.py` into `logic/controllers/cloud/store/` (imports `logic.controllers.cloud.store.*`).
- **Moved:** the authentication modules into `logic/controllers/cloud/auth/`: `credentials.py`, `config.py` (cloud profiles) and `secret_refs.py`. Imports are `logic.controllers.cloud.auth.*`; `profiles.py` (control-plane and store URIs) stays in `logic/controllers/cloud/`.
- **CR-03 (#123): release image and registries**:
  - `infra/cloud/image/Dockerfile`: built from a release tree (`infra/cloud/image/build.sh` builds the tree with `build_release.py` and tags the image `wsmart-route:<commit>`, with `-dirty` for a modified tree). Extras `solvers,cloud` by default; `EXTRAS=...,neural` adds the Attention Model's stack. The dependencies are a cached layer, installed from the release `pyproject.toml` before the code is copied. Neither data nor model weights are in the image. It runs as a non-root user, with `/app` and `/work` group-0 writable so any UID works, and its entry point is the task runtime. About 1.7 GB.
  - `logic/controllers/cloud/images.py`: `DockerRegistry` tags and pushes with the Docker CLI and returns the pushed `sha256:` digest. `artifact_registry()`, `ecr()` (password on stdin only) and `acr()` set up the login, and `pinned()` gives the immutable `<repository>@<digest>` reference jobs should use.
  - The release now ships `logic.controllers.cloud` (CLI, runtime, local backend, control planes) with a `[cloud]` extra of the provider SDKs. The extra is defined in `logic/release.yaml` only: adding it to `logic/pyproject.toml` re-resolves 11 existing packages of the development lock (starlette 1.x, fastapi, zenml, pyjwt, grpcio, opentelemetry, websockets) and drops `fastapi-utils`.
  - The task runtime runs `main.py` in a checkout and `__main__.py` in a release tree. It copies the simulator's output to its own stdout, so the cloud log has it, uploads each attempt's log to `work/<job>/<task>/logs/`, and puts the redacted tail of the log in the failure message.
  - `infra/cloud/image/smoke.py`: stages one one-day Rio Maior 100 cell into a local store, records it in a SQLite control plane, and runs the image under an arbitrary UID with only the state directory mounted. It passed: exit 0, with both simulation logs in the results index.
- **CR-13/CR-15 review fixes** (Studio cloud backend):
  - The Tauri commands call `python -m logic.controllers.cloud --project-root <repo>` instead of `python main.py cloud`.
  - The job poller keeps at least 1 s between polls, delays rather than bursts after a slow poll, and tags each watcher with a generation. Before, a finished poller could unregister a newer watcher of the same job, which then could not be stopped.
  - `CloudDownloadResult` carries the optional `file_list`.
- **CR-06 (#126): AWS backends, S3 object store, and Secrets Manager** in `logic/controllers/cloud/providers/aws/`:
  - `batch.py`: `AwsBatchBackend` for AWS Batch jobs (`ComputeKind.BATCH`), supporting container overrides, native secret resolution via Secrets Manager ARNs (`secrets[].valueFrom`), wall-time limits, spot preemption mapping, and cancellation.
  - `serverless.py`: `EcsFargateBackend` for AWS ECS Fargate tasks (`ComputeKind.SERVERLESS`), supporting Fargate Spot capacity providers, native secret references, and spot interruption handling.
  - `vm.py`: `Ec2Backend` for EC2 instances (`ComputeKind.VM`), supporting spot market options, UserData bootstrap injection, spot termination detection, and cancellation.
  - `storage.py`: `S3ObjectStore` implementing `ObjectStore` with content-addressing support and parent directory creation.
  - `secrets.py`: `AwsSecretsManagerStore` implementing `SecretStore` with automatic secret creation and updates.
  - `registry.py`: `EcrImageRegistry` for Amazon ECR image push and tag digest resolution.
  - `pricing.py`: static price table for AWS instance types and Fargate across regions.
  - `_sdk.py`: lazy `boto3` client factory preventing SDK imports at module load time.
  - Provider registration in `logic/controllers/cloud/providers/__init__.py` (`BACKENDS` dictionary and dynamic `load_backend`).
  - Unit and contract tests in `logic/test/unit/controllers/cloud/providers/test_aws_contracts.py` passing `backend_contract`, `object_store_contract`, and `secret_store_contract` with in-process fakes.
- **CR-16 (#136): Studio Cloud Analysis Dashboard** in `app/src/pages/cloud/CloudAnalysis.tsx`:
  - `CloudAnalysis.tsx`: Post-run aggregate analytics dashboard for cloud simulation jobs. Includes master job picker, top-level KPI cards (total tasks, wall-clock time, total compute hours, concurrency speedup factor, accrued cost), and multi-tab exploration (`overview`, `waterfall`, `tasks`, `cost`).
  - `CostAccrualCard.tsx`: Financial telemetry and budget tracking aligning with CR-10. Displays accrued cost across compute duration, pre-run budget estimate with variance, and spot capacity savings breakdown vs on-demand baseline. Gracefully handles uncalibrated/unbounded estimates ("Unbounded / No Cap").
  - `PredictionAccuracyCard.tsx`: Task duration prediction accuracy card aligning with CR-11 runtime surrogates. Analyzes observed vs predicted execution time, mean absolute percentage error, ±15% tolerance counts, and prediction bias. Cleanly renders "Pending CR-11 Calibration" when pre-run predictions are absent.
  - `TaskDurationWaterfall.tsx`: Gantt-style ECharts waterfall visualization plotting task durations color-coded by terminal execution state (`succeeded`, `failed`, `cancelled`, `preempted`).
  - Task Inspection Table: Interactive, sortable, and searchable task grid with policy identification, state filters, task log inspection via `TaskLogsModal`, and browser-based CSV export (`exportTasksToCsv`).
  - Financial & Spot Breakdown tab: In-depth infrastructure analysis detailing compute engine type, region, spot tier discount percentages, and cloud billing governance notes.
  - `useCloudAnalysisStore`: Zustand persistence store for active analysis tabs, selected jobs, task filters, and column sorting preferences.
  - Navigation integration in `Sidebar.tsx` (under Analysis section), `App.tsx` (lazy route), and `pagePrefetch.ts`.
  - Full suite of Vitest unit and component tests (`CostAccrualCard.test.tsx`, `PredictionAccuracyCard.test.tsx`, `cloudAnalysis.test.ts`, `cloudAnalysisStore.test.ts`).
- **CR-07 (#127): Azure backends, Blob store, and Key Vault** in `logic/controllers/cloud/providers/azure/`:
  - `batch.py`: `AzureBatchBackend` for Azure Batch jobs (`ComputeKind.BATCH`), supporting container tasks, native Key Vault secret references (`@Microsoft.KeyVault(SecretUri=...)`), wall-clock time constraints, and low-priority / spot node preemption mapping to `AttemptState.PREEMPTED`.
  - `serverless.py`: `ContainerAppsJobsBackend` for Azure Container Apps Jobs (`ComputeKind.SERVERLESS`), supporting template overrides, native secret references (`secretRef`), CPU/memory limits, cancellation, and spot eviction mapping.
  - `vm.py`: `AzureVmBackend` for Azure Virtual Machines (`ComputeKind.VM`), supporting spot priority with deallocate eviction policy, custom data script bootstrap, instance status polling, and spot eviction detection.
  - `storage.py`: `BlobObjectStore` implementing `ObjectStore` backed by Azure Blob Storage with content-addressing and prefix filtering.
  - `secrets.py`: `KeyVaultSecretStore` implementing `SecretStore` backed by Azure Key Vault with name sanitization and secret lifecycle handling.
  - `registry.py`: `AcrImageRegistry` for Azure Container Registry image tag resolution and manifest push.
  - `pricing.py`: static hourly price table for Azure VM types (D-series, F-series, B-series) and Container Apps across regions.
  - `_sdk.py`: lazy Azure SDK client factories without module-level imports.
  - Unit and contract tests in `logic/test/unit/controllers/cloud/providers/test_azure_contracts.py` passing `backend_contract`, `object_store_contract`, and `secret_store_contract` with in-process fakes.
- **Roadmap:** `docs/moon/roadmaps/cloud.md` (§J), linked from `docs/moon/ROADMAP.md`: the v1.1.0 cloud runner design, its 17 work packages (#121–#137, epic #120) and future work.
- **Agent brief:** `.agent/tasks/cloud-runner-2026-10-19.md`, with Claude's and Gemini's lanes, and a kickoff entry on the agent bus.


### Package layout (`2026-10-19`)

Directories renamed after what they hold. No behaviour changes.

#### Changed

- **HPO search spaces** (`logic/src/pipeline/simulations/hpo/`): `filters/` → `packing/`, `interceptors/` → `route_improvement/`, `jobs/` → `routing/`, `rules/` → `acceptance_criteria/`. The directory constants in `search_spaces.py` follow (`PACKING_DIR`, `ROUTE_IMPROVEMENT_DIR`, `ROUTING_DIR`, `ACCEPTANCE_CRITERIA_DIR`), as do the loaded spaces (`PACKING_SPACES`, `ROUTE_IMPROVEMENT_SPACES`, `ROUTING_SPACES`, `ACCEPTANCE_CRITERIA_SPACES`). `tools/check_hpo_jobs.py` and `tools/convert_policy_yaml.py` read `hpo/routing/`.
- **HPO component names** (`search_spaces.py`) follow the directories:
  - `get_component_search_space` component types: `"job"` → `"routing"`, `"filter"` → `"packing"`, `"interceptor"` → `"route_improvement"`, `"rule"` → `"acceptance_criteria"`;
  - `compose_search_space` parameters: `job`/`filter`/`interceptor`/`rule` → `routing`/`packing`/`route_improvement`/`acceptance_criteria`, and the `*_keywords` parameters likewise;
  - template generators: `generate_policy_filters` → `generate_routing_templates`, `generate_packing_jobs` → `generate_packing_templates`, `generate_route_improvement_interceptors` → `generate_route_improvement_templates`, `generate_acceptance_criteria_rules` → `generate_acceptance_criteria_templates`.
  The `hpo_sim` config fields (`selection_name`, `improver_name`, `acceptance_name` and their `*_keywords`) are unchanged.
- **Small simulator modules merged** (`logic/src/pipeline/simulations/`):
  - `rng.py` (day random streams) and `problem_flags.py` (`per_trip_shift`) → `day_setup.py`;
  - `solver_status.py` (per-day solver status and certificate) and `aggregation.py` (sample aggregation) → `reporting.py`.
  The functions keep their names. `tools/gates/capture.py` falls back to `solver_status` on trees with the old layout, such as the gate baseline.
- **Moved:** `pipeline/simulations/failure_analyzer.py` → `tracking/logging/modules/failure_analyzer.py`, next to `failure_emit.py`, which serialises its summaries. It analyses a finished day; it is not a step of the simulation.
- **Moved:** `logic/src/envs` → `logic/src/pipeline/rl/envs`, and its tests from `logic/test/unit/envs` to `logic/test/unit/pipeline/rl/envs`. Imports are `logic.src.pipeline.rl.envs.*`. `logic/release.yaml`, the release build check in `logic/package/build_release.py`, the docs and the notebooks use the new path; `docs/api/envs.md` is folded into `docs/api/pipeline.md`. No saved model checkpoint pickles a reference to the old module path (all 374 `.pt`/`.ckpt` files checked).
- **Renamed:** `logic/src/pipeline/rl` → `logic/src/pipeline/learning` (tests: `logic/test/unit/pipeline/rl` → `logic/test/unit/pipeline/learning`). The environments are therefore at `logic.src.pipeline.learning.envs`. Registry targets, `logic/release.yaml`, `logic/package/build_release.py`, docs and notebooks are updated; the PyInstaller specs collect every `logic` submodule and need no change. The `rl` config group and `cfg.rl` keys are unchanged.
- **Renamed:** `logic/src/policies/helpers/reinforcement_learning` → `logic/src/policies/helpers/learning`. The config module `configs/policies/other/reinforcement_learning.py` keeps its name.

### SANS operators moved to `helpers/operators` (`2026-10-18`)

The neighbourhood and perturbation operators of `simulated_annealing_neighborhood_search` now live under `logic/src/policies/helpers/operators`. Operators that did the same thing are merged. Every SANS and LAC call site keeps its exact behaviour, including the random numbers it draws: a characterization run of all 61 operator call forms over 150 seeds matched the previous outputs and generator states. The SA engines, their dispatch tables (`heuristics/sans_operators.py`, `search/random_search.py`), the objectives and the refinement drivers stay in the SANS package.

#### Changed

- **Random moves and swaps** (`perturbation_shaking/sans/`):
  - Merged: `move_1_route`, `swap_1_route`, `move_2_routes` and `swap_2_routes` were the n = 1 cases of `move_n_route_random`, `swap_n_route_random`, `move_n_2_routes_random` and `swap_n_2_routes_random`. They are deleted; callers pass `n=1`.
  - Merged: the new engine's `mutate_route_by_swapping_bins` into `swap_n_route_random`. It gains `n_choices` and `min_route_len`.
  - Merged: the new engine's consecutive segment swap into `swap_n_2_routes_consecutive`. The new flag `segment_may_cover_end_depot` is True for LAC, whose segment can carry a route's closing depot, and False for the new engine.
  - Moved from `heuristics/sans_neighborhoods.py`:
    - `relocate_within_route` and `or_opt_move` to `intra_move.py`;
    - `move_between_routes` to `inter_move.py`;
    - `cross_exchange` to `inter_swap.py`, renamed `exchange_route_tails`.
  - Moved from `heuristics/sans_perturbations.py`, renamed so they no longer share names with the different intra-route operators:
    - `move_n_route_random` → `transfer_n_bins_random`;
    - `move_n_route_consecutive` → `transfer_n_bins_consecutive`;
    - `swap_n_route_random` → `exchange_n_bins_random`.
  - Moved from `common/routes.py`: `rearrange_part_route`, to `intra_move.py`.
- **Objective-driven passes** now take the objective as two callables, `profit_fn` and `real_profit_fn`, so `helpers/operators` does not import SANS objectives. `refinement/` builds them from the penalised and real profit.
  - Merged: `search/deterministic.py::local_search_2` and `search/reversed.py::local_search_reversed`, which differed only in scan order, into `improvement_descent/relocate_descent.py::relocate_descent(..., reverse=False)`.
  - Moved: `select/greedy.py::insert_bins` to `recreate_repair/bin_pool.py`, and `remove_bins_end` to `destroy_ruin/bin_pool.py`.
  - Merged: `common/routes.py::two_opt_uncross_arc`, a first-improvement 2-opt, into `intra_route_local_search/k_opt.py::two_opt_route`. That function gains `max_iter=None` (run until no reversal improves) and `tol`; SANS passes `max_iter=None, tol=0.0`. The routes are the same on the 150 characterization cases and on 400 cases built with co-located bins to force ties.
  - Moved: `common/solution_initialization.py` to `solution_initialization/zone_si.py`.
- **Regression test:** `logic/test/unit/policies/test_sans_operator_parity.py` pins every SANS and LAC call form, and the two refinement drivers, to hashes taken from the pre-refactor tree. Each hash covers the operator's result and the generator's next draw.
- **Docs:** `docs/api/policies.md` regenerated, and the SANS tree in `docs/modules/POLICIES_MODULE.md` updated.
- **Moved to other families:**
  - `get_2opt_neighbors` → `evolutionary_mutation/random_2opt.py`;
  - `organize_route` → `solution_initialization/nearest_neighbor_si.py`.
- **Bin-pool removal and reinsertion** (`destroy_ruin/bin_pool.py`, `recreate_repair/bin_pool.py`). These replace `select/random.py`, `select/consecutive.py` and `heuristics/sans_perturbations.py`, and take `insert_bin_in_route` from `heuristics/sans_neighborhoods.py`.
  - Merged: LAC's `remove_bin` into `remove_n_bins_random`, which gains `n`, `require_spare_bin`, `max_resamples` and `prune_chosen_empty_route`. LAC calls it with `n=1`, the guard, the unbounded redraw and the empty-route pruning that `remove_bin` had.
  - Merged: LAC's `add_bin` into `add_n_bins_random(n=1)`.
  - Renamed: the new engine's pool operators, which differ from LAC's (they return copies and check capacity):
    - `remove_n_bins_random` → `remove_n_bins_across_routes`;
    - `remove_n_bins_consecutive` → `remove_segment_without_mandatory`;
    - `add_n_bins_random` → `insert_n_bins_cheapest`;
    - `add_n_bins_consecutive` → `insert_n_bins_block`;
    - `add_route_with_removed_bins_random` → `add_route_from_pool_capacitated`.
  - Moved: LAC's route splits `add_route_random` and `add_route_consecutive` to `perturbation_shaking/sans/inter_move.py`.
- **Removed:**
  - `heuristics/sans_neighborhoods.py`, `heuristics/sans_perturbations.py`, `select/` and `search/deterministic.py` / `search/reversed.py`, now empty;
  - `get_neighbors`, which nothing called;
  - `add_route_with_removed_bins_consecutive` of the new engine, which only called the random version.

### ACO-HH no longer drops mandatory bins (`2026-10-18`)

#### Fixed

- **The profit-aware kick dropped mandatory nodes.** `kick_profit` reinserted removed nodes only when their insertion was profitable, so an unprofitable mandatory bin stayed unvisited. ACO-HH scores plans by routing cost alone, so it kept the shorter plan. On random 40-node instances with 15 mandatory bins, ACO-HH with its default operators missed 60 mandatory visits over 10 seeds, 41 of them with `kick` alone.
  - `_profit_reinsertion` now always reinserts mandatory nodes: at their most profitable feasible position, or else on a new route.
  - `perturb` and `perturb_profit` no longer swap a mandatory node out for an unvisited one.
  - `HyperHeuristicACO._is_feasible` rejects any plan that misses a mandatory node, so such a plan can never become the incumbent.

### `envs/tasks` and `envs/generators` folded into `envs` (`2026-10-18`)

#### Changed

- **One module per problem: `logic/src/envs/<problem>.py`** (`ptp`, `cptp`, `tcptp`, `cvrp`, `tsp`, `atsp`, `op`, `pctsp`, `spctsp`, `pdp`, `irp`, `thop`). Each holds its instance generator, environment and task class; the generators of `envs/generators/` were merged into these modules.
- **Shared code:**
  - `envs/tasks/base.py` (`BaseProblem`) → `envs/base/problem.py`.
  - `envs/generators/base.py` (`Generator`, `RoutingGenerator`) → `envs/base/generator.py`.
  - `envs/tasks/common/`, `envs/tasks/core/` and `envs/tasks/spec.py` → `envs/common/`, `envs/core/` and `envs/spec.py`.
- **`envs/tasks/` and `envs/generators/` are deleted.** `logic.src.envs` now serves everything both packages exported, lazily: the env, generator and task classes, `get_env`, and `get_generator`. `GENERATOR_REGISTRY` is now a mapping that imports a class on lookup.
- **Updated to match:**
  - every import and patch target in `logic/`;
  - `logic/release.yaml` and the release check in `build_release.py`;
  - `docs/modules/ENVS_MODULE.md` and `docs/api`.

### Route improvers price moves with directed distances (`2026-10-18`)

#### Fixed

- **Fast-TSP drove small trips backwards.** fast-tsp 0.1.5 treats its matrix as symmetric. For tours of up to 20 nodes it returns the optimum of the *transposed* matrix, and the wrapper (`travelling_salesman_problem/tsp.py::find_route`) drove that order forward. On the paper's 30-day outputs, the reverse of a Fast-TSP trip of up to 19 bins was shorter in 93% of trips.
  - `find_route` now drives the cheaper direction on the true matrix, then polishes the route with `directed_two_opt_route`.
  - It also no longer fails on an empty bin list.
- **2-opt, 3-opt and k-opt ignored the direction of reversed segments.** `k_opt.py` priced a reversal by its exchanged edges only. On an asymmetric matrix an "improving" move could lengthen the route.
  - The new `reversal_delta` adds the cost of driving the reversed interior backwards. It is zero, and skipped, for symmetric matrices; the symmetry check is cached on the `LocalSearch`.
  - Applied in `_apply_2opt`, `_apply_3opt`, the k-opt configuration search and `two_opt_route`.
- **The CLS 2-opt sweep never broke the first depot edge**, so it could not reverse a route's leading segment or a whole route. `SweepLocalSearch.two_opt_intra` now starts from the depot.
- Every `LocalSearch` that uses these operators (HGS, ILS, FILO, SA and the improvers) now evaluates them correctly on road matrices. Results on asymmetric instances change. The archived paper outputs were produced before this fix.

#### Added

- `directed_two_opt_route`: best-improvement 2-opt with exact directed move costs.
- `test_directed_route_moves.py`: every applied move must shorten the directed route; `find_route` must drive a directed ring forward.

### One local search implementation (`2026-10-18`)

#### Changed

- **`local_search_manager.py` is deleted; `LocalSearch` (`local_search_base.py`) is the single local search state.** The manager duplicated the base's state and caches (routes, loads, distance access) in a standalone class. Its named first-improvement sweeps (`relocate`, `swap`, `two_opt_intra`, `or_opt`, `two_opt_star`, `swap_star`, `cross_exchange_op`, ..., `lns`, `ges`, `lkh_refinement`) now live in `SweepLocalSearch(LocalSearch)` (`local_search_sweeps.py`, exported by the package).
  - It takes the manager's constructor arguments, plus `mandatory_nodes` and `penalty_capacity`, and adds `optimize()`: descend with a list of sweeps until none improves.
  - It builds no granular neighbour lists, so moves do not maintain the top-insertion cache.
  - Its callers switch from `LocalSearchManager` to it: the classical, random, Or-opt, cross-exchange and guided local search improvers, HGS-ADC and GIHH.
  - **Same moves as before.** On seeded random instances, every sweep applies the same sequence of moves as the manager did (checked for all 21 operators), except `swap_star` and `relocate` (both fixed below).

#### Fixed

- **`swap_star` raised `AttributeError`** under the manager, which had no `top_insertions`, whenever two non-empty routes existed. The improvers that wrap their loop in `try` (classical local search with `ls_operator: all`, random local search) then returned the tour unimproved. HGS-ADC's intra-day education step raised whenever a day had two or more routes. It now runs. The paper's improvers (`ls_operator: 2opt` and fast TSP) never called it.
- **The `relocate` sweep cycled.** It also tried relocating a node after itself. `move_relocate` priced that as a gain, but applying it swaps the node with its successor, a longer tour, which the next sweep undid. A descent then alternated between the two tours until its iteration cap. `move_relocate` now rejects `v == u`. `LocalSearch`'s own neighbour loop never pairs a node with itself, so it is unaffected. `relocate` was in the classical improver's `all` suite, random local search and guided local search, not in the paper's improvers.

### Studio startup crash on empty plugin config (`2026-10-18`)

#### Fixed

- **WSmart-Route Studio exited on launch** (`just studio` / `tauri dev`) with `PluginInitialization("notification", "invalid type: map, expected unit")`. `app/src-tauri/tauri.conf.json` had `"notification": {}` (and the same empty object for `dialog` and `store`). Those plugins deserialize their config as unit (`null` / omitted), not a map; they are already registered in `lib.rs`. The empty keys are removed. `shell` and `updater` keep the config they actually read.


### One generator class per problem (`2026-10-18`)

#### Changed

- **The instance generators live only in `logic/src/envs/generators/`.** `envs/tasks/` held a second `PTPGenerator`, `CPTPGenerator`, `TCPTPGenerator`, `CVRPGenerator`, `OPGenerator`, `PDPGenerator` and `TSPGenerator`, used by the environments, while `get_generator`/`GENERATOR_REGISTRY` returned the `envs/generators/` classes of the same names. Each pair is now one class in `envs/generators/<problem>.py`. The task modules import it (as `atsp`, `irp`, `pctsp` and `thop` already did) and keep re-exporting the name.
  - **Defaults reproduce what the environments generated.** With default arguments and the same torch seed, every merged generator returns bit-identical instances to the old env generator (checked for all seven).
  - **What the `envs/generators/` copies added is kept as options:** `depot_type` (now applied; it was stored but ignored by the env's PTP generator), CVRP `vehicle_capacity`, OP `prize_type` (`"dist"`, `"unif"`, `"const"`, plus the env's continuous `"uniform"` default) and `max_length=None` for the size table, and an even `num_loc` for PDP. The env-side additions are kept too: `dm_filepath` (road distances) and the `capacity`/`max_waste` fields.
  - **Defaults that changed for `get_generator` callers**, which used to get the `envs/generators/` defaults: PTP `depot_type` `"random"` (was `"corner"`); CVRP `num_loc` 50 and `max_demand` 9 (were 20 and 10); OP `prize_type="uniform"` and `max_length=2.0` (were `"dist"` and the table); TSP draws the depot after the customers.
  - **PTP waste:** a non-uniform `waste_distribution` (e.g. `gamma1`, `dist`) without an empirical grid is now drawn from that distribution. The env generator used to fall back to uniform silently. Distance-dependent waste is computed on the instance's own coordinates instead of a fresh, unrelated draw.
  - `RoutingGenerator` moved to `envs/generators/base.py` (re-exported by `envs/base/routing.py`) and gained `depot_type`. `CPTPGenerator` is new in `envs/generators/cptp.py`, and the registry's `"cptp"` entry is that class instead of `PTPGenerator`. `Generator.to()` keeps `dm_filepath`.
- `docs/api` regenerated.

#### Fixed

- **Paths left by the move of `envs/tasks/profitable_tour/` into `envs/tasks/`:**
  - The PTP, CPTP and TCPTP `env_target`s in `released_specs.py` (`PROBLEMS["ptp"].get_env()` raised `ImportError`).
  - The lazy `envs.tasks` exports.
  - The `closed_tour_length` import of `tasks/cvrp.py`.
  - Fifteen test modules.

### BPC-CPTP fork replaces the cptp submodule (`2026-10-18`)

#### Changed

- **`logic/submodules/cptp` → `logic/submodules/BPC-CPTP`**, now tracking the fork https://github.com/ACFHarbinger/BPC-CPTP.git in place of https://github.com/spoorendonk/cptp.git. It is pinned at the same commit (`de21477`), so the pricer's code is unchanged and the package is still imported as `cptp`. `logic/pyproject.toml` (`[tool.uv.sources]`), `uv.lock` and the docstrings and messages that name the path are updated. Install it with `uv pip install logic/submodules/BPC-CPTP`.

### BPC: CPTP branch-and-cut pricing engine (`2026-10-18`)

#### Added

- **`logic/submodules/cptp`**: the open CPTP branch-and-cut of Spoorendonk (2026, arXiv:2607.04497; v0.2.0 plus docs/build commits, `de21477`), a re-implementation of Jepsen et al. (2014) on HiGHS. It is the optional extra `cptp` (`uv pip install logic/submodules/cptp`; needs CMake and a C++23 compiler) and is not part of the 1.0.0 release.
- **`pricing_engine`** (BPC, default `labelling`; plus `cptp_threads`, default 1). With `cptp`, the pricing subproblem goes to the branch-and-cut (`pricing/cptp_pricer.py`):
  - **Mapping.** Node coverage and node-LCI duals become node profits; RCC and edge-clique duals become edge costs; the fleet dual is a constant.
  - **Fallback to labelling.** A call falls back when it holds something CPTP cannot express (Farkas pricing, must-use arcs, Ryan-Foster branches, subset-row, rank-1, multistar or arc-LCI duals), when the tour has no orientation avoiding the node's forbidden arcs, or when the tour is already a master column (a column at its upper bound keeps a positive reduced cost, and CPTP returns one tour per call).
  - **Cut families.** Only those the pricer can express are separated: rounded capacity, knapsack cover, edge clique, fleet cover and node-based physical LCIs (`branch_and_price_and_cut/engines.py`).
  - **Asymmetric matrices.** These are priced on `(d + dᵀ)/2`; each route is oriented on the true costs and its reduced cost recomputed exactly. Pricing is exact on symmetric matrices only.
  - **Extra columns.** Each CPTP tour also yields its drop-one-bin neighbours that keep a positive reduced cost: CPTP returns one tour per call, while the labelling returns up to `max_routes`.
  - **Timeouts.** Only a CPTP solve with no tour counts as a pricing timeout.
  - **Fallback days are recorded.** When no integer node is reached, BPC publishes `bpc:fallback_greedy` or `bpc:fallback_rmp_ip` as the day's `solver_status`, so the frequency of its fallback plan is countable from the day logs (for either pricing engine).
  - **`build_simulation_summary.py --override ROOT`** replaces the rows of the same configurations with those built from another output root, such as a one-algorithm rerun.
  - **`cptp_options`** passes cptp solver options. The default caps its capacity-aware labelling at 100,000 queue pops: at cptp's default of 1,000,000 that preprocessing ignores the time limit, and it took about 35 s of a 49 s call at 350 nodes.
  - **Logging.** The engine logs its call, outcome and fallback counts per solve.

### Modules over 500 code lines split (`2026-10-18`)

#### Changed

- **Every `logic/` file is now under 500 code lines** (`logic/validation/count_loc.py`, Code column; it was 15 files). Behaviour is unchanged; the old module paths keep re-exporting what callers import.
  - `search/cutting_planes.py` → `cut_engine_base.py`, `cuts_capacity.py`, `cuts_subset_row.py`, `cuts_fleet_cover.py` (the factory and composite stay).
  - `pricing/solver.py` → `labeling.py` (`RCSPPLabelingMixin`) and `pricing_types.py`.
  - Single-class solvers get mixins in sibling modules: ACO-HH `bootstrap.py`; ALNS-SARSA `alns_sarsa_operators.py`; tabu search `moves.py`, `memory.py`, `ls_adapter.py`; branch-and-cut `cut_callbacks.py`, `lagrangian_bound.py`; LKH-3 ALNS `alns_operators.py`, `alns_routing.py`.
  - Function modules: POPMUSIC `pmedian.py`, `subproblem.py`; LKH-3 `lkh3_alns_entry.py`; SWC-TCF `formulations.py`; MS-BPC-SP `node_selection.py`; results tables `tracking/logging/modules/summary_display.py`.
  - `logic/gen/gen_paper_latex.py` → `paper_data.py`, `paper_figures.py`, `paper_diagrams.py`, `paper_networks.py`. Its tables and figures are byte-identical to before.
  - `logic/package/build_release.py` → `release_closure.py`, `release_staging.py`, `release_files.py`.
  - `test_validation_checkers.py` → plus `test_validation_graph_checkers.py`, `test_validation_code_checkers.py`.
- The two tests that patched a moved name now patch its new module (`formulations.quicksum`, `release_closure.REPO`/`LOGIC`/`STAGES`).
- `docs/api` regenerated.

### ESPPRC env module removed (`2026-10-18`)

#### Changed

- **`logic/src/envs/tasks/espprc.py` is deleted.** Its shared machinery moved to `logic/src/envs/base/`: `resources.py` (`Resource`, `LoadResource`, `ShiftTimeResource`) and `routing.py` (`RoutingEnv`, `RoutingGenerator`, formerly `ESPPRCEnv`/`ESPPRCGenerator`). Every env (PTP, CPTP, TCPTP, CVRP and the unreleased ones) now subclasses `RoutingEnv`; the `espprc` entry of the env registry is gone. The release manifest seeds the two new modules. The trained models do not record env module paths, so they load unchanged.

### BPC greedy warm-start switch (`2026-10-18`)

#### Added

- **`greedy_warm_start`** (BPC and MS-BPC-SP, default `true`). Off, the restricted master starts with an empty column pool: no greedy mandatory-bin routes and no optional-bin seed routes, and column generation (Phase I Farkas pricing while mandatory bins are uncovered) builds the whole pool. `profit_aware_operators` still picks the greedy variant when the switch is on. Both engines now share `master_problem/warm_start.py`, which also gives MS-BPC-SP the copy fix BPC already had: the optional seeding no longer mutates the mandatory columns' node lists in place.

### 1.0.0 release package: build fixes found by running the release (`2026-10-17`)

#### Fixed

- **Default policy YAML crashed every simulation.** The routing base took a flat config's `acceptance_criteria` list for the policy section and raised "policy YAML must be a mapping after P-13" on day 1 (`base_routing_policy.py`). Stage keys are no longer section candidates.
- **PG-CLNS adapter failed to load.** The simulator's `_AttrDict` sub-configs reached `OmegaConf.create`, which rejects dict subclasses. `_structured_load` now converts them to plain containers first.
- **Relative `sim.output_dir` split one run across two folders.** The run context, `compose_dirpath` and the analysis scanner wrote and read under `assets/<output_dir>`, while the summary writer used `<output_dir>`. A relative output folder therefore crashed at the end of every run; the paper runs used absolute paths. All four now resolve against the repository root.
- **Python 3.11+ import error.** Five config dataclasses (`HGSConfig`, `RTSConfig`, `KSparseACOConfig`, `GIHHConfig`, `SAConfig`) used an `AcceptanceConfig(...)` instance as a field default; they now use `default_factory`.
- **Last-minute settings capture (#61).** The `consumer_last_minute` live-parameter record, dropped by WP-09, is restored. It is looked up through `sys.modules`, so packing stays torch-free.
- **AM packing variant.** `policy_na.yaml` asked for a `default` variant that `pk_lookahead.yaml` does not define, so its results were logged as `pk_lookahead_lookahead_na_amgat_none` rather than `lookahead_na_amgat_none`, which the summary builder skips. It now uses the `lookahead` variant; every released policy's variant references were checked.
- **Training failed in the release.** `run_training` imported `run_hpo`, which the release prunes; it is now imported only when `hpo.n_trials > 0`. The released vectorised packing strategies (`lookahead`, `last_minute`, `service_level`, `regular`, `spec`) are loaded by string target, so the builder pruned them; they are now release seeds. `plotext` is capped below 6 (6.x removed `plt.clf`, which the display callbacks use; main's lock already had 5.3.2). REINFORCE and imitation training now run from the delivered package.
- **Defaults.** The AM policy's default `model_path` now points to the paper model (`cptp100_riomaior_plastic/emp/amgat0-il-rl`), and the imitation expert defaults to `profitable_nn`, the paper's expert.

#### Changed

- `logic/package/build_release.py`:
  - ships `__main__.py`;
  - drops the task YAMLs' defaults for unreleased policies (Hydra could not compose `test_sim` otherwise);
  - keeps every `configs/policies/other` file a shipped policy references, plus the `ac_*`/`ri_*` files of released keys;
  - carries the PyG index into the release `pyproject.toml`;
  - lets `release.yaml` add requirements to the core dependencies or to any extra.
- `release.yaml` adds the packages the shipped code imports but the release did not declare: `plotext` and `seaborn` (core), `pyomo` (solvers), `lightning`, `tensorboard`, `loss-landscapes` and `plotly` (neural).
- The setup scripts (`desktop/*/setup_package.*`) target the 1.0.0 layout: a root `pyproject.toml`, and `uv sync --extra solvers --extra neural`.
- `ci/build_windows_exe.sh` reads the 1.0.0 root `pyproject.toml` (core, `solvers` and `neural`; without torch-scatter/torch-sparse, which have no Windows wheels, and Pyomo).
- The release's package-data now covers the data files the shipped code reads (`src/**/*.sql`, `*.json`, `*.html`, `py.typed`); before, only `popup.html` was installed and a non-editable install lacked the tracking queries. The builder also drops the data files of excluded packages (the simulation-HPO rules and jobs).
- `ci/simulator_release.spec` builds the release executable. Unlike the minimal spec, it bundles OR-Tools, PyVRP, Fast-TSP and matplotlib, the tracking SQL/JSON files, SciPy's dynamically imported `scipy._external` modules, and the sources of `torch_geometric` and `lightning`, which TorchScript and `inspect` read at import.
- Tests updated: output paths, a removed `get_cmap` patch, and the HPO health test skipping without optuna. The unit suite gives 2,204 passed, 4 skipped, with no deselection. Every one of the ten released routing algorithms completes a 3-day simulation from the built release in its own Python 3.12 environment.

#### Open

- The release's curated test set (`logic/test/release_tests.txt`) does not run from the built tree: its `conftest.py` is pruned (it registers fixture plugins of unreleased modules), and other tests use unreleased acceptance criteria or HPO. The delivered package ships without `logic/test`, as the previous one did.

### 1.0.0 refactor merged into `main` (`2026-10-16`)

#### Changed

- `main` now holds the 1.0.0 refactor: `refactor/v1-integration` (WP-01 to WP-13) plus Codex's review fixes (`refactor/v1-review-fixes`, including its corrected Q-14 harness), merged with `main`'s paper and AM work.
  - Merge conflicts: the paper generator keeps the AM, static-reference and exclusion handling under the refactor's routing names (`table_routing`, `ROUTING_SPEC`, `routing_labels`). The routing table is now `Tables/results_routing.tex` (paper submodule `f6a2fcd`).
  - The `wsmart_bin_analysis` submodule now lives in `packages/`.
- Three unit tests were updated to the refactored simulator and structured configs: the checkpoint-error result fields, the opt-in fill-matrix export, and HGS/ALNS `engine` defaults. The full unit suite gives 2,203 passed, 4 skipped and 1 failed.
  - The failure is the slow integration test `test_live_test_sim_captures_ms_ri_ac`. It fails the same way on the pre-merge `main`, so the merge didn't cause it.

#### Open

- Q-14: the ALNS replay matches on every tour, status and physical KPI. The strict comparison still fails on one-ULP kg and kg/km differences and on the candidate-only `mean.n_failed`/`mean.n_success` keys.

### Attention Model results in the simulation paper (`2026-10-16`)

#### Added

- The Attention Model's results in the simulation paper (submodule `a963a7f`). The AM row of the routing table, and its blocks under the packing, improver, horizon and scenario tables, are generated from data. The AM sentences in the abstract, results, discussion and conclusion are filled in.
- `docs/private/global/simulation/rerun-2026-10-16/`: the summary vintage the paper now reads, built from `outputs/simulations`. It adds the AM rows (60 per horizon) and PSOMA's rerun rows; its README lists the differences from `rerun-2026-10-13`.
- `logic/gen/build_simulation_summary.py`: the AM (`na_amgat`, labelled `AM`) is now a summarised routing algorithm.
- `logic/gen/gen_paper_latex.py`:
  - keeps AM runs out of the classical aggregates and cell medians;
  - puts the AM through the same exclusion rule, measured against the classical cell median;
  - renders the AM as a separate block under each table (`learned_runs`, template `blocks`);
  - keeps the static-reference placeholder rows (`pending_blocks` in the config) until those results are summarised.
  - removes a stale `results_excluded.tex` when no run is excluded (with `--force`).

- **Static reference in the paper.** The summary builder reads the `static_cvrp` runs (`constructor` = `CVRP`, `strategy` = `Regular-3/4/5`, `improver` = `None`). The generator reports them as their own block under the routing table, outside every classical aggregate. The paper's static-reference text is filled in, and no TODOs remain.
- **Fill-trajectory figure regenerated.** It now reads the scenario's deposits from the demand dataset. The old code looked for fill histories under the removed `assets/output`, and those workbooks hold the deposits rather than fill levels, so it always skipped. The paper's claim that LM-CF70 never lets a bin approach capacity is corrected.

#### Changed

- **Paper runtime description.** The paper now describes runtime as what this simulator records: each run's policy time (packing, routing and improvement summed over days, without simulator work). It previously called it archived elapsed time including overhead. The same change applies to the table notes and the scaling-figure caption.
- **PSOMA's rerun replaces its earlier runs in the paper.** That rerun followed the budget and seed-particle fixes `244b7479e` and `e97180567`. The earlier Figueira da Foz Empirical runs fell 25–44% short and removed that cell for every algorithm; the rerun runs do not. So all 480 classical 30-day runs and all 174 horizon pairs are now retained, and every classical number in the results text is updated.
  - Finding change: the packing range of mean efficiency (2.38 kg/km) is now wider than the routing range (1.12). The text no longer says the two ranges are similar.
- **Figueira da Foz 350 Empirical AM model retrained.** The new model used four times the training instances, imitation over all 31 days, and fine-tuning on days 17–31. Its 20 cells were re-simulated. The paper's methodology and parameter appendix state the recipe.
- `gen_paper_latex.py` now reads `rerun-2026-10-16` by default; it used to read the September summaries. The constructor, horizon and scenario table notes now compute their counts instead of hard-coding them. The default figures directory is now `Images/Generated`, the one the paper uses.

### Grant renewal-report generator (`2026-10-15`)

#### Added

- `logic/gen/gen_grant_report.py`: generates the grant holder's renewal report ("Relatório de Actividades do Bolseiro") as an editable `.docx`, with the IST logo (`logic/gen/svg/Tecnico_ID_Principal_RBG-COR.svg`, rasterised with Inkscape or ImageMagick) in the page header, a centred title (and an optional subtitle), the period, the grant holder's full name and grant ID, the research report (a list of paragraphs, with `line_spacing` set in the JSON) and a signature block (label, a line to sign on, and the printed name). The package is written as plain Office Open XML with the standard library, so there is no new Python dependency. Default output: `docs/private/reports/relatorio_renovacao_bolsa.docx`. The generator does not overwrite an existing report (which may hold your edits) without `--force`.
- `logic/gen/json/grant_report.json`: the period (Setembro de 2025 a Setembro de 2026), the full name, the grant ID (2025.06860.BDANA), the signature and a one-page Portuguese research report for the WSmart Route+ work at INESC-ID: the simulator and the work with the partner company, exact and heuristic methods, the neural models, the survey and the papers.
- `tools/gen/justfile`, imported in the root justfile as `gen` and listed in `just help`: `grant-report` (regenerates the report, overwriting it), `paper-tables`, `paper-pdf` and `paper`, `sim-summary` and `dist-matrix`, all running the `logic/gen` scripts (the Studio's deck and analysis generators run from inside the app).

### Simulator 1.0.0 Refactor Proposal Review (`2026-10-15`)

#### Added

- Architectural analysis and proposal additions for Simulator 1.0.0 refactor (`.agent/tasks/simulator-v1-refactor-proposal-2026-10-15.md`, draft 8):
  - `P-20`: Identified 6 live call sites invoking legacy unvectorized `BaseProblem.get_costs`; recommended immediate clean break removal.
  - `P-22`: Proposed unified `NodeEmbeddingModule(static_dim, dynamic_dim, embed_dim)` with lazy input projection.
  - `P-23`: Formulated vectorized tensor execution contract for `Resource` abstractions (`Capacity`, `TimeWindow`, `Pickups`, `Battery`) under `ESPPRCEnv`.
  - `P-24`: Proposed standard neural policy output contract (`{"actions": Tensor, "log_likelihood": Tensor, "reward": Optional[Tensor]}`).
  - `P-25`: Proposed lazy subnet and policy factory registry to avoid eager heavyweight PyG imports.
  - `P-51`: Proposed unified offline expert trajectory cache format (`.td`) for deterministic multi-day imitation (`fill_from: expert`).
  - Submitted round-1 reviewer questions `RQ-gemini-1` through `RQ-gemini-4`.
### Integration reconciliation (Codex, 2026-10-16; uncommitted review fixes)

- Reconciled RL-GD-HH's `gd` name and absolute target parameters; preserved legacy config translation. Acceptance prototypes are isolated per solve. ALNS uses a nested calibrated BMC schedule; PG-CLNS retains its separate legacy schedule and solver RNG.
- Corrected config/YAML drift, context action access and active tracking after removal of `reward`. Restored classical imports without torch and completed release wheel dependencies and YAML package data. A fresh base-plus-solvers installation passes the blocked-neural smoke.
- Gate overrides now use Hydra syntax and materialize each target checkout's full YAML. Partial bodies had selected different defaults. BPC daily proof records pair an actual incumbent and exhausted-root bound with a problem fingerprint; missing proof cannot pass the fallback.
- Prepared an explicit baseline-only adapter correction: adding the runtime seed no longer hides `vrpp: true` or mutates the reusable policy wrapper. The harness requires its exact source hash over `d82170cf0` and records that amendment in frozen evidence. See `tools/gates/BASELINE.md` and the adjacent patch; no optimization loops are amended.
- Verification: fixture-enabled unit suite 2,124 passed, 4 skipped, 79 deselected; updated gate selection 47 passed; baseline dispatch regression 4 passed; HPO 96/96 and banned-Params guard pass. Full Q-14 calibration and paired/archived gates remain required before merge.

### Integration of the 1.0.0 refactor (`refactor/v1-integration`, `2026-10-15`)

WP-01 to WP-13 merged in report §12 order, each package with Codex's review edits committed on its branch first (WP-08 and P-21/P-42 were wholly uncommitted; WP-08 is stacked on P-21/P-42). Not merged into `main` (Q-15).

#### Fixed

- **Merges.** WP-02 and WP-10 both made the training facades lazy; the superset lazy maps are kept, with WP-02's expert map in `rl/core/imitation.py`. WP-11 was built on WP-05, not WP-06: its MS-BPC-SP engine keeps the WP-06 `MSBPCSPConfig`, and its `vrpp` reads follow WP-08's `current_expand_pool()` (P-43). WP-12's CVRP `solution_limit` moves from the deleted `CVRPParams` onto `CVRPConfig`. WP-13's docstring trim keeps the integrated code on conflict, and the trimmer and API generator are re-run on the merged tree.
- **WP-06 leftovers on the released path.** `NeuralAgentConfig.from_config` was dropped while the neural agent still called it, and it read neither the plain-mapping YAML (P-13) nor a direct `decoding` mapping, so `policy_na.yaml`'s `beam_width: 5` became 1. The JGO/JSA/NDS-BRKGA joint policies called a method their base lacks. Duplicate and unused imports from the scripted migration are removed.
- **Released envs (CR-12).** PTP masks empty bins again unless they are mandatory, and TCPTP ends a day only when no customer is reachable on a fresh trip; WP-07 had made both inherit ESPPRC defaults (selectable empty bins; one trip per day). 42 seeded rollouts of the pre-WP-07 PTP/CPTP/TCPTP/CVRP envs at `a5eb27eec` now replay identically (18 differed). `test_released_env_semantics.py`.
- **Release build (CR-15).** The closure indexes lazy maps instead of following every entry, so the release ships the paper stack only (`models/core`: `attention_model`; `rl/core`: `reinforce`, `imitation`, `losses`; 1,850 → 1,222 files). The imitation factory is a seed, `packages/wsmart_bin_analysis` ships vendored with matplotlib as a dependency, and the P-24 `policy_contract` is a seed. Test selection follows `from <pkg> import <submodule>`. The smoke now also resolves every released routing key and imports the simulator from the built tree. `--check`, `--build` and `--smoke` pass.
- **Regression gates (CR-13).** SANS has no `time_limit > 0` guard, so the gate's `time_limit: 0` stopped it at once; its cell now runs with no clock and `iterations_per_T: 200`, and keeps the paper schedule in the time-capped repetitions. Calibration runs on `d82170cf0` (branch `gates/baseline-a5eb27eec-cvrp-knob`): `a5eb27eec` plus only the default-off CVRP knob, with the harness. The guard still refuses any other code.
- **Tests.** The export-profile script tests WP-02 retired are removed; tests follow the Q-7b docstring rule, the lazy imitation import and the `pk_*` packing files.

#### Changed

- **P-51 (owner amendment, CR-16).** The precompute switch (`rl.imitation.precompute`) and the state-keyed expert cache are the delivered interface. There is no `gen_data expert` command or `.td` `expert_actions` field: the cache key is the exact training state, so precomputed and live solves cannot disagree.

#### Open

- **CR-14 (P-11 for ALNS).** `ALNSConfig` still carries `start_temp`, `cooling_rate` and `start_temp_control`. In ALNS they only switch the Ropke & Pisinger calibration, but PG-CLNS reads `start_temp` and `cooling_rate` live through WP-11's `LegacySAAcceptance`, which draws from the solver's own random stream. Moving them needs a PG-CLNS LNS config, a BMC `start_temp_control` applied in `setup()`, and YAML/HPO migration, and bit-identity has to be shown by the Q-14 gates, which wait for `sim chain done`.
- **Q-16a.** The released routing adapters (through `base_routing_policy` → `simulations.repository`) and the day-loop states still import torch: 28 modules on the simulator path import the torch stack at module level. The release installs without torch, but running the simulator needs the `neural` extra.
- **CR-13.** Per-instance/day BPC certificates for RQ-kimi-1 need solver instrumentation in both trees. No heavy gate or calibration has run.
- **Unreleased code.** WP-06 completion (the dropped Params methods, acceptance criteria built per solve) and CR-12's unreleased envs (OP, PDP, IRP, ThOP, ATSP, PCTSP, SPCTSP) are on `refactor/v1-review-fixes` for Codex review. 64 unit tests fail on this branch and 29 with `refactor/v1-review-fixes` on top, all predating integration: strict YAML keys of unreleased policies, WP-08 context and `expand_pool` follow-ups, and tests of the unreleased layer.
- **Paper.** The simulation paper still `\input`s `Tables/results_constructors`; it is renamed to `results_routing` when the refactor merges into `main`, because the AM tables are generated with main's code until then.

### Review fixes for the unreleased layer (`refactor/v1-review-fixes`, `2026-10-15`)

For Codex review before it merges into the integration branch.

#### Fixed

- **WP-06 completion (Q-8b).** `tools/migrate_unreleased_params.py` merged each `params.py`'s fields into its config but not its methods. 81 configs lost `from_config`, `to_dict`, `__post_init__` (default acceptance criteria such as TS's aspiration and RL-GD-HH's great deluge, operator weights, validation) and derived properties, while 19 adapters still called `XConfig.from_config`. The methods are ported verbatim from the WP-05 tree (the migration's input), with `Params` names renamed and the removed `vrpp` flag (P-43) dropped. Twelve multi-period and hyper-heuristic adapters declare their config class, so `_effective_config` returns the typed config instead of a dict.
- **Acceptance per solve (P-11).** `AcceptanceCriterionFactory.build` turns a criterion, an `AcceptanceConfig` or a mapping into a fresh criterion. GIHH, RL-GD-HH, ILS-RVND-SP, LB-VNS, ILS, TS and VNS use it instead of calling `setup()` on the config.
- **Unreleased envs (CR-12).** OP regains its reach-and-return `max_length` budget and prize reward, and PDP its pickup-before-delivery precedence. IRP, ThOP, ATSP and PCTSP/SPCTSP regain their original generators and dynamics on their ESPPRC classes. Seeded rollouts of the pre-WP-07 `envs/routing/*.py` replay identically. PCTSP's reward accepts the single per-step action, which crashed the original. `test_unreleased_env_semantics.py`.

### Cross-package review (Codex, `2026-10-15`)

- Codex review: root registry tag queries deduplicate class identity across child registries/aliases, and get_all preserves the union of child tags. Added a regression test.
- Codex review: release closure retains ancestor initializers and deferred runtime imports; multiline lazy maps are recognized. Builds reject populated/repository destinations instead of deleting them. Added isolated closure and destination tests.
- Codex review: invalid user values no longer bypass structured validation through the dataclass fallback; fallback is restricted to schema construction. Added type-validation regression tests.
- Codex review: check_hpo_jobs.py adds the repository to sys.path for its documented direct CLI/CI invocation.
- Codex review: structured-load value errors propagate instead of bypassing validation; the HPO key-check CLI resolves repository imports when executed by filename.
- Codex review: CVRP LoadResource consumes ESPPRC's depot-aligned waste vector, preserving customer-only demand input and avoiding mask dimension mismatch. Added a two-trip reset/capacity/completion test. Unreleased environment semantic migration remains blocked (see review report).
- Codex review: improvement precedes repair/collection; pipeline validation happens before RNG mutation and receives sim.day_pipeline. Archive conversion distinguishes daily, sample and std horizons and rejects overlapping trees. Propagated CVRP depot-aligned resources and structured-value validation fixes.
- Codex review: propagated the structured-loader correction so invalid user values cannot bypass OmegaConf validation. Focused BPC/MS-BPC-SP regression selection passes.
- Codex review: gates fail on incomplete/ambiguous sweeps and absent/unequal/non-finite certificates; retain physical time comparisons and the heavy lock. Calibration uses separate time-capped repeats; logs use canonical keys and recursive lowercase policy lookup. Fixed OR-Tools SAT worker parameter nesting. Heavy gates remain unrun.
- Codex review: mirrored release-builder ancestor/deferred-import/lazy-map fixes and non-destructive destination checks with isolated tests.
- Codex review: propagated the CVRP depot-aligned resource-key correction and two-trip regression test into the P-21/P-42 copy.

### Registry, lazy imports and facades (`2026-10-15`)

#### Changed

- **Breaking:** one `Registry` (`logic/src/utils/registry.py`) replaces the five stage registries and the `GlobalRegistry` implementation. Name lookup, `create`, and tag queries (`query_intersection` / `query_union` / `query_difference` / `get_all`) live on the same type. Stage instances are `ROUTING`, `PACKING`, `IMPROVEMENT`, `ACCEPTANCE` and `JOINT`; `ALL` is their union. The old class names (`RoutingAlgorithmRegistry`, `PackingRegistry`, …) are aliases of those instances. `GlobalRegistry` is a thin wrapper over `ALL` so existing `@GlobalRegistry.register(...)` decorators still resolve.
- **Breaking:** `Registry.create` is the only constructor. Factories look up a name and call `create`; they no longer keep their own class maps. Improver aliases (`cls`, `ftsp`, …) and packing name remaps live in the generated manifests / `CONFIG_MAPPING`.
- **Breaking:** parent policy, model, env and interface packages no longer import their implementations eagerly. `from logic.src.policies.routing.meta_heuristics import policy_hgs` still works, but only that module is loaded. Registration is a generated `_manifest.py` of `module:Class` targets (`tools/gen_registry_manifests.py`), not an import of every adapter.
- **Breaking:** `logic.src.envs.problems` is removed. Import problem classes from `logic.src.envs.tasks` (`PTP`, `CPTP`, `TCPTP`).
- **Breaking:** RL expert configs in `logic.src.configs.rl.policies` are renamed to `VectorALNSConfig`, `VectorHGSConfig`, `VectorILSConfig`, `VectorHGSALNSConfig` and `VectorHVPLConfig`. The old names remain as aliases on the package and on each module so Hydra `_target_: logic.src.configs.rl.policies.HGSALNSConfig` checkpoints still load.
- `logic.src.envs.temporal` moved to `logic.src.constants.temporal`. `get_default_temporal_params` is imported from there.

#### Added

- `logic/test/unit/utils/test_registry.py` covers name lookup, `create`, tag queries, Hydra aliases, and the torch-free classical import path.

### Release manifest and package build (`2026-10-15`)

#### Added

- `logic/release.yaml`: the 1.0.0 release manifest, the single source of truth for the released package. It lists the released problems (PTP, CPTP, TCPTP, CVRP), the released registry keys per stage (owner ruling Q-1/Q-2), the infrastructure seeds whose import closure ships (simulator, features, REINFORCE + imitation trainers, AM-GAT, the four envs, the profitable-NN expert, vectorised packing, the CLI), and the excluded packages (both HPO stacks and `rl.meta`, Q-6/Q-12a). `wsmart_bin_analysis` is documented as pending its own package (RQ-grok-7): `bins/base.py` still imports `GridBase` from it, so excluding it today fails the build loudly.
- `logic/package/build_release.py`: builds the package from `main` per the manifest (Q-10a). `--check` validates every manifest key against the generated registry manifests; `--build` stages a pruned copy (filtered registry manifests, released YAMLs only, collection-validated release test set, generated release pyproject and `docs/release.md` index) and verifies that no module-level import or manifest target escapes the tree; `--smoke` runs the torch-free classical smoke (released keys resolve and instantiate, unreleased keys do not, `torch` never enters `sys.modules`).
- Release CI (`.github/workflows/package-and-build.yml` and its `.gitea`/`.forgejo` mirrors, `.gitlab-ci.yml`): manifest check, release build, staged-tree smoke, and a torch-free **installed-package** smoke (`pip install --no-deps` into a fresh venv without torch, Q-16a). The release artifact is uploaded.

#### Changed

- **Breaking:** `logic/pyproject.toml` base dependencies are torch-free; the training stack (torch, torchvision, torchrl, tensordict, torch-geometric, torch-scatter, torch-sparse, pytorch-lightning, wandb) moves to the `neural` extra: `pip install wsmart-route[neural]` (Q-16a). CI syncs `--all-extras`; local training environments use `uv sync --extra neural` (or `--all-extras`). `uv.lock` regenerated.
- **Breaking (import-time only):** eager imports that pulled unreleased or out-of-scope code became lazy, with no API change (all names still resolve; they load on first access): `pipeline/rl/core/__init__.py` and `pipeline/rl/__init__.py` (RL algorithms and the meta stack), `policies/vector/__init__.py` (model policies and vectorised solvers), `models/core/__init__.py`, `models/subnets/{encoders,decoders,modules}/__init__.py` (P-25), `utils/model/loader.py` (encoder factories), `pipeline/rl/core/imitation.py` (expert policies load by config class name), `pipeline/features/train/model_factory/{builder,imitation}.py`, and `controllers/cli/drive_parser.py` (cloud import deferred). `pipeline/features/train/__init__.py` resolves `run_hpo` lazily, so importing the training package no longer requires the HPO stack.
- The release build drops `from .hpo import run_hpo` from the staged `features/train/__init__.py` surface: HPO is not in 1.0.0 (Q-6), and accessing `run_hpo` in the released package raises `ModuleNotFoundError`.

#### Removed

- **Breaking:** the acronym-driven export flow is retired (Q-10a): `ci/packager.py`, `ci/analyze_deps.py`, `ci/export_config.json`, `ci/workflow_config.json`, `ci/export_profiles/` (the `minimal-ptp` patch no longer applied to `main`), `logic/package/prune_codebase.py`, the 16 `logic/package/remove_*.py` scripts, `apply_export_profile.py`, `make_export_profile.py`, `cleanup_helper.py`, and their two tests. The `algo-export*` branch trigger is gone from all CI mirrors. The export tag `export/minimal-20260926` and the `wsr-package` worktree remain for history; the integrator deletes them.

#### Migration

- Release scope questions go in `logic/release.yaml`, not in CI job inputs. Components are addressed by registry key; the BB/BC/BPC acronyms live only in the Studio app's display map (`app/src/gen/config/simulation_metadata.json`).
- The release test set is written to `logic/test/release_tests.txt` (collection-validated) and `release_tests.excluded.txt` (excluded with reasons) inside the built tree. Running it as a CI gate becomes possible once the P-06 test pruning lands: some kept tests assert the dev tree's full registry, which the filtered release registry intentionally violates.

### Config format and leftover names (`2026-10-15`)

#### Changed

- **Breaking:** policy YAML is a plain mapping (`alns.custom.time_limit`), not a list of single-key maps. `tools/convert_policy_yaml.py` converted the 96 `policy_*.yaml` files, the `other/*.yaml` packing/improver/acceptance files, and the SRC/ARCO HPO job keys. Unknown solver keys fail at `OmegaConf.structured` load (`_structured_load` in `base_routing_policy.py`). Stage file-refs (`packing`, `route_improvement`, `acceptance_criteria`) stay as `{other/pk_*.yaml: [variant, …]}` until a later inlining pass. `_flatten_raw_config` and `_flatten_config` are gone; the day loop unwraps engine wrappers with `_as_policy_mapping`.
- **Breaking:** packing leftovers from the packing/routing rename. `ms_*.yaml` is `pk_*.yaml`; `packing/selection_*.py` is `packing/<name>.py`; `selection_factory.py` / `selection_registry.py` are `factory.py` / `registry.py`; `IPackingAlgorithm.select_bins` is `pack`; `SelectionContext` is `PackingContext`; `SearchPhase.SELECTION/CONSTRUCTION` are `PACKING/ROUTING` (string values too); SRC/ARCO `constructors` is `routing_algorithms`; `list_available_ms_strategies` is `list_available_packing_strategies`; paper table identifiers `constructor_labels` / `CONSTRUCTOR_SPEC` / `table_constructors` / `results_constructors.tex` are `routing_*`.
- **Breaking:** packing YAML and dataclasses use `threshold` (and `combined_strategies`) instead of `frequency` / `confidence_factor` / `learned_threshold` / `n_sectors` / `period` / `strategies`. The packing factory no longer remaps those names.

#### Added

- `logic/test/unit/policies/test_structured_policy_load.py` covers structured ALNS load and unknown-key failure.

  - **Look-Ahead selection**: read `logic/src/policies/mandatory_selection/selection_lookahead.py` and replaced slide 7's 2-line LA equation with the full 6-step derivation (flag $\mathcal{M}_0$ → check empty → simulate collection → compute next collection day $nc$ → check $nc\leq t$ → bundle bins overflowing before $nc$), matched 1:1 to `pack`; added a new native flowchart diagram (`lookahead_flow.png`, ports the reference flowchart) illustrating the same 6 phases with the branch/loop structure
### Parameter-usage audit and HPO job keys (`2026-10-15`)

#### Added

- `tools/audit_param_usage.py` (P-55): the corrected three-pattern usage audit — attribute reads, dict-key reads and params.py-internal wiring, over the policy directory first and the whole tree second, for params-only AND shared params/config fields. Verdicts: USED / USED-WIDE / INTERNAL-ONLY / INERT, plus a "solver attr read" column so plumbing-only fields (constructed by a dispatcher, never read by the solver — the released-HGS class) are visible at a glance. No false INERT by construction: the wide scope is searched before any INERT verdict. Generated evidence: `.agent/tasks/simulator-v1-refactor-proposal-2026-10-15-appendix-usage.md` (94 classes, 1,131 fields, 13 INERT, 37 released-set plumbing-only fields). This audit gates the P-10 pass: nothing it marks USED / USED-WIDE / INTERNAL-ONLY may be deleted; INERT fields are deleted with the engine literal promoted to a named constant (owner ruling RQ-mistral-1).
- `tools/check_hpo_jobs.py` (P-56): `--check` requires every HPO job key to resolve to a live field of its policy's config dataclass (or a pipeline-owned section) and every job default to equal the config default; `--convert` rewrites the jobs (file renames, key renames, stale-key drops, default alignment). Wired into the quality job of all four CI files (`.github`/`.gitea`/`.forgejo` `ci.yml`, `.gitlab-ci.yml`). The check turns every future config rename or field removal into a loud CI failure instead of a silently dropped tuning key (the `policy_aco_hh.yaml` precedent).

#### Changed

- **Breaking (HPO jobs):** the 96 simulator HPO job JSONs are converted against the live configs (owner ruling RQ-mistral-2): `adp.json` → `adp_rollout.json` (the job file name is the policy key); stale keys with no live config field are removed (`aks.time_limit_stage_1`, `bc.use_exact_separation`, `bpc.max_cut_iterations`/`use_spatial_partitioning`/`enable_hybrid_search`, `na.model`/`reward`/`vrpp`, `swc_tcf.delta`); job defaults aligned with the config defaults (`alns.start_temp` 100.0 → 0.0, the calibrated default that ran in the paper; `aco_hh.sequence_length`/`operators` → null; `na.seed` → 42). Both HPO stacks remain out of the 1.0.0 release (Q-6); the jobs stay loadable in `main`.

### Released routing configs: one source of truth (WP-04, `2026-10-15`)

#### Changed

- **Breaking:** BPC, SWC-TCF, ACO-HH and CVRP delete their `params.py` layer (P-10). Adapters and engines read the typed config dataclass (`BPCConfig`, `SWCTCFConfig`, `HyperHeuristicACOConfig`, `CVRPConfig`); `XParams.from_config(...)` and the `getattr(cfg, "field", default)` fallbacks are gone. `BaseRoutingPolicy._effective_config(values)` (new) materializes the per-call area overrides and the simulator seed onto the typed config, so the solver's view is one typed object. Engine entry points still accept a plain dict and coerce it with `_structured_load` for direct callers and tests.
- **Breaking:** config defaults settled per field to the values that ran in the paper (P-10 discipline: the policy YAML wins).
  - `BPCConfig`: `cutting_planes` `"rcc"` → `"all"`, `rc_tolerance` `1e-5` → `1e-6`, `profit_aware_operators` `False` → `True`, `max_cg_iterations` `50` → `150`, `max_cuts_per_iteration` `5` → `10`, `max_routes_per_pricing` `5` → `50`, `max_bb_nodes` `1000` → `2000`, `optimality_gap` `1e-4` → `0.005`, `early_termination_gap` `1e-3` → `0.01`, `lr_pre_pruning` `False` → `True`, `lr_max_subgradient_iters` `30` → `40`, `lr_op_time_limit` `3.0` → `5.0`, `lr_warm_start_cg` `False` → `True`, `ng_neighborhood_size` `8` → `12`, `cut_orthogonality_threshold` `0.8` → `0.7`.
  - `SWCTCFConfig`: `framework` `"ortools"` → `"gurobi"`, `time_limit` `600.0` → `60.0`; gains the params-only fields `asymmetric` (`False`) and `no_solution_time_factor` (`3.0`).
  - `HyperHeuristicACOConfig`: `n_ants` `20` → `10`, `rho` `0.1` → `0.5`, `time_limit` `30.0` → `60.0`, `profit_aware_operators` `False` → `True`; gains the params-only field `eta_decay` (`0.5`, pheromone visibility decay).
  - `CVRPConfig`: `seed` `None` → `42` — the deleted `CVRPParams.from_config` upgraded a `None` config seed to `42` (`seed=42 if seed is None else seed`), so 42 is the value that actually ran; `time_limit`/`engine` already matched the YAML (`60.0`/`"ortools"`). The CVRP adapter also no longer bypasses the simulator's per-call values when a typed config exists (`self._config or values`), so an injected simulator seed now reaches the solver as intended; with no injected seed the settled default 42 preserves the paper runs.
- **Breaking:** seeds for BPC / SWC-TCF / ACO-HH stay `None` by default (the deleted params defaults `42` never ran: the simulator injects its seed over the `None` config seed, and the adapters read the injected value). The simulator seeding contract is now the only seed mechanism.
- **Breaking (P-52):** SWC-TCF `gurobi.py` no longer falls back to `SWCTCFParams.<field>` class attributes at solve time (11 sites) and `policy_aco_hh.py` no longer carries its 19 `values.get(..., literal)` definitions; all read the threaded config. `MAX_ARC_DISTANCE_KM` moved from the deleted SWC-TCF `params.py` into `_tcf_data.py` (sole consumer).
- **Breaking (Q-8b, unreleased importers migrated in the same change):** EGH `stage_bpc.py` and LASM `stage_lbbd.py` construct `BPCConfig` instead of `BPCParams`; HVPL (`params.py`, `policy_hvpl.py`) constructs `HyperHeuristicACOConfig` and imports `OPERATOR_NAMES` from `hyper_operators`; the B&B tree helper annotates `BPCConfig`; MS-BPC-SP docstrings reference `MSBPCSPParams`. Tests that built the deleted classes now build the configs (`test_bpc_tree_params.py`, `test_swc_tcf_backends.py`, `test_swc_tcf_paper_model.py`, `test_wave3_hh.py`, `test_aco_hh_fixes.py`, `test_hvpl.py`, `test_cvrp_engines.py`); the params↔config parity assertions are pruned with the params half (P-06).

#### Added

- `tools/check_params_refs.py` (P-52 durable guard): bans references to the deleted `Params` symbols (`BPCParams`, `SWCTCFParams`, `HyperACOParams`, `CVRPParams`) anywhere in `logic/src` outside `configs/` and tests. WP-05/WP-06 extend the list until the last `params.py` is gone. Wired into the quality job of all four CI files.

### Released routing configs: ALNS family and HGS (WP-05, `2026-10-15`)

#### Changed

- **Breaking:** ALNS, HGS, PG-CLNS, PSOMA and SANS delete their `params.py` layer (P-10). Solvers, adapters and dispatchers read the typed config dataclasses (`ALNSConfig`, `HGSConfig`, `PGCLNSConfig`, `PSOMAConfig`, `SANSConfig`); `XParams.from_config(...)` and the `getattr(cfg, "field", default)` fallbacks are gone.
- **Breaking:** Config dataclasses settled per field to paper run values (YAML-wins):
  - `ALNSConfig`: dropped inert `min_removal` (paper literal 4 promoted to `MIN_REMOVAL_LOWER_BOUND` constant in `alns.py`), set `profit_aware_operators=True`, added operator fields (`start_temp_control`, `xi`, `segment_size`, `noise_factor`, `worst_removal_randomness`, `shaw_randomization`, `max_removal_cap`, `regret_pool`, `sigma_1`, `sigma_2`, `sigma_3`).
  - `HGSConfig`: set `time_limit=60.0`, `restart_timer=60.0`, `profit_aware_operators=True`, deleted 8 dead fields (`nb_granular`, `mutation_rate`, `local_search_iterations`, `use_cross_exchange`, `use_lambda_interchange`, `lambda_max`, `use_ejection_chains`, `use_3opt`), added `lambda_param` alias to `n_offspring`.
  - `KSparseACOConfig`: added `q0: float = 0.9` for PG-CLNS / ACO variant parity.
  - `PSOMAConfig`: set `profit_aware_operators=True`, added `sa_steps: int = 5000` and `seed_nearest_neighbour: bool = True`.
  - `SANSConfig`: added `V: float = 0.0`, `shift_duration: float = 28800.0`, `vrpp: bool = True`.
  - `PGCLNSConfig`: dropped `min_removal` from default nested `lns` factory.
- **Breaking:** Fresh per-solve acceptance criteria instances decoupled from declarative configs (P-11); solvers construct `self.acceptance_criterion` via `AcceptanceCriterionFactory` at runtime.
- **Breaking (Q-8b, unreleased consumers migrated in the same change):**
  - Matheuristics: `exact_guided_heuristic/` (`dispatcher.py`, `stage_alns.py`, `params.py`), `learning_allocated_sequential_matheuristic/` (`dispatcher.py`, `stage_lbbd.py`, `params.py`), `partial_optimization_metaheuristic_under_special_intensification_conditions/` (`solver.py`).
  - Metaheuristics: `hybrid_genetic_search_with_adaptive_large_neighborhood_search/` (`params.py`, `policy_hgs_alns.py`), `hybrid_memetic_large_neighborhood_search/` (`params.py`, `policy_hmlns.py`), `hybrid_volleyball_premier_league/` (`params.py`, `policy_hvpl.py`), `adaptive_large_neighborhood_search_with_inter_period_operators/` (`params.py`).
  - Learning heuristics: `reinforcement_learning_hybrid_volleyball_premier_league/` (`params.py`, `policy_rl_hvpl.py`).
- **Tests (P-06):** Pruned `*Params` references from tests (`test_alns_start_temperature.py`, `test_pg_clns_clock.py`, `test_psoma_time_budget.py`, `test_psoma_seeding.py`, `test_policies_aux.py`, `test_ms_ri_ac_yaml_propagation.py`, `test_parameter_loader_independence.py`, `test_sans_base_routing_integration.py`).

#### Added

- `tools/check_params_refs.py`: extended `BANNED_SYMBOLS` to include `ALNSParams`, `HGSParams`, `PGCLNSParams`, `ACOParams`, `LNSParams`, `PSOMAParams`, `SANSParams`.

### Scripted migration of the unreleased policies (`2026-10-15`)

#### Changed

- **Breaking:** the unreleased policies read the typed config dataclass; their `params.py` layer is deleted (Q-8b, WP-06). `tools/migrate_unreleased_params.py` migrated 84 of the 86 remaining params modules in one scripted pass:
  - **Groups A/B/D** (a wired config exists): params-only fields are appended to the config dataclass with their params defaults (shared-field defaults are kept as the config's; the value that runs is the YAML's), and every `XParams` reference is rewritten to the config class.
  - **Group C** (no config anywhere): the params dataclass moved to `logic/src/configs/policies/<key>.py` as `<Key>Config` with `from_config` dropped (new: `LnsMipConfig`, `TpksConfig`).
  - **Group C-unwired** (a config existed but the adapter returned `None`): params-only fields merged into the existing config and the adapter wired — `sa`, `na`, the `mp_*` multi-period family, the hyper-heuristics, `cgh`, `lrh`, `rfo`, `hgs_adc`, `rl_alns` now have typed configs their HPO jobs validate against.
  - Adapter `XParams.from_config(values)` calls become `self._effective_config(values)` (the base filters and structured-loads the same values, with the simulator seed).
- **Breaking:** two config files gained deferred imports instead of import-time cycles: `configs/policies/{ils_rvnd_sp,rl_gd_hh}.py` import the acceptance factory inside `__post_init__` (P-11 relocation is still pending for them). `configs/policies/rl_hvpl.py` references `KSparseACOConfig` from `aco_ks` instead of the deleted `KSACOParams`.
- Two params modules remain as the hand-fix queue (Q-8b decision point): `calm` (six nested dataclasses) and `hmlns` (`MACOParams` + `HybridMemeticLargeNeighborhoodSearchParams`); their references still resolve.

#### Removed

- 84 `params.py` modules under `logic/src/policies/` (−9.7k lines gross). `tools/check_params_refs.py` bans all 95 deleted `*Params` symbols (WP-04/05/06); with `calm` and `hmlns` still to migrate, the final ban is on any `*Params` symbol once they land.

#### Migration

- HPO jobs (P-56 gate): `na.seed` default aligned to the config's `None` (the simulator injects the seed); `sa.iterations_per_temp` dropped — a stale key the config-less SA path never validated, now loud because SA is wired. All 96 jobs check green.
- Cross-params imports (`KSACOParams` in `rl_hvpl`, the aco-ks operator config in eight solvers) now import from `logic.src.configs.policies.*`.

### Envs on the ESPPRC base (WP-07, `2026-10-15`)

#### Added
- **P-23 (Canonical ESPPRC Base & Declarative Resources):** `ESPPRCEnv` in `logic/src/envs/tasks/espprc.py` implementing forward-only routing dynamics, elementarity tracking, and declarative `Resource` hierarchy (`Resource`, `LoadResource`, `ShiftTimeResource`, `ESPPRCGenerator`).
  - Strict batched PyTorch tensor support `[B, ...]` without per-instance loops (RQ-gemini-1).
  - Pure NumPy support for classical heuristic solvers (RQ-codex-1).
  - Explicit customer-count and depot-placement schema support (`[B, 1, 2]` depot + `[B, N, 2]` locs or `[B, N+1, 2]` prepended depot) with road-distance cache.
- **P-21 (Unified Economic Objective):** `ProblemSpec` in `logic/src/envs/tasks/spec.py` and `evaluate_profitable_tour` in `logic/src/envs/tasks/profitable_tour/common.py` providing unified economic objective calculations (EUR), cleanly separating route evaluation/feasibility and physical units.
- **P-44 / Q-3 (Problem Family Hierarchy):** Problem subpackages (`profitable_tour/` containing `ptp`, `cptp`, `tcptp`), self-contained problem generators in task files, and shared `ESPPRCGenerator`.

#### Changed
- **P-20 (Unified Route Evaluator):** Replaced legacy `get_costs` across all 6 live neural caller sites (`pointer_network/model.py`, `gat/decoder.py`, `glimpse/decoder.py`, `eval/engine.py`, `epoch.py`, `neural_agent/batch.py`) with `evaluate_tour`, returning standardized outcome dictionaries (`cost`, `reward`, `length`, `waste`, `overflows`, `valid`, `total`, and optional `time`).
- **P-22 (Embedding Deduplication & Checkpoint Parity):** `CPTPInitEmbedding` subclasses `PTPInitEmbedding`, and `CPTPContextEmbedder` subclasses `PTPContextEmbedder`. Retains 100% bitwise `state_dict` key compatibility and exact greedy action parity with released AM checkpoints.
- **All 12 Problem Tasks Sit on ESPPRC:**
  - Released: `PTP`, `CPTP`, `TCPTP`, `CVRP`
  - Unreleased: `TSP`, `ATSP`, `OP`, `PCTSP`, `SPCTSP`, `PDP`, `IRP`, `ThOP`

#### Removed
- **P-20 (Clean Break):** Deleted legacy `logic/src/envs/routing/` completely (13 legacy files removed).

### Shared objective and KPI contracts (P-21/P-42, `2026-10-15`)

- Added backend-neutral `profit_objective`, `distance_objective`, and injected `Economics`; scalar/NumPy/Torch callers retain batch shape and tensor gradients. ESPPRC reward and profitable-tour/CVRP evaluation use the shared objectives.
- Added Torch-free KPI definitions and `DayOutcome`, day-to-sample aggregation and equal-weight successful-sample summaries. Ratios use per-sample sums; failed samples are excluded and counted. Zero-distance ratio compatibility includes a count of defined ratios. Policy seconds, day wall seconds and physical shift hours stay distinct; simulated and collection days have separate keys.
- Made task exports lazy so importing the shared objective/KPI modules does not require neural dependencies. Released problem specs select canonical simulation metric names; RL reward diagnostics remain separate.
- Fixed WP-07 road-tour evaluation to count every directed depot return/departure and close open paths once; profitable and CVRP evaluators share this calculation. Added regression cases for asymmetric multitrip, open, padded, single-action and empty paths.
- Simulator/result-writer adoption remains in the separately owned WP-08 package; solver internal search scores are not rewritten by this change.

### Modular simulator (WP-08, `2026-10-15`)
Codex P-21/P-42 objectives and KPI primitives (`envs/tasks/common/objectives.py`, `metrics.py`) are on this branch and are what the day loop calls. They were not part of the WP-07 tip.

#### Added

- Day pipeline is a validated dependency graph (`logic/src/pipeline/simulations/pipeline.py`). Each action declares `timed`, `reads`, and `writes`. Fill precedes packing, routing precedes improvement and collection, and collection publishes final `profit` and `cost` before logging.
- Shared fill seed `(seed + sample_id * 1_000_003 + day) % 2**31`. Policy RNG does not reseed `bins.rng`.
- `ProblemSpec.per_trip_shift` (TCPTP only) and `expand_pool`. Released specs load from `released_specs.py` without importing the neural task module. `run_day` publishes `expand_pool` on a `ContextVar`.
- `tools/convert_sim_outputs.py` maps archived `days` to `collection_days`, stamps `days_simulated` from the run config, and drops `reward`. It was not run on the archives.
- `sim.export_fill_matrix` (default off), `sim.pool_start_method` (`forkserver` is Linux opt-in; default stays `spawn`), and `sim.day_pipeline`.

#### Changed

- **Breaking:** day logs drop `reward` and the ambiguous `days` key. Records carry `days_simulated` and `collection_days`. `profit` stays in EUR. RL `reward` tensors are unchanged.
- `get_daily_results` writes a `DayOutcome` for the executed plan. Failed samples keep partial KPIs and stay out of means (`n_failed`). An all-failed cell stores `None` estimates.
- `SimulationDayContext` is no longer a `Mapping`. Dict callers go through `bind_context`.
- Classical bin history and the simulator worker seed import PyTorch only when a tensor device or a neural seed is needed. Pandas stays required.
- `wsmart_bin_analysis` moved to `packages/wsmart_bin_analysis` (import `packages.wsmart_bin_analysis`). `packages/__init__.py` stays absent because `packages/*` is gitignored. GitLab `GIT_SUBMODULE_PATHS` follows the new path. WP-02 owns the release manifest.

#### Removed

- Per-policy `vrpp` bool. Solvers read `current_expand_pool()`. `alns_vrpp` is a different field and remains.

### Packing: NumPy definition and Torch adapter (WP-09, `2026-10-15`)

#### Added
- **P-37 / RQ-codex-1:** one shared definition per released packing rule in `logic/src/policies/packing/spec.py` (NumPy, percent units). Classical wrappers (`last_minute`, `regular`, `service_level`, `lookahead`) call that spec. Batched Torch adapters live in `packing/vectorized/spec.py` (`[B, N]`, depot at 0, fraction units) and are used by the neural selectors. Parity tests keep the two backends in agreement, including Regular.
- **RQ-cursor-3:** unreleased `eoq_last_minute` (`LastMinuteEOQSelection`) keeps the Harris EOQ trigger in `main`. `pk_eoq_last_minute.yaml` now names that key.

#### Changed
- **Breaking:** 1.0.0 `last_minute` is fill-threshold only. `use_eoq_threshold` on a last-minute context is ignored; use `eoq_last_minute` for the EOQ path. Revenue and profit-per-km still read the flag.
- Released packing `pack()` no longer imports the simulator (`_record_live_params`) or Torch. `vectorized/__init__.py` and `get_vectorized_selector` load adapters lazily.

#### Migration
- Replace `strategy: last_minute` plus `use_eoq_threshold: true` with `strategy: eoq_last_minute`. Paper CF70/CF90 last-minute YAML is unchanged.

- **P-05:** `packing/vectorized/manager.py` no longer imports `models.meta.hrl_manager` at module level. `ManagerSelector` is dropped from the packing / vectorized public lazy maps; the factory still string-loads it for in-tree `strategy: manager`. Importing the released vectorized selectors does not load the HRL manager.

### Neural training stack (WP-10, `2026-10-15`)

#### Changed
- **P-50 (batched expert):** the profitable nearest-neighbour expert (`policies/vector/profitable_nearest_neighbour.py`) solves the whole batch at once with NumPy instead of one instance at a time in Python. Its tours are identical to the original's: the nearest-neighbour tie-break, the first-improvement 2-opt order and the float32/float64 arithmetic are kept. Checked on 4,000 synthetic instances and on the Rio Maior 100/170 and Figueira da Foz 350 road matrices (emp and gamma3, 1, 3 and 7 days of waste), with no difference. Per instance, 170 gamma3 goes from 66 ms to 3.6 ms and 350 gamma3 from 280 ms to 15 ms. `test_profitable_nearest_neighbour_parity.py` keeps the original as the reference. The per-instance `_solve` and `_two_opt` methods are replaced by `solve(dms, wastes, revenue, cost)`; no migration is needed outside the expert.

#### Added
- **Q-23 (`rl.imitation.fill_from`):** chooses whose routes empty the bins between time-training days during imitation. `model` (default) keeps the model's sampled rollout, as in the paper models; nothing changes for existing runs. `expert` uses the expert's tours and skips the model's sampled decode in training (P-50 item 1): the step returns the expert's solve, the loss reuses it, and `train/reward`, the rollout metrics and the rollout's `train/log_likelihood`/`entropy` are not logged (`train/expert_reward` and the teacher-forced `train/log_likelihood` are). The skipped decode no longer draws random numbers, so an `expert` run's sampling stream differs from a `model` run's. Validation and test still decode the model. The training step's rollout is now the overridable `StepMixin._rollout(td, env, phase)`.

- **P-51 (expert solve cache):** `rl.imitation.expert_cache: <dir>` keeps the profitable nearest-neighbour expert's tours on disk (`policies/vector/expert_cache.py`, shards `shard-*.pt`, written at the end of each training day). A record's key hashes everything the tour depends on: the distance matrix, the waste of every node, the capacity, the revenue and cost weights, `two_opt_passes` and the expert version. A state that differs in any of them is solved again, so a record is never reused for another state. Only tours are stored; the actions and the reward are rebuilt from the current state on every use, so `weighted_nll` gets the same reward as without the cache. With `fill_from: expert` a rerun or a resumed run reuses every solve of the days it already trained.
- **P-51 (offline expert solves):** `rl.imitation.precompute: true` (with `fill_from: expert` and `expert_cache`) only fills the cache: the training run solves every state with the expert and advances the bins with the expert's tours, but skips the model's training passes. A later training run with the same dataset and settings finds every solve. This replaces the `gen_data expert` command and the `.td` `expert_actions` field proposed in the report, because the cache is keyed by the states that training itself produces (Codex's P-51 correction).
- **P-24 (policy forward contract):** `models/common/policy_contract.py` writes down the forward contract that the released Attention Model, the imitation expert and their consumers (training step, imitation loss, simulator neural agent) already share: `policy(td, env, strategy="sampling"|"greedy", actions=None)` returning a dict with `reward` `[batch]`, `actions` `[batch, steps]` (long, depot-padded), `log_likelihood` `[batch]` (with gradient in training and teacher forcing) and optional `entropy` and `td`. `check_policy_output` validates an output; `test_policy_contract.py` checks the AM in sampling, greedy and teacher-forced mode (teacher forcing reproduces the sampled log-likelihood) and the expert with and without the cache. The calling convention and the dict container are unchanged; the report's `decode_type` default is not adopted (Codex's P-24 correction).

- **P-24:** under teacher forcing the Attention Model returned `entropy` as the integer 0, and a batch already done at reset returned an integer `log_likelihood`; both are now `[batch]` zero tensors. The released checkpoints (Rio Maior 100 emp, Figueira da Foz 350 gamma3) give bitwise-identical greedy actions, rewards and log-likelihoods before and after this package.
- **Q-12 (paper stack only in the import closure):** importing the training and evaluation features now loads only the Attention Model, REINFORCE, imitation and its expert. Before, it loaded every model family in `models/core` (DACT, GLOP, MatNet, MoE, N2S and others), every RL algorithm, `rl.meta`, `rl.hpo` and the vectorised HGS/ALNS/ACO experts. The release build keeps the import closure of its seeds, so these would have shipped (777 loaded `logic.src` modules before, 621 after). `test_paper_stack_imports.py` pins this. Changes:
  - `policies/vector/__init__.py`, `models/core/__init__.py`, `pipeline/rl/__init__.py` and `pipeline/rl/core/__init__.py` are lazy facades (WP-01's `lazy_getattr`). `from <package> import Name` works as before.
  - The training builder's model map, hybrid policy and `MetaRLModule` load on use, and the algorithm factories load by name: `model_factory.registry.get_algorithm_factory(name)`.
  - `features.train.run_hpo` loads on use.
  - The imitation trainer imports only the profitable nearest-neighbour expert eagerly; the other experts load when configured.
  - **Migration:** `RL_ALGORITHM_REGISTRY` and `model_factory.registry._ALGO_REGISTRY` values are now `"module:attribute"` strings; use `get_rl_algorithm(name)` / `get_algorithm_factory(name)` to get the class or factory. `_create_adaptive_imitation` moved to `model_factory/adaptive_imitation.py`.

### Solver consolidation: one engine contract per solver family (WP-11, `2026-10-15`)

#### Changed

- **Breaking (P-30):** Routing solvers take one value object — `solver(problem: SubProblem, cfg)`. The positional 7-argument constructors are gone; `run_*` dispatchers (`run_alns`, `run_hgs`, `run_bpc`, `run_swc_tcf_optimizer`, `run_ms_bpc_sp`) take the same `(problem, cfg)` pair and return the `(routes, profit, cost)` triple (`SolveResult` NamedTuple, unpacking-compatible). Released adapters are `GenericRoutingPolicy` subclasses declaring `_solver` + `_config_class()` only (PSOMA, PG-CLNS, ALNS, HGS, BPC, SWC-TCF, MS-BPC-SP, ACO-HH); adapters with genuinely different behaviour (SANS dispatcher contract, CVRP engine dispatch, the neural NA adapter) stay explicit. The 58 cookie-cutter unreleased adapters are deliberately not converted (released-set-first ruling, RQ-claude-3 pattern); they keep working on `BaseRoutingPolicy` unchanged.
- **Breaking (P-31):** One LNS engine for the released ALNS and the PG-CLNS LNS: new `meta_heuristics/large_neighborhood_search/` package (`LNSBase` loop, `RopkePisingerSegments` and `EmaWeightUpdate` weight strategies, `LegacySAAcceptance` for the pre-P-11 SA trajectory). PG-CLNS keeps its own frozen operator copies (owner ruling 2026-10-07); `ALNSSolver`'s surface (constructor, mutable `destroy_ops`/`repair_ops`, overridable `_update_weights`, `solve(initial_solution=...)`) is preserved for its unreleased consumers.
- **Breaking (P-33):** MS-BPC-SP re-platformed onto the shared column-generation core per RQ-kimi-2: its private CG loop and pricing/cycle helper twins (~1,050 of 2,008 lines) are deleted in favour of `column_generation_loop(..., strict_certification=True)`; `MSBPCSPParams`/`params.py` deleted — `MSBPCSPConfig` (settled to its policy YAML, which matches the values that ran) is the single source of truth; the shared core gained the `strict_certification` flag (MS node-certification semantics: unexhausted Farkas/pricing and skipped descendant pricing leave the node incomplete instead of accepting an uncertified bound; default path byte-stable). Also in the shared helpers: the `rf_conflicts` sign corrected to `not together` (latent — never fed the label-correcting algorithm) and the strong integrality audit (`require_elementary=True`: elementarity + node-usage set-partitioning check) ported into `smoothing.is_solution_integer`, used by MS-BPC-SP. Released BPC stays byte-stable on the default (weak) check; adopting the strong check for BPC is a deferred exactness upgrade pending owner sign-off.
- **Breaking (P-34):** `PopulationSearch` template (`meta_heuristics/population_search.py`: initialise / vary / educate / replace / track-best / two-level stopping hooks) with the released PSOMA ported onto it; HGS/GA/PSO fit the hooks and are deferred.
- **Breaking (P-35):** `HyperHeuristicSearch` route-level HH template (`hyper_heuristics/hyper_heuristic_base.py`: select-move / apply-move / accept / rank / incumbent / post-iteration / selection-state hooks) with the released ACO-HH ported onto it. Public surface preserved: `pheromone` + `deposit_edge`/`evaporate_all` (HVPL, RL-HVPL), `construct`, `tau`, `operator_names`, `sequence_length`, `_select_sequence`, `_elitism_sync_count`; dual RNG streams and their consumption order unchanged (determinism tests green); learning state persists across `solve()` calls.
- **Breaking (Q-8b, unreleased consumers migrated in the same change):** `hybrid_memetic_large_neighborhood_search`, `hybrid_volleyball_premier_league`, `hybrid_genetic_search_with_adaptive_large_neighborhood_search` (HGS-ALNS), `partial_optimization_metaheuristic_under_special_intensification_conditions` (POMSIC), `learning_allocated_sequential_matheuristic` (LASM `stage_lbbd`), `exact_guided_heuristic` (`stage_alns`, `stage_bpc`), `reinforcement_learning_hybrid_volleyball_premier_league` (RL-HVPL), and `logic/benchmark/{benchmark_suite,baseline_benchmarks}.py`.
- **Tests (P-06):** engine/dispatcher constructions across the policy tests moved to `build_subproblem(...)`; the fork-locking MS mock tests re-aimed at the shared loop's `strict_certification` mode; unit tests of the deleted fork's private helpers (`_reset_master_constraints`, `_apply_branching_to_master`, `_perform_strong_branching`, `_compute_lr_bound_at_node`) pruned with the fork — the full-engine brute-force tests cover the same mechanics; adapter patch points moved to the `GenericRoutingPolicy._solver` class attribute.

#### Added

- `routing/base/sub_problem.py`: `SubProblem` value object, `SolveResult` NamedTuple, `Solver` protocol, `build_subproblem(...)` helper.
- `routing/base/generic_policy.py`: `GenericRoutingPolicy` — one generic adapter doing effective-config + sub-problem assembly + solve + finalize behind `_make_subproblem`/`_solve`/`_finalize` hooks.
- `logic/test/unit/policies/solvers/test_bpc_exactness_gate.py`: the RQ-kimi-1 exact-solver gate — incumbent equals the brute-force optimum within 1e-6 and a valid certificate (logged z_UB at/above the incumbent, or exact optimum agreement on trivially-converged instances) for both `run_bpc` and `run_ms_bpc_sp` on small fixtures.
- `tools/check_params_refs.py` ban list extended with `MSBPCSPParams`.

- **Consolidation (coerce_config):** Canonical `coerce_config` lives in `routing/base/sub_problem.py` and is exported through `routing/base`; `large_neighborhood_search/engine.py` re-exports it and `PopulationSearch._coerce_config` delegates to it, eliminating duplication across meta-heuristic engine bases.
- **Fix (HGS local search iterations):** Restored `local_search_iterations: int = 500` to `HGSConfig` with defensive `getattr(self.params, "local_search_iterations", 500)` in `LocalSearch._optimize_internal`, resolving the `TestHGSSolverContract::test_hgs_contract[custom]` contract failure.
- `routing/base/sub_problem.py`: `SubProblem` value object, `SolveResult` NamedTuple, `Solver` protocol, `build_subproblem(...)` helper, `coerce_config(...)` helper.
- **Breaking (P-33):** MS-BPC-SP re-platformed onto BPC plus a set-partitioning stage per RQ-kimi-2. Both engines drive one B&B walk (`search_bpc_tree` in `helpers/solvers_and_matheuristics/search/bpc_tree.py`) and the shared `column_generation_loop`. MS keeps knapsack pre-selection, index remap, Phase 3 regret-2, Phase 5 set-partitioning, and the empty-result retries; its certificates stay on `MSBPCSPConfig` plus `BPCSearchOptions(strict_certification=True, require_elementary=True, incomplete_node_ip=True)` — `MSBPCSPConfig` is not coerced into `BPCConfig`. `MSBPCSPParams`/`params.py` deleted. The shared CG core gained `strict_certification` (MS node-certification: unexhausted Farkas/pricing leaves the node incomplete). Strong integrality (`require_elementary=True`) lives in `smoothing.is_solution_integer` and is used by MS; released BPC stays on the default weak check.
### Regression gates harness, WP-12 #117 (`2026-10-15`)

#### Added

- Q-14 gate harness in `tools/gates/`: `cells.py` (10 gate-(a) cells — each released routing algorithm x last-minute_cf70 x CLS on Rio Maior 100 over 5 days, with per-solver timing-off overrides), `compare.py` (bit-for-bit compare excluding timing/metadata, tolerance and RQ-kimi-1 certificate verdicts, RQ-codex-2 tolerance freezing), `run_gate.py` (`overrides` dry print plus heavy-gated `calibrate`, `gate-a`, `gate-b`, refusing to run before `sim chain done` and on anything but the unchanged baseline code for calibration).
- Deterministic CVRP budget for gate-(a): optional `solution_limit` on `find_routes_ortools` (single search worker, no wall clock; `None` default preserves the `time_limit` path), threaded through `CVRPPolicy` from `CVRPParams.solution_limit` / `CVRPConfig.solution_limit`. Behaviour-preserving: defaults keep current behaviour. Note for WP-04 integration: `CVRPConfig` on the WP-04 branch needs the same one-line field.
- Durable tests: `logic/test/unit/gates/test_gate_compare.py` (15 tests, pass) plus `solution_limit` plumbing/determinism tests in `test_cvrp_engines.py` (added, not run in the worktree: the empty `wsmart_bin_analysis` submodule breaks the `logic.src.policies` import chain there; they run where the submodule exists).

#### Changed

- Nothing behavioural. Heavy calibration and gate runs wait for `sim chain done` (Q-15); tolerances are frozen from unchanged-code repeats before any behaviour-preserving package merges.

### Docs and cleanup (`2026-10-15`, WP-13)

#### Added

- `tools/trim_docstrings.py` (P-03): the mechanical trimmer for the generated docstring boilerplate — `Returns:`/`Yields:` sections that only document None, `Example:` sections that only `>>> import` the module, and `Args:` entries restating the signature's type hints (`name (Type): ...` becomes `name: ...`). One pass over `logic/src`: 333 files trimmed, 589 lines removed, zero parse failures, idempotent on rerun. `--check` is CI-able.
- `tools/gen_api_docs.py` (Q-22): generates `docs/api/` — one deterministic markdown page per package (`configs`, `constants`, `data`, `enums`, `envs`, `interfaces`, `models`, `pipeline`, `policies`, `utils`, `controllers`, …) listing every public module with its docstring summary and public class/function inventory. The output is sorted, so a diff against the committed copy detects drift.
- `docs/api/` (12 package pages + index): the generated API reference. The hand-written `docs/modules/*.md` keep their narrative but carry a banner marking them as not the source of truth (Q-22a: keep the hand-written guides short, keep `docs/moon/` internal).

#### Changed

- **Breaking (doc rule, Q-7b):** both docstring checkers now enforce the relaxed rule instead of the batch-generated style. `check_docstrings.py`: public modules/classes/functions need a docstring; private helpers (leading underscore) and `__init__` (covered by the class docstring) need nothing. `check_google_style.py`: module `Attributes:`/`Example:` and class `Attributes:` sections are optional; an `Args:` section is optional and must not restate type hints; a `Returns:`/`Yields:` section that documents `None` is a defect. The remaining findings under the new rule are pre-existing content debt (127 public functions without docstrings, mostly `enums/registry.py`, `envs/tasks/espprc.py`, and the `wsmart_bin_analysis` submodule), not new regressions — recorded for the per-package review.
- `logic/release.yaml` (WP-02 follow-up): the env seeds follow the WP-07 layout — `envs.routing.*` is gone; the released problems sit on `logic.src.envs.tasks.{ptp,cptp,tcptp,cvrp}` plus the `envs.tasks.espprc` base/generator (P-23, Q-3a), and `envs/generators/` is unreleased-only. `logic.controllers` seeds are the CLI's three modules. Pending notes: add `models.common.policy_contract` when WP-10 integrates (the P-24 contract), and activate the `wsmart_bin_analysis` exclusion when WP-08's `packages/` move lands. The builder also learned WP-03's `pk_*.yaml` packing-config names and the `packing/base/registry.py` rename; `--check`, `--build` (941 files, closure verified) and the torch-free `--smoke` all pass on the merged WP-07 chain.
- `rl4co/` clone deleted from the repository root (Q-21): untracked reference material, 172 MB. WSmart-Route stays an independent fork; nothing in the tree referenced it.

#### Removed

- 589 docstring boilerplate lines across 333 files (the trimmer's first pass).

### WP-13 follow-ups (`2026-10-15`)

- `logic/release.yaml`: `logic.cloud`, `logic.benchmark` and `logic.test` added to `excluded_packages`. The `sync_drive`/`benchmark`/`test_suite` CLI jobs reach them only through deferred imports (`drive_parser.py`, `parser_dispatch.py`, `jobs/ops_runner.py`), and the three packages live outside the staged tops (`src`/`configs`/`controllers`), so they can never ship with the package. Under the reviewed closure semantics (deferred imports are followed) the build fails staged verification through `test_cloud_sync.py`; under the previous semantics they were kept-but-unstaged, a silent defect — the release CLI's `test_suite`/`sync_drive`/`benchmark` jobs would have crashed at runtime. The exclusions declare them opt-out paths (unavailable in the release, matching the builder's documented deferred-import semantics); the curated release test files (P-06) are unaffected, and the staged `logic/test/__init__.py` ships as a package marker with the runner import stripped. Check/build/smoke green on the WP-13 tree under both closure semantics; the verified HEAD release-test selection is preserved.
- `tools/trim_docstrings.py`: the walk now skips the pinned `wsmart_bin_analysis` submodule — its content belongs to its own repository and cannot ship through a superproject commit, so `--apply` would otherwise leave the submodule dirty on every run (the first pass had trimmed 14 of its files; those uncommitted trims were reverted from the worktree copy).

### Packing and routing names (`2026-10-15`)

#### Changed

- **Breaking:** the policy stages are renamed. Mandatory selection strategies are now packing algorithms and route constructors are now routing algorithms (`f6efa50c0`).
  - Packages: `policies/mandatory_selection` is now `policies/packing`, `policies/route_construction` is now `policies/routing`, `policies/selection_and_construction` is now `policies/packing_and_routing`, and `policies/vector/hgs_core` is now `policies/vector/core`.
  - Policy configs: the YAML and dataclass key `mandatory_selection` is now `packing`, so `train.policy.mandatory_selection=null` becomes `train.policy.packing=null`.
  - Classes: `MandatorySelectionConfig` is now `PackingConfig`, `IMandatorySelectionStrategy` is now `IPackingAlgorithm`, `IRouteConstructor` is now `IRoutingAlgorithm`, and `RouteConstructorRegistry` is now `RoutingAlgorithmRegistry`.
  - Tags: `PolicyTag.SELECTION`, `CONSTRUCTION` and `JOINT` are now `PACKING`, `ROUTING` and `PACKING_AND_ROUTING`.
  - Old configs and pickled checkpoints that name the old modules or keys no longer load.
- The docs, app, notebooks, flowcharts, paper table generator and papers use the new terms (`d0a1a2ad9`).
- **Breaking:** the travel-time matrix module `logic/src/data/time` moved to `logic/src/data/processor/temporal.py`. `compute_time_matrix` is now imported from `logic.src.data.processor.temporal`.

### CPTP training on the real network (`2026-10-13` to `2026-10-14`)

#### Fixed

- A CPTP day ends on an explicit stop: a depot visit unloads, choosing the depot from the depot ends the day, and the end is permanent. Before, a day ended only when the whole batch was at the depot, so every instance visited every bin (`233dfad33`).
- Time training keeps the dataset order, so the bins a route collected are emptied in the same rows the next day (`537bd766f`).
- Training instances use the real network (`focus_graph`) and the simulator's vehicle capacity for the area and waste type divided by 100, e.g. 73.68 bins for Rio Maior plastic. The empirical and Gamma samplers draw a new day on each call; before, every day of a multi-day dataset was the same (`f2ddc6909`, `7ed5178f0`).
- Real-network instances are costed in road km from `dm_filepath`, and capacity is derived from the simulator when not set (`7ed5178f0`).
- The CPTP reward is in EUR: `cost_weight` 1.0 per road km and `waste_weight` 27.7 per bin fill (0.5837 EUR/kg x 47.5 kg). This replaces the 9.0 recorded on 2026-10-13 (`4dcdedf4d`).
- `load_model` rebuilt trained policies with the legacy depot projection, so evaluated routes differed from those of the trained network (#104, `a0d09d327`).
- The imitation factories pass the full training setup to the module. They passed a short list without `cfg` and `train_time`, so imitation trained one batch per epoch outside time training (`edcef3544`).

#### Added

- `init_accumulation_days` and `state_refresh_fraction` per curriculum graph: time training starts from 1 to N days of waste and redraws a share of the instances each epoch, so the bin state does not settle at one day's waste (`d9f605599`, `6e619d368`).
- `rl.normalize_advantage` divides the REINFORCE advantage by its batch standard deviation (`d9f605599`).
- Profitable nearest-neighbour expert for imitation (`models/policies@rl.imitation.policy_config=profitable_nn`): a nearest-neighbour tour over the bins with waste on road km, 2-opt, then removal of each bin whose detour costs more than its waste earns; it stays at the depot when the tour does not pay (`edcef3544`).
- `train.policy.model.load_path` starts training from saved policy weights (`edcef3544`).

### CPTP time training, survey paper and test coverage (`2026-10-13`)

#### Fixed

- CPTP training reward balance: `waste_weight` is 9.0 in `logic/configs/envs/cptp.yaml`. The previous 1.0 paid 0.045 per bin against about 0.045 km per bin of travel, so the reward was negative for almost any tour. The value comes from the 50-bin instance: about 20 km per full day against 2.28 units of fill gives about 9 (`96eea4d82`).
- Vehicle capacity on the TensorDict training path is in the same unit as the waste it carries. Waste is a bin-fill fraction (`MAX_WASTE`), while the capacity was the percent default (100), so the route-capacity limit could never bind for CPTP and CVRP training data (`logic/src/data/generators/builders.py`).
- Training checkpoints are named by epoch and step. The filename referenced `val_reward`, which is never logged (the monitor is `val/reward`), so every checkpoint was named `val_reward=0.0000` (`logic/src/pipeline/rl/common/trainer.py`).
- The neural policy comments describe the saved model directory the loader reads: `args.json` and the newest `epoch-*.pt` (`logic/configs/policies/policy_na.yaml`).
- The CVaR selection comment described the VaR <= 0 branch as an upper bound. The code computes CVaR of max(0, X) exactly, and the comment now says so (`logic/src/policies/mandatory_selection/selection_cvar.py`).

#### Added

- Regression tests: capacity and waste share units on the TensorDict path (`logic/test/unit/data/test_builder_capacity_units.py`); CVaR selection threshold against quadrature (`logic/test/unit/policies/mandatory_selection/test_selection_cvar.py`); the 2-opt\* gain equals the real route-cost change (`logic/test/unit/policies/helpers/test_k_opt_star_delta.py`).

#### Changed

- Survey paper branch (submodule, not merged): corrections to the 2-opt\* reconnection, the set-partitioning master, mandatory selection (12 items), neural and routing marks, registered metaheuristic variants, the INFORMS document class, and the author block. Consolidated proposal and paper roadmap in `.agent/tasks/`.

### Combinatorial Optimization survey proposal — 2026-10-13

- Added Codex's round-2 contribution in `.agent/cache/co-survey-round2-shared-2026-10-13.md`: field-wide structure, source-linked content changes, mathematical corrections and revisions to earlier implementation-oriented recommendations.
- Added Gemini's round-2 contribution in `.agent/cache/co-survey-round2-shared-2026-10-13.md`: neural operational taxonomy (§5: L2C, L2P-O, L2P-M, L2I, Neuro-MIP), unified diffusion/consistency comparison box (§6.4, owner #12), four-tier LLM optimization taxonomy (§6.6, owner #11), refutation of the uncertified neural trilemma, and epistemic boundaries on generative models.
- Recorded the findings on the agent bus. This is a proposal under the owner's updated scope; the survey manuscript and code remain unchanged.

### SWC-TCF retries, asymmetric arcs and Drive sync (`2026-10-12`)

#### Added

- `asymmetric: true` for the SWC-TCF paper formulation: directed arcs, each traversed at its own cost, and a load bounded by Q on used arcs. The default (`false`) keeps the undirected published model. On symmetric matrices both give the same optimal profit; on asymmetric matrices the reported profit equals the recomputed profit of the returned route.
- `logic/cloud`: copy-only `rclone` wrapper that synchronises `data/` and `assets/output/` with a Google Drive folder, with `uploader` and `downloader` entry points. Symlinked directories are refused, dry runs are supported, and nothing is deleted on either side.
- `python main.py sync_drive --direction upload|download [--target data|output|tracking|all] [--dry-run]` runs the same sync from the main CLI.

#### Fixed

- SWC-TCF days that Gurobi reports infeasible at the tight feasibility tolerance are re-solved at the default tolerance, and days that reach the time limit without an incumbent are re-solved once with three times the limit. Retries do not chain, and the retry markers are kept in the per-day solver status.
- The time-limit retry multiplier is a per-run setting, `no_solution_time_factor` (default 3), so a run can raise the budget through its policy configuration.

### Research-code integration and tracker reconciliation — 2026-10-04

- Recorded the integrated round-5 fixes: generator import and ARCO entry
  (`1a2a9430e`), old-SANS uncross deadline (`dd113dc89`), ILS/VNS shared
  operators (`ee1c76586`), security utilities (`23a1a1fd2`), DACT/NeuOpt/N2S
  decoders with review amendments (`a840a9370`), and bound-test strengthening
  (`d894bc60f`).
- Reconciled the [issue #61 tracker](../errors/ROADMAP.md) with the resolved
  WCVRP and VRPP decisions (`792193bbb`), preserving historical pass notes.
- Recorded three reviewed patches as pending integration: directory-source
  validation, N2S action documentation, and generator lint cleanup. They are
  not described as shipped fixes.
- Retained the Figueira selection-provenance blocker and missing local 90-day
  archives as rerun limitations. No new experiment results or issue closures
  are implied by this documentation update.


### Added

#### CTOP travel-time matrices and simulation completion (`2026-09-07`)

- Added `logic/src/data/time.compute_time_matrix`: uniform 35 km/h travel
  or ID-aligned, directed CSV travel times, with explicit seconds/minutes/hours
  conversion and validation. Bin-only dashboard files use distance/speed for
  missing depot legs. Missing customer IDs raise an error.
- Extended repository temporal parameter loading to return a travel-time matrix
  when coordinates and distances are supplied, preserving the no-argument
  scalar API. Added simulation shift, speed, service, and graph time-file options.
- CTOP now splits every policy's final route after improvement using both
  capacity and time, preserves depot stops, rejects individually infeasible
  customers, and validates before changing bin contents. Time metrics use the
  same matrix, including return legs and zero service-time overrides.
- Resumed simulations reload time matrices against restored coordinates.
  CTOP environment masks, resource updates, and task evaluation accept directed
  time matrices in hours. Added regression coverage for these paths.
- Saved daily tours retain intermediate depot visits, making split-trip
  capacity and time checks reproducible from the logs.
- Validation: CTOP/simulator/TSP regression tests pass; two-day real-data
  simulations completed with uniform speed and a labelled seconds matrix.
  The broad suite ran with 1,334 passing, three skipped, and four training
  failures (tracking SQLite access and non-OmegaConf configuration inputs).
  Focused lint/type checks pass; repository-wide checks still report unrelated
  lint and typing issues. The standard test-suite wrapper also has a relocated
  pytest launcher issue in this checkout; verification used `uv run python -m pytest`.

#### Shared research, codebase, and manuscript audit (`2026-08-28`)

- Added `.agent/reports/shared/COMPREHENSIVE_REPORT.md` as the collaborative
  evidence-backed report of record for the MPVRPP research, archived
  experiment, framework implementation, and paper. The initial audit includes
  a claim-to-artifact matrix, prioritized amendment ledger, reproducibility
  requirements, and explicit disagreement tracking.
- Verified two new publication blockers from primary artifacts: all 36 archived
  30-day configurations use `sim.n_vehicles: 0`, contradicting the paper's
  single-vehicle description under the code's automatic/unlimited routing
  semantics; and the tracked 174-row 90-day set does not reproduce either a
  per-scenario or global-policy Pareto carry-forward rule.

#### Hypothesis property-based testing suite for bit-packing utilities (`2026-08-27`, §B.1)

- Added `logic/test/properties/test_boolmask_properties.py` using Hypothesis to formally verify `_pad_mask` 8-byte alignment, `_mask_bool2byte` / `_mask_byte2bool` lossless roundtrips, and `mask_bool2long` / `mask_long2bool` bit-packing invariants across arbitrary graph sizes.

#### Comprehensive performance benchmark suite runner (`2026-08-27`, §E.1)

- Added `logic/benchmark/run_all.py` and root recipe `just run-benchmark` executing neural latency/throughput scaling, vectorized local search throughput, and exact/metaheuristic OR solver performance with structured metric logging.
- Fixed CUDA device generator synchronization in `RandomLocalSearchPolicy` and `ImprovementPolicy`.

#### Reproducible paper build target (`2026-08-27`)

- Added `just paper` shorthand recipe to regenerate simulation summary tables, protocol diagrams, network maps, and compile `paper.pdf` in one command (§I.2).

#### Policy & operator review audit refresh (`2026-08-27`, #62)

- Refreshed all four review documents in `docs/moon/review/` against the modern modular directory structure, expanding analysis to 80+ routing algorithms, 85+ operators, 23 standard acceptance criteria, and CTOP temporal/profit duality (§C.8).

### Changed

#### Capacitated Team Orienteering naming (`2026-09-07`)

- Renamed the time-constrained capacitated problem to Capacitated Team
  Orienteering Problem (CTOP), including Python classes, module filenames,
  registries, Hydra configuration, tests, and documentation. Use `ctop` as
  the problem identifier and `envs@train.env=ctop` for training.

#### Paper citation and scholarly-positioning audit (`2026-08-28`)

- Verified the paper's publication-facing references against primary publisher
  records and corrected Wentges, Lysgaard, Barnhart, Sun, Kool, and Ma
  metadata or citation keys. Stable author–year keys now replace the remaining
  generic and hash-like exports, and DOI fields no longer contain nested URLs.
- Positioned the MPVRPP explicitly beside periodic VRP, inventory routing,
  team orienteering, and multi-period profitable routing, with an additional
  good-laboratory-practice anchor for the framework's component-level audit.
- Corrected the regional-network provenance: both panels use OpenStreetMap
  road geometry, while the route matrices use Google Maps for Rio Maior and
  OpenStreetMap for Figueira da Foz. Added contributor and ODbL attribution.
- Withdrew the shared report's RCP-034 finding after the primary Lin et al.
  article confirmed that it explicitly uses Farkas pricing for an infeasible
  restricted master problem. The rebuilt PDF has no undefined citations or
  references; generated files under `Tables/` were not edited.

### Performance

#### GPU peak memory monitoring & profiling (`2026-08-27`, §F.2)

- Added `GPUMemoryMonitor` Lightning callback (`logic/src/pipeline/callbacks/pytorch/gpu_memory_monitor.py`) to reset CUDA peak statistics at epoch start and log `memory/peak_allocated_mb` and `memory/peak_reserved_mb` at epoch end. Auto-registered in `WSTrainer` on CUDA devices.

#### Evaluation engine inference optimization (`2026-08-27`, §F.1)

- Adopted `torch.inference_mode()` across all evaluation evaluators (`GreedyEval`, `SamplingEval`, `AugmentationEval`, `MultiStartEval`, `MultiStartAugmentEval`), eliminating PyTorch version tracking and view mutation overhead during evaluation rollouts.

### Fixed

#### Sym-NCO problem-symmetricity loss was a no-op (`2026-08-27`)

- `SymNCO.shared_step` commented "1. Problem symmetricity loss" but never
  called `problem_symmetricity_loss`; `loss_ps` stayed 0, so the paper's
  instance-augmentation consistency term did not train. Wired the call
  (augmentation axis, dim=1) next to the already-live solution-symmetricity
  term. `train/loss_inv` now logs the computed invariance loss instead of
  the leftover zero tensor. Regression:
  `test_shared_step_includes_problem_symmetricity_loss`.
- `AttentionModel.__init__` default `n_encode_layers` aligned to 3 (Kool
  2019 / `am.yaml`); was 2. Tests that need a thinner net already pass the
  count explicitly.
- `AttentionModelPolicy` default `hidden_dim` aligned to 512 (Kool 2019
  encoder FF width / `am.yaml`); was 128. Integration tests already pass
  128 explicitly.

#### Hydra task aliases (`2026-08-27`)

- The documented `evaluation` and `sim_hpo` CLI aliases now select the real
  Hydra task groups (`eval` and `hpo_sim`) instead of composing a missing
  configuration group and failing before dispatch.

#### Train curriculum configuration (`2026-08-27`)

- Curriculum training now reads and injects graph settings under Hydra's
  task-scoped `train.env`, preventing the silent one-epoch fallback caused by
  looking for an uncomposed root `env` key.

#### Evaluation device validation (`2026-08-27`)

- CPU-only evaluation now reports that multiprocessing requires CUDA instead
  of dividing by a zero CUDA-device count.

#### Data-generation tracking (`2026-08-27`)

- Failed dataset-generation runs are now tagged `failed`, rather than being
  incorrectly recorded as completed.

#### Simulator validation (`2026-08-27`)

- The simulator engine now delegates to the public configuration validator;
  both paths normalize numeric and scientific-notation edge thresholds alike.

#### CTOP feasibility and state-resume review (`2026-08-27`)

- CTOP now validates the full depot-to-depot trip, including a trailing or
  explicit return to the depot. Static scoring and the live action mask use a
  supplied road-distance matrix for travel time, with the same inclusive
  `1e-6` feasibility tolerance.
- CTOP defaults now live in the routing package rather than importing the
  simulation repository during environment construction, which removes an
  import cycle that made `CTOPGenerator` unusable.
- Resetting an initialized CVRPP/CTOP state preserves its remaining capacity
  and time resources, so resumed search keeps the original depot coordinate
  and resource accounting.
- CTOP now resolves to the CVRPP initial, context, state, and edge embedding
  components, allowing neural training and critic construction; the
  drift-report CLI also accepts `--problem ctop`.
- `CTOPEnv` had no `__init__` override, so it inherited `VRPPEnv`'s, which
  always builds a plain `VRPPGenerator`. `get_env("ctop",
  shift_hours=6.5, ...)` (including Hydra-composed overrides) silently
  swallowed the three temporal kwargs in `VRPPGenerator.**kwargs`; they
  never reached the environment and `_reset_instance` fell back to
  `get_default_temporal_params()` regardless of what was requested.
  Confirmed live before the fix (`env.generator` was a `VRPPGenerator`
  with no `shift_hours` attribute at all). Added
  `CTOPEnv.__init__`, building a `CTOPGenerator` and passing it through
  to `VRPPEnv.__init__` (which then skips its own generator construction
  since one is already supplied). Regression test added:
  `test_get_env_factory_builds_a_ctop_generator_not_a_plain_vrpp_one`.

### Added

#### CTOP multi-day simulator support and dual constraint enforcement (`2026-08-27`)

- **Simulation Problem Registration & Validation**: Registered `ctop` across `logic/src/constants/simulation.py`, `logic/src/configs/tasks/sim.py`, and `logic/src/pipeline/features/test/validation.py` / `engine.py`.
- **Temporal Parameter Loading & Simulation Context**: Automated retrieval of working shift duration (`shift_hours`), average vehicle speed (`avg_speed_kmh`), and per-bin service time (`service_time_h`) in `InitializingState`, forwarded across `SimulationDayContext` and `SearchContext`.
- **Dual Constraint Route Construction**: Extended `get_multi_tour` and `BaseRoutingPolicy` to enforce both vehicle waste capacity ($Q$) and shift time budget ($T_{\max}$) with depot returns.
- **Operational Metric & KPI Tracking**: Implemented additive tracking for `time_spent = (raw_km / avg_speed_kmh) + (ncol * service_time_h)` in `CollectAction`, `LogAction`, and summary tables in `logic/src/tracking/logging/modules/analysis.py`.
- **Unit Testing**: Added `logic/test/unit/pipeline/simulations/test_ctop_simulation.py` covering validation, dual constraint splitting, operational time calculation, and shift budget enforcement.

#### Grok joins in place of Opencode; continuous work queued for all three agents (`2026-08-27`)

- Grok replaces Opencode in the agent rotation. `git/messages/opencode_coauthor.msg`
  renamed to `grok_coauthor.msg`; `.agent/bus/AGENT_BUS.md`'s roster table
  updated with current assignments (it had gone stale — Agy's row still said
  "website visual design" while actually on CTOP simulator work). Opencode's
  prior website-interactive lane is left as historical record
  (`.agent/tasks/opencode-website-interactive.md`), not rewritten.
- Issue #60 (CTOP Hydra configs) reassigned Opencode → Grok, same scope.
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

#### Capacitated Team Orienteering Problem (CTOP) — foundation (`2026-08-27`)

- New problem type: CTOP = CVRPP's existing objective and per-trip vehicle
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
- `logic/src/envs/tasks/ctop.py`, `logic/src/envs/routing/ctop.py`,
  `logic/src/envs/generators/ctop.py` (`CTOP(CVRPP)`, `CTOPEnv(CVRPPEnv)`,
  `CTOPGenerator(VRPPGenerator)`), registered across `ENV_REGISTRY`,
  `GENERATOR_REGISTRY`, the `envs.problems` facade, and
  `utils.model.problem_factory.load_problem`. `logic/configs/envs/ctop.yaml`
  added, mirroring `vrpp.yaml`.
- KPI tracking made additive rather than fixed-schema: `cost_dict`/results
  dicts now carry a `"time"` key only when the problem tracks one (ctop),
  tolerated as absent everywhere else — the plug-and-play seam for future
  problem-specific KPIs.
- **Bug found and fixed, independent of CTOP**: `CVRPPEnv._step_instance`
  never called `super()._step_instance()` — `current_node`/`visited`/
  `tour_length`/`tour` never advanced past their reset values for any
  cvrpp rollout. Capacity tracking ran, but the agent's position and the
  episode's own state were frozen after the first action. Confirmed via
  direct before/after rollout test. Any committed cvrpp results predating
  this fix should be treated as suspect.
- Delegated the rest of CTOP: #58 (Codex — adversarial review of the RL-envs
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
  not PDP). Sym-NCO subsequently 4/5 after wiring the dead
  `problem_symmetricity_loss` (see Fixed). Line-check of the PDFs:
  `MATNet.pdf` is a PV-forecasting namesake, not Kwon 2021; `NARGNN.pdf`
  is Li/Chen/Koltun GCN+tree-search while `core/nargnn/` is a TSP heatmap
  (score 2/5 vs that PDF). DR-ALNS Table 1 matches the 7-d state 1:1.
- `bibliography/models/README.md` (#64): filename → actual PDF title → intended
  citation.
- Replaced the two wrong files from arXiv (first pages verified):
  `MATNet.pdf` is Kwon et al. 2021 (arXiv:2106.11113);
  `NARGNN.pdf` is Joshi, Laurent & Bresson 2019 (arXiv:1906.01227).
  Previous copies kept under `bibliography/models/_mismatched/`.
  NARGNN re-scored 4/5 against Joshi (15 GCN layers vs 30, REINFORCE vs
  supervised BCE, greedy vs beam-1280).

#### CTOP Hydra config tree (#60, training/eval side) (`2026-08-27`)

- `logic/configs/envs/ctop.yaml` rewritten to match the landed design
  (`CTOP = CVRPP capacity + independent time budget`). The previous header
  still described the discarded uncapacitated/VRPP-with-time-replacing-
  capacity draft, and the file had no `capacity:` key — `VRPPGenerator`
  would then default to `1.0` instead of CVRPP's `100.0`.
- `EnvConfig` now declares optional `shift_hours` / `avg_speed_kmh` /
  `service_time_h`. `ctop.yaml` lists them as `null` (Python default) so a
  Hydra CLI override such as `train.env.shift_hours=6.5` is a real key
  change rather than a struct-missing error.
- Task YAMLs document the Hydra *group swap* needed to actually load
  `ctop.yaml`: `envs@train.env=ctop`, `envs@eval.env=ctop` (plus the
  hardcoded `eval.env.name` / `eval.problem` lockstep), `envs@hpo.env=ctop`,
  `envs@meta_rl.env=ctop` (plus `meta_rl.env.name`), `envs@hpo_sim.env=ctop`,
  `gen_data data.problem=ctop`, `slurm` `problem=ctop`. `test_sim.yaml` and
  `logic/configs/policies/*.yaml` are untouched — blocked on #59.
- `tracking/*.yaml` have no per-problem metric lists; ctop's additive
  `"time"` KPI is code-side (`eval/engine.py`) and needs no tracking YAML.
- `docs/CONFIGURATION_GUIDE.md` env table and CLI examples now use the
  real group-override syntax (`envs@<task>.env=`) instead of the
  non-composed `envs=vrpp` form.
- After #59 landed: `test_sim.yaml` documents `sim.problem=ctop` as the
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

### CO survey catalogue checkpoint (`2026-10-13`, Codex)

- On isolated paper branch `survey/proposal-2026-10-13`, added the canonical 164-record catalogue, generated five table views, 82-entry bibliography with metadata provenance, citation corrections and provisional abstract (`7e07531`, plus whitespace cleanup).
- Validated a 114-page INFORMS build and catalogue consistency; source/claim review remains open for 88 records. Details and deferred work are in the paper's `WORKLOG.md`.
- No WSmart-Route implementation edits, shared pytest run, push, merge or submodule-pointer update.
