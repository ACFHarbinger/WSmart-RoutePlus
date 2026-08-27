> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## B — Architecture

### §B.1 — Unit Test Coverage Uplift

**Pain**: The CI pipeline enforces a coverage threshold (60%), but 218 test files across `logic/test/` cover primarily the high-level pipeline and environment modules. Core sub-components (masking utilities in `boolmask.py`, tensor-dict protocol, individual attention modules) lack unit-level isolation tests.

**Options**

- **A** — Audit uncovered lines with `coverage report --show-missing`; write targeted parametric tests (`@pytest.mark.parametrize`) for the utility and module layers until coverage reaches 75%. `[Quick Win]`
- **B** — Add mutation testing (`mutmut`) to the CI pipeline to distinguish tests that merely execute code from those that actually detect bugs.
- **C** — Set per-module coverage floors in `.coveragerc` (e.g., `logic/src/utils/` ≥ 80%, `logic/src/models/modules/` ≥ 70%) to prevent regressions in well-tested modules while allowing lower thresholds in exploratory code.
- **D** — Generate property-based tests with Hypothesis for mathematical invariants (e.g., `boolmask` bit-packing lossless roundtrips and padding alignment in `test_boolmask_properties.py`, decoding top-k/top-p invariants, reward scaler normalization invariants). `[Completed 2026-08-27]`

**Recommendation**: **Option C** immediately (configuration change, no new tests needed), then **Option A** to fill gaps. **Option D** `[Delivered]` is a high-value investment for mathematical correctness guarantees.

**Effort × Impact**: Low effort (Options A/C) / High impact

---

### §B.2 — Benchmark Regression CI Gate

**Pain**: The CI only checks code quality and unit correctness. There is no automated performance baseline: a refactor that accidentally degrades inference speed by 30% or increases peak GPU memory will merge silently.

**Options**

- **A** — Add a `benchmark` job to `ci.yml` that runs `pytest --benchmark-only` (pytest-benchmark) on a small fixed dataset; compare against a stored baseline JSON; fail if any metric regresses > 10%.
- **B** — Use `asv` (airspeed velocity) for a more mature benchmark suite with statistical confidence and HTML reports. Higher setup cost.
- **C** — Track benchmark results as GitHub Actions artefacts and comment the delta on every PR using `github-action-benchmark`. `[Quick Win]` for visibility without hard failure gates.
- **D** — Run benchmarks only on `push` to `main` (not on every PR) to keep CI fast; store results in a `gh-pages` branch.

**Recommendation**: **Option C** first (adds visibility with minimal CI cost), then **Option A** to enforce regression gates once baseline values are stable.

**Effort × Impact**: Low effort / High impact

---

### §B.3 — Policy Plugin System

**Pain**: Adding a new classical policy (e.g., a new metaheuristic) requires modifying multiple files: the policy registry, the CLI argument parser, the Studio dropdown list, and the simulation runner. There is no single registration point.

**Options**

- **A** — Define a `@register_policy(name, problem_types)` decorator that writes to a module-level dict in `logic/src/policies/__init__.py`; CLI and Studio query this dict at runtime. `[Quick Win]`
- **B** — Use Python entry points (`pyproject.toml` `[project.entry-points]`) for full plugin isolation; external packages can register policies without modifying the core codebase.
- **C** — Use a YAML-driven policy manifest (`assets/configs/policies.yaml`) that maps names to fully-qualified class paths; load via `importlib`.
- **D** — Use Hydra's `_target_` instantiation pattern (already in use for models) to register and instantiate policies, achieving consistency with the existing config system.

**Recommendation**: **Option D** is the most architecturally consistent choice given that Hydra is already the config backbone. **Option A** is a useful quick bridge while Option D is designed.

**Effort × Impact**: Medium effort / High impact

---

### §B.4 — Structured Logging Consolidation

**Pain**: The codebase has three parallel logging mechanisms: Python's `logging` module with a `logstash` handler and JSON formatter (in `logic/src/tracking/logging/`), `print()` statements scattered throughout model code (380 files mix both), and the simulation's own JSON file output. This makes log aggregation and filtering inconsistent.

**Options**

- **A** — Run `grep -rn "print(" logic/src/ | grep -v test | grep -v "#"` to enumerate all non-test print calls; replace with `logger.debug()` or `logger.info()`. `[Quick Win]`
- **B** — Introduce `structlog` as a unified structured logging backend; all existing `logging.getLogger()` calls are wrapped by a `structlog.BoundLogger`.
- **C** — Add a `LoggingConfig` dataclass to the Hydra config tree controlling per-module log levels, output sinks (file, stdout, logstash), and JSON vs. plain format — without changing any log call sites.
- **D** — Integrate OpenTelemetry tracing for end-to-end span propagation across training → evaluation → simulation pipeline stages.

**Recommendation**: **Option A** is immediate hygiene. **Option C** gives operators control without touching 380 files. **Option B** is the right long-term architecture once the volume justifies it.

**Effort × Impact**: Low–Medium effort / Medium impact

---

### §B.5 — Type Safety Migration: Strict MyPy

**Pain**: MyPy runs in CI but with `continue-on-error: true`, meaning type errors are never blocking. The 379 typed files use inconsistent annotation patterns, and the complex tensor-dict protocol in `logic/src/interfaces/` is only partially typed.

**Options**

- **A** — Enable `--strict` mode for a well-contained subpackage first (`logic/src/utils/`); fix all errors there; expand gradually. Remove `continue-on-error` for that subpackage. `[Quick Win]`
- **B** — Add `py.typed` marker and ship inline type stubs for the `logic` package.
- **C** — Use `pyright` (faster, better PyTorch generics support) alongside MyPy; make pyright the blocking check and MyPy the advisory check (both pyright and pylance can be used as the engine for type checking using Pyrefly).
- **D** — Use `beartype` for runtime type enforcement at public API boundaries (interfaces module). Catches issues that static analysis misses.

**Recommendation**: **Option A** for gradual strictness adoption; **Option D** as a runtime safety net for the interfaces layer where type errors cause silent mathematical bugs.

**Effort × Impact**: Medium effort / High impact

---

### §B.6 — Environment Plugin System (Analogous to §B.3)

**Pain**: Adding a new problem environment (e.g., a new VRP variant) requires modifying `logic/src/envs/problems.py`, the CLI parser, the data generator, and the Studio environment selector — no single registration point.

**Options**

- **A** — Define a `@register_env(name, problem_class)` decorator and a central env registry; CLI/Studio consult it at startup.
- **B** — Use Hydra `_target_` pattern: each env is a config group entry under `conf/env/`, instantiated via `hydra.utils.instantiate()`. Fully consistent with existing model instantiation.
- **C** — Define a `ProblemManifest` dataclass that each env module exports; a loader discovers them via `importlib.metadata`.

**Recommendation**: **Option B** — already the pattern for models; extending it to environments achieves full symmetry across the config system.

**Effort × Impact**: Medium effort / High impact

---

### §B.7 — Circular Import Prevention

**Pain**: With 1,825 Python files, implicit inter-module dependencies are likely. Circular imports surface at runtime as `ImportError` or `AttributeError` and are hard to track down post-hoc.

**Options**

- **A** — Add `pydeps` to CI (`uv run pydeps logic/src --max-bacon 3 --no-show`) to generate a dependency graph; fail if cycles are detected. `[Quick Win]`
- **B** — Enforce import order via `isort` + `ruff` rules `I` (already partially configured); add a custom `ruff` rule that flags cross-layer imports (logic → gui).
- **C** — Introduce `__all__` definitions in every `__init__.py` to make the public surface explicit and prevent accidental internal imports.

**Recommendation**: **Option A** for automated detection, **Option B** for prevention. Both are low-cost additions to CI.

**Effort × Impact**: Very Low effort / Medium impact

---

### §B.8 — Async Task & Worker Standardization

**Pain**: The existing PySide6 GUI background workers (`data_loader_worker.py`, `chart_worker.py`, `file_tailer_worker.py`) each implement `QThread` independently with inconsistent error propagation, progress signal patterns, and cancellation logic. During the Tauri migration (§G), these Qt-specific workers are replaced; however, the Python logic layer still spawns background operations (data loading, simulation orchestration, training) that need consistent cancellation and progress-reporting contracts.

**Options**

- **A** — For the transitional PySide6 GUI: define a `BaseWorker(QThread)` in `gui/src/helpers/base_worker.py` with: `progress = Signal(int)`, `error = Signal(str)`, `result = Signal(object)`, `_cancelled: bool`, and a `cancel()` method. Subclasses override `run_task()`. Superseded once §G Phase 15 is complete. `[Quick Win]`
- **B** — For the Tauri backend: define a Rust `AsyncTask` trait with `run()`, `cancel()`, and a `progress_channel: Sender<f32>`. All long-running Rust commands implement this trait; progress events are forwarded to the frontend via Tauri's event system.
- **C** — For the Python logic layer: introduce a `BackgroundTask` protocol class with `run()`, `cancel()`, `on_progress(callback)` methods used consistently across simulation, training, and data generation entry points. The Tauri backend's Rust layer calls the Python subprocess and receives structured progress lines from stdout.
- **D** — Use Python `concurrent.futures.ThreadPoolExecutor` managed by a Rust-aware bridge class that maps futures to Tauri async commands.

**Recommendation**: **Option A** for the PySide6 transitional period, **Option B + C** for the Tauri architecture. The Rust trait (B) standardizes the Tauri command layer; the Python protocol (C) standardizes the subprocess-facing API so Rust can stream progress without format-specific parsing.

**Effort × Impact**: Low effort (Option A) / Medium effort (B + C) / High impact

---

### Effort × Impact Matrix — Architecture

| Item                                                    | Effort   | Impact | Priority                  |
| ------------------------------------------------------- | -------- | ------ | ------------------------- |
| §B.7 Option A (pydeps CI)                               | Very Low | Medium | P0 `[Quick Win]`          |
| §B.1 Option C (per-module coverage floors)              | Very Low | High   | P0 `[Quick Win]`          |
| §B.4 Option A (remove print() calls)                    | Low      | Medium | P0 `[Quick Win]`          |
| §B.8 Option A (BaseWorker, transitional)                | Low      | Medium | P1 `[Quick Win]`          |
| §B.5 Option A (strict MyPy, utils subpackage)           | Medium   | High   | P1                        |
| §B.2 Option C (benchmark visibility)                    | Low      | High   | P1                        |
| §B.8 Option B+C (Tauri async trait + Python protocol)   | Medium   | High   | P2 `[Blocked]` §G Phase 0 |
| §B.3 Option D (Hydra policy plugin)                     | Medium   | High   | P2                        |
| §B.6 Option B (Hydra env plugin)                        | Medium   | High   | P2                        |
| §B.1 Option D (Hypothesis property tests)               | High     | High   | P2 `[Research]`           |
| §B.2 Option A (benchmark regression gate)               | Medium   | High   | P2                        |

---

